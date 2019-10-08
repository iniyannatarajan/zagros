#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pyrap.tables as pt

from vardefs import *
from priors import Priors
from africanus.rime.cuda import phase_delay, predict_vis
from africanus.coordinates import radec_to_lm
from africanus.model.coherency.cuda import convert # convert between correlations and Stokes parameters
from africanus.model.shape import gaussian as gaussian_shape

# Global variables related to input data
data_vis = None # variable to hold input data matrix
data_uvw = None
data_uvw_cp = None
data_ant1 = None
data_ant2 = None
data_inttime = None
data_flag = None
data_flag_row = None

data_nant = None
data_nbl = None
data_uniqtimes = None
data_ntime = None
data_uniqtime_indices = None

data_nchan = None
data_chanwidth = None
data_chan_freq=None # Frequency of each channel. NOTE: Can handle only one SPW.
data_chan_freq_cp=None # Frequency of each channel. NOTE: Can handle only one SPW.

# Global variables to be computed / used for bookkeeping
baseline_dict = None # Constructed in main()
init_loglike = False # To initialise the loglike function
ndata_unflgged = None
per_bl_sig = None
weight_vector = None
einschema = None

# Other global vars that will be set through command-line
hypo = None
#npsrc = None
#ngsrc = None

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms", help="Input MS name")
    p.add_argument("col", help="Name of the data column from MS")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Necessary to compare"
                        "codex vis against MeqTrees-generated vis")
    p.add_argument('--hypo', type=int, choices=[0,1,2], required=True)
    #p.add_argument('--npsrc', type=int, required=True)
    #p.add_argument('--ngsrc', type=int, required=True)
    p.add_argument('--npar', type=int, required=True)
    p.add_argument('--basedir', type=str, required=True)
    p.add_argument('--fileroot', type=str, required=True)
    return p

def pol_to_rec(amp, phase):
    """
    Converts a complex number from polar to cartesian coordinates
    Parameters
    ----------
    amp: Amplitude of a complex number
    phase: phase of a complex number in degrees

    Returns
    -------
    re, im: Real and imaginary parts of a complex number

    """
    re = amp*np.cos(phase*np.pi/180.0)
    im = amp*np.sin(phase*np.pi/180.0)
    return re, im

# INI: Flicked from MeqSilhouette
def make_baseline_dictionary(ant_unique):
    return dict([((x, y), np.where((data_ant1 == x) & (data_ant2 == y))[0]) for x in ant_unique for y in ant_unique if y > x])

# INI: For handling different correlation schema; not used as of now
def corr_schema():
    """
    Parameters
    ----------
    None

    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    corrs = pol.NUM_CORR.values
    corr_types = pol.CORR_TYPE.values

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)

def einsum_schema(hypo):
    """
    Returns an einsum schema suitable for multiplying per-baseline
    phase and brightness terms.
    Parameters
    ----------
    None

    Returns
    -------
    einsum_schema : str
    """
    corrs = data_vis.shape[2]

    if corrs == 4:
        if hypo == 0:
            return "srf, sij -> srfij"
        elif hypo == 1:
            return "srf, srf, sij -> srfij"
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)

def loglike(theta):
    """
    Compute the loglikelihood function.
    NOTE: Not called directly by user code; the function signature must
          correspond to the requirements of the numerical sampler used.
    Parameters
    ----------
    theta : Input parameter vector

    Returns
    -------
    loglike : float
    """

    global init_loglike, ndata_unflagged, per_bl_sig, weight_vector, data_vis, einschema

    if init_loglike == False:

        # Find total number of visibilities
        ndata = data_vis.shape[0]*data_vis.shape[1]*data_vis.shape[2]*2 # 8 because each polarisation has two real numbers (real & imaginary)
        flag_ll = np.logical_not(data_flag[:,0,0])
        ndata_unflagged = ndata - np.where(flag_ll == False)[0].shape[0] * 8
        print ('Percentage of unflagged visibilities: ', ndata_unflagged, '/', ndata, '=', (ndata_unflagged/ndata)*100)

        # Set visibility weights
        weight_vector=np.zeros(data_vis.shape, dtype='float') # ndata/2 because the weight_vector is the same for both real and imag parts of the vis.
        if not sigmaSim:
            per_bl_sig = np.zeros((data_nbl))
            bl_incr = 0;
            for a1 in np.arange(data_nant):
              for a2 in np.arange(a1+1,data_nant):
                #per_bl_sig[bl_incr] = np.sqrt((sefds[a1]*sefds[a2])/(data_chanwidth*data_inttime[bl_incr])) # INI: Removed the sq(2) from the denom. It's for 2 pols.
                per_bl_sig[bl_incr] = (1.0/corr_eff) * np.sqrt((sefds[a1]*sefds[a2])/(2*data_chanwidth*data_inttime[bl_incr])) # INI: Added the sq(2) bcoz MeqS uses this convention
                weight_vector[baseline_dict[(a1,a2)]] = 1.0 / np.power(per_bl_sig[bl_incr], 2)
                bl_incr += 1;
        else:
            weight_vector[:] = 1.0 /np.power(sigmaSim, 2)

        weight_vector *= np.logical_not(data_flag)
        weight_vector = cp.array(weight_vector.reshape((data_vis.shape[0], data_vis.shape[1], 2, 2)))

        # Compute einsum schema
        einschema = einsum_schema(hypo)

        init_loglike = True # loglike initialised; will not enter on subsequent iterations

    # Set up arrays necessary for forward modelling
    # Set up the phase delay matrix
    lm = cp.array([[theta[1], theta[2]]])
    phase = phase_delay(lm, data_uvw_cp, data_chan_freq_cp)

    if hypo == 1:
        # Set up the shape matrix for Gaussian sources
        gauss_shape = gaussian_shape(data_uvw, data_chan_freq, np.array([[theta[3], theta[4], theta[5]]]))
        gauss_shape = cp.array(gauss_shape)

    # Set up the brightness matrix
    stokes = cp.array([[theta[0], 0, 0, 0]])
    brightness =  convert(stokes, ['I', 'Q', 'U', 'V'], [['RR', 'RL'], ['LR', 'LL']])

    '''print ('einschema: ', einschema)
    print ('phase.shape: ', phase.shape)
    print ('gauss_shape.shape: ', gauss_shape.shape)
    print ('brightness.shape: ', brightness.shape)'''

    # Compute the source coherency matrix (the uncorrupted visibilities, except for the phase delay)
    if hypo == 0:
        source_coh_matrix =  cp.einsum(einschema, phase, brightness)
    elif hypo == 1:
        source_coh_matrix =  cp.einsum(einschema, phase, gauss_shape, brightness)

    #print('srccoh shape: ', source_coh_matrix.shape)

    ### Uncomment the following and assign sampled complex gains per ant/chan/time to the Jones matrices
    # Set up the G-Jones matrices
    die_jones = cp.zeros((data_ntime, data_nant, data_nchan, 2, 2), dtype=cp.complex)
    if hypo == 0:
        for ant in np.arange(data_nant):
          for chan in np.arange(data_nchan):
              delayterm = theta[ant+12]*(chan-refchan_delay)*data_chanwidth # delayterm in 'turns'; 17th chan (index 16) freq is the reference frequency.
              pherr = theta[ant+3] + delayterm*360 # convert 'turns' to degrees; pherr = pec_ph + delay + rate; rates are zero
              re, im = pol_to_rec(1,pherr)
              die_jones[:, ant, chan, 0, 0] = die_jones[:, ant, chan, 1, 1] = re + 1j*im
    elif hypo == 1:
        for ant in np.arange(data_nant):
          for chan in np.arange(data_nchan):
              delayterm = theta[ant+15]*(chan-refchan_delay)*data_chanwidth # delayterm in 'turns'; 17th chan (index 16) freq is the reference frequency.
              pherr = theta[ant+6] + delayterm*360 # convert 'turns' to degrees; pherr = pec_ph + delay + rate; rates are zero
              re, im = pol_to_rec(1,pherr)
              die_jones[:, ant, chan, 0, 0] = die_jones[:, ant, chan, 1, 1] = re + 1j*im
              
    # Predict (forward model) visibilities
    # If the die_jones matrix has been declared above, assign it to both the kwargs die1_jones and die2_jones in predict_vis()
    model_vis = predict_vis(data_uniqtime_indices, data_ant1, data_ant2, die1_jones=die_jones, dde1_jones=None, source_coh=source_coh_matrix, dde2_jones=None, die2_jones=die_jones, base_vis=None)

    # Compute chi-squared and loglikelihood
    diff = model_vis - data_vis.reshape((data_vis.shape[0], data_vis.shape[1], 2, 2))
    chi2 = cp.sum((diff.real*diff.real+diff.imag*diff.imag) * weight_vector)
    loglike = cp.float(-chi2/2.0 - cp.log(2*cp.pi*(1.0/weight_vector.flatten()[cp.nonzero(weight_vector.flatten())])).sum())

    return loglike, []

#------------------------------------------------------------------------------
pri=None
def prior_transform(hcube):
    """
    Transform the unit hypercube into the prior ranges and distributions requested (refer to Nested sampling papers).
    NOTE: Not called directly by user code; the function signature must
          correspond to the requirements of the numerical sampler used.
    Parameters
    ----------
    hcube: Input hyercube

    Returns
    -------
    theta: Sampled parameter vector

    """

    global pri;
    if pri is None: pri=Priors()

    theta = []

    if hypo == 0: # 21 parameters
        theta.append(pri.GeneralPrior(hcube[0],'U',Smin,Smax))
        theta.append(0)#pri.GeneralPrior(hcube[1],'U',dxmin,dxmax))
        theta.append(0)#pri.GeneralPrior(hcube[2],'U',dymin,dymax))
        theta.append(pri.GeneralPrior(hcube[3],'U', phasemin, phasemax))
        theta.append(pri.GeneralPrior(hcube[4],'U', phasemin, phasemax))
        theta.append(0) # referenced to the third station (LMT) by default
        for ant in range(6,6+(data_nant-3)):
            theta.append(pri.GeneralPrior(hcube[ant], 'U', phasemin, phasemax))
        theta.append(pri.GeneralPrior(hcube[12], 'U', delaymin, delaymax))
        theta.append(pri.GeneralPrior(hcube[13], 'U', delaymin, delaymax))
        theta.append(0) # referenced to the third station (LMT) by default
        for ant in range(15,15+(data_nant-3)):
            theta.append(pri.GeneralPrior(hcube[ant],'U', delaymin, delaymax))

    elif hypo == 1: # 24 parameters
        theta.append(pri.GeneralPrior(hcube[0],'U',Smin,Smax))
        theta.append(0)#pri.GeneralPrior(hcube[1],'U',dxmin,dxmax))
        theta.append(0)#pri.GeneralPrior(hcube[2],'U',dymin,dymax))
        theta.append(pri.GeneralPrior(hcube[3],'U',e1min,e1max))
        theta.append(pri.GeneralPrior(hcube[4],'U',e2min,theta[3])) # ALWAYS LESS THAN theta[3]
        theta.append(pri.GeneralPrior(hcube[5],'U',pamin,pamax))
        theta.append(pri.GeneralPrior(hcube[6],'U', phasemin, phasemax))
        theta.append(pri.GeneralPrior(hcube[7],'U', phasemin, phasemax))
        theta.append(0) # referenced to the third station (LMT) by default
        for ant in range(9,9+(data_nant-3)):
            theta.append(pri.GeneralPrior(hcube[ant], 'U', phasemin, phasemax))
        theta.append(pri.GeneralPrior(hcube[15], 'U', delaymin, delaymax))
        theta.append(pri.GeneralPrior(hcube[16], 'U', delaymin, delaymax))
        theta.append(0) # referenced to the third station (LMT) by default
        for ant in range(18,18+(data_nant-3)):
            theta.append(pri.GeneralPrior(hcube[ant],'U', delaymin, delaymax))

    else:
        print('*** WARNING: Illegal hypothesis')
        return None

    return theta
#------------------------------------------------------------------------------

def main(args):

    global hypo, data_vis, data_uvw, data_uvw_cp, data_nant, data_nbl, data_uniqtimes, data_uniqtime_indices, data_ntime, data_inttime, \
            data_chan_freq, data_chan_freq_cp, data_nchan, data_chanwidth, data_flag, data_flag_row, data_ant1, data_ant2, baseline_dict

    # Set command line parameters
    hypo = args.hypo
    #npsrc = args.npsrc
    #ngsrc = args.ngsrc

    ####### Read data from MS
    tab = pt.table(args.ms).query("ANTENNA1 != ANTENNA2"); # INI: always exclude autocorrs; this code DOES NOT work for autocorrs
    data_vis = tab.getcol(args.col)
    data_ant1 = tab.getcol('ANTENNA1')
    data_ant2 = tab.getcol('ANTENNA2')
    ant_unique = np.unique(np.hstack((data_ant1, data_ant2)))
    baseline_dict = make_baseline_dictionary(ant_unique)

    # Read uvw coordinates; nececssary for computing the source coherency matrix
    data_uvw = tab.getcol('UVW')
    if args.invert_uvw: data_uvw = -data_uvw # Invert uvw coordinates for comparison with MeqTrees

    # get data from ANTENNA subtable
    anttab = pt.table(args.ms+'/ANTENNA')
    stations = anttab.getcol('STATION')
    data_nant = len(stations)
    data_nbl = int((data_nant*(data_nant-1))/2)
    anttab.close()

    # Obtain indices of unique times in 'TIME' column
    data_uniqtimes, data_uniqtime_indices = np.unique(tab.getcol('TIME'), return_inverse=True)
    data_ntime = data_uniqtimes.shape[0]
    data_inttime = tab.getcol('EXPOSURE', 0, data_nbl)

    # Get flag info from MS
    data_flag = tab.getcol('FLAG')
    data_flag_row = tab.getcol('FLAG_ROW')
    data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

    tab.close()

    # get frequency info from SPECTRAL_WINDOW subtable
    freqtab = pt.table(args.ms+'/SPECTRAL_WINDOW')
    data_chan_freq = freqtab.getcol('CHAN_FREQ')[0]
    data_nchan = freqtab.getcol('NUM_CHAN')[0]
    data_chanwidth = freqtab.getcol('CHAN_WIDTH')[0,0];
    freqtab.close();

    # Move necessary arrays to cupy from numpy
    data_vis = cp.array(data_vis)
    data_ant1 = cp.array(data_ant1)
    data_ant2 = cp.array(data_ant2)
    data_uvw_cp = cp.array(data_uvw)
    data_uniqtime_indices = cp.array(data_uniqtime_indices, dtype=cp.int32)
    data_chan_freq_cp = cp.array(data_chan_freq)

    '''# Set up pypolychord
    settings = PolyChordSettings(args.npar, 0)
    settings.base_dir = args.basedir
    settings.file_root = args.fileroot
    settings.nlive = nlive
    settings.num_repeats = num_repeats
    settings.precision_criterion = evtol
    settings.do_clustering = False # check whether this works with MPI
    settings.read_resume = False
    settings.seed = seed    

    ppc.run_polychord(loglike, args.npar, 0, settings, prior=prior_transform)'''

    # Make a callable for running PolyChord
    my_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(loglike, prior_transform, args.npar)

    settings_dict = {'file_root': args.fileroot,
                     'base_dir': args.basedir,
                     'seed': seed}
    
    comm = MPI.COMM_WORLD

    # Run dyPolyChord
    dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict, ninit=nlive_init, nlive_const=nlive, comm=comm)

    return 0

if __name__ == '__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
