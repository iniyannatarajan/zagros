# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pyrap.tables as pt

from vardefs import *
from priors import Priors
from africanus.rime.cuda import phase_delay, predict_vis

# Global variables related to input data
data_vis = None # variable to hold input data matrix
data_nant=None
data_nbl=None
data_inttime=None
data_uniqtime_index=None
data_nchan=None
data_chanwidth=None
data_flag=None
data_flag_row=None
data_ant1=None
data_ant2=None

# Global variables to be computed / used for bookkeeping
init_loglike=False # To initialise the loglike function
ndata_unflgged=None
baselinenoise = None
weight_vector=None

# Other global vars that will be set through command-line
hypo=None
npsrc=None
ngsrc=None

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms", help="Input MS name")
    p.add_argument("col", help="Name of the data column from MS")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Necessary to compare"
                        "codex vis against MeqTrees-generated vis")
    p.add_argument('--hypo', type=int, choices=[0,1,2], required=True)
    p.add_argument('--npsrc', type=int, required=True)
    p.add_argument('--ngsrc', type=int, required=True)
    p.add_argument('--npar', type=int, required=True)
    p.add_argument('--basedir', type=str, required=True)
    p.add_argument('--fileroot', type=str, required=True)

    return p

def pol_to_rec(amp, phase):
    re = amp*np.cos(phase*np.pi/180.0)
    im = amp*np.sin(phase*np.pi/180.0)
    return re, im

def loglike(theta):
    """
    Compute the loglikelihood function
    """

    if init_loglike == False:

        # Find total number of visibilities
        ndata = data_vis.shape[0]*data_vis.shape[1]*data_vis.shape[2]*2 # 8 because each polarisation has two real numbers (real & imaginary)
        flag_ll = np.logical_not(data_flag[:,0,0])
        ndata_flagged = np.where(flag_ll == False)[0].shape[0] * 8
        ndata_unflagged = ndata - ndata_flagged
        print 'Number of unflagged visibilities: ', ndata, '-', ndata_flagged, '=', ndata_unflagged

        # Set visibility weights
        if use_weight_vector:
            weight_vector=np.zeros(ndata/2) # ndata/2 because the weight_vector is the same for both real and imag parts of the vis.
            baselinenoise = np.zeros((data_nbl))
            basecount = 0;
            for i in np.arange(data_nant):
              for j in np.arange(i+1,data_nant):
                sefd = np.sqrt(sefdlist[i]*sefdlist[j]);
                #baselinenoise[basecount] = sefd/math.sqrt(data_chanwidth*data_inttime[basecount]) #INI:Removed the sq(2) from the denom. It's for 2 pols.
                baselinenoise[basecount] = sefd/math.sqrt(2*data_chanwidth*chanwid*data_inttime[basecount]) #INI:ADDED the sq(2) bcoz MeqS uses this convention
                basecount += 1;

            for ibl in np.arange(data_nbl):
                weight_vector[:,:,ibl,:] = 1.0 / (baselinenoise[ibl]*baselinenoise[ibl])



    return loglike, []

#------------------------------------------------------------------------------
pri=None
def prior_transform(hcube):
    """
    Transform the unit hypercube into the prior ranges and distributions requested
    """

    global pri;
    if pri is None: pri=Priors()

    theta = []

    if hypo == 0:
        theta.append(pri.GeneralPrior(hcube[0],'U',Smin,Smax))
        theta.append(pri.GeneralPrior(hcube[1],'U',dxmin,dxmax))
        theta.append(pri.GeneralPrior(hcube[2],'U',dymin,dymax))

    else:
        print('*** WARNING: Illegal hypothesis')
        return None

    return theta
#------------------------------------------------------------------------------

def main(args):

    global data_vis, dta_nant, data_nbl, data_timearr, data_ntime, data_inttime, data_nchan, data_chanwidth, data_flag, data_flag_row, data_ant1, data_ant2

    ####### Read data from MS
    tab = pt.table(args.ms).query("ANTENNA1 != ANTENNA2"); # INI: always exclude autocorrs; this code DOES NOT work for autocorrs
    data_vis = tab.getcol(args.col)

    # Create baseline noise array ------------------------

    anttab = pt.table(args.ms+'/ANTENNA')
    stations = anttab.getcol('STATION')
    data_nant = len(stations)
    data_nbl = int((data_nant*(data_nant-1))/2)
    anttab.close()
 
    _, data_uniqtime_index = np.unique(tab.getcol('TIME'), return_inverse=True) # INI: Obtain the indices of all the unique times
    data_ntime = data_timearr.shape[0]
    data_inttime = tab.getcol('EXPOSURE', 0, data_nbl) # jan26, for VLBA sims

    # get channel width
    freqtab = pt.table(args.ms+'/SPECTRAL_WINDOW')
    data_chanwidth = freqtab.getcol('CHAN_WIDTH')[0,0];
    data_nchan = freqtab.getcol('NUM_CHAN')[0]
    freqtab.close();

    # get flags from MS
    data_flag = tab.getcol('FLAG')
    data_flag_row = tab.getcol('FLAG_ROW')
    data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

    # Set up arrays necessary for predict_vis
    data_ant1 = tab.getcol('ANTENNA1')
    data_ant1 = tab.getcol('ANTENNA1')

    tab.close()

    # Set up pypolychord
    settings = PolyChordSettings(args.npar, 0)
    settings.base_dir = args.basedir
    settings.file_root = args.fileroot
    settings.nlive = nlive
    settings.num_repeats = num_repeats
    settings.precision_criterion = evtol
    settings.do_clustering = False # check whether this works with MPI
    settings.read_resume = False
    settings.seed = seed    

    #ppc.run_polychord(loglike, args.npar, 0, settings, prior=prior_transform)
    loglike(0)

    return 0

if __name__ == '__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
