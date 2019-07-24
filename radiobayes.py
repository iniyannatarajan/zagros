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
data_nbl=None
data_timearr=None
data_ntime=None
data_inttime=None
data_nchan=None
data_chanwidth=None
data_flag=None
data_flag_row=None
data_ant1=None
data_ant2=None

# Global variables to be computed
baselinenoise = None

#other global vars that will be set through command-line
hypo=None
npsrc=None
ngsrc=None
ndata_valid=None


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

    # Find total number of visibilities
    ndata = 8*data_ntime*data_nbl*data_nchan # 8 because each polarisation has two real numbers (real & imaginary)
    flag_ll = np.logical_not(data_flag[:,0,0])
    number_flagged = np.where(flag_ll == False)[0].shape[0] * 8
    ndata_valid = ndata - number_flagged
    print 'Number of valid visibilities: ', ndata,'-',number_flagged,'=',ndata_valid

    # Set visibility weights
    if use_weight_vector:
        weight_vector=np.zeros(ndata/2).astype(slvr.ft)\
            .reshape(slvr.weight_vector_shape) # ndata/2.0 because each vis. is (real+i*imag); so now, the array has ntime*nbl*nchan*4 elem. 
        baselinenoise = np.zeros((slvr.nbl))
        basecount = 0;
        for i in np.arange(slvr.na):
          for j in np.arange(i+1,slvr.na):
            sefd = np.sqrt(sefdlist[i]*sefdlist[j]);
            #baselinenoise[basecount] = sefd/math.sqrt(chanwid*dt_val[basecount]) #INI:Removed the sq(2) from the denom. It's for 2 pols.
            baselinenoise[basecount] = sefd/math.sqrt(2*chanwid*dt_val[basecount]) #INI:ADDED the sq(2) bcoz MeqS uses this convention
            basecount += 1;

        for ibl in np.arange(slvr.nbl):
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

    global data_vis, data_nbl, data_timearr, data_ntime, data_inttime, data_nchan, data_chanwidth, data_flag, data_flag_row

    ####### Read data from MS
    tab = pt.table(args.ms).query("ANTENNA1 != ANTENNA2"); # INI: always exclude autocorrs; this code DOES NOT work for autocorrs
    data_vis = tab.getcol(args.col)

    # Create baseline noise array ------------------------

    anttab = pt.table(args.ms+'/ANTENNA')
    stations = anttab.getcol('STATION')
    nant = len(stations)
    data_nbl = int((nant*(nant-1))/2)
    anttab.close()
 
    data_timearr = np.unique(tab.getcol('TIME'))
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
