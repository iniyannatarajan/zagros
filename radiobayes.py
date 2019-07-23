# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import pyrap.tables as pt

from vardefs import *
from priors import Priors
from africanus.rime.cuda import phase_delay, predict_vis

# Global variables
ms_data = None # variable to hold input data matrix

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms", help="Input MS name")
    p.add_argument("col", help="Name of the data column from MS")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Necessary to compare"
                        "codex vis against MeqTrees-generated vis")
    p.add_argument('--hypo', type=int, choices=[0,1,2], required=True)
    p.add_argument('--npsrc', type=int, default=0, required=True)
    p.add_argument('--ngsrc', type=int, default=1, required=True)
    p.add_argument('--npar', type=int, default=6, required=True)
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

    global ms_data

    ####### Read data from MS
    tab = pt.table(args.ms)
    ms_data = tab.getcol(args.col)
    tab.close()

    # Create baseline noise array ------------------------

    anttab = pt.table(args.ms+'/ANTENNA')
    stations = anttab.getcol('STATION')
    numants = len(stations)
    numbaselines = int((numants*(numants-1))/2)
    print('nbl: ',numbaselines)
    anttab.close()
 
    tab = pt.table(args.ms).query("ANTENNA1 != ANTENNA2"); # INI: exclude autocorrs
    dt_val = tab.getcol('EXPOSURE',0,numbaselines) # jan26, for VLBA sims
    print('dt_val: ',dt_val)

    #get channel width
    freqtab = pt.table(args.ms+'/SPECTRAL_WINDOW')
    chanwid = freqtab.getcol('CHAN_WIDTH')[0,0];
    nchan = freqtab.getcol('NUM_CHAN')[0]
    print("chanwid (Hz), numants, nchan: ", chanwid, numants, nchan)
    freqtab.close();

    #get flags from MS
    flag = tab.getcol('FLAG')
    flag_row = tab.getcol('FLAG_ROW')
    flag = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

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

    #ppc.run_polychord(loglike, n_params, 0, settings, prior=prior_transform)
    loglike(0)

    return 0

if __name__ == '__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
