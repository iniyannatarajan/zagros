"""
Settings file for radiobayes.py. Import this in radiobayes.py to access the variables.
"""

import time
import os
import sys
import scipy.constants as sc
import pyrap.tables as pt
import pypolychord as ppc
from pypolychord.settings import PolyChordSettings
import numpy as np
import cupy as cp
#from mpi4py import MPI
from math import sqrt
import logging

#-------------------------------------------------------------------------------
# define some constants

deg2rad = sc.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;

sqrtTwo=sqrt(2.0)

#------------------------------------------------------------------------------
# Variables for assigning weights to visibilities

sigmaSim=None #Error on each visibility that goes into the predictions - should be the same as SIMULATION NOISE; None -> fit it
compute_weight_vector=True
sefds=np.array([6000,1300,560,220,2000,1600,5000,1600,4500]) # station SEFDs in Jy - from EHT2017_station_info


# Codex-africanus settings

# pypolychord settings
nlive=375 # Number of live points for ppc
num_repeats = 30 # Recommended (not less than 2*nDims)
evtol=0.5 # Evidence tolerance for ppc
seed=42

#-------------------------------------------------------------------------------

# Priors - MUST BE CHANGED FOR EVERY MODEL THAT IS TESTED

# for uniform priors
Smin=0; Smax=2 # Jy
#delaymin=-200.0*1e-12; delaymax=200e-12; # in seconds; the actual delays
dxmin=-50; dxmax= 50; # uas
dymin=-50; dymax= 50; # uas
#e1min = 0e-5 * arcsec2rad; e1max = 4e-5 * arcsec2rad;
#e2min = -4e-5 * arcsec2rad; e2max = 4e-5 * arcsec2rad;

#for Gaussian priors
#Smu=0.149124;Ssigma=1e-4;

#refchan_delay=0 # index (in the MS spw table) of reference channel for delays
