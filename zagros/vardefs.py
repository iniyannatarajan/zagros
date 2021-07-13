"""
Settings file for zagros.py. Import this in zagros.py to access the variables.
"""

#------------------------------------------------------------------------------
# Variables for assigning weights to visibilities

noise_per_vis = 0.5 # error on each visibility in Jy. None -> fit it
sefds = [6000,1300,560,220,2000,1600,5000,1600,4500] # station SEFDs in Jy - from EHT2017_station_info
corr_eff = 0.88

# NestedSampler settings
nlive_factor = 100
seed = 42
termination_frac = 0.1

#-------------------------------------------------------------------------------

# Priors - MUST BE CHANGED FOR EVERY MODEL THAT IS TESTED

# for uniform priors
Smin = 0; Smax = 2 # Jy
dxmin = -10; dxmax = 10 # uas
dymin = -10; dymax = 10 # uas
e1min = 0; e1max = 25 # uas
e2min = 0; e2max = 25 # uas
pamin = 0; pamax = 0 # degrees
