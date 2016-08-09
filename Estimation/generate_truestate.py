# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:00:57 2016

@author: Ren
"""

import matplotlib.pyplot as plt
from numpy import *
import sys
import os

##########################################################################################

def extract_true(directory, filename):
    densityTrue = zeros((720,27))
    k = 0
    f = open(directory+filename,'r')
    for line in f:
        myFocus = line.strip('\n').split(',')
        myFocus = [float(i) for i in myFocus]
        densityTrue[k] = array(myFocus)
        k = k+1
    return densityTrue

def plot_true(densityTrue, directorySave, PR, seed):
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.hold(True)
    plt.imshow(densityTrue, aspect = 'auto',origin='lower',interpolation='nearest')        
    plt.ylabel('Time Step',fontsize=20)
    plt.xlabel('Cell Number',fontsize=20)
    plt.colorbar()
    plt.savefig(directorySave + 'PlotTrueDensity_'+str(PR)+'_'+str(seed)+'.pdf', bbox_inches='tight')
    plt.clf()

##########################################################################################

# the set of penetration rate
PRset = [0, 25, 50, 75, 100]
# sensor location
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]

directorySave = os.getcwd()+'/DATA/'

##########################################################################################

# extract true

directoryLoadTrue = os.getcwd()+'/../Simulation/true_states/'

for PR in PRset:

    for seed in sensorLocationSeed:

        filename = 'truestate_5s179m_sce'+str(PR)+'_seed'+str(seed)+'_density.txt'
    
        densityTrue = extract_true(directoryLoadTrue, filename)

        save(directorySave+'TrueDensity_'+str(PR)+'_'+str(seed), densityTrue)    

        plot_true(densityTrue, directorySave, PR, seed)





