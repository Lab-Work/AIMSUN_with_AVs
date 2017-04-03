import matplotlib.pyplot as plt
from numpy import *
import sys
import os

"""
This script copies and formats the true states data to the DATA folder needed in the estimator.
"""
__aurthor__ = 'Ren Wang and Yanning Li'

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
# Set the senarios
PRset = [0, 25, 50, 75, 100]
# Set the seeds
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]
# Set the output directory
directorySave = os.getcwd()+'/DATA/'
# Set the input directory 
directoryLoadTrue = os.getcwd()+'/../Simulation/true_states/'
##########################################################################################
# format true states for each scenario and seed
for PR in PRset:

    for seed in sensorLocationSeed:

        filename = 'truestate_5s179m_sce'+str(PR)+'_seed'+str(seed)+'_density.txt'
    
        densityTrue = extract_true(directoryLoadTrue, filename)

        save(directorySave+'TrueDensity_'+str(PR)+'_'+str(seed), densityTrue)    

        plot_true(densityTrue, directorySave, PR, seed)





