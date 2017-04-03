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

def extract_data(directoryLoad, filename):
    
    densityStore = zeros(720)
    wStore = zeros(720)
    k = 0

    header = True
    f = open(directoryLoad+filename,'r')
    for line in f:
        if header: # skip header and the first empty line
            header = False
            f.next()
        else:      
            myFocus = line.strip('\n').split(',')
            w = float(myFocus[1])
            flow = float(myFocus[2])
            speed = float(myFocus[3])

            density = flow/speed
            k = k+4            
            densityStore[k] = density
            wStore[k] = w            

    return densityStore, wStore



##########################################################################################

# the set of penetration rate
PRset = [0, 25, 50, 75, 100]
# sensor location
sensorLocationSet = [0,1,2,3]
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]

#PRset = [0]
## sensor location
#sensorLocationSet = [0,1,2,3]
#sensorLocationSeed = [1355]


directorySave = os.getcwd()+'/DATA/'

##########################################################################################
# extract measurements and boundary



directoryLoadMea = os.getcwd()+'/../Simulation/detector_data/'



for PR in PRset:
    
    for seed in sensorLocationSeed:

        densityMea = zeros((720, 27))
        boundary = zeros((720,4))

        for sl in sensorLocationSet:
            
            filename = 'sim_sce'+str(PR)+'_seed'+str(seed)+'_PM'+str(sl)+'.csv'               
            
            densityStore, wStore = extract_data(directoryLoadMea, filename)
            
            if sl == 0:
                boundary[:,0] = densityStore
                boundary[:,1] = wStore        
            elif sl == 3:
                boundary[:,2] = densityStore
                boundary[:,3] = wStore                    
            elif sl == 1:
                densityMea[:,8] = densityStore
            elif sl == 2:
                densityMea[:,17] = densityStore   
            else:
                print 'check, undefined case'
                
        for k in range(0,720,4):
            boundary[k+1:k+4] = boundary[k] 
    
        save(directorySave+'boundary_'+str(PR)+'_'+str(seed), boundary)
        save(directorySave+'mea_'+str(PR)+'_'+str(seed), densityMea)



