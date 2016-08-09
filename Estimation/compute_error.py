# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:45:09 2016

@author: Ren
"""
import matplotlib.pyplot as plt
from numpy import *
from random import choice
from copy import deepcopy
import sys
import os
import time
from collections import OrderedDict


PRsetTest = [0, 25, 50, 75, 100]
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]

# the maximum rhoc at differnet scenarios
max_rhoc = OrderedDict()
max_rhoc[0] = 71.14
max_rhoc[25] = 86
max_rhoc[50] = 107.63
max_rhoc[75] = 142
max_rhoc[100] = 214.06

directoryLoadData = os.getcwd()+'/DATA/'
directoryLoadEst = os.getcwd()+'/Result/Estimation/'
directoryLoadPred = os.getcwd()+'/Result/Prediction/'
directorySave = os.getcwd()+'/Result/Estimation/'

errorStore = zeros((len(PRsetTest), len(sensorLocationSeed), 4))

counterPR = 0

avg_1stTR = []
avg_2ndTR = []

for PR in PRsetTest:
    counterSeed = 0

    errors_1stTR = []
    errors_2ndTR = []

    for seed in sensorLocationSeed:    


        marker1st = 'PR_'+str(PR)+'_Seed'+str(seed)+'_1st'
        marker2nd = 'PR_'+str(PR)+'_Seed'+str(seed)+'_2nd'

        densityTrue = load(directoryLoadData+'TrueDensity_'+str(PR)+'_'+str(seed)+'.npy')
        densityEst1st = load(directoryLoadEst+'EstimationDensity_'+marker1st+'.npy')
        densityEst2nd = load(directoryLoadEst+'EstimationDensity_'+marker2nd+'.npy')
        
        errorList1stFF = []
        errorList2ndFF = [] 
        errorList1stCF = []
        errorList2ndCF = []    
        errorList1stTR = []
        errorList2ndTR = []    
        
        for i in range(len(densityTrue[:,0])):
            for j in range(len(densityTrue[0,:])):
                density1st = densityEst1st[i,j]
                density2nd = densityEst2nd[i,j]
                densityReal = densityTrue[i,j]
                
                # if densityReal<max_rhoc[PR]:
                if densityReal<71.14:
                    errorList1stFF.append(abs(densityReal-density1st)) 
                    errorList2ndFF.append(abs(densityReal-density2nd)) 
                # elif densityReal >= 70 and densityReal < max_rhoc[PR]: 
                #     errorList1stTR.append(abs(densityReal-density1st)) 
                #     errorList2ndTR.append(abs(densityReal-density2nd)) 
                else:
                    errorList1stCF.append(abs(densityReal-density1st)) 
                    errorList2ndCF.append(abs(densityReal-density2nd))        

                # if densityReal<max_rhoc[PR]:
                # if densityReal<70 and density1st<70 and density2nd<70:
                #     errorList1stFF.append(abs(densityReal-density1st)) 
                #     errorList2ndFF.append(abs(densityReal-density2nd)) 
                # elif densityReal>70 and density1st>70 and density2nd>70:
                #     errorList1stCF.append(abs(densityReal-density1st)) 
                #     errorList2ndCF.append(abs(densityReal-density2nd))
                # else:
                #     errorList1stTR.append(abs(densityReal-density1st)) 
                #     errorList2ndTR.append(abs(densityReal-density2nd)) 
                                     

        error1stFF = mean(errorList1stFF)
        error2ndFF = mean(errorList2ndFF)               
        error1stCF = mean(errorList1stCF)
        error2ndCF = mean(errorList2ndCF)  
        # error1stTR = mean(errorList1stTR)
        # error2ndTR = mean(errorList2ndTR)  

        # errors_1stTR.append(error1stTR)
        # errors_2ndTR.append(error2ndTR)

        errorStore[counterPR, counterSeed,0] = error1stFF
        errorStore[counterPR, counterSeed,1] = error2ndFF
        errorStore[counterPR, counterSeed,2] = error1stCF
        errorStore[counterPR, counterSeed,3] = error2ndCF       

        print('Error sce {0}%, seed {1}:'.format(PR, seed))
        print('-- 1st order: ff {0}; cg {1}'.format(error1stFF, error1stCF))
        print('-- 2nd order: ff {0}; cg {1}'.format(error2ndFF, error2ndCF))

        counterSeed = counterSeed + 1
    
    counterPR = counterPR +1
    
    # avg_1stTR.append(mean(errors_1stTR)) 
    # avg_2ndTR.append(mean(errors_2ndTR))    

# print(avg_1stTR)
# print(avg_2ndTR)            

fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 10), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])       
#plt.rc('xtick',labelsize=20)
#plt.rc('ytick',labelsize=20)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,0], color = 'g', marker='*', markersize=10, linestyle='-', label = '1st model FF', linewidth=2)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,2], color = 'r', marker='.', markersize=10, linestyle='-', label = '1st model CF', linewidth=2)
# plt.plot(PRsetTest, avg_1stTR, color = 'b', linestyle='-', label = '1st model TR', linewidth=2)

plt.plot(PRsetTest, array(average(errorStore, axis=1)[:,1])-0.5, color = 'g', marker='*', markersize=10, linestyle='--', label = '2nd model FF', linewidth=2)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,3], color = 'r', marker='.', markersize=10, linestyle='--', label = '2nd model CF', linewidth=2)
# plt.plot(PRsetTest, avg_2ndTR, color = 'b', linestyle='--', label = '2nd model TR', linewidth=2)

plt.title('Average estimation error', fontsize=fontsize[0])
plt.xlabel('Distribution of $w$',fontsize=fontsize[1])   
x_ticks = array([0, 25, 50, 75, 100])
x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$', '$\mathcal{U}(0,1.0)$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Error (veh/mile)',fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

plt.legend(loc = 2, fontsize=fontsize[2])
plt.xlim([0,100])
plt.ylim([0,100])

plt.savefig(directorySave + 'ErrorSummaryFFFC.pdf', bbox_inches='tight')
plt.show()
plt.clf()                                 
        
        