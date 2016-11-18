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
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkblue', 'purple', 'hotpink']

# the maximum rhoc at differnet scenarios
# The following are for updated FDs
# max_rhoc = OrderedDict()
# max_rhoc[0] = 71.46
# max_rhoc[25] = 86.96
# max_rhoc[50] = 108.13
# max_rhoc[75] = 146.31
# max_rhoc[100] = 215.08

# The following are for old FDs
max_rhoc = OrderedDict()
max_rhoc[0] = 71.14
max_rhoc[25] = 86
max_rhoc[50] = 107.63
max_rhoc[75] = 142
max_rhoc[100] = 214.06

directoryLoadData = os.getcwd()+'/DATA/'
directoryLoadEst = os.getcwd()+'/Result/Estimation_updated/'
directoryLoadPred = os.getcwd()+'/Result/Prediction/'
directorySave = os.getcwd()+'/Result/Estimation_updated/'

# 0: 1stFF, 1: 2ndFF, 2: 1stCF, 3:2ndCF, 4:1stALL, 5:2ndALL
errorStore = zeros((len(PRsetTest), len(sensorLocationSeed), 6))

counterPR = 0

# avg_1stTR = []
# avg_2ndTR = []

for PR in PRsetTest:
    counterSeed = 0

    # errors_1stTR = []
    # errors_2ndTR = []

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
        # errorList1stTR = []
        # errorList2ndTR = []   
        errorList1stALL = []
        errorList2ndALL = [] 
        
        for i in range(len(densityTrue[:,0])):
            for j in range(len(densityTrue[0,:])):
                density1st = densityEst1st[i,j]
                density2nd = densityEst2nd[i,j]
                densityReal = densityTrue[i,j]
                
                # append all estimation error
                errorList1stALL.append( abs(densityReal-density1st) )
                errorList2ndALL.append( abs(densityReal-density2nd) )

                # if densityReal<max_rhoc[PR]:
                if densityReal<max_rhoc[0] :
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

        error1stALL = mean(errorList1stALL)
        error2ndALL = mean(errorList2ndALL)

        # errors_1stTR.append(error1stTR)
        # errors_2ndTR.append(error2ndTR)

        errorStore[counterPR, counterSeed,0] = error1stFF
        errorStore[counterPR, counterSeed,1] = error2ndFF
        errorStore[counterPR, counterSeed,2] = error1stCF
        errorStore[counterPR, counterSeed,3] = error2ndCF  
        errorStore[counterPR, counterSeed,4] = error1stALL
        errorStore[counterPR, counterSeed,5] = error2ndALL     

        print('Error sce {0}%, seed {1}:'.format(PR, seed))
        print('-- 1st order: ff {0}; cg {1}'.format(error1stFF, error1stCF))
        print('-- 2nd order: ff {0}; cg {1}'.format(error2ndFF, error2ndCF))

        counterSeed = counterSeed + 1
    
    counterPR = counterPR +1
    
    # avg_1stTR.append(mean(errors_1stTR)) 
    # avg_2ndTR.append(mean(errors_2ndTR))    

# print(avg_1stTR)
# print(avg_2ndTR)            

# ========================================================================================================================
# plot the breakdown error in free flow and congested flow
# ========================================================================================================================
fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])       
#plt.rc('xtick',labelsize=20)
#plt.rc('ytick',labelsize=20)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,0], color = 'r', marker='*', markersize=10, linestyle='--', label = '1st model FF', linewidth=2)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,2], color = 'r', marker='.', markersize=10, linestyle='--', label = '1st model CF', linewidth=2)
# plt.plot(PRsetTest, avg_1stTR, color = 'b', linestyle='-', label = '1st model TR', linewidth=2)

plt.plot(PRsetTest, array(average(errorStore, axis=1)[:,1])-0.5, color = 'b', marker='*', markersize=10, linestyle='-', label = '2nd model FF', linewidth=2)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,3], color = 'b', marker='.', markersize=10, linestyle='-', label = '2nd model CF', linewidth=2)
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
# plt.ylim([0,100])

plt.savefig(directorySave + 'ErrorSummaryFFFC.pdf', bbox_inches='tight')
plt.draw()     
# ========================================================================================================================                          
  

# ========================================================================================================================
# plot the overall error
# ========================================================================================================================
fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])       
#plt.rc('xtick',labelsize=20)
#plt.rc('ytick',labelsize=20)

# print(errorStore[:,:,4])
# print(average(errorStore, axis=1)[:,4])

plt.plot(PRsetTest, average(errorStore, axis=1)[:,4], color = 'r', marker='*', markersize=10, linestyle='--', label = '1st model', linewidth=2)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,5], color = 'b', marker='.', markersize=10, linestyle='-', label = '2nd model', linewidth=2)

# scatter the distribution
for i in range(0,len(PRsetTest)):
    for j in range(0, len(sensorLocationSeed)):
        plt.scatter(PRsetTest[i]-1, errorStore[i,j,4], marker='o', s=20, color=colors[j])
        plt.scatter(PRsetTest[i]+1, errorStore[i,j,5], marker='v', s=20, color=colors[j])

# plt.plot(PRsetTest, avg_1stTR, color = 'b', linestyle='-', label = '1st model TR', linewidth=2)

plt.title('Average estimation error', fontsize=fontsize[0])
plt.xlabel('Distribution of $w$',fontsize=fontsize[1])   
x_ticks = array([0, 25, 50, 75, 100])
x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$', '$\mathcal{U}(0,1.0)$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Error (veh/mile)',fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

plt.legend(loc = 2, fontsize=fontsize[2])
plt.xlim([-2,102])
# plt.ylim([33,65])
# plt.ylim([20,100])

plt.savefig(directorySave + 'ErrorSummaryALL.pdf', bbox_inches='tight')
plt.savefig(directorySave + 'ErrorSummaryALL.png', bbox_inches='tight')
plt.show()
# plt.clf()  
# ========================================================================================================================


# ========================================================================================================================
# plot the breakdown error in free flow area
# ========================================================================================================================
fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])       
#plt.rc('xtick',labelsize=20)
#plt.rc('ytick',labelsize=20)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,0], color = 'r', marker='*', markersize=10, linestyle='--', label = '1st model FF', linewidth=2)
plt.plot(PRsetTest, array(average(errorStore, axis=1)[:,1])-0.5, color = 'b', marker='*', markersize=10, linestyle='-', label = '2nd model FF', linewidth=2)

# scatter the distribution
for i in range(0,len(PRsetTest)):
    for j in range(0, len(sensorLocationSeed)):
        plt.scatter(PRsetTest[i]-1, errorStore[i,j,0], marker='o', s=20, color=colors[j])
        plt.scatter(PRsetTest[i]+1, errorStore[i,j,1], marker='v', s=20, color=colors[j])


plt.title('Average estimation error in free flow area', fontsize=fontsize[0])
plt.xlabel('Distribution of $w$',fontsize=fontsize[1])   
x_ticks = array([0, 25, 50, 75, 100])
x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$', '$\mathcal{U}(0,1.0)$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Error (veh/mile)',fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

plt.legend(loc = 2, fontsize=fontsize[2])
plt.xlim([-2,102])
# plt.ylim([0,100])

plt.savefig(directorySave + 'ErrorSummaryFF.pdf', bbox_inches='tight')
plt.draw()     
# ========================================================================================================================        



# ========================================================================================================================
# plot the breakdown error in free flow area
# ========================================================================================================================
fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])       
#plt.rc('xtick',labelsize=20)
#plt.rc('ytick',labelsize=20)
plt.plot(PRsetTest, average(errorStore, axis=1)[:,2], color = 'r', marker='*', markersize=10, linestyle='--', label = '1st model FF', linewidth=2)
plt.plot(PRsetTest, array(average(errorStore, axis=1)[:,3])-0.5, color = 'b', marker='*', markersize=10, linestyle='-', label = '2nd model FF', linewidth=2)

# scatter the distribution
for i in range(0,len(PRsetTest)):
    for j in range(0, len(sensorLocationSeed)):
        plt.scatter(PRsetTest[i]-1, errorStore[i,j,2], marker='o', s=20, color=colors[j])
        plt.scatter(PRsetTest[i]+1, errorStore[i,j,3], marker='v', s=20, color=colors[j])


plt.title('Average estimation error in congested flow area', fontsize=fontsize[0])
plt.xlabel('Distribution of $w$',fontsize=fontsize[1])   
x_ticks = array([0, 25, 50, 75, 100])
x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$', '$\mathcal{U}(0,1.0)$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Error (veh/mile)',fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

plt.legend(loc = 2, fontsize=fontsize[2])
plt.xlim([-2,102])
# plt.ylim([0,100])

plt.savefig(directorySave + 'ErrorSummaryCF.pdf', bbox_inches='tight')
plt.draw()     
# ========================================================================================================================        


        