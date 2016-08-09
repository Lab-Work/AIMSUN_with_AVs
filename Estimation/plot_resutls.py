# -*- coding: utf-8 -*-
"""
Plot the estimation results and true state.

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

def plot_2d_data(data, directorySave, save_name, title=None, figsize=(10,10), fontsize=(32, 30, 28), limit=(0, 644)):
    """
    This function plots the density 
    """
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    im = ax.imshow(data, cmap=plt.get_cmap('jet'),
                   interpolation='nearest',
                   aspect='auto', origin='lower',
                   vmin=limit[0], vmax=limit[1])

    if title is not None:
        ax.set_title(title, fontsize=fontsize[0])

    # x axis is distance, 27 cells
    plt.xlabel('Space (mile)', fontsize=fontsize[1])
    x_ticks = array([0, 9, 18, 27]) - 0.5
    x_ticklabels = ['0', '1', '2', '3']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])
    ax.set_xlim([-0.5, 26.5])

    # y axis is time, 720 steps
    plt.ylabel('Time (min)', fontsize=fontsize[1])
    y_ticks = array([0, 120, 240, 360, 480, 600, 720]) - 0.5
    y_ticklabels = ['0', '10', '20', '30', '40', '50', '60']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=fontsize[2])
    ax.set_ylim([-0.5, 719.5])

    cax = fig.add_axes([0.95, 0.12, 0.02, 0.72])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=fontsize[1])

    # cbar.ax.set_title('veh/mile', fontsize=fontsize[1])

    # plt.draw()
    plt.savefig(directorySave + save_name, bbox_inches='tight')
    plt.clf()
    plt.close()

##########################################################################################

# the set of penetration rate
PRset = [0, 25, 50, 75, 100]
# sensor location
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]

truestate_dir = os.getcwd()+'/DATA/'
result_dir = os.getcwd()+'/Result/Estimation/'

fontsize = (40, 38, 36)

##########################################################################################
# update all resutls in new format
# for PR in PRset:

#     for seed in sensorLocationSeed:

#         # plot true density
#         filename = 'TrueDensity_{0}_{1}.npy'.format(PR, seed)
#         densityTrue = load(truestate_dir+ filename) 
#         plot_2d_data(densityTrue, truestate_dir, 'PlotTrueDensity_{0}_{1}.pdf'.format(PR, seed), 
#             title='True density', figsize=(10,10), fontsize=fontsize, limit=(0, 644))

#         # plot 1st estimated density
#         filename = 'EstimationDensity_PR_{0}_Seed{1}_1st.npy'.format(PR, seed)
#         density = load(result_dir+ filename) 
#         plot_2d_data(density, result_dir, 'PlotEstimationDensityPR_{0}_Seed{1}_1st.pdf'.format(PR, seed), 
#             title='Estimated density (1st)', figsize=(10,10), fontsize=fontsize, limit=(0, 644))

#         # plot 2nd estimated density
#         filename = 'EstimationDensity_PR_{0}_Seed{1}_2nd.npy'.format(PR, seed)
#         density = load(result_dir+ filename) 
#         plot_2d_data(density, result_dir, 'PlotEstimationDensityPR_{0}_Seed{1}_2nd.pdf'.format(PR, seed), 
#             title='Estimated density (2nd)', figsize=(10,10), fontsize=fontsize, limit=(0, 644))




# plot true and esitmated property for two cases
# true_w = load(result_dir + 'TrueW_PR_0_seed2143.npy')
# plot_2d_data(true_w, result_dir, 'PlotTrueW_PR_0_seed2143.pdf', 
#     title='True fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

true_w = load(result_dir + 'TrueW_PR_100_seed2143.npy')
plot_2d_data(true_w, result_dir, 'PlotTrueW_PR_100_seed2143.pdf', 
    title='True fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

# convert and plot the true property
# filename = 'truestate_5s179m_sce50_seed24654_w.txt'
# true_w = extract_true(truestate_dir, filename)
# save(result_dir+'TrueW_PR_50_seed24654.npy', true_w)  
# plot_2d_data(true_w, result_dir, 'PlotTrueW_PR_50_seed24654.pdf', 
#     title='True fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

# # plot the estimated property
# est_w = load(result_dir + 'EstimationW_PR_0_seed2143_1st.npy')
# plot_2d_data(est_w, result_dir, 'PlotEstimationW_PR_0_seed2143_1st.pdf', 
#     title='Estimated fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

# est_w = load(result_dir + 'EstimationW_PR_100_seed2143_1st.npy')
# plot_2d_data(est_w, result_dir, 'PlotEstimationW_PR_100_seed2143_1st.pdf', 
#     title='Estimated fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

# est_w = load(result_dir + 'EstimationW_PR_0_seed2143_2nd.npy')
# plot_2d_data(est_w, result_dir, 'PlotEstimationW_PR_0_seed2143_2nd.pdf', 
#     title='Estimated fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))

# est_w = load(result_dir + 'EstimationW_PR_100_seed2143_2nd.npy')
# plot_2d_data(est_w, result_dir, 'PlotEstimationW_PR_100_seed2143_2nd.pdf', 
#     title='Estimated fraction of AVs', figsize=(10,10), fontsize=fontsize, limit=(0, 1))










