import os
from collections import OrderedDict
from os.path import exists
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys


"""
This is just a quick script for plotting the sensitivity analysis result
"""

# ==========================================================================
# The sensitivity analysis result is obtained from the server.
# ==========================================================================
err_1st = {}
err_1st['calibrated'] = np.array([44.08, 51.96, 52.07, 32.29, 35.60, 45.21, 59.37, 40.26, 43.80, 56.53])
err_1st['vm_5l'] = np.array([43.28, 50.42, 51.57, 31.37, 33.95, 49.63, 59.00, 41.70, 42.97, 54.91])
err_1st['vm_5h'] = np.array([43.99, 51.11, 53.33, 32.84, 37.11, 46.25, 61.93, 40.42, 46.28, 55.01])
err_1st['rhom_5l'] = np.array([49.13, 58.39, 57.99, 36.31, 44.53, 53.53, 61.82, 42.58, 48.79, 60.97])
err_1st['rhom_5h'] = np.array([39.44, 45.13, 47.25, 31.00, 30.78, 43.21, 54.46, 40.46, 41.50, 50.46])
err_1st['2nd_5l'] = np.array([45.34, 50.14, 56.76, 31.43, 36.54, 49.14, 59.29, 40.20, 45.19, 54.56])
err_1st['2nd_5h'] = np.array([44.87, 47.97, 49.83, 31.21, 34.44, 45.38, 60.18, 39.80, 46.65, 56.77])
err_1st['rhoc75_5l'] = np.array([46.75, 53.70, 56.68, 37.74, 41.71, 50.64, 62.22, 40.51, 48.08, 60.14])
err_1st['rhoc75_5h'] = np.array([40.11, 44.98, 45.15, 30.69, 29.76, 40.80, 56.22, 39.63, 42.75, 48.10])

err_2nd = {}
err_2nd['calibrated'] = np.array([38.44, 40.66, 33.88, 32.12, 33.79, 37.42, 45.60, 36.46, 33.15, 40.42])
err_2nd['vm_5l'] = np.array([41.34, 39.11, 37.25, 33.74, 35.23, 35.53, 45.37, 36.63, 31.37, 39.24])
err_2nd['vm_5h'] = np.array([41.04, 40.18, 36.98, 36.60, 35.22, 40.27, 47.44, 37.53, 31.35, 37.88])
err_2nd['rhom_5l'] = np.array([47.57, 44.44, 42.88, 38.39, 41.48, 46.90, 52.58, 38.68, 36.32, 47.45])
err_2nd['rhom_5h'] = np.array([36.99, 34.98, 33.69, 29.72, 33.18, 33.54, 43.27, 35.63, 32.42, 39.71])
err_2nd['2nd_5l'] = np.array([44.77, 44.07, 39.19, 39.43, 40.30, 44.96, 47.28, 38.32, 33.95, 45.12])
err_2nd['2nd_5h'] = np.array([35.83, 35.14, 32.83, 30.80, 31.91, 32.24, 44.24, 34.57, 30.06, 36.35])
err_2nd['rhoc75_5l'] = np.array([39.76, 39.41, 35.02, 34.77, 35.62, 37.67, 45.12, 35.54, 31.94, 39.43])
err_2nd['rhoc75_5h'] = np.array([41.00, 36.76, 35.50, 34.03, 37.30, 37.19, 43.52, 36.07, 33.75, 41.27])


# ==========================================================================
# The sensitivity analysis result after removing the rhoc_2nd for 1st model and rhoc_1st for 2nd 
# ==========================================================================
err_1st = {}
err_1st['calibrated'] = np.array([44.08, 51.96, 52.07, 32.29, 35.60, 45.21, 59.37, 40.26, 43.80, 56.53])
err_1st['vm_5l'] = np.array([43.28, 50.42, 51.57, 31.37, 33.95, 49.63, 59.00, 41.70, 42.97, 54.91])
err_1st['vm_5h'] = np.array([43.99, 51.11, 53.33, 32.84, 37.11, 46.25, 61.93, 40.42, 46.28, 55.01])
err_1st['rhom_5l'] = np.array([49.13, 58.39, 57.99, 36.31, 44.53, 53.53, 61.82, 42.58, 48.79, 60.97])
err_1st['rhom_5h'] = np.array([39.44, 45.13, 47.25, 31.00, 30.78, 43.21, 54.46, 40.46, 41.50, 50.46])
err_1st['2nd_5l'] = np.array([44.08, 51.96, 52.07, 32.29, 35.60, 45.21, 59.37, 40.26, 43.80, 56.53])    # substituted by calibrated
err_1st['2nd_5h'] = np.array([44.08, 51.96, 52.07, 32.29, 35.60, 45.21, 59.37, 40.26, 43.80, 56.53])    # substituted by calibrated
err_1st['rhoc75_5l'] = np.array([46.75, 53.70, 56.68, 37.74, 41.71, 50.64, 62.22, 40.51, 48.08, 60.14])
err_1st['rhoc75_5h'] = np.array([40.11, 44.98, 45.15, 30.69, 29.76, 40.80, 56.22, 39.63, 42.75, 48.10])

err_2nd = {}
err_2nd['calibrated'] = np.array([38.44, 40.66, 33.88, 32.12, 33.79, 37.42, 45.60, 36.46, 33.15, 40.42])
err_2nd['vm_5l'] = np.array([41.34, 39.11, 37.25, 33.74, 35.23, 35.53, 45.37, 36.63, 31.37, 39.24])
err_2nd['vm_5h'] = np.array([41.04, 40.18, 36.98, 36.60, 35.22, 40.27, 47.44, 37.53, 31.35, 37.88])
err_2nd['rhom_5l'] = np.array([47.57, 44.44, 42.88, 38.39, 41.48, 46.90, 52.58, 38.68, 36.32, 47.45])
err_2nd['rhom_5h'] = np.array([36.99, 34.98, 33.69, 29.72, 33.18, 33.54, 43.27, 35.63, 32.42, 39.71])
err_2nd['2nd_5l'] = np.array([44.77, 44.07, 39.19, 39.43, 40.30, 44.96, 47.28, 38.32, 33.95, 45.12])
err_2nd['2nd_5h'] = np.array([35.83, 35.14, 32.83, 30.80, 31.91, 32.24, 44.24, 34.57, 30.06, 36.35])
err_2nd['rhoc75_5l'] = np.array([38.44, 40.66, 33.88, 32.12, 33.79, 37.42, 45.60, 36.46, 33.15, 40.42])   # substituted by calibrated
err_2nd['rhoc75_5h'] = np.array([38.44, 40.66, 33.88, 32.12, 33.79, 37.42, 45.60, 36.46, 33.15, 40.42])   # substituted by calibrated

# normalize the error
sces = ['calibrated', 'vm_5l', 'vm_5h', 'rhom_5l', 'rhom_5h', '2nd_5l', '2nd_5h', 'rhoc75_5l', 'rhoc75_5h']

perc_impr = {}
mean_perc_impr = []
mean_norm_1st = []
mean_norm_2nd = []
for sce in sces:
    mean_norm_1st.append( np.average(err_1st[sce]) )
    mean_norm_2nd.append( np.average(err_2nd[sce]) )

    # compute the improvement.
    perc_impr[sce] = 100*(err_1st[sce]-err_2nd[sce])/err_1st[sce]
    mean_perc_impr.append( np.average(perc_impr[sce]) )

print(mean_perc_impr)

# remove the 1st result for rhoc_2nd and 2nd result for rhoc_1st
mean_norm_1st = np.asarray(mean_norm_1st)
mean_norm_2nd = np.asarray(mean_norm_2nd)
mean_norm_1st[5:7] = None
mean_norm_2nd[7:9] = None

# ==========================================================================
# plot the error result
# ==========================================================================
fontsize = (36, 32, 28)
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkblue', 'purple', 'hotpink']
# plt.rc('xtick',labelsize=20)
# plt.rc('ytick',labelsize=20)

# plt.scatter(np.arange(0, len(sces)), mean_norm_1st, color='r', marker='v', s=70,
#           label='1st model')
# plt.scatter(np.arange(0, len(sces)), mean_norm_2nd, color='b', marker='x', s=60,
#           label='2nd model')



width = 0.2
bar_1st = ax.bar(np.arange(0, len(sces))-width, mean_norm_1st, width, color='r')
bar_2nd = ax.bar(np.arange(0, len(sces)), mean_norm_2nd, width, color='b')


# print('1st:')
# print(mean_norm_1st)
# print('2nd:')
# print(mean_norm_2nd)

# scatter the distribution
if False:
    for i, sce in enumerate(sces):
        for j in range(0, 10):
            # only scatter the averaged result for each simulation
            plt.scatter((i-0.1), err_1st[sce][j], marker='o', s=20, color=colors[j])
            plt.scatter((i+0.1), err_2nd[sce][j], marker='v', s=20, color=colors[j])

plt.title('Estimation error', fontsize=fontsize[0])
plt.xlabel('Sets of FD parameters', fontsize=fontsize[1])
x_ticks = np.arange(0, len(sces))
x_ticklabels = ['None', '$v_{m}^{-}$', '$v_{m}^{+}$', r'$\rho_{m}^{-}$', r'$\rho_{m}^{+}$',
                r'$\rho_{c}^{2nd -}$', r'$\rho_{c}^{2nd +}$',
                r'$\rho_{c}^{1st -}$', r'$\rho_{c}^{1st +}$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Estimation error (veh/mile)', fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

# plt.legend(loc=1, fontsize=fontsize[2])
ax.legend((bar_1st, bar_2nd),
                      ('1st order model', '2nd order model'), prop={'size': 24}, loc='best')

plt.xlim([-0.2, 8.2])
plt.ylim([28, 65])

plt.savefig('sa.pdf', bbox_inches='tight')


# ==========================================================================
# plot the percent improvement
# ==========================================================================
fontsize = (36, 32, 28)
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkblue', 'purple', 'hotpink']
# plt.rc('xtick',labelsize=20)
# plt.rc('ytick',labelsize=20)


plt.bar(np.arange(0, len(sces)), mean_perc_impr, width, color='r')

# scatter the distribution
if False:
    for i, sce in enumerate(sces):
        for j in range(0, 10):
            # only scatter the averaged result for each simulation
            plt.scatter((i-0.1), perc_impr[sce][j], marker='o', s=20, color=colors[j])

plt.title('Performance improvement', fontsize=fontsize[0])
plt.xlabel('Perturbation of parameters', fontsize=fontsize[1])
x_ticks = np.arange(0, len(sces))
x_ticklabels = ['None', '$v_{m}^{-}$', '$v_{m}^{+}$', r'$\rho_{m}^{-}$', r'$\rho_{m}^{+}$',
                r'$\rho_{c}^{2nd -}$', r'$\rho_{c}^{2nd +}$',
                r'$\rho_{c}^{1st -}$', r'$\rho_{c}^{1st +}$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

plt.ylabel('Improvement (%)', fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

# plt.legend(loc=1, fontsize=fontsize[2])
plt.xlim([-0.2, 8.2])
plt.ylim([0, 30])

plt.savefig('sa_perc.pdf', bbox_inches='tight')
