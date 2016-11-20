# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:45:09 2016

@author: Ren and Yanning
"""
import os
from collections import OrderedDict
from os.path import exists
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *
import sys


# ==============================================================================
# ==============================================================================
# Start of the main function
# ==============================================================================
# ==============================================================================
def main(argv):
    PRsetTest = [0, 25, 50, 75, 100]
    # seed 24232 seems to be a bad simulation where the queue extended beyonded the entrance
    sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]
    # sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24654, 45234, 59230]
    # sensorLocationSeed = [24232]    #24654
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkblue', 'purple', 'hotpink']
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'darkblue', 'purple', 'hotpink']
    seed_id = 0
    # the runs of the particle filter
    # runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    runs = [0]

    # the maximum rhoc at differnet scenarios
    max_rhoc = OrderedDict()
    max_rhoc[0] = 71.14
    max_rhoc[25] = 86.57
    max_rhoc[50] = 107.63
    max_rhoc[75] = 145.63   # 145.63; old: 142
    max_rhoc[100] = 214.06

    directoryLoadData = os.getcwd() + '/DATA/'
    # directoryLoadPred = os.getcwd() + '/Result/Prediction/'
    directoryLoadPrefix = os.getcwd() + '/Result/Estimation_corrected_1st_rhoc100_5h_'
    directoryLoad = os.getcwd() + '/Result/Estimation_corrected_1st_rhoc100_5h/'
    directorySave = os.getcwd() + '/Result/Estimation_corrected_1st_rhoc100_5h/'
    savename = '0_1st_rhoc100_5h_'

    if not exists(directorySave):
        os.makedirs(directorySave)

    # compute all the errors
    error_array_1st_all, error_array_1st_ff, error_array_1st_cf, error_array_2nd_all, error_array_2nd_ff, error_array_2nd_cf = \
        compute_mae(directoryLoadData, directoryLoad, directoryLoadPrefix, PRsetTest, sensorLocationSeed, runs, max_rhoc[0])

    # ==============================================================================
    # plot the average error for one run with seed scatters
    scatter_seeds_err(directorySave, PRsetTest, sensorLocationSeed,
                      error_array_1st_all, 0, '1st model',
                      error_array_2nd_all, 0, '2nd model',
                      colors, 'Average estimation error', savename + 'err_all')

    scatter_seeds_err(directorySave, PRsetTest, sensorLocationSeed,
                      error_array_1st_ff, 0, '1st free flow',
                      error_array_2nd_ff, 0, '2nd free flow',
                      colors, 'Average estimation error in free flow', savename+'err_ff')

    scatter_seeds_err(directorySave, PRsetTest, sensorLocationSeed,
                      error_array_1st_cf, 0, '1st congested flow',
                      error_array_2nd_cf, 0, '2nd congested flow',
                      colors, 'Average estimation error in congested flow', savename+'err_cf')

    # ==============================================================================
    # plot the average error for one seed with different runs
    # scatter_runs_err(directorySave, PRsetTest, runs,
    #                  error_array_1st_all, seed_id, '1st model',
    #                  error_array_2nd_all, seed_id, '2nd model',
    #                  colors, 'Average error with different runs', savename+'err_all')
    #
    # scatter_runs_err(directorySave, PRsetTest, runs,
    #                  error_array_1st_ff, seed_id, '1st free flow',
    #                  error_array_2nd_ff, seed_id, '2nd free flow',
    #                  colors, 'Average error with different runs in free flow', savename+'err_ff')
    #
    # scatter_runs_err(directorySave, PRsetTest, runs,
    #                  error_array_1st_cf, seed_id, '1st congested flow',
    #                  error_array_2nd_cf, seed_id, '2nd congested flow',
    #                  colors, 'Average error with different runs in congested flow', savename+'err_cf')

    # ==============================================================================
    perc_improv_all, perc_improv_ff, perc_improv_cf = \
        compute_perc_improv(PRsetTest, sensorLocationSeed, runs,
                            error_array_1st_all, error_array_1st_ff, error_array_1st_cf,
                            error_array_2nd_all, error_array_2nd_ff, error_array_2nd_cf)
    #
    scatter_seeds_perc_improve(directorySave, PRsetTest, sensorLocationSeed,
                               perc_improv_all, 0, colors, 'Improvemnt of 2nd over 1st', savename+'err_perc_all')

    scatter_seeds_perc_improve(directorySave, PRsetTest, sensorLocationSeed,
                               perc_improv_ff, 0, colors, 'Improvemnt of 2nd over 1st in free flow', savename+'err_perc_ff')

    scatter_seeds_perc_improve(directorySave, PRsetTest, sensorLocationSeed,
                               perc_improv_cf, 0, colors, 'Improvemnt of 2nd over 1st in congested flow', savename+'err_perc_cf')

    # ==============================================================================
    # plot multiple seeds with multiple runs
    # scatter_err(directorySave, PRsetTest, sensorLocationSeed, runs,
    #             error_array_1st_all, '1st model',
    #             error_array_2nd_all, '2nd model', colors, 'Mean estimation error', 'err_summary_all')
    # scatter_err(directorySave, PRsetTest, sensorLocationSeed, runs,
    #             error_array_1st_ff, '1st free flow',
    #             error_array_2nd_ff, '2nd free flow', colors, 'Mean estimation error in free flow', 'err_summary_ff')
    # scatter_err(directorySave, PRsetTest, sensorLocationSeed, runs,
    #             error_array_1st_cf, '1st congested flow',
    #             error_array_2nd_cf, '2nd congested flow', colors, 'Mean estimation error in congested flow', 'err_summary_cf')

    # plot the percent improvement
    # scatter_perc_improve(directorySave, PRsetTest, sensorLocationSeed, runs, perc_improv_all, colors,
    #                      'Improvement of 2nd over 1st', 'err_summary_perc_all')
    # scatter_perc_improve(directorySave, PRsetTest, sensorLocationSeed, runs, perc_improv_ff, colors,
    #                      'Improvement of 2nd over 1st in free flow', 'err_summary_perc_ff')
    # scatter_perc_improve(directorySave, PRsetTest, sensorLocationSeed, runs, perc_improv_cf, colors,
    #                      'Improvement of 2nd over 1st in congested flow', 'err_summary_perc_cf')


# ========================================================================================================================
# Compute the mean absolute error
# ========================================================================================================================
def compute_mae(directoryLoadData, directoryLoad, directoryLoadPrefix, PRsetTest, sensorLocationSeed, runs, rhoc_thres):
    """
    This function computes the mean absolute error.
    - return six 3d arrays: err_1st_all, err_1st_ff, err_1st_cf, err_2n_all, err_2n_ff, error_array_2nd_cf
    - each array is num_variability x num_seed_sims x num_runs
    """

    # the variable saving all errors
    # num_variability x num_seed_sims x num_runs
    _error_array_1st_all = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    _error_array_1st_ff = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    _error_array_1st_cf = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))

    _error_array_2nd_all = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    _error_array_2nd_ff = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    _error_array_2nd_cf = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))

    # avg_1stTR = []
    # avg_2ndTR = []
    for counterRun, run in enumerate(runs):
        # for each run of the particle filter, find the correct folder

        # disable the folder selection if directory load is set
        if directoryLoad is not None:
            directoryLoadEst = directoryLoad
        elif directoryLoadPrefix is not None:
            directoryLoadEst = directoryLoadPrefix + str(run) + '/'
        else:
            raise Exception('Incorrect estimation data folder. Check directoryLoad')

        for counterPR, PR in enumerate(PRsetTest):

            # errors_1stTR = []
            # errors_2ndTR = []

            for counterSeed, seed in enumerate(sensorLocationSeed):
                # for each seed simulation

                marker1st = 'PR_' + str(PR) + '_Seed' + str(seed) + '_1st'
                marker2nd = 'PR_' + str(PR) + '_Seed' + str(seed) + '_2nd'

                densityTrue = load(directoryLoadData + 'TrueDensity_' + str(PR) + '_' + str(seed) + '.npy')
                densityEst1st = load(directoryLoadEst + 'EstimationDensity_' + marker1st + '.npy')
                densityEst2nd = load(directoryLoadEst + 'EstimationDensity_' + marker2nd + '.npy')

                errorList1stFF = []
                errorList2ndFF = []

                errorList1stCF = []
                errorList2ndCF = []

                # errorList1stTR = []
                # errorList2ndTR = []
                errorList1stALL = []
                errorList2ndALL = []

                for i in range(len(densityTrue[:, 0])):
                    for j in range(len(densityTrue[0, :])):
                        density1st = densityEst1st[i, j]
                        density2nd = densityEst2nd[i, j]
                        densityReal = densityTrue[i, j]

                        # append all estimation error
                        errorList1stALL.append(abs(densityReal - density1st))
                        errorList2ndALL.append(abs(densityReal - density2nd))

                        # if densityReal<max_rhoc[PR]:
                        if densityReal < rhoc_thres:
                            errorList1stFF.append(abs(densityReal - density1st))
                            errorList2ndFF.append(abs(densityReal - density2nd))
                        # elif densityReal >= 70 and densityReal < max_rhoc[PR]:
                        #     errorList1stTR.append(abs(densityReal-density1st))
                        #     errorList2ndTR.append(abs(densityReal-density2nd))
                        else:
                            errorList1stCF.append(abs(densityReal - density1st))
                            errorList2ndCF.append(abs(densityReal - density2nd))

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

                _error_array_1st_ff[counterPR, counterSeed, counterRun] = error1stFF
                _error_array_1st_cf[counterPR, counterSeed, counterRun] = error1stCF
                _error_array_1st_all[counterPR, counterSeed, counterRun] = error1stALL

                _error_array_2nd_ff[counterPR, counterSeed, counterRun] = error2ndFF
                _error_array_2nd_cf[counterPR, counterSeed, counterRun] = error2ndCF
                _error_array_2nd_all[counterPR, counterSeed, counterRun] = error2ndALL

                print('Error sce {0}%, seed {1}, run {2}:'.format(PR, seed, run))
                print('-- 1st order: ff {0}; cg {1}; all {2}'.format(error1stFF, error1stCF, error1stALL))
                print('-- 2nd order: ff {0}; cg {1}; all {2}'.format(error2ndFF, error2ndCF, error2ndALL))


                # avg_1stTR.append(mean(errors_1stTR))
                # avg_2ndTR.append(mean(errors_2ndTR))
    return _error_array_1st_all, _error_array_1st_ff, _error_array_1st_cf, \
           _error_array_2nd_all, _error_array_2nd_ff, _error_array_2nd_cf
    # print(avg_1stTR)
    # print(avg_2ndTR)


# ========================================================================================================================
# Compute the percent improvement of 2nd over 1st
# ========================================================================================================================
def compute_perc_improv(PRsetTest, sensorLocationSeed, runs,
                        error_array_1st_all, error_array_1st_ff, error_array_1st_cf,
                        error_array_2nd_all, error_array_2nd_ff, error_array_2nd_cf):
    """
    Computes the percent improvement of 2nd over 1st
    - returns 3 3d arrays: perc_improv_all, perc_improv_ff, perc_improv_cf
    - each array is num_variability x num_seed_sims x num_runs
    """

    percentImprovement_all = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    percentImprovement_ff = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))
    percentImprovement_cf = zeros((len(PRsetTest), len(sensorLocationSeed), len(runs)))

    for sce in range(0, len(PRsetTest)):
        for seed in range(0, len(sensorLocationSeed)):
            for i in range(0, len(runs)):
                percentImprovement_all[sce, seed, i] = \
                    (error_array_1st_all[sce, seed, i] - error_array_2nd_all[sce, seed, i]) / error_array_1st_all[
                        sce, seed, i]

                percentImprovement_ff[sce, seed, i] = \
                    (error_array_1st_ff[sce, seed, i] - error_array_2nd_ff[sce, seed, i]) / error_array_1st_ff[
                        sce, seed, i]

                percentImprovement_cf[sce, seed, i] = \
                    (error_array_1st_cf[sce, seed, i] - error_array_2nd_cf[sce, seed, i]) / error_array_1st_cf[
                        sce, seed, i]

    # change to percent
    percentImprovement_all = 100 * percentImprovement_all
    percentImprovement_cf = 100 * percentImprovement_cf
    percentImprovement_ff = 100 * percentImprovement_ff

    return percentImprovement_all, percentImprovement_ff, percentImprovement_cf


# ======================================================================================================================
# Plot the error for a single run with different seeds
# ======================================================================================================================
def scatter_seeds_err(directorySave, PRsetTest, sensorLocationSeed,
                      error_array_1, run_id_1, label_1,
                      error_array_2, run_id_2, label_2,
                      colors, title, savename):
    fontsize = (36, 32, 28)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick',labelsize=20)

    plt.plot(PRsetTest, average(error_array_1, axis=1)[:, run_id_1], color='r', marker='*', markersize=10,
             linestyle='--', label=label_1, linewidth=2)
    plt.plot(PRsetTest, average(error_array_2, axis=1)[:, run_id_2], color='b', marker='.', markersize=10,
             linestyle='-', label=label_2, linewidth=2)

    # scatter the distribution
    for i in range(0, len(PRsetTest)):
        for j in range(0, len(sensorLocationSeed)):
            plt.scatter(PRsetTest[i] - 1, error_array_1[i, j, run_id_1], marker='o', s=20, color=colors[j])
            plt.scatter(PRsetTest[i] + 1, error_array_2[i, j, run_id_2], marker='v', s=20, color=colors[j])

    plt.title(title, fontsize=fontsize[0])
    plt.xlabel('Distribution of $w$', fontsize=fontsize[1])
    x_ticks = array([0, 25, 50, 75, 100])
    x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$',
                    '$\mathcal{U}(0,1.0)$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

    plt.ylabel('Error (veh/mile)', fontsize=fontsize[1])
    ax.tick_params(labelsize=fontsize[2])

    plt.legend(loc=2, fontsize=fontsize[2])
    plt.xlim([-2, 102])
    # plt.ylim([0,75])

    plt.savefig(directorySave + '{0}.pdf'.format(savename), bbox_inches='tight')


# ======================================================================================================================
# Plot the percent improvement for a single run with different seeds
# ======================================================================================================================
def scatter_seeds_perc_improve(directorySave, PRsetTest, sensorLocationSeed,
                               perc_improv, run_id,
                               colors, title, savename):
    fontsize = (36, 32, 28)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick',labelsize=20)

    plt.plot(PRsetTest, average(perc_improv, axis=1)[:, run_id], color='r', marker='*', markersize=10, linestyle='--',
             linewidth=2)

    # scatter the distribution
    for i in range(0, len(PRsetTest)):
        for j in range(0, len(sensorLocationSeed)):
            plt.scatter(PRsetTest[i] - 1, perc_improv[i, j, run_id], marker='o', s=20, color=colors[j])
            # plt.scatter(PRsetTest[i] + 1, perc_improv[i, j, run_id], marker='v', s=20, color=colors[j])

    plt.title(title, fontsize=fontsize[0])
    plt.xlabel('Distribution of $w$', fontsize=fontsize[1])
    x_ticks = array([0, 25, 50, 75, 100])
    x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$',
                    '$\mathcal{U}(0,1.0)$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

    plt.ylabel('Percent', fontsize=fontsize[1])
    ax.tick_params(labelsize=fontsize[2])

    plt.legend(loc=2, fontsize=fontsize[2])
    plt.xlim([-2, 102])
    # plt.ylim([33,65])

    plt.savefig(directorySave + '{0}.pdf'.format(savename), bbox_inches='tight')


# ======================================================================================================================
# Plot the error for a single run with different seeds
# ======================================================================================================================
def scatter_runs_err(directorySave, PRsetTest, runs,
                     error_array_1, seed_id_1, label_1,
                     error_array_2, seed_id_2, label_2,
                     colors, title, savename):
    fontsize = (36, 32, 28)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick',labelsize=20)

    plt.plot(PRsetTest, average(error_array_1, axis=2)[:, seed_id_1], color='r', marker='*', markersize=10,
             linestyle='--', label=label_1, linewidth=2)
    plt.plot(PRsetTest, average(error_array_2, axis=2)[:, seed_id_2], color='b', marker='.', markersize=10,
             linestyle='-', label=label_2, linewidth=2)

    # scatter the distribution
    for i in range(0, len(PRsetTest)):
        for j in range(0, len(runs)):
            plt.scatter(PRsetTest[i] - 1, error_array_1[i, seed_id_1, j], marker='o', s=20, color=colors[j])
            plt.scatter(PRsetTest[i] + 1, error_array_2[i, seed_id_2, j], marker='v', s=20, color=colors[j])

    plt.title(title, fontsize=fontsize[0])
    plt.xlabel('Distribution of $w$', fontsize=fontsize[1])
    x_ticks = array([0, 25, 50, 75, 100])
    x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$',
                    '$\mathcal{U}(0,1.0)$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

    plt.ylabel('Error (veh/mile)', fontsize=fontsize[1])
    ax.tick_params(labelsize=fontsize[2])

    plt.legend(loc=2, fontsize=fontsize[2])
    plt.xlim([-2, 102])
    # plt.ylim([33,65])

    plt.savefig(directorySave + '{0}.pdf'.format(savename), bbox_inches='tight')


# ======================================================================================================================
# Plot the error for multiple seeds, with multiple runs
# ======================================================================================================================
def scatter_err(directorySave, PRsetTest, sensorLocationSeed, runs,
                      error_array_1, label_1,
                      error_array_2, label_2,
                      colors, title, savename):
    fontsize = (36, 32, 28)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick',labelsize=20)

    plt.plot(PRsetTest, average( average(error_array_1, axis=1), axis=1), color='r', marker='*', markersize=10,
             linestyle='--', label=label_1, linewidth=2)
    plt.plot(PRsetTest, average( average(error_array_2, axis=1), axis=1), color='b', marker='.', markersize=10,
             linestyle='-', label=label_2, linewidth=2)

    # scatter the distribution
    for i in range(0, len(PRsetTest)):
        for j in range(0, len(sensorLocationSeed)):
            for r in range(0, len(runs)):
                plt.scatter(PRsetTest[i] - 1, error_array_1[i, j, r], marker='o', s=20, color=colors[j])
                plt.scatter(PRsetTest[i] + 1, error_array_2[i, j, r], marker='v', s=20, color=colors[j])

    plt.title(title, fontsize=fontsize[0])
    plt.xlabel('Distribution of $w$', fontsize=fontsize[1])
    x_ticks = array([0, 25, 50, 75, 100])
    x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$',
                    '$\mathcal{U}(0,1.0)$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

    plt.ylabel('Error (veh/mile)', fontsize=fontsize[1])
    ax.tick_params(labelsize=fontsize[2])

    plt.legend(loc=2, fontsize=fontsize[2])
    plt.xlim([-2, 102])
    # plt.ylim([0,75])

    plt.savefig(directorySave + '{0}.pdf'.format(savename), bbox_inches='tight')


# ======================================================================================================================
# Plot the percent improvement for differnet seeds and runs
# ======================================================================================================================
def scatter_perc_improve(directorySave, PRsetTest, sensorLocationSeed, runs,
                               perc_improv,
                               colors, title, savename):
    fontsize = (36, 32, 28)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick',labelsize=20)

    plt.plot(PRsetTest, average(average(perc_improv, axis=1), axis=1), color='r', marker='*',
             markersize=10, linestyle='--', linewidth=2)

    # scatter the distribution
    for i in range(0, len(PRsetTest)):
        for j in range(0, len(sensorLocationSeed)):
            for r in range(0, len(runs)):
                plt.scatter(PRsetTest[i] - 1, perc_improv[i, j, r], marker='o', s=20, color=colors[j])
            # plt.scatter(PRsetTest[i] + 1, perc_improv[i, j, run_id], marker='v', s=20, color=colors[j])

    plt.title(title, fontsize=fontsize[0])
    plt.xlabel('Distribution of $w$', fontsize=fontsize[1])
    x_ticks = array([0, 25, 50, 75, 100])
    x_ticklabels = ['$\mathcal{U}(0,0)$', '$\mathcal{U}(0,0.25)$', '$\mathcal{U}(0,0.5)$', '$\mathcal{U}(0,0.75)$',
                    '$\mathcal{U}(0,1.0)$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize[2])

    plt.ylabel('Percent', fontsize=fontsize[1])
    ax.tick_params(labelsize=fontsize[2])

    plt.legend(loc=2, fontsize=fontsize[2])
    plt.xlim([-2, 102])
    # plt.ylim([33,65])

    plt.savefig(directorySave + '{0}.pdf'.format(savename), bbox_inches='tight')









if __name__ == "__main__":
    sys.exit(main(sys.argv))
