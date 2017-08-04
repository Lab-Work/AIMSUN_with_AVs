import os
import sys
import time
from os.path import exists
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *

sys.path.append(os.getcwd() + '/TrafficModel/')
from ql_model_normal import *

"""
This script is the main function for running the particle filter estimation algorithm. 
Make sure the parameters in main() function are correctly set.
"""
__aurthor__ = 'Ren Wang and Yanning Li'

###############################################################################################################
def main(argv):

    # ========================================================================================================
    # Specify the fundamental diagram parameters which is calibrated in the simulation phase.
    # The gparameters can be perturbed to perform the sensitivity analysis
    # ========================================================================================================
    ## 2nd model parameters
    rhoc_0 = 71.14          # 67.58 (-5%); 71.14; 74.70 (+5%)
    rhoc_1 = 71.14          # 67.58 (-5%); 71.14; 74.70 (+5%); 78.25 (+10%); 85.37 (+20%);
    rhoc_10 = 76.27         # 72.46 (-5%); 76.27; 80.08 (+5%); 83.90 (+10%); 91.52 (+20%);
    rhoc_20 = 83.77         # 79.58 (-5%); 83.77; 87.96 (+5%); 92.15 (+10%); 100.52 (+20%);
    rhoc_25 = 86.57         # 82.24 (-5%); 86.57; 90.90 (+5%); 95.23 (+10%); 103.88 (+20%);
    rhoc_30 = 91.23         # 86.67 (-5%); 91.23; 95.79 (+5%); 100.35 (+10%); 109.48 (+20%);
    rhoc_40 = 97.53         # 92.65 (-5%); 97.53; 102.41 (+5%); 107.28 (+10%); 117.04 (+20%);
    rhoc_50 = 107.63        # 102.25 (-5%); 107.63; 113.01 (+5%); 118.40 (+10%); 129.16 (+20%);
    rhoc_60 = 116.85        # 111.01 (-5%); 116.85; 122.69 (+5%); 128.54 (+10%); 140.22 (+20%); 
    rhoc_70 = 134.65        # 127.92 (-5%); 134.65; 141.38 (+5%); 148.12 (+10%); 161.58 (+20%);
    rhoc_75 = 145.63        # 138.35 (-5%); 145.63; 152.91 (+5%); 160.19 (+10%); 174.76 (+20%);
    rhoc_80 = 151.56        # 143.98 (-5%); 151.56; 159.14 (+5%); 166.72 (+10%); 181.87 (+20%); 
    rhoc_90 = 183.24        # 174.08 (-5%); 183.24; 192.40 (+5%); 201.56 (+10%); 219.89 (+20%);
    rhoc_99 = 214.06        # 203.36 (-5%); 214.06; 224.76 (+5%); 235.47 (+10%); 256.87 (+20%);
    rhoc_100 = 214.06       # 203.36 (-5%); 214.06; 224.76 (+5%); 235.47 (+10%); 256.87 (+20%); 

    rhom_all = 644          # 611.8 (-5%); 644; 676.2 (+5%)

    vmax_all = 76.28        # 72.47 (-5%); 76.28;  80.1 (+5%)

    beta = 600              # beta parameter is set as a relatively large constant to reduce the curvature

    fdpNB = rhoc_0, rhoc_10, rhoc_20, rhoc_30, rhoc_40, rhoc_50, rhoc_60, \
            rhoc_70, rhoc_80, rhoc_90, rhoc_100, rhom_all, vmax_all, beta,

    # ========================================================================================================
    # 1st model parameters
    PRset = [0, 5, 15, 25, 35, 45, 50, 55, 65, 75, 85, 95, 100]
    rhoc_0 = 71.14          # 71.14; 74.7 (+5%);  
    rhoc_5 = 71.98
    rhoc_15 = 73.97
    rhoc_25 = 76.76         # 72.92 (-5%); 76.76; 80.60 (+5%)
    rhoc_35 = 79.87
    rhoc_45 = 82.91
    rhoc_50 = 84.29         # 80.08 (-5%); 84.29; 88.5 (+5%);  
    rhoc_55 = 85.38
    rhoc_65 = 88.81
    rhoc_75 = 92.94         # 88.3 (-5%); 92.94; 97.59 (+5%)
    rhoc_85 = 98.28
    rhoc_95 = 105.8
    rhoc_100 = 119.46       #113.52 (-5%); 119.49; 125.46 (+5%);  update    119.49 ; old 105.8

    rhoc1st = rhoc_0, rhoc_5, rhoc_15, rhoc_25, rhoc_35, rhoc_45, \
              rhoc_50, rhoc_55, rhoc_65, rhoc_75, rhoc_85, rhoc_95, rhoc_100

    prRhocDict = generate_prRhocDict(PRset, rhoc1st)

    # ========================================================================================================
    # set the number of samples and test sets
    # Approximately, the run time of the estimator for one replication: 
    #       50:5 min; 100: 8 min; 500: 43 min; 1000: 86 min
    samples = [500]
    
    # Run the estimator for the following scenario and seed
    # PRsetTest: the experiment with penetration rate of AVs 75%
    PRsetTest = [75]
    # PRsetTest = [0, 25, 50, 75, 100]

    # sensorLocationSeed: the random seed for generating the experiment.
    sensorLocationSeed = [2352, 59230]
    # full: sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]


    # ========================================================================================================
    directoryLoad = os.getcwd() + '/DATA/'

    for sample in samples:

        for i in range(0, 10):
            # Perform 10 runs of the PF to remove the stochasticity of teh filter

            directorySave = os.getcwd() + '/Result/Estimation_corrected_{0}/'.format(i)

            if not exists(directorySave):
                os.makedirs(directorySave)

            t_start = time.time()

            run_estimators(fdpNB, prRhocDict, directoryLoad, directorySave, sample, sensorLocationSeed, PRsetTest)

            t_end = time.time()

            print('Finished PF run {0}: {1} s'.format(i, t_end - t_start))


def init_weight(sample):
    ## particle weight update function
    weight = 1.0 / sample * ones(sample)
    return weight


def init_state(DensityMeaInit, cellNumber, laneNumber, sample):
    mean = (DensityMeaInit[0, 8] + DensityMeaInit[0, 17]) / 2
    std = 0.05 * mean
    state = random.normal(mean, std, (sample, cellNumber))
    return state


def plot_density(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.clim(0.0, 560)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar()
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_error(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar()
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_property(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.clim(0.0, 1.0)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar(ticks=[0.0, 0.5, 1.0])
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_sampleMatch(sampleMatch, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(sampleMatch, range(len(sampleMatch)))
    plt.ylabel('Time Step', fontsize=20)
    plt.xlabel('Number of distinct particles', fontsize=20)
    plt.xticks([500, 1000, 1500, 2000, 2500])
    plt.yticks([0, 30, 60, 90, 120, 150, 180])
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


def compute_likelihood(state, densityMea, densityMeaMean, densityMeaStd, sample, sensorLocation):
    hasMeasurement = True
    if densityMea[sensorLocation[0]] == 0 and densityMea[sensorLocation[1]] == 0:
        hasMeasurement = False
        likelihood = nan

    else:
        likelihood = zeros(int(sample))
        for j in range(sample):
            modelLikelihood = 1.0
            for i in range(len(sensorLocation)):
                diff = densityMea[sensorLocation[i]] - state[j, sensorLocation[i]]
                modelLikelihood = modelLikelihood * 1.0 / (densityMeaStd * sqrt(2 * pi)) * exp(
                    -(diff - densityMeaMean) * (diff - densityMeaMean) / (2 * densityMeaStd * densityMeaStd))
            likelihood[j] = modelLikelihood
        likelihood = likelihood / sum(likelihood)
    return likelihood, hasMeasurement


def update_weight(likelihood, weight):
    ## particle weight update function
    weight = likelihood * weight
    weight = weight / sum(weight)
    return weight


def resampling(state, w, sample, cellNumber, weight):
    Cum = cumsum(weight)
    stateCopy = state.copy()
    wCopy = w.copy()

    step = 1.0 / sample
    i = 1
    u1 = random.uniform(0, step)
    for j in range(sample):
        uj = u1 + step * (j - 1)
        while uj > Cum[i]:
            i = i + 1
        state[j] = stateCopy[i]
        w[j] = wCopy[i]
    return (state, w)


def rhoc_to_w(fdpNB, PR, prRhocDict):
    rhoc = prRhocDict[PR]

    for i in range(10):
        # print('{0}<={1}<={2}'.format(fdpNB[i], rhoc, fdpNB[i+1]))

        if fdpNB[i] <= rhoc <= fdpNB[i + 1]:
            rhoc_min = fdpNB[i]
            rhoc_max = fdpNB[i + 1]
            if i == 0:
                w_min = 0.01
                w_max = 0.1
            elif i == 9:
                w_min = 0.9
                w_max = 0.99
            else:
                w_min = i / 10.0
                w_max = (i + 1) / 10.0
    w = (rhoc - rhoc_min) / (rhoc_max - rhoc_min) * (w_max - w_min) + w_min
    return w


def generate_prRhocDict(PRset, rhoc1st):
    prRhocDict = dict()
    i = 0
    for PR in PRset:
        prRhocDict[PR] = rhoc1st[i]
        i = i + 1
    return prRhocDict


###############################################################################################################

def run_estimators(fdpNB, prRhocDict, directoryLoad, directorySave, sample, sensorLocationSeed, PRsetTest):
    """
    This function can be used to run the PF or for forward simulation.
    :param fdpNB: the parameters for the second order model
    :param prRhocDict: the rhoc parameters for the first order model
    :param directoryLoad: the directory for loading data
    :param directorySave: the directory for saving results
    :param sample: the number of samples used in the PF
    :param sensorLocationSeed: list of int, the seeds of different simulations
        [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]
    :param PRsetTest: list of int, the scenarios with different penetrations of AVs
        [0, 25, 50, 75, 100]
    :return:
    """

    # ==========================================================================================
    # set the noise parameters
    modelNoiseMean = 0.0
    modelNoiseStd = 10.0

    densityMeaMean = 0.0
    densityMeaStd = 10.0

    # ==========================================================================================
    # discretization
    dt = 5.0
    dx = 0.1111
    length = 3.0
    cellNumber = int(floor(length / dx))

    # ==========================================================================================
    # simulation parameter
    Lambda = dt / 3600 / dx
    timeStep = int(3600 / dt)

    trafficModelSet = ['1st', '2nd']

    errorStore = zeros((len(PRsetTest), len(sensorLocationSeed), 2))
    count1 = 0
    count2 = 0

    sensorLocation = [8, 17]
    savefigure = True

    # ==========================================================================================
    # run PF for each penetration rate experiments
    for PR in PRsetTest:

        seedCount = 0

        # run PF for each simulation seed
        for seed in sensorLocationSeed:

            # load the true density and measurements
            densityTrue = load(directoryLoad + 'TrueDensity_' + str(PR) + '_' + str(seed) + '.npy')
            densityMeasurement = load(directoryLoad + 'mea_' + str(PR) + '_' + str(seed) + '.npy')

            # run PF for both 1st order model and 2nd order model
            for modelMarker in trafficModelSet:

                marker = 'PR_' + str(PR) + '_Seed' + str(seed) + '_' + modelMarker

                # ==========================================================================================
                # Load the boundary condition
                boundary = load(directoryLoad + 'boundary_' + str(PR) + '_' + str(seed) + '.npy')
                if modelMarker == '1st':
                    wBoundary1st = rhoc_to_w(fdpNB, PR, prRhocDict)
                    boundary[:, 1] = wBoundary1st
                    boundary[:, 3] = wBoundary1st

                # ==========================================================================================
                # Creat array to save results
                estimatedState = zeros((timeStep, cellNumber))
                estimatedw = zeros((timeStep, cellNumber))

                # ==========================================================================================
                # Initialization
                state = 1.0 * ones((sample, cellNumber))
                w = 0.5 * ones((sample, cellNumber))
                weight = init_weight(sample)

                estimatedState[0] = average(state, axis=0)
                estimatedw[0] = boundary[0, 1] * ones(cellNumber)

                # ==========================================================================================
                # run for each time step
                for k in range(1, timeStep):

                    # --------------------------------------------------------------------------------------
                    # set the in and out flow boundary condition: [inflow, inflow_w, outflow, outflow_w]
                    bdl = boundary[k, 0], boundary[k, 1]
                    bdr = boundary[k, 2], boundary[k, 3]

                    # --------------------------------------------------------------------------------------
                    # forward propagate each sample
                    for j in range(sample):
                        state[j], w[j] = ctm_2ql(state[j], w[j], fdpNB, Lambda, bdl, bdr, modelNoiseMean, modelNoiseStd,
                                                 inflow=-1.0, outflow=-1.0)

                    # NOTE: set True to run PF for estimation.
                    #       set False to run forward simulation
                    if True:
                        # --------------------------------------------------------------------------------------
                        # get density measurement and run measurement correction
                        densityMea = densityMeasurement[k]

                        likelihood, hasMeasurement = compute_likelihood(state, densityMea, densityMeaMean, densityMeaStd,
                                                                        sample, sensorLocation)

                        if hasMeasurement:
                            # resample weight and update esitimated state and w
                            weight = update_weight(likelihood, weight)
                            state, w = resampling(state, w, sample, cellNumber, weight)

                        # --------------------------------------------------------------------------------------
                        # reinitialize the weight for the next step
                        weight = init_weight(sample)

                        # --------------------------------------------------------------------------------------
                        # save the estimated state in
                        estimatedState[k] = average(state, axis=0)
                        estimatedw[k] = average(w, axis=0)
                    else:
                        estimatedState[k] = state
                        estimatedw[k] = w

                error = average(abs(estimatedState - densityTrue))

                print marker, error

                errorStore[count1, seedCount, count2] = error
                count2 = count2 + 1

                if count2 == 2:
                    count2 = 0

                plot_density(densityTrue, True, 'PlotTrue' + marker, directorySave)
                plot_density(estimatedState, True, 'PlotEstimationDensity' + marker, directorySave)
                plot_property(estimatedw, True, 'PlotEstimationW' + marker, directorySave)
                save(directorySave + 'EstimationDensity_' + marker, estimatedState)
                save(directorySave + 'EstimationW_' + marker, estimatedw)
                save(directorySave + 'TrueDensity_' + marker, densityTrue)

            seedCount = seedCount + 1

            print('Finished sce {0}, seed {1}'.format(PR, seed))

        count1 = count1 + 1

    errorStoreAveSeed = average(errorStore, axis=1)

    #            
    # save(directorySave+'ErrorSummary', errorStore)
    #            
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(PRsetTest, errorStoreAveSeed[:, 0], color='b', label='1st model')
    plt.plot(PRsetTest, errorStoreAveSeed[:, 1], color='r', label='2nd model')
    plt.ylabel('Error (veh/mile)', fontsize=20)
    plt.xlabel('Variance of AVs (%)', fontsize=20)
    plt.legend(loc=4)
    plt.xlim([0, 100])
    plt.ylim([0, 75])
    plt.savefig(directorySave + 'ErrorSummary.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()




