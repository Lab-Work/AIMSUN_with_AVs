import matplotlib.pyplot as plt
from numpy import *
import os


PRsetTest = [0, 25, 50, 75, 100]
sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]


directoryLoadData = os.getcwd() + '/DATA/'
directoryLoadPrefix = os.getcwd() + '/Result/Estimation_corrected_0/'

error_store_1st = zeros( (len(PRsetTest), len(sensorLocationSeed)))
error_store_2nd = zeros( (len(PRsetTest), len(sensorLocationSeed)))

for PR_counter, PR in enumerate(PRsetTest):

    for seed_counter, seed in enumerate(sensorLocationSeed):

        density_true = load(directoryLoadData + 'TrueDensity_' + str(PR) + '_' + str(seed) + '.npy')
        density_est_1st = load(directoryLoadPrefix + 'EstimationDensity_' +
                               'PR_' + str(PR) + '_Seed' + str(seed) + '_1st' + '.npy')
        density_est_2nd = load(directoryLoadPrefix + 'EstimationDensity_' +
                               'PR_' + str(PR) + '_Seed' + str(seed) + '_2nd' + '.npy')

        error = average(abs(density_est_1st - density_true))
        error_store_1st[PR_counter, seed_counter] = error

        error = average(abs(density_est_2nd - density_true))
        error_store_2nd[PR_counter, seed_counter] = error


error_avg_1st = average( error_store_1st, axis=1 )
error_avg_2nd = average( error_store_2nd, axis=1 )


plt.plot(PRsetTest, error_avg_1st, color='b', label='1st model')
plt.plot(PRsetTest, error_avg_2nd, color='r', label='2nd model')

plt.savefig(directoryLoadPrefix + 'err_test_all.pdf', bbox_inches='tight')