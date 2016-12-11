import csv
import sqlite3
import sys
from collections import OrderedDict
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy import optimize

__author__ = 'Yanning Li'

"""
This script is used to calibrate the FD using the detector data.
"""
# ======================================================================================
# Configure the directories
# ======================================================================================
# directory = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32':
    config_file = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\configuration_file.txt'
    folder_dir = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\FD_calibration\\'
    logger_path = folder_dir + 'Logs\\'
elif sys.platform == 'darwin':
    config_file = '../configuration_file.txt'
    folder_dir = '../FD_calibration/'
    logger_path = folder_dir + 'Logs/'


# ======================================================================================
# End of configurations in this block
# ======================================================================================


def main(argv):
    config = load_configuration(config_file)
    # ============================================================
    # get all detector data
    file_list = []
    for pAV in config['fd_scenarios']:

        if not exists(get_file_name('det_data', pAV)):
            # print('not exist')
            print('Extracting data from scenario {0}'.format(pAV))
            # extract from sqlite and calibrate FD.
            extract_clean_det_data(pAV, config)

        file_list.append(get_file_name('det_data', pAV))

    # ============================================================
    # calibrate the second order FDs
    # fd_2nd_freeflow_thres = [40, 40, 40, 40, 40, 40,
    #                          40, 40, 45, 45, 45]
    fd_2nd_pav_list = [[0.0, 0.0], [0.07, 0.13], [0.17, 0.23], [0.22, 0.28],
                       [0.27, 0.33], [0.37, 0.43], [0.47, 0.53],
                       [0.57, 0.63], [0.67, 0.73], [0.72, 0.78], [0.77, 0.83],
                       [0.87, 0.93], [1.0, 1.0]]
    fd_2nd_freeflow_thres = [40, 40, 40, 40, 
                             40, 40, 40,
                             40, 40, 40, 40, 
                             40, 40]
    # fd_2nd_freeflow_thres = [40, 40]
    # fd_2nd_pav_list = [[0.22, 0.28], [0.72, 0.78]]


    preset_vm_beta = [76.28, 600]
    # preset_vm_beta = None
    rho_max = 644  # veh/mile
    # preset_vm_beta = calibrate_NC_QLFD(file_list, fd_2nd_pav_list,
    #                                    fd_2nd_freeflow_thres, rho_max,
    #                                    preset_vm_beta, save_fig=True, order='second')

    # ============================================================
    # calibrate the first order FDs
    if True:
        # fd_1st_pav_list = [[0.0, 0.0], [0.0, 0.05], [0.0, 0.15], [0.0, 0.25], [0.0, 0.35], 
        #                    [0.0, 0.45], [0.0, 0.5], [0.0, 0.55], [0.0, 0.65], [0.0, 0.75], 
        #                    [0.0, 0.85], [0.0, 0.95], [0.0, 1.0]]
        # fd_1st_freeflow_thres = [40, 40, 40, 40, 40, 
        #                          40, 40, 40, 40, 40, 
        #                          40, 40, 40]
        fd_1st_pav_list = [[0.0, 0.25], [0.0, 0.5]]
        fd_1st_freeflow_thres = [40, 40]
        calibrate_NC_QLFD(file_list, fd_1st_pav_list, fd_1st_freeflow_thres,
                          rho_max, preset_vm_beta, save_fig=True, order='first')


def load_configuration(file_name):
    """
    This function loads the configuraiton of the simulation
    :param file_name: the file path
    :return: a config dict;
    """

    if exists(file_name):

        config = OrderedDict()

        with open(file_name, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                else:
                    items = line.split(':')
                    config[items[0]] = __assertType(items[1])
        return config

    else:
        raise Exception('Failed to find configuration file {0}'.format(file_name))


def get_file_name(file_type, sce=10):
    """
    Standardize the naming convention for fd calibration
    :param file_type: the file type to be saved
    :param sce: scenario #, str, e.g. 23
    :return: name str
    """
    if sys.platform == 'win32':
        if file_type == 'det_data':
            return folder_dir + 'detector_data\\fd_sce{0}.csv'.format(sce)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files\\fd_sce{0}.sqlite'.format(sce)
        elif file_type == 'raw_data':
            return folder_dir + 'detector_data\\raw_fd_sce{0}.csv'.format(sce)
        elif file_type == '1stFDs':
            return folder_dir + 'calibrated_FDs\\1stFD_sce{0}.png'.format(sce)
        elif file_type == '2ndFDs':
            return folder_dir + 'calibrated_FDs\\2ndFD_sce{0}.png'.format(sce)
        elif file_type == '2ndFDall':
            return folder_dir + 'calibrated_FDs\\2ndFD_all.png'.format(sce)
        else:
            raise Exception('Unrecognized file type for naming.')
    elif sys.platform == 'darwin':
        if file_type == 'det_data':
            return folder_dir + 'detector_data/fd_sce{0}.csv'.format(sce)
        elif file_type == 'raw_data':
            return folder_dir + 'detector_data/raw_fd_sce{0}.csv'.format(sce)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files/fd_sce{0}.sqlite'.format(sce)
        elif file_type == '1stFDs':
            return folder_dir + 'calibrated_FDs/1stFD_sce{0}.png'.format(sce)
        elif file_type == '2ndFDs':
            return folder_dir + 'calibrated_FDs/2ndFD_sce{0}.png'.format(sce)
        elif file_type == '2ndFDall':
            return folder_dir + 'calibrated_FDs/2ndFD_all.png'.format(sce)
        else:
            raise Exception('Unrecognized file type for naming.')


def calibrate_NC_QLFD(cleaned_data_files, pAV_ranges, freeflow_thres,
                      rho_m, preset_vm_beta,
                      save_fig=False,
                      order='first'):
    """
    This function calibrates the collapsed non-constrained quadratic linear fundamental diagram.
        - calibrate the w for the first pAV_range (largest) congested data point
        - Use all freeflow data with even weights in all pAV_ranges and fit the curve. Directly fit a quadratic curve.
        - Calibrate the w for each other pAV_range using the congested data point.
        - Intersect the w's with the quadratic curve and find out the intersection at each.
        The freeflow regime is fitted by a quadratic function while the congested regime is fitted by a linear function
    :param cleaned_data_files: list of file names with columns [pav, flow (veh/hr), speed (mph), density (veh/mile)]
    :param pAV_range:[ [min, max] ], only use the data that falls in this range. The first range should be the largest
        range which is used to calibrated the quadratic curve.
    :param freeflow_thres: in mph, the freeflow threshold
    :param rho_max: in veh/mile, the maximum density
    :param preset_vm_beta: tuple, (vm, beta) in mph, Fit a quadratic curve if none
    :param save_fig: True or False
    :param order: 'first' or 'second' order fundamental diagram, just for naming the file
    :return:
    """
    # =========================================================
    # read all the data files
    # p_av, flow, speed, density
    data = []
    for cleaned_file in cleaned_data_files:
        print('Reading data file {0}...\n'.format(cleaned_file))
        f_data = open(cleaned_file, 'r')
        r_data = csv.reader(f_data)
        r_data.next()

        for row in r_data:
            data.append(row)
        f_data.close()

    # compute the density by rho = q/v
    data = np.array(data).astype(float)
    # remove the nan values
    # data = data[ ~np.isnan(data[:,2]) ,:]

    # =========================================================
    # Calibration procedure.
    # If second order model:
    #   - Fit the collapsed free flow quadratic curve using all freeflow data from all penetrations.
    #   - Then fit each penetration individually using the same left side quadratic curve
    # If first order model:
    #   - Left side is same as the second order model
    #   - Fit congested regime accordingly.

    # -------------------------------------------------
    # Step 1: If second order, fit the quadratic curve
    # -------------------------------------------------
    if preset_vm_beta is not None:
        # for both first order and second order model, if there is preset vm and beta, then use preset
        all_vm = preset_vm_beta[0]
        all_beta = preset_vm_beta[1]

    else:
        # otherwise, need to refit all vm and beta while calibrating the second order model.
        if order == 'second':
            # Get all free flow data points in the ranges to fit the collapsed quadratic curve for the second order model
            # get the data at all ranges.
            print('\nCalibrating collapsed freeflow curve for second order model:')
            all_ff_data = []
            num_points = []
            for i, p in enumerate(pAV_ranges):
                ffInRange = ((data[:, 0] >= p[0]) & (data[:, 0] <= p[1]) & (data[:, 2] >= freeflow_thres[i]))
                print('--- Got {0} data point for {1} pAV'.format(sum(ffInRange), p))

                num_points.append(sum(ffInRange))

            # weight the free flow data by density bins
            max_num_pts = np.max(num_points)
            for i, p in enumerate(pAV_ranges):

                ffInRange = ((data[:, 0] >= p[0]) & (data[:, 0] <= p[1]) & (data[:, 2] >= freeflow_thres[i]))
                scale = np.round(max_num_pts / num_points[i])
                print('--- pAV {0}: Scale {1} data point by {2} to {3} data points'.format(p, num_points[i], scale,
                                                                                       scale * num_points[i]))
                for s in range(0, scale):
                    if all_ff_data == []:
                        all_ff_data = data[ffInRange, :]
                    else:
                        all_ff_data = np.vstack([all_ff_data, data[ffInRange, :]])

            all_ff_pAV = all_ff_data[:, 0]
            all_ff_flow = all_ff_data[:, 1]
            all_ff_speed = all_ff_data[:, 2]
            all_ff_density = all_ff_flow / all_ff_speed

            # now fit a quadratic function with intercepts at origin and has related coefficient
            # the quadratic function:
            # q = (v_m^2 - 2wv_m)/(4wrho_m)*rho^2 + v_m*rho
            beta_preset = 600
            # funcQuadFit = lambda vm_beta, rho: vm_beta[0] * rho - np.power(rho, 2) * vm_beta[0] / vm_beta[1]
            funcQuadFit = lambda vm_beta, rho: vm_beta[0] * rho - np.power(rho, 2) * vm_beta[0] / beta_preset
            funErr = lambda vm_beta, rho, q: funcQuadFit(vm_beta, rho) - q
            vm_beta_init = [80, 600]  # initial guess of vm is 60

            vm_beta_est, success = optimize.leastsq(funErr, vm_beta_init, args=(all_ff_density, all_ff_flow))

            all_vm = vm_beta_est[0]
            all_beta = beta_preset
            # all_beta = vm_beta_est[1]
        else:
            raise Exception('Please calibrate second order FD first to get vm and beta')
        

    # -------------------------------------------------
    # Step 1: Fit congested part and find the maximum flow for each penetration rate
    # -------------------------------------------------
    # define the free flow quadratic part
    funcQuadAll = lambda vm, beta_para, rho: vm * rho - np.power(rho, 2) * vm / beta_para

    # Visualize all fundamental diagram on a single plot
    if order == 'second':
        fig_all, ax = plt.subplots(figsize=(15, 10))
        cm = plt.cm.get_cmap('jet')

    for i, pAV_range in enumerate(pAV_ranges):
        # calibrate the fundamental diagram for each pAV_range

        inRange = (data[:, 0] >= pAV_range[0]) & (data[:, 0] <= pAV_range[1])

        # convert data to mph, veh/mile
        pAV = data[inRange, 0]
        flow = data[inRange, 1]
        speed = data[inRange, 2]
        density = flow / speed

        print('Calibrating {2} order FD for {0} pAV using {1} data points...'.format(pAV_range, flow.size,
                                                                                     order))
        # ---------------------------------------------------------
        # fit the congested regime
        freeFlow = (speed >= freeflow_thres[i])
        shifted_dens = density[~freeFlow] - rho_m
        w, _, _, _ = np.linalg.lstsq(shifted_dens[:, np.newaxis], flow[~freeFlow])
        w = w[0]

        # compute q_max and rho_c
        # beta = 4*w*rho_m/(2*w - vm_est)
        rho_c = (-(w * all_beta - all_vm * all_beta) -
                 np.sqrt(np.power(w * all_beta - all_vm * all_beta, 2) - 4 * all_vm * (-w * all_beta * rho_m))) / (
                    2 * all_vm)
        q_max = funcQuadAll(all_vm, all_beta, rho_c)

        print('Computed the critical density for {0} pAV'.format(pAV_range))

        print('-- freeflow: q = v_m*rho - v_m*rho^2/beta')
        print('-- congflow: q = w(rho - rho_m)')
        print('------ v_m: {0}'.format(all_vm))
        print('------ beta: {0}'.format(all_beta))
        print('------ w: {0}'.format(w))
        print('------ rho_m: {0}'.format(rho_m))
        print('------ rho_c: {0}'.format(rho_c))
        print('------ q_m:{0}'.format(q_max))

        # =========================================================
        # Visualize the fundamental diagram
        fig_window = plt.figure(figsize=(15, 10))
        fig = fig_window.add_subplot(111)

        # freeflow side
        plt.scatter(density[freeFlow], flow[freeFlow], c='k', linewidths=0, marker='o')
        dens = np.linspace(0, rho_c, 100)
        plt.plot(dens, funcQuadAll(all_vm, all_beta, dens), 'r-', linewidth=2.0)

        # congested side
        plt.scatter(density[~freeFlow], flow[~freeFlow], c='k', linewidths=0, marker='o')
        dens = np.linspace(rho_c, rho_m, 100)
        plt.plot(dens, w * (dens - rho_m), 'r-', linewidth=2.0)

        if order == 'first':
            plt.title('1st order with AV penetration in {0}'.format(pAV_range), fontsize=24)
        elif order == 'second':
            plt.title('2nd order at AV penetration {0}'.format(pAV_range), fontsize=24)

        plt.xlabel('Traffic density (veh/mile)', fontsize=24)
        plt.ylabel('Traffic flow (veh/hr)', fontsize=24)

        text_str = r'freeflow: $q = v_m\rho - v_m\rho^2/\beta$' + '\n' \
                                                                  r'congflow: $q = w(\rho - \rho_m)$' + '\n' + \
                   r' $v_m$=   {0} mph'.format(np.round(all_vm, 2)) + '\n' + \
                   r' $\beta$=    {0} '.format(np.round(all_beta, 2)) + '\n' + \
                   r' $w$=    {0} mph'.format(np.round(w, 2)) + '\n' + \
                   r' $\rho_c$=   {0} veh/mile'.format(np.round(rho_c, 2)) + '\n' + \
                   r' $\rho_m$=   {0} veh/mile'.format(np.round(rho_m, 2)) + '\n' + \
                   r' $q_m$=   {0} veh/hr'.format(np.round(q_max, 2))

        anchored_text = AnchoredText(text_str, loc=1)
        fig.add_artist(anchored_text)

        plt.grid(True)

        if save_fig is True:
            if order == 'second':
                pav_name = int((pAV_range[0] + pAV_range[1]) * 50)
                plt.savefig('{0}'.format(get_file_name('2ndFDs', pav_name)), bbox_inches='tight')
            elif order == 'first':
                pav_name = int(pAV_range[1]*100)
                plt.savefig('{0}'.format(get_file_name('1stFDs', pav_name)), bbox_inches='tight')
            else:
                raise Exception('unrecognized FD order')
            plt.clf()
            plt.close()
        else:
            plt.draw()

        # =========================================================
        # add plot to the larger figure
        if order == 'second':
            sc = ax.scatter(density[freeFlow], flow[freeFlow], c=pAV[freeFlow],
                            vmin=0.0, vmax=1.0, cmap=cm, linewidths=0, marker='o')
            dens = np.linspace(0, rho_c, 100)
            ax.plot(dens, funcQuadAll(all_vm, all_beta, dens), 'r-', linewidth=2.0)
            ax.scatter(density[~freeFlow], flow[~freeFlow], c=pAV[~freeFlow],
                       vmin=0.0, vmax=1.0, cmap=cm, linewidths=0, marker='o')
            dens = np.linspace(rho_c, rho_m, 100)
            ax.plot(dens, w * (dens - rho_m), 'r-', linewidth=2.0)

    if order == 'second':
        ax.set_title('2nd order FD all', fontsize=24)

        ax.set_xlabel('Traffic density (veh/mile)', fontsize=24)
        ax.set_ylabel('Traffic flow (veh/hr)', fontsize=24)
        cbar_ax = fig_all.add_axes([0.92, 0.15, 0.02, 0.7])
        fig_all.colorbar(sc, cax=cbar_ax)
        ax.grid(True)
        if save_fig is True:
            plt.savefig('{0}'.format(get_file_name('2ndFDall')), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()

    # return the vm and beta calibrated in second order
    if order == 'second':
        return all_vm, all_beta


def __assertType(string):
    """
    This function asserts the correct type of the string value:
        - int if the string is a int number
        - list of ints if the string is a comma separated list
        - float if the string can be converted to a float;
        - list of floats if the string is a comma separated list
        - string otherwise
    :param string: the input string
    :return: a float or the original string
    """
    converted_list = []
    items = string.strip().split(',')
    flag_float = False

    for v in items:

        # if int, convert to int
        try:
            int(v)
            converted_list.append(int(v))
            continue
        except ValueError:
            pass

        # if float, convert to int
        try:
            float(v)
            converted_list.append(float(v))
            flag_float = True
            continue
        except ValueError:
            pass

        # otherwise, just append the original string
        converted_list.append(v)

    if flag_float is True:
        # if one element is float, then convert all to float
        try:
            converted_list = [float(i) for i in converted_list]
        except ValueError:
            raise Exception('Error: List can not be converted to float: {0}.'.format(converted_list))

    if len(converted_list) == 1:
        return converted_list[0]
    else:
        return converted_list


def extract_clean_det_data(sce, config):
    """
    This function extracts the detector data from sqlite, formmat it and clean it.
    :param sce: str, or int, the scenario id for naming
    :param config: the configuration loaded from config file
    :return: saved in det_data folder
    """

    # =================================================================
    # extract raw data from sqlite
    if True:
        con = sqlite3.connect(get_file_name('sqlite', sce))
        cur = con.cursor()
        sqlite_data = cur.execute("SELECT * FROM MIDETEC")

        # get column names
        headers = [description[0] for description in cur.description]

        with open(get_file_name('raw_data',sce),'wb') as f_raw:
            writer = csv.writer(f_raw)
            writer.writerow(headers)
            writer.writerows(sqlite_data)

    # =================================================================
    # load csv and loop through csv file, which is much faster than looping through sqlite
    # extract the specific columns to the following format
    # columns: p_rate, [flow(veh/hr), speed(kph), density(veh/km)],
    with open(get_file_name('raw_data', sce), 'r') as f_raw:
        raw_data = csv.reader(f_raw)
        headers = raw_data.next()
        col = {}
        for i, x in enumerate(headers):
            if x == 'did':
                col['rep_id'] = i
            elif x == 'oid':
                col['det_id'] = i
            elif x == 'sid':
                col['veh_type'] = i
            elif x == 'ent':
                col['step'] = i
            elif x == 'flow':
                col['flow'] = i
            elif x == 'speed':
                col['speed'] = i
            elif x == 'density':
                col['density'] = i

        # save the data in a sorted dictionary
        # data['rep_id']['det_id']['step'] = [pAV, flow(veh/hr), speed(kph), density(veh/km)]
        data = OrderedDict()
        idx_pAV = 0
        idx_flow = 1
        idx_speed = 2
        idx_density = 3
        idx_flow_AV = 4
        len_idx = 5

        # skipped unnecessary things

        i = 0
        det_used = [str(d) for d in config['det_used']]
        reps_to_skip = [str(r) for r in config['reps_to_skip']]
        for row in raw_data:
            # skip the lines for repliation 798, which is used to generate the initial congested state
            rep_id = row[col['rep_id']]
            det_id = row[col['det_id']]
            step = int(row[col['step']])

            sys.stdout.write('\r')
            sys.stdout.write('Status: processing row {0}'.format(i))
            sys.stdout.flush()
            i += 1
            # -------------------------------------------------------------
            # skip freeflow replications
            # if rep_id == '670' or rep_id == '671' or rep_id == '672' or rep_id == '673' or rep_id == '674':
            #     continue

            # skip detectors
            if det_id not in det_used:
                # print('skipeed detector {0}'.format(det_id))
                continue

            # skip unneeded data
            # those are the average replication or the warm up replication.
            if rep_id in reps_to_skip:
                # print('skipped replication {0}'.format(rep_id))
                continue

            # this is the cumulative flow and speed
            if step == 0:
                # print('skipped cumulative flow')
                continue
            # -------------------------------------------------------------
            # print('eff')
            if rep_id not in data.keys():
                data[rep_id] = OrderedDict()

            if det_id not in data[rep_id].keys():
                data[rep_id][det_id] = OrderedDict()

            if step not in data[rep_id][det_id].keys():
                data[rep_id][det_id][step] = np.zeros(len_idx)

            # save different types of vehicles
            if row[col['veh_type']] == '0':
                # the total flow, speed, density
                data[rep_id][det_id][step][idx_flow] = float(row[col['flow']])
                data[rep_id][det_id][step][idx_speed] = float(row[col['speed']])
                data[rep_id][det_id][step][idx_density] = float(row[col['density']])
            elif row[col['veh_type']] == str(config['av_type']):
                # the row for AV
                data[rep_id][det_id][step][idx_flow_AV] = float(row[col['flow']])

    # now compute the penetration rate
    for rep_id in data.keys():
        for det_id in data[rep_id].keys():
            for step in data[rep_id][det_id].keys():

                if data[rep_id][det_id][step][idx_flow] != 0:
                    data[rep_id][det_id][step][idx_pAV] = data[rep_id][det_id][step][idx_flow_AV] \
                                                          / data[rep_id][det_id][step][idx_flow]
            # Now sort the steps
            data[rep_id][det_id] = OrderedDict(sorted(data[rep_id][det_id].items()))

    # save in to formatted data for post processing
    formatted_data = []
    for rep_id in data.keys():
        for det_id in data[rep_id].keys():
            for step in data[rep_id][det_id].keys():
                formatted_data.append(data[rep_id][det_id][step][:-1])
    formatted_data = np.array(formatted_data).astype(float)

    print('\nFinished formatting data')
    print('formated data size: {0}'.format(formatted_data.shape))

    # =================================================================
    # clean the formatted data
    # - removing points with zero speed. (completely stopped traffic, density will be inf by q/v)
    # - removing data point where both flow and speed are below threshold.
    #     The reason is the flow measurement has an error +- 360. In severe congestion, the speed is small,
    #     e.g, 1 mph. and flow is small e.g. 720 veh/hr, which gives a reasonable density 720 veh/mile. But
    #     if the flow is off by 360, the density will be 360 veh/mile.
    nonZeroSpeed = (formatted_data[:, 2] > 0)

    # convert data to mph, and veh/mile
    formatted_data[:, 2] = formatted_data[:, 2] / 1.609
    formatted_data[:, 3] = formatted_data[:, 3] * 1.609
    flow = formatted_data[:, 1]
    speed = formatted_data[:, 2]
    cleanData = ~((flow <= config['flow_thres']) & (speed <= config['speed_thres'])) & nonZeroSpeed
    cleaned_data = formatted_data[cleanData, :]

    # =================================================================
    # output cleaned detector data in file
    cleaned_header = ['pav', 'flow(veh/hr)', 'speed(mph)', 'density(veh/mile)']
    with open(get_file_name('det_data', sce), 'w+') as f_clean:
        f_clean.write(','.join(i for i in cleaned_header) + '\n')
        for row in range(0, cleaned_data.shape[0]):
            f_clean.write(','.join(str(i) for i in cleaned_data[row, :]) + '\n')


if __name__ == "__main__":
    sys.exit(main(sys.argv))
