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
This script is used to extract the detector data (including AV penetration rate.)
"""
# ======================================================================================
# Configure the directories
# ======================================================================================
# directory = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32':
    config_file = '..\\configuration_file.txt'
    folder_dir = '..\\Simulation\\'
    logger_path = folder_dir + 'Logs\\'
elif sys.platform == 'darwin':
    config_file = '../configuration_file.txt'
    folder_dir = '../simulation/'
    logger_path = folder_dir + 'Logs/'


# ======================================================================================
# End of configurations in this block
# ======================================================================================


def main(argv):
    config = load_configuration(config_file)
    # ============================================================
    # get all detector data
    file_list = []

    # in case not a list
    if type(config['sim_scenarios']) is not list:
        config['sim_scenarios'] = [config['sim_scenarios']]
    if type(config['sim_seeds']) is not list:
        config['sim_seeds'] = [config['sim_seeds']]

    for pAV in config['sim_scenarios']:

        for seed in config['sim_seeds']:

            if not exists(get_file_name('det_data', pAV, seed)+'PM0.csv'):

                print('\nExtracting detector data from scenario {0} seed {1}...'.format(pAV, seed))
                # extract from sqlite and calibrate FD.
                extract_det_data(pAV, seed, config)
            else:
                print('\nDetector data for scenario {0} seed {1} previously generated'.format(pAV, seed))


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


def get_file_name(file_type, sce, seed):
    """
    Standardize the naming convention for fd calibration
    :param file_type: the file type to be saved
    :param sce: scenario #, str, e.g. 23
    :return: name str
    """
    if sys.platform == 'win32':
        if file_type == 'det_data':
            return folder_dir + 'detector_data\\sim_sce{0}_seed{1}_'.format(sce, seed)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'raw_data':
            return folder_dir + 'raw_det_data\\raw_sim_sce{0}_seed{1}.csv'.format(sce, seed)
        else:
            raise Exception('Unrecognized file type for naming.')
    elif sys.platform == 'darwin':
        if file_type == 'det_data':
            return folder_dir + 'detector_data/sim_sce{0}_seed{1}_'.format(sce, seed)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'raw_data':
            return folder_dir + 'raw_det_data/raw_fd_sce{0}_seed{1}.csv'.format(sce, seed)
        else:
            raise Exception('Unrecognized file type for naming.')


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


def extract_det_data(sce, seed, config):
    """
    This function extracts the detector data from sqlite, formmat it and clean it.
    :param sce: str, or int, the scenario id for naming
    :param config: the configuration loaded from config file
    :return: saved in det_data folder
    """

    # =================================================================
    # extract raw data from sqlite
    if True:
        print('extracting MIDETEC from {0}'.format(get_file_name('sqlite', sce, seed)))
        con = sqlite3.connect(get_file_name('sqlite', sce, seed))
        cur = con.cursor()
        sqlite_data = cur.execute("SELECT * FROM MIDETEC")

        # get column names
        headers = [description[0] for description in cur.description]

        with open(get_file_name('raw_data', sce, seed),'wb') as f_raw:
            writer = csv.writer(f_raw)
            writer.writerow(headers)
            writer.writerows(sqlite_data)

    # =================================================================
    # load csv and loop through csv file, which is much faster than looping through sqlite
    # extract the specific columns to the following format
    # columns: p_rate, [flow(veh/hr), speed(kph), density(veh/km)],
    with open(get_file_name('raw_data', sce, seed), 'r') as f_raw:
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
        # det_used = [str(d) for d in config['det_used']]
        sim_rep = str(config['sim_replication'])
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
            # if det_id not in det_used:
            #     # print('skipeed detector {0}'.format(det_id))
            #     continue

            # skip unneeded data
            # those are the average replication or the warm up replication.
            if rep_id != sim_rep:
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
                # convert speed to mph
                data[rep_id][det_id][step][idx_speed] = float(row[col['speed']])/1.609
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


    # ==========================================================
    # save in to detector data set for each detector
    # set the time stamps, in seconds
    num_intervals = int(config['num_intervals'])
    interval_dur = float(config['interval_duration'])
    step_dur = float(config['detection_cycle'])
    num_steps = int(num_intervals*interval_dur/step_dur)
    _times = np.arange(0, num_steps) + 1.0
    timestamps = _times * step_dur

    # get the detector names for naming
    det_names = config['det_names']
    det_ids = [str(d) for d in config['det_used']]
    det = OrderedDict()
    for i,d in enumerate(det_ids):
        det[d] = det_names[i]
    det_data_prefix = get_file_name('det_data', sce, seed)

    if len(data.keys()) > 1:
        raise Exception('Only support one replication per simulation, which is highly recommended for efficiency.')
    for rep_id in data.keys():
        for det_id in data[rep_id].keys():

            with open(det_data_prefix+det[det_id]+'.csv', 'w+') as f:
                f.write('timestamps(s),penetration_AV,flow(veh/hr),speed(mph)\n')

                for step in data[rep_id][det_id].keys():
                    f.write(str(timestamps[step-1]) + ',' + ','.join(str(i) for i in data[rep_id][det_id][step][:-2] ) + '\n')


if __name__ == "__main__":
    sys.exit(main(sys.argv))
