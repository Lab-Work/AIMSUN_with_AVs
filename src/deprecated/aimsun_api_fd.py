from copy import deepcopy
import sys
from datetime import datetime
import time
from datetime import timedelta
from matplotlib.offsetbox import AnchoredText
import sqlite3
import csv
from AIMSUNFUNCTIONS_V4 import *

__author__ = 'Yanning Li'

"""
This script is used to run precconfigured aimsun files, gets the detector data.
- It loads multiple AIMSUN files.
- It simulates the loaded ang files with no additional configurations. The simulated detector data are in sqlite file.
- It exports the detector data to csv, which will be cleaned and saved.
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


# Load a network using ANGConsole
def main(argv):
    # ==========================================================
    # start simulation logger
    start_time = datetime.now()
    start_cmd_logger(logger_path, start_time)
    print_cmd('=========================Start AIMSUN simulation for FD calibration=========================')
    print_cmd('Started at {0}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # ==========================================================
    # load configurations to simulate
    config = load_configuration(config_file)
    if isinstance(config['sim_scenarios'], int):
        config['sim_scenarios'] = [config['sim_scenarios']]

    # ==========================================================
    # simulate each ang file and extract detector data
    finished_simulation = []
    for pav in config['sim_scenarios']:
        run_time = datetime.now() - start_time
        print_cmd('\n------------------------------------------------------')
        print_cmd('Simulation run time  : {0}'.format(str(run_time)))
        print_cmd('Simulated:{0}'.format(finished_simulation))
        print_cmd('-------- Simulating pav {0} ----------'.format(pav))
        print_cmd('------------------------------------------------------')

        # ==========================================================
        # check if previously simulated
        if exists(get_file_name('fdsimdone', pav)):
            print_cmd('Status: scenario {0} has '.format(pav) +
                      'been simulated previously.')

            if ~exists(get_file_name('det_data', pav)):
                extract_clean_det_data(pav, config)
            finished_simulation.append(pav)
            continue

        # ==========================================================
        # if original aimsun ang file is not in the cmd input line
        if (len(argv) < 2):
            print_cmd('Usage: aconsole.exe -script SCRIPT ANG_FILE')
            return -1

        # ==========================================================
        # save into another separate aimsun ang file
        console = ANGConsole()
        if console.open(argv[1]):
            print_cmd('\nAimsun opening {0} ...\n'.format(argv[1]))

            # -----------------------------------------------------
            # set up AIMSUN
            model = GKSystem.getSystem().getActiveModel()

            # load all replications
            cong_avg = model.getCatalog().findByName(QString('cong_avg'))
            free_avg = model.getCatalog().findByName(QString('free_avg'))

            # plugin is a module which can compute the average for the GKExperimentResult object
            plugin = GKSystem.getSystem().getPlugin("GGetram")
            # print('plugin:{0}'.format(plugin))
            simulator = plugin.getCreateSimulator(model)
            # print('plugin get simulator: {0}'.format(plugin.getSimulator()))
            # old way for creating simulator
            # simulator = create_simulator(model)

            # -----------------------------------------------------
            # simulate all replications; TODO: to test if this works
            print_cmd('Simulating congestion...')
            simulate_experiment(simulator, cong_avg)
            print_cmd('Simulating free flow...')
            simulate_experiment(simulator, free_avg)

            print_cmd('---- Finished simulating pav_{0}_seed{1}'.format(pav))
            print_cmd('\nAimsun is now closing...\n')
            console.close()
            time.sleep(3)  # pause a while to make sure aimsun closed

            # -----------------------------------------------------
            # export sqlite MIDETEC to csv files
            extract_clean_det_data(pav, config)

            finished_simulation.append(pav)
        else:
            console.getLog().addError("Could not open")

    # stop logger and save into file
    stop_cmd_logger()


def extract_clean_det_data(sce, config):
    """
    This function extracts the detector data from sqlite, formmat it and clean it.
    :param sce: str, or int, the scenario id for naming
    :param config: the configuration loaded from config file
    :return: saved in det_data folder
    """
    # extract raw data
    con = sqlite3.connect(get_file_name('sqlite', sce))
    cur = con.cursor()
    raw_data = cur.execute("SELECT * FROM MIDETEC")

    # get column names
    headers = [description[0] for description in cur.description]

    # =================================================================
    # extract the specific columns to the following format
    # columns: p_rate, [flow(veh/hr), speed(kph), density(veh/km)],
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

    for row in raw_data:
        # skip the lines for repliation 798, which is used to generate the initial congested state
        rep_id = row[col['rep_id']]
        det_id = row[col['det_id']]
        step = int(row[col['step']])

        # -------------------------------------------------------------
        # skip freeflow replications
        # if rep_id == '670' or rep_id == '671' or rep_id == '672' or rep_id == '673' or rep_id == '674':
        #     continue

        # skip detectors
        if det_id not in config['det_used']:
            # print('skipeed detector {0}'.format(det_id))
            continue

        # -------------------------------------------------------------
        # skip unneeded data
        # those are the average replication or the warm up replication.
        if rep_id in config['reps_to_skip']:
            # print('skipped replication {0}'.format(rep_id))
            continue

        # this is the cumulative flow and speed
        if step == 0:
            # print('skipped cumulative flow')
            continue
        # -------------------------------------------------------------

        if rep_id not in data.keys():
            data[ rep_id ] = OrderedDict()

        if det_id not in data[rep_id].keys():
            data[rep_id][det_id] = OrderedDict()

        if step not in data[rep_id][det_id].keys():
            data[rep_id][det_id][step] = np.zeros(len_idx)

        # now save the data into the corresponding location
        if row[col['veh_type']] == 0:
            # the total flow, speed, density
            data[rep_id][det_id][step][idx_flow] = float( row[col['flow']] )
            data[rep_id][det_id][step][idx_speed] = float( row[col['speed']] )
            data[rep_id][det_id][step][idx_density] = float( row[col['density']] )
        elif row[col['veh_type']] == config['av_type']:
            # the row for AV
            data[rep_id][det_id][step][idx_flow_AV] = float( row[col['flow']] )

        # Now compute the penetration ratio
        for rep_id in data.keys():
            for det_id in data[rep_id].keys():
                for step in data[rep_id][det_id].keys():

                    if data[rep_id][det_id][step][idx_flow] != 0:
                        data[rep_id][det_id][step][idx_pAV] = data[rep_id][det_id][step][idx_flow_AV]\
                                                              /data[rep_id][det_id][step][idx_flow]
                # Now sort the steps
                data[rep_id][det_id] = OrderedDict( sorted( data[rep_id][det_id].items() ) )

    # save in to formatted data for post processing
    formatted_data = []
    for rep_id in data.keys():
        for det_id in data[rep_id].keys():
            for step in data[rep_id][det_id].keys():
                formatted_data.append([data[rep_id][det_id][step][:-1]])
    formatted_data = np.array(formatted_data).astype(float)

    # =================================================================
    # clean the formatted data
    # - removing points with zero speed. (completely stopped traffic, density will be inf by q/v)
    # - removing data point where both flow and speed are below threshold.
    #     The reason is the flow measurement has an error +- 360. In severe congestion, the speed is small,
    #     e.g, 1 mph. and flow is small e.g. 720 veh/hr, which gives a reasonable density 720 veh/mile. But
    #     if the flow is off by 360, the density will be 360 veh/mile.
    nonZeroSpeed = (formatted_data[:,2] > 0)

    # convert data to mph, and veh/mile
    formatted_data[:,2] = formatted_data[:,2]/1.609
    formatted_data[:,3] = formatted_data[:,3]*1.609
    flow = formatted_data[:,1]
    speed = formatted_data[:,2]
    cleanData = ~( (flow <= config['flow_thres']) & (speed <= config['speed_thres']) ) & nonZeroSpeed
    cleaned_data = formatted_data[cleanData,:]

    # =================================================================
    # output cleaned detector data in file
    cleaned_header = ['pav', 'flow(veh/hr)', 'speed(mph)', 'density(veh/mile)']
    with open(get_file_name('det_data', sce), 'w+') as f_clean:
        f_clean.write( ','.join( i for i in cleaned_header ) + '\n' )
        for row in range(0, cleaned_data.shape[0] ):
            f_clean.write( ','.join( str(i) for i in cleaned_data[row,:]) + '\n' )


def get_file_name(file_type, sce=1):
    """
    Standardize the naming convention for fd calibration
    :param file_type: the file type to be saved
    :param sce: scenario #, str
    :return: name str
    """
    if sys.platform == 'win32':
        if file_type == 'aimsun':
            return folder_dir + 'aimsun_files\\fd_sce{0}.ang'.format(sce)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files\\fd_sce{0}.sqlite'.format(sce)
        elif file_type == 'fdsimdone':
            return folder_dir + 'aimsun_files\\fd_sce{0}_SimDone.txt'.format(sce)
        elif file_type == 'det_data':
            return folder_dir + 'detector_data\\fd_sce{0}.csv'.format(sce)
        else:
            raise Exception('Unrecognized file type for naming.')
    elif sys.platform == 'darwin':
        if file_type == 'aimsun':
            return folder_dir + 'aimsun_files/fd_sce{0}.ang'.format(sce)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files/fd_sce{0}.sqlite'.format(sce)
        elif file_type == 'fdsimdone':
            return folder_dir + 'aimsun_files/fd_sce{0}_SimDone.txt'.format(sce)
        elif file_type == 'det_data':
            return folder_dir + 'detector_data/fd_sce{0}.csv'.format(sce)
        else:
            raise Exception('Unrecognized file type for naming.')


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


if __name__ == "__main__":
    sys.exit(main(sys.argv))
