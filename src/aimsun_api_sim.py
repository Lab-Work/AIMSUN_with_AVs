from copy import deepcopy
import sys
from datetime import datetime
import time
from datetime import timedelta
from AIMSUNFUNCTIONS_V4 import *

__author__ = 'Yanning Li'

"""
This script is used to automate AIMSUN simulation for different scenarios
- It generates demand with maximum inflow and varying AV percentages from an uniform distribution at the inflow.
- It simulates the traffic in AIMSUN and saved detector data to corresponding folder.
- it saves the trajectory data into sqlite. Once done, it will write to a flag file: sim_sce#_seed#_SimDone.txt
"""

# ======================================================================================
# Configure the directories
# ======================================================================================
# directory = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32':
    config_file = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\configuration_file.txt'
    folder_dir = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\Simulation\\'
    logger_path = folder_dir + 'Logs\\'
elif sys.platform == 'darwin':
    config_file = '../configuration_file.txt'
    folder_dir = '../Simulation/'
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
    print_cmd('=========================Start AIMSUN simulation for Scenarios=========================')
    print_cmd('Started at {0}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # ==========================================================
    # load configurations to simulate
    config = load_configuration(config_file)
    if isinstance(config['sim_scenarios'], int):
        config['sim_scenarios'] = [config['sim_scenarios']]
    if isinstance(config['sim_seeds'], int):
        config['sim_seeds'] = [config['sim_seeds']]

    # ==========================================================
    # simulate for each scenario and seed
    finished_simulation = OrderedDict()
    for pav in config['sim_scenarios']:

        # keep a record of finished simulations
        finished_simulation[pav] = []

        for seed in config['sim_seeds']:

            run_time = datetime.now() - start_time
            print_cmd('\n------------------------------------------------------')
            print_cmd('Simulation run time  : {0}'.format(str(run_time)))
            print_cmd('Simulated:')
            for key in finished_simulation.keys():
                print_cmd('---- pav {0}: {1}'.format(key, finished_simulation[key]))
            print_cmd('-------- Simulating pav {0}, seed {1} ----------'.format(pav, seed))
            print_cmd('------------------------------------------------------')

            # ==========================================================
            # check if previously simulated
            if exists(get_file_name('simdone', pav, seed)):
                print_cmd('Status: scenario {0} with seed {1} has '.format(pav, seed) +
                          'been simulated previously.')
                finished_simulation[pav].append(seed)
                continue

            # ==========================================================
            # if original aimsun ang file is not in the cmd input line
            if (len(argv) < 2):
                print_cmd('Usage: aconsole.exe -script SCRIPT ANG_FILE')
                return -1

            # ==========================================================
            # generate and reset the demand in AIMSUN
            generate_demand_in_pav_range(pav, seed, config)

            # ==========================================================
            # save into another separate aimsun ang file
            console = ANGConsole()
            if console.open(argv[1]):
                print_cmd('\nAimsun opening {0} ...\n'.format(argv[1]))

                # -----------------------------------------------------
                # set up AIMSUN
                model = GKSystem.getSystem().getActiveModel()

                # TODO: load demand from matrix file
                demand = import_od_to_aimsun(get_file_name('demand', pav, seed), model,
                                             'sim_demand')

                # create the scenario, experiment
                scenario_name = 'sim_sce'

                det_cycle_str = time.strftime("%H:%M:%S", time.gmtime(config['detection_cycle']) )
                scenario = setup_scenario(model, scenario_name, demand,
                                          det_interval= det_cycle_str)
                print('status: set scenario')
                exp_name = 'sim_exp'
                experiment = setup_experiment(model, exp_name, scenario)
                print('status: set experiment')

                # create one replication
                avg_result = setup_replication(model, experiment, 1, [seed])
                print('status: finished setup of rep')

                # plugin is a module which can compute the average for the GKExperimentResult object
                plugin = GKSystem.getSystem().getPlugin("GGetram")
                # print('plugin:{0}'.format(plugin))
                simulator = plugin.getCreateSimulator(model)
                # print('plugin get simulator: {0}'.format(plugin.getSimulator()))

                # old way for creating simulator
                # simulator = create_simulator(model)
                # -----------------------------------------------------
                # Now save to file and prepare to simulate
                console.save(get_file_name('aimsun', pav, seed))
                print_cmd('Saved AIMSUN file {0}'.format(get_file_name('aimsun', pav, seed)))

                simulate_experiment(simulator, avg_result)

                # -----------------------------------------------------
                # get and save the detector data
                all_det_data = extract_detector_data(model, avg_result, plugin)
                save_detector_data(pav, seed, all_det_data, config)

                print_cmd('=======================================================')
                print_cmd('---- Finished simulating pav_{0}_seed{1}'.format(pav, seed))
                print_cmd('\nAimsun is now closing...\n')
                console.close()

                # -----------------------------------------------------
                # flag the end of the simulation
                with open(get_file_name('simdone', pav, seed), 'w+') as f:
                    f.write('AIMSUN simulation finished for scenario with {0}% AVs and seed {1}'.format(pav, seed))

                finished_simulation[pav].append(seed)
                time.sleep(3)  # pause a while to make sure aimsun closed
            else:
                console.getLog().addError("Could not open")

        print_cmd('Finished simulating pav_{0} with seeds: {1}'.format(pav, config['sim_seeds']))
    # stop logger and save into file
    stop_cmd_logger()



def generate_demand_in_pav_range(pav, seed, config):
    """
    This function generates the demand for scenario pav with seed
    The maximum flow at different penetration that can be generated is:
    pAv =  [0.01,    0.1,   0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    0.99]
    flow = [3600,   3780    3520    4400    3740    3000    3700    4300,   4420    4960    6600]
    select [3600,   3600    3600    3600    3600    3600    4000    4400    4400    5000    6600]
    :param pav: str, e.g. 35, percentage of autonomous vehicles
    :param seed: str, random seed which is used for naming purpose
    :param config: the configuration dict
    :return: demand is saved in the corresponding file
    """

    # interpolate the maximum inflow under different pavs
    _pavs = np.array(config['pAVs']).astype(float)/100.0
    _flows = np.array(config['max_inflows']).astype(float)

    # max_flow = interpolate.interp1d(_pavs, _flows)    # scipy broke on the laptop, use numpy
    max_flow = lambda pav_100: np.interp( pav_100/100.0, _pavs, _flows)

    us_centroid = int(config['us_centroid'])
    ds_centroid = int(config['ds_centroid'])
    car_id = int(config['car_id'])
    av_id = int(config['av_id'])

    max_av = float(pav)

    # Set the duration of each simulation step
    duration = int(config['interval_duration'])
    num_dur = int(config['num_intervals'])

    # this is the initial time for the simulation
    t_ini = datetime(2016, 6, 6, 0, 0, 0, 0)

    with open(get_file_name('demand', pav, seed), 'w+') as f_OD:
        tmp_id = 0
        t_start = t_ini
        for i in range(0, num_dur):
            # Get the duration of this state
            dt = timedelta(seconds=duration)
            dt_str = (t_ini + dt).strftime('%H:%M:%S')

            # generate a random pav
            seed_pav = int(random.uniform(0, max_av))
            seed_flow = max_flow(seed_pav)

            # this is the number of cars and AVs that will be sent during this time interval
            count_car = seed_flow * (1 - seed_pav / 100.0) * duration / 3600
            count_av = seed_flow * (seed_pav / 100.0) * duration / 3600

            # write the car matrix
            tmp_id += 1
            f_OD.write('{0} car_{1}_{2}\n'.format(tmp_id, t_start.strftime('%H:%M'), seed_pav))
            f_OD.write('{0} Car\n'.format(car_id))
            f_OD.write(t_start.strftime('%H:%M:%S') + '\n')
            f_OD.write(dt_str + '\n')
            f_OD.write('{0} {1} {2}\n'.format(us_centroid, ds_centroid, count_car))
            f_OD.write('\n')

            # write the AV matrix
            tmp_id += 1
            f_OD.write('{0} AV_{1}_{2}\n'.format(tmp_id, t_start.strftime('%H:%M'), seed_pav))
            f_OD.write('{0} AV\n'.format(av_id))
            f_OD.write(t_start.strftime('%H:%M:%S') + '\n')
            f_OD.write(dt_str + '\n')
            f_OD.write('{0} {1} {2}\n'.format(us_centroid, ds_centroid, count_av))
            f_OD.write('\n')

            # update the start time
            t_start += dt




def generate_demand_at_pav_val(pav, seed, config):
    """
    This function generates the demand for scenario pav with seed
    The maximum flow at different penetration that can be generated is:
    pAv =  [0.01,    0.1,   0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    0.99]
    flow = [3600,   3780    3520    4400    3740    3000    3700    4300,   4420    4960    6600]
    select [3600,   3600    3600    3600    3600    3600    4000    4400    4400    5000    6600]
    :param pav: str, e.g. 35, percentage of autonomous vehicles
    :param seed: str, random seed which is used for naming purpose
    :param config: the configuration dict
    :return: demand is saved in the corresponding file
    """

    # interpolate the maximum inflow under different pavs
    _pavs = np.array(config['pAVs']).astype(float)/100.0
    _flows = np.array(config['max_inflows']).astype(float)

    # max_flow = interpolate.interp1d(_pavs, _flows)    # scipy broke on the laptop, use numpy
    max_flow = lambda pav_100: np.interp( pav_100/100.0, _pavs, _flows)

    us_centroid = int(config['us_centroid'])
    ds_centroid = int(config['ds_centroid'])
    car_id = int(config['car_id'])
    av_id = int(config['av_id'])

    max_av = float(pav)

    # Set the duration of each simulation step
    duration = int(config['interval_duration'])
    num_dur = int(config['num_intervals'])

    # this is the initial time for the simulation
    t_ini = datetime(2016, 6, 6, 0, 0, 0, 0)

    with open(get_file_name('demand', pav, seed), 'w+') as f_OD:
        tmp_id = 0
        t_start = t_ini
        for i in range(0, num_dur):
            # Get the duration of this state
            dt = timedelta(seconds=duration)
            dt_str = (t_ini + dt).strftime('%H:%M:%S')

            # generate a random pav
            seed_pav = max_av
            seed_flow = max_flow(seed_pav)

            # this is the number of cars and AVs that will be sent during this time interval
            count_car = seed_flow * (1 - seed_pav / 100.0) * duration / 3600
            count_av = seed_flow * (seed_pav / 100.0) * duration / 3600

            # write the car matrix
            tmp_id += 1
            f_OD.write('{0} car_{1}_{2}\n'.format(tmp_id, t_start.strftime('%H:%M'), seed_pav))
            f_OD.write('{0} Car\n'.format(car_id))
            f_OD.write(t_start.strftime('%H:%M:%S') + '\n')
            f_OD.write(dt_str + '\n')
            f_OD.write('{0} {1} {2}\n'.format(us_centroid, ds_centroid, count_car))
            f_OD.write('\n')

            # write the AV matrix
            tmp_id += 1
            f_OD.write('{0} AV_{1}_{2}\n'.format(tmp_id, t_start.strftime('%H:%M'), seed_pav))
            f_OD.write('{0} AV\n'.format(av_id))
            f_OD.write(t_start.strftime('%H:%M:%S') + '\n')
            f_OD.write(dt_str + '\n')
            f_OD.write('{0} {1} {2}\n'.format(us_centroid, ds_centroid, count_av))
            f_OD.write('\n')

            # update the start time
            t_start += dt




def get_file_name(file_type, sce, seed, det='pm0', grid=(5, 178)):
    """
    Standardize the naming convenstion
    :param sce: scenario #, str
    :param seed: seed #, str
    :param det: detector id
    :param grid: tuple, (s, m), the trues tate grid
    :return: name str
    """
    if sys.platform == 'win32':
        if file_type == 'aimsun':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}.ang'.format(sce, seed)
        elif file_type == 'demand':
            return folder_dir + 'demand_data\\sim_sce{0}_seed{1}_matrix.txt'.format(sce, seed)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'simdone':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}_SimDone.txt'.format(sce, seed)
        elif file_type == 'traj_data':
            return folder_dir + 'traj_data\\sim_sce{0}_seed{1}.csv'.format(sce, seed)
        elif file_type == 'convertdone':
            return folder_dir + 'traj_data\\sim_sce{0}_seed{1}_ConvertDone.txt'.format(sce, seed)
        elif file_type == 'det_data':
            return folder_dir + 'detector_data\\sim_sce{0}_seed{1}_{2}.csv'.format(sce, seed, det)
        elif file_type == 'truespeed':
            return folder_dir + 'true_states\\truestate_sce{0}_seed{1}_speed.csv'.format(sce, seed)
        elif file_type == 'truedensity':
            return folder_dir + 'true_states\\truestate_sce{0}_seed{1}_density.csv'.format(sce, seed)
    elif sys.platform == 'darwin':
        if file_type == 'aimsun':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}.ang'.format(sce, seed)
        elif file_type == 'demand':
            return folder_dir + 'demand_data/sim_sce{0}_seed{1}_matrix.txt'.format(sce, seed)
        elif file_type == 'sqlite':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'simdone':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}_SimDone.txt'.format(sce, seed)
        elif file_type == 'traj_data':
            return folder_dir + 'traj_data/sim_sce{0}_seed{1}.csv'.format(sce, seed)
        elif file_type == 'convertdone':
            return folder_dir + 'traj_data/sim_sce{0}_seed{1}_ConvertDone.txt'.format(sce, seed)
        elif file_type == 'det_data':
            return folder_dir + 'detector_data/sim_sce{0}_seed{1}_{2}.csv'.format(sce, seed, det)
        elif file_type == 'truespeed':
            return folder_dir + 'true_states/truestate_sce{0}_seed{1}_{2}s{3}m_speed.csv'.format(sce, seed,
                                                                                                 grid[0], grid[1])
        elif file_type == 'truedensity':
            return folder_dir + 'true_states/truestate_sce{0}_seed{1}{2}s{3}m__density.csv'.format(sce, seed,
                                                                                                   grid[0], grid[1])


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


def save_detector_data(sce, seed, detector_data, config):
    """
    This function saves the data from detectors
    :param sce: str, the scenario id, pav, e.g., 35
    :param seed: str, the seed id for naming
    :param detector_data: dict from AIMSUN
        detector_data[det_id] = [ [speed(mph)], [count] ]
    :param config: the configuration dict
    :return: saved in detector files
    """

    # set the time stamps, in seconds
    num_intervals = int(config['num_intervals'])
    interval_dur = float(config['interval_duration'])

    # detection cycle
    step_dur = float(config['detection_cycle'])
    num_steps = int(num_intervals*interval_dur/step_dur)

    _times = np.arange(0, num_steps) + 1.0
    timestamps = _times * step_dur

    for det in detector_data.keys():
        # save the detector data in corresponding files
        with open(get_file_name('det_data', sce, seed, det), 'w+') as f_det:
            f_det.write('timestamps(s), speed (mph), count\n')  # header
            for i in range(0, num_steps):
                f_det.write('{0},{1},{2}\n'.format(timestamps[i],
                                                 detector_data[det][0][i],
                                                 detector_data[det][1][i]))


def import_od_to_aimsun(od_file, model, demand_name):
    """
    This function loads the OD matrix to AIMSUN from the OD matrix file
    :param od_file: the OD matrix file in AIMSUN required format
    :param demand_name: the demand name specified in AIMSUN
    :return: the demand
    """
    centroid_config_id = 336

    centroidConf = model.getCatalog().find(centroid_config_id)

    # Set up a Traffic Demand and add it to the model
    if demand_name is not None:
        demand = model.getCatalog().findByName(QString(demand_name))
        print_cmd('Loaded aimsun demand {0}'.format(demand_name))
    else:
        demand = None

    if demand is None or demand.isA(QString("GKTrafficDemand")) is False:
        # create new demand
        demand = GKSystem.getSystem().newObject("GKTrafficDemand", model, -1, True)
        demand.setName(QString(demand_name))
        print_cmd('Demand {0} not found. Creating new one...'.format(demand_name))

        # create under demand folder
        folder = __getDemandFolder(model)
        folder.append(demand)

    # clear schedule in demand to prevent overlapping
    demand.removeSchedule()

    # load the matrix to centroid configurations
    state = 0
    matrix = None
    for line in open(od_file, "r").readlines():
        line = line.strip()
        if len(line) == 0:
            state = 0
        else:
            if state == 0:
                state = 1
                # read identifier and name
                matrixId = int(line[:line.index(' ')])
                matrixName = line[line.index(' ') + 1:]
                matrix = centroidConf.getModel().getCatalog().find(matrixId)
                if matrix == None or matrix.isA("GKODMatrix") == False:
                    matrix = GKSystem.getSystem().newObject("GKODMatrix", model)
                    matrix.setName(matrixName)
                    centroidConf.addODMatrix(matrix)
                else:
                    print "New Matrix"
            elif state == 1:
                state = 2
                # read vehicle id and name
                vehId = int(line[:line.index(' ')])
                vehName = line[line.index(' ') + 1:]
                vehicle = centroidConf.getModel().getCatalog().find(vehId)
                if vehicle == None:
                    vehicle = centroidConf.getModel().getCatalog().findByName(vehName)
                if vehicle != None:
                    matrix.setVehicle(vehicle)
            elif state == 2:
                state = 3
                # From Time
                matrix.setFrom(QTime.fromString(line, Qt.ISODate))
            elif state == 3:
                state = 4
                # Duration
                matrix.setDuration(GKTimeDuration.fromString(line))
            elif state == 4:
                # Trips
                tokens = line.split(" ")
                if len(tokens) == 3:
                    fromCentroid = centroidConf.getModel().getCatalog().find(int(tokens[0]))
                    toCentroid = centroidConf.getModel().getCatalog().find(int(tokens[1]))
                    trips = float(tokens[2])
                    # Set the value if the section is valid
                    if fromCentroid != None and toCentroid != None:
                        matrix.setTrips(fromCentroid, toCentroid, trips)

                        # ==========================================================
                        # finished reading one matrix, now set to demand
                        schedule = GKScheduleDemandItem()
                        schedule.setTrafficDemandItem(matrix)

                        # set the time and duration
                        hr = matrix.getFrom().hour()
                        minute = matrix.getFrom().minute()
                        sec = matrix.getFrom().second()

                        duration = matrix.getDuration().toSeconds()[0]

                        if duration < 3600:
                            print('{4}: setting schedule {0}:{1}:{2}, duration {3}'.format(
                                hr, minute, sec, duration, matrixName,))
                            schedule.setFrom(3600 * hr + 60 * minute + sec)
                            schedule.setDuration(duration)
                            demand.addToSchedule(schedule)

    return demand


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
