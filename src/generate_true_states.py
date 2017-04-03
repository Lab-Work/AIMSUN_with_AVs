import sys
import time
from collections import OrderedDict
from os.path import exists
import sqlite3
import csv

import matplotlib.pyplot as plt
import numpy as np

from Virtual_Sensors import Virtual_Sensors

__author__ = 'Yanning Li'
"""
This is the script for visualizing the simulated traffic states using trajectory data generated from AIMSUN.
"""


# =====================================================================================================
# Configure folder path
# =====================================================================================================

if sys.platform == 'win32':
    config_file = 'C:\\Users\Admin\\Dropbox\\RenAimsunSimulation\\configuration_file.txt'
    folder_dir = 'C:\\Users\\Admin\\Dropbox\\RenAimsunSimulation\\Simulation\\'
    # config_file = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\configuration_file.txt'
    # folder_dir = 'C:\\Users\\TrafficControl\\Dropbox\\RenAimsunSimulation\\Simulation\\'
    logger_path = folder_dir + 'Logs\\'
elif sys.platform == 'darwin':
    config_file = '../configuration_file.txt'
    folder_dir = '../Simulation/'
    logger_path = folder_dir + 'Logs/'


# =====================================================================================================
# No modification after this.
# =====================================================================================================

def main(argv):
    config = load_configuration(config_file)
    plot_all = False

    dt = config['dt']
    dx = config['hwy_length'] / config['num_cells']
    grid_res = (dt, dx)

    workzone = 'I00'
    # load the work zone configuration.
    time_grid, space_grid, workzone_topo, aimsun_start_dur_step = \
        load_workzone(folder_dir + 'topology.txt', grid_res)

    # print('Loaded the work zone topology.')
    # print('-- time_grid: {0}'.format(time_grid))
    # print('-- space_grid: {0}'.format(space_grid))

    # ========================================================================
    # Generate the true state for all simulation scenarios and seeds
    generated_truestates = OrderedDict()
    for sce in config['sim_scenarios']:
        if sce not in generated_truestates.keys():
            generated_truestates[sce] = []

        for seed in config['sim_seeds']:

            # Generate a Virtual_Sensor class for processing trajectory
            truestate_generator = Virtual_Sensors(workzone,
                                          workzone_topo['sections'],
                                          workzone_topo['fwy_sec_order'],
                                          workzone_topo['replications'],
                                          aimsun_start_dur_step,
                                          [space_grid[0], space_grid[-1]])

            truestate_file_prefix = get_file_name('truestate_prefix', sce, seed,
                                                  grid=grid_res)
            # check if previously generated
            if exists(truestate_file_prefix + '_density.txt'):
                print('\nTrue state for sce {0} seed {1} previousely genereated'.format(sce, seed))
                generated_truestates[sce].append(seed)

                if plot_all:
                    # plot the generated true speed
                    title = 'true speed sce {0} seed {1}'.format(sce, seed)
                    true_speed_file = get_file_name('truestate_prefix', sce, seed, grid=grid_res) + '_speed.txt'
                    plot_true_speed_for_rep(true_speed_file, unit='imperial', limit=[0, 80], title=title, save_fig=True)

                    # plot the generated true density
                    title = 'true density sce {0} seed {1}'.format(sce, seed)
                    true_density_file = get_file_name('truestate_prefix', sce, seed, grid=grid_res) + '_density.txt'
                    plot_true_density_for_rep(true_density_file, unit='imperial', limit=[0, 644], title=title, save_fig=True)

                continue

            # ==================================================
            # convert sqlite to trajectory data if csv not there
            print('\nGenerating true state for for sce {0} seed {1} ...'.format(sce, seed))
            if not exists(get_file_name('traj_data',sce, seed)):

                # ==================================================
                # generate true state for this scenario and seed
                while not exists(get_file_name('simdone', sce, seed)):
                    print('AIMSUN simulation for sce {0} seed {1} has not finished...'.format(sce, seed))
                    time.sleep(10)

                extract_traj_to_csv(get_file_name('sqlite', sce, seed),
                                    get_file_name('traj_data', sce, seed))

            traj_file = get_file_name('traj_data', sce, seed)
            veh_type_file = get_file_name('veh_type', sce, seed)
            # true states are saved in mph and veh/mile
            truestate_generator.generate_true_states_data(grid_res, traj_file, veh_type_file,
                                                          truestate_file_prefix)

            # ==================================================
            # plot and save true state
            unit = 'imperial'
            speed_bar = [0, 80]  # limit used for visualization,
            density_bar = [0, 644]  # veh/mile

            # plot the generated true speed
            title = 'true speed sce {0} seed {1}'.format(sce, seed)
            true_speed_file = get_file_name('truestate_prefix', sce, seed, grid=grid_res) + '_speed.txt'
            plot_true_speed_for_rep(true_speed_file, unit='imperial', limit=[0, 80], title=title, save_fig=True)

            # plot the generated true density
            title = 'true density sce {0} seed {1}'.format(sce, seed)
            true_density_file = get_file_name('truestate_prefix', sce, seed, grid=grid_res) + '_density.txt'
            plot_true_density_for_rep(true_density_file, unit='imperial', limit=[0, 644], title=title, save_fig=True)

            generated_truestates[sce].append(seed)

    plt.show()


def get_file_name(file_type, sce, seed, grid=(5, 178)):
    """
    Standardize the naming convenstion
    :param sce: scenario #, str
    :param seed: seed #, str
    :param det: detector id
    :param grid: tuple, (s, m), the trues tate grid
    :return: name str
    """
    if sys.platform == 'win32':
        if file_type == 'sqlite':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'simdone':
            return folder_dir + 'aimsun_files\\sim_sce{0}_seed{1}_SimDone.txt'.format(sce, seed)
        elif file_type == 'traj_data':
            return folder_dir + 'traj_data\\sim_sce{0}_seed{1}.csv'.format(sce, seed)
        elif file_type == 'truestate_prefix':
            return folder_dir + 'true_states\\truestate_{2}s{3}m_sce{0}_seed{1}'.format(sce, seed,
                                                                                    grid[0], int(grid[1]))
        elif file_type == 'veh_type':
            return folder_dir + 'traj_data\\sim_sce{0}_seed{1}_vehtype.csv'.format(sce, seed)
        else:
            raise Exception('Unrecognized file type for naming.')

    elif sys.platform == 'darwin':
        if file_type == 'sqlite':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}.sqlite'.format(sce, seed)
        elif file_type == 'simdone':
            return folder_dir + 'aimsun_files/sim_sce{0}_seed{1}_SimDone.txt'.format(sce, seed)
        elif file_type == 'traj_data':
            return folder_dir + 'traj_data/sim_sce{0}_seed{1}.csv'.format(sce, seed)
        elif file_type == 'truestate_prefix':
            return folder_dir + 'true_states/truestate_{2}s{3}m_sce{0}_seed{1}'.format(sce, seed,
                                                                                        grid[0], int(grid[1]))
        elif file_type == 'veh_type':
            return folder_dir + 'traj_data/sim_sce{0}_seed{1}_vehtype.csv'.format(sce, seed)
        else:
            raise Exception('Unrecognized file type for naming.')


def extract_traj_to_csv(infile, outfile):
    """
    This function extracts trajectory data from infile sqlite to csv
    :param infile: sqlite file name
    :param outfile: csv file name
    :return:
    """
    if exists(outfile):
        print('trjectory file {0} has been previously genreated'.format(outfile))
    else:
        con = sqlite3.connect(infile)
        cur = con.cursor()
        sqlite_data = cur.execute("SELECT * FROM MIVEHDETAILEDTRAJECTORY")

        with open(outfile, 'wb') as f_raw:
            writer = csv.writer(f_raw)
            writer.writerows(sqlite_data)


def load_workzone(topo_file, grid_res):
    """
    This function loads the topology of the network
    :param topo_file: the file path
    :param grid_res: (seconds, meters) tuple for the resolution of the visualization
    :return: t_grid, x_grid, topology of the network including the location of ramps, freeway sec order and replications
    """
    start_dur_step = None
    t_grid = None
    x_grid = None

    workzone_topo = {'sections': OrderedDict(),
                     'loc_onramp': None,
                     'loc_offramp': None,
                     'fwy_sec_order': None,
                     'replications': None}

    f_topo = open(topo_file, 'r')

    for line in f_topo:

        if line[0] == '#':
            continue

        if 'sec_id' in line:
            items = line.strip().split(';')
            sec_id = int(items[0].split(':')[1])
            workzone_topo['sections'][sec_id] = {}

            # add length
            workzone_topo['sections'][sec_id][items[1].split(':')[0]] = float(items[1].split(':')[1])

            if 'upstream' in line:
                workzone_topo['sections'][sec_id]['connections'] = {}
                for i in range(2, len(items)):
                    entry = items[i].split(':')
                    if entry[0] == 'upstream':
                        workzone_topo['sections'][sec_id]['connections']['upstream'] = [int(j) for j in
                                                                                        entry[1].split(',')]
                    elif entry[0] == 'downstream':
                        workzone_topo['sections'][sec_id]['connections']['downstream'] = [int(j) for j in
                                                                                          entry[1].split(',')]

        elif 'fwy_sec_order' in line:
            items = line.strip().split(':')
            workzone_topo['fwy_sec_order'] = [int(i) for i in items[1].split(',')]

        elif 'loc_start_end' in line:
            items = line.strip().split(':')
            start = float(items[1].split(',')[0])
            end = float(items[1].split(',')[1])

            # get x_grid
            x_grid = np.arange(end, start, - grid_res[1])
            if x_grid[-1] != start:
                x_grid = np.concatenate([x_grid, np.array([start])])
            x_grid = x_grid[::-1]

            # round every digit to two decimals
            for i in range(0, len(x_grid)):
                x_grid[i] = round(x_grid[i], 2)

        elif 'time_start_dur_step' in line:
            start_dur_step = [float(i) for i in line.split(':')[1].split(',')]
            t_grid = np.arange(0, start_dur_step[1], grid_res[0])
            if t_grid[-1] != start_dur_step[1]:
                t_grid = np.concatenate([t_grid, np.array([start_dur_step[1]])])

            # round every digit to two decimals
            for i in range(0, len(t_grid)):
                t_grid[i] = round(t_grid[i], 2)

        elif 'replications' in line:
            items = line.strip().split(':')
            workzone_topo['replications'] = [int(i) for i in items[1].split(',')]

    f_topo.close()

    # the following entries must be set in the topology file
    if t_grid is None or x_grid is None or workzone_topo['replications'] is None:
        raise Exception('Error: loc_start_end, simulation_duration, replications must be set in workzone topology.')
    else:
        t_grid = t_grid.tolist()
        x_grid = x_grid.tolist()

    return t_grid, x_grid, workzone_topo, start_dur_step


def plot_true_speed_for_rep(file_name, unit='imperial', limit=(0, 40), title=None,
                            save_fig=True, fig_size=(18,8), fontsize=(36,34,32)):
    """
    This function plots the true speed profile in the specified unit
    :param file_name: the true speed data in mph.
    :param unit: 'metric', 'imperial'; respectively 'm, s, m/s', and 'mile, hour, mph'
    :param limit: The limit of the colorbar in above units
    :return: A figure profile with x-axis being the time, and y-axis being the space. (flow direction upwards)
    """

    # read the result from file
    true_speed_file = file_name

    speed_data = np.genfromtxt(true_speed_file, delimiter=',')
    speed_data = np.matrix(speed_data).T

    if unit == 'metric':
        speed_data = speed_data*1609.0/3600.0
        speed = np.flipud(speed_data)
        unit_str = 'm/s'
    elif unit == 'imperial':
        speed = np.flipud(speed_data)
        unit_str = 'mph'
    else:
        raise Exception('Error: Unrecognized unit for plotting speed.')

    fig = plt.figure(figsize=fig_size, dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(speed, cmap=plt.get_cmap('jet_r'),
                   interpolation='nearest',
                   aspect='auto',
                   vmin=limit[0], vmax=limit[1])
    ax.autoscale(False)

    ax.set_title('{0} ({1})'.format(title, unit_str), fontsize=fontsize[0])

    plt.xlabel('Time', fontsize=fontsize[1])
    plt.ylabel('Space, traffic direction $\mathbf{\Rightarrow}$')

    cax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    fig.colorbar(im, cax=cax, orientation='vertical')
    if save_fig is True:
        fig_name = file_name.strip('txt') + 'png'
        plt.savefig(fig_name, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()


def plot_true_density_for_rep(file_name, unit='imperial', limit=(0, 40),
                              title=None, save_fig=True):
    """
    This function plots the true speed profile in the specified unit
    :param file_name: the true density data, in veh/mile
    :param unit: 'metric', 'imperial'; respectively 'm, s, m/s', and 'mile, hour, mph'
    :param limit: The limit of the colorbar in above units
    :return: A figure profile with x-axis being the time, and y-axis being the space. (flow direction upwards)
    """

    # read the result from file
    true_density_file = file_name

    density_data = np.genfromtxt(true_density_file, delimiter=',')
    density_data = np.matrix(density_data).T

    if unit == 'metric':
        # all internal values are in metric, so plot directly
        density_data = density_data/1609.0
        density = np.flipud(density_data)
        unit_str = 'veh/m'
    elif unit == 'imperial':
        density = np.flipud(density_data)
        # limit = self.__metric2imperial(np.array(limit), 'speed')
        unit_str = 'veh/mile'
    else:
        raise Exception('Error: Unrecognized unit for plotting speed.')

    fig = plt.figure(figsize=(18, 8), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(density, cmap=plt.get_cmap('jet'),
                   interpolation='nearest',
                   aspect='auto',
                   vmin=limit[0], vmax=limit[1])
    ax.autoscale(False)

    ax.set_title('{0} ({1})'.format(title, unit_str))
    plt.xlabel('Time')
    plt.ylabel('Space, traffic direction $\mathbf{\Rightarrow}$')
    cax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    fig.colorbar(im, cax=cax, orientation='vertical')

    if save_fig is True:
        fig_name = file_name.strip('txt') + 'png'
        plt.savefig(fig_name, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()


def __metric2imperial(value=np.zeros((1, 1)), option='speed'):
    """
    A utility function which converts the metric (m, s, m/s) to imperial (mile, hour, m/h)
    :param value: float, np.array, or np.matrix. the to be converted value
    :param option: 'speed', 'density'
    :return: converted value
    """

    if type(value) is float or type(value) is np.float64 \
            or type(value) is np.ndarray or type(value) is np.matrix:
        if option == 'speed':
            return value * 3600.0 / 1609.34
        elif option == 'density':
            return value * 1609.34
        elif option == 'distance':
            return value / 1609.34
        else:
            raise Exception('Error: Unrecognized unit conversion option.')
    else:
        print(type(value))
        raise Exception('Error: Unrecognized value type for unit conversion.')


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

        # convert sim_scenarios, sim_seeds to lists
        if type(config['sim_scenarios']) is not list:
            config['sim_scenarios'] = [config['sim_scenarios']]
        if type(config['sim_seeds']) is not list:
            config['sim_seeds'] = [config['sim_seeds']]

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
