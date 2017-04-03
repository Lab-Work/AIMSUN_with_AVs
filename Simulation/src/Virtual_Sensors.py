# Author: Juan Carlos Martinez
# 3/21/2016


import csv
import sys
import time
import warnings
from collections import OrderedDict
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

warnings.filterwarnings("error")

"""
    class virtual_sensors

        PUBLIC

            __init__(self, work_zone, time_step, directory, sections, main_grid_order):
                loads the default simulation and virtual sensor parameters
                loads the trajetories data
                organizes the data

            generate_virtual_sensors_data()
                parses the information of the sensors specified in the to_generate file
                builds a crosses matrix that holds the crossing information for a sensor
                generates the virtual_sensor based on that crossing matrix
                save the results in the appropriate files

            generate_true_states_data(grid_resolution, plot)
                generate maps for speed and density
                generate true state for travel time # PENDING
                save the results in the appropriate files and plot if specified


        PRIVATE

            __load_data
                loads the trajectories data

            __organize_data
                organizes the data by the following dictionary structure:
                self.__organized_data[replication][vehicle][timestamp]
                the value for each key is an array with the following entries:
                [ent, section, section_distance, travelled_distance, lane_idx, speed, acceleration]

            __build_volume_sensor
                build the sensor crosses and readings for each simulation replication
                save the generated data in the appropriate files

            __build_sensor_crosses
                build the sensor crosses matrix
                use binary search to find the crossing time for the upstream bound of the reading area
                iterate after the found timestamp to find the crossing of the downstream bound

            __add_lane_to_sensor_location
                compute the locations of the upstream and downstream visibility boundaries of a sensor

            __build_volume_sensor_readings
                obtain noise parameters, check for occlusion and build sensor readings

            __occlusion_check
                check if occlusion occurs for a given vehicle cross on a sensor visibility region

            __build_travel_time_sensor
                build sensor crossings at the upstream and downstream sensor
                build the travel time pair readings
                save the generated data in the appropriate file

            __build_travel_time_sensor_readings
               obtain travel time noise parameters
               filter vehicles recorded based on penetration rate
               add the travel times of individual vehicles and compute means for each interval

            __build_true_speed_density_states
                set sensor locations (cells) along the main road
                call helper function to build true state at each cell
                plot if specified by the user

            __build_true_speed_density_states_readings
                organize the sensor locations
                generate no-noise sensor readings for each location along the road
                stack the data into speed and density stacks (each row is a time interval)


        """


class Virtual_Sensors:
    def __init__(self, work_zone, sections, main_grid_order,
                 replications,
                 aimsun_start_dur_step=(55800, 9000, 1.0), space_start_end=None, ):
        """

        :param work_zone: str, 'I57', 'I80'
        :param time_step:
        :param log_dir: the top level directory of the logs.
        :param data_dir: the top level directory of the actual data, default E:\\Workzone\\
        :param sections:
        :param main_grid_order:
        :param aimsun_start_dur_step: [start, dur, step ] the start timestamp, duration and step length in seconds
        :param space_start_end: the absolute start and end location of the space domain under investigation
        :param config: the configuration
        :return:
        """

        print('\n')
        print('Constructing virtual_sensors class...')

        self.__work_zone = work_zone
        self.__time_step = aimsun_start_dur_step[2]
        self.__sections = sections
        self.__main_grid_order = main_grid_order
        self.__space_start = space_start_end[0]
        self.__space_end = space_start_end[1]
        self.replications = replications

        self.__start_timestamp = aimsun_start_dur_step[0]
        self.__end_timestamp = self.__start_timestamp + aimsun_start_dur_step[1]

        # raw data from sqlite table
        self.__organized_data = None
        self.__raw_data = None
        self.__veh_type = OrderedDict()

    def generate_true_states_data(self, grid_resolution, traj_file, veh_type_file, true_prefix):
        """
        This function loads the trajectory sqlite file and genreate the true states (speed and density in imperial) in
        true_prefix files
        :param grid_resolution: (s,m) the resolution of the true grid
        :param traj_file: trajectory data file csv
        :param veh_type_file: csv file for the MIVEHTRAJECTORY table
        :param true_prefix: prefix_density.txt; prefix_speed.txt
        :return:
        """

        # =================================================
        # load the sqlite file, and get the detailed trajectory table
        if self.__organized_data is None:
            time0 = time.time()
            print('Status: loading data...')
            self.__load_data(traj_file, veh_type_file)
            time1 = time.time()
            print('Status: Loaded trajectory data. Took {0:.2f} s. Organizing...'.format(time1-time0))
            self.__organize_data()

            time1 = time.time()
            elapsed_time = '{0:.2f}'.format(time1 - time0)
            print('Elapsed time: ' + elapsed_time + ' s')

        time0 = time.time()
        print('\n')
        print('Generating true states data...')
        self.__build_true_flow_density_speed_states(grid_resolution, true_prefix)
        time1 = time.time()
        elapsed_time = '{0:.2f}'.format(time1 - time0)
        print('Elapsed time: ' + elapsed_time + ' s')

    def __parse_sensor_line(self, sensor):
        """
        This function parses the sensor file whose data need to be generated
        :param sensor: the sensor line
        :return:
        """
        # Parse, load and update sensor parameters
        sensor_parameters = OrderedDict()
        sensor_parameters['custom'] = OrderedDict()
        for couple in sensor.split(';'):
            category = couple.split(':', 1)[0].rstrip()
            value = couple.split(':', 1)[1].rstrip()
            if value[0] == '[':
                value = [float(i) for i in value.strip('[]').split(',')]
            elif value.find('.') != -1:
                value = float(value)
            elif value.isdigit():
                value = int(value)
            elif value == 'True':  # the occlusion
                value = True
            elif value == 'False':
                value = False

            if (category != 'type' and category != 'id' and category != 'section' and category != 'distance' and
                        category != 'section_up' and category != 'section_down' and category != 'distance_up' and category != 'distance_down'
                ):
                sensor_parameters['custom'][category] = value
            else:
                sensor_parameters[category] = value
        sensor_parameters['default'] = self.__default_parameters[sensor_parameters['type']].copy()
        if sensor_parameters['custom']:
            for category in sensor_parameters['custom']:
                sensor_parameters['default'][category] = sensor_parameters['custom'][category]

        print('Status: -- generating data for sensor {0}...'.format(sensor_parameters['id']))

        return sensor_parameters


    def __load_data(self, traj_file, veh_type_file):

        self.__raw_data = np.zeros((0,11))
        for rep in self.replications:
            self.__raw_data = np.concatenate( [self.__raw_data,
                                               np.genfromtxt(traj_file, dtype='str', delimiter=',')])

        # read the veh_type table file and save in to dictionary
        with open(veh_type_file, 'r') as f_veh_type:
            for line in f_veh_type:
                # each line:
                # rep_id, veh_id, type_id, origin_id, destination_id, entranceTime, exitTime, travelTime, delayTime
                if len(line) == 0:
                    continue

                items = line.strip().split(',')
                if int(items[0]) not in self.replications:
                    # skip unrelated replications
                    continue

                # keep the vehicle id and type both as int
                self.__veh_type[int(items[1])] = int(items[2])


    def __organize_data(self):

        self.__organized_data = OrderedDict()

        # variable that keeps track how far in the network a vehicle has traversed
        length_prev_sections_and_tapers = 0.0

        for idx, line in enumerate(self.__raw_data):

            replication = int(line[0])
            vehicle = int(line[1])
            ent = int(line[2])
            section = int(line[3])

            # we only input the few sections that covers the road segment under estimation.
            if section not in self.__sections.keys():
                continue

            lane_idx = int(line[4])
            timestamp = float(line[7])
            speed = float(line[8])
            travelled_distance = float(line[9])
            acceleration = float(line[10])

            if replication not in self.__organized_data:
                self.__organized_data[replication] = OrderedDict()

            if vehicle not in self.__organized_data[replication]:
                self.__organized_data[replication][vehicle] = OrderedDict()

            if idx:
                prev_line = self.__raw_data[idx - 1, :]
                prev_replication = int(prev_line[0])
                prev_vehicle = int(prev_line[1])
                prev_section = int(prev_line[3])
                prev_timestamp = float(prev_line[7])
                prev_travelled_distance = float(prev_line[9])

                if replication != prev_replication or vehicle != prev_vehicle:
                    length_prev_sections_and_tapers = 0.0

                elif section != prev_section:

                    # reconstruct the trajectories on the taper
                    length_prev_sections_and_tapers = length_prev_sections_and_tapers + self.__sections[prev_section][
                        'length']
                    prev_section_remaining_length = length_prev_sections_and_tapers - prev_travelled_distance

                    for node_section in self.__sections:
                        # find the right taper based on the sections upstream and downstream
                        # zero acceleration at the tapers is assumed (constant velocity)
                        if 'connections' in self.__sections[node_section]:
                            if (
                                            prev_section in self.__sections[node_section]['connections']['upstream'] and
                                            section in self.__sections[node_section]['connections']['downstream']
                            ):
                                node_lane_idx = lane_idx
                                node_speed_m_s = (travelled_distance - prev_travelled_distance) / (
                                    timestamp - prev_timestamp)
                                node_speed_km_h = float('{0:.2f}'.format(node_speed_m_s * 3600 / 1000))
                                node_acceleration_m_s2 = 0.0
                                node_timestamps = np.arange(prev_timestamp + self.__time_step,
                                                            timestamp, self.__time_step)
                                for node_timestamp in node_timestamps:
                                    node_timestamp = float('{0:.2f}'.format(node_timestamp))
                                    # avoid floating point rounding error
                                    if abs(timestamp - node_timestamp) < self.__time_step / 2:
                                        pass
                                    else:
                                        node_section_distance = float('{0:.2f}'.format(node_speed_m_s * (
                                            node_timestamp - prev_timestamp) - prev_section_remaining_length))
                                        node_travelled_distance = float('{0:.2f}'.format(
                                            prev_travelled_distance + prev_section_remaining_length + node_section_distance))
                                        # if the vehicle is still on the previous section
                                        if node_section_distance < 0:
                                            selected_section = prev_section
                                            selected_section_distance = float('{0:.2f}'.format(
                                                self.__sections[prev_section]['length'] + node_section_distance))
                                        else:
                                            selected_section = node_section
                                            selected_section_distance = node_section_distance
                                        self.__organized_data[replication][vehicle][node_timestamp] = [-1,
                                                                                                       selected_section,
                                                                                                       selected_section_distance,
                                                                                                       node_travelled_distance,
                                                                                                       node_lane_idx,
                                                                                                       node_speed_km_h,
                                                                                                       node_acceleration_m_s2]
                                length_prev_sections_and_tapers = length_prev_sections_and_tapers + \
                                                                  self.__sections[node_section]['length']
                                # break and resume with next section
                                break

            section_distance = float('{0:.2f}'.format(travelled_distance - length_prev_sections_and_tapers))
            # set trajectory points that have section_distance exceeding the specified section distance to have the maximum
            # allowable value
            # this takes care of possible inconcistencies in the trajectory data
            if section_distance > self.__sections[section]['length']:
                section_distance = self.__sections[section]['length']

            self.__organized_data[replication][vehicle][timestamp] = [ent, section, section_distance,
                                                                      travelled_distance, lane_idx, speed, acceleration]


    def __build_true_flow_density_speed_states(self, resolution, truestate_file_prefix, av_type=436):

        """
        This function generates matrices that hold the true traffic state on
        a road network based on Edie's definitions
        :param resolution: Tuple [aggregation in seconds, aggregation distance in meters]
        :param replication: Replication number
        :param plot: Boolean (True if plots are requested, False otherwise)
        :return:
        """

        # extract resolution and construct cell space and time boundaries
        agg_sec = resolution[0]
        agg_dist = resolution[1]
        cell_area = agg_sec * agg_dist
        main_len = 0.0
        section_bounds = [0.0]
        for section in self.__main_grid_order:
            main_len = main_len + self.__sections[section]['length']
            section_bounds.append(float('{0:.2f}'.format(main_len)))

        # get the space cell grids only for the section investigation
        num_pt = round((self.__space_end - self.__space_start) / agg_dist) + 1
        space_cell_bounds = np.linspace(self.__space_start, self.__space_end, num_pt)
        # print('VS: space_cell_bounds: {0}'.format(space_cell_bounds))

        # get the time cell grids only for the time domain under investigation
        num_pt = round((self.__end_timestamp - self.__start_timestamp) / agg_sec) + 1
        time_cell_bounds = np.linspace(self.__start_timestamp, self.__end_timestamp, num_pt) - self.__start_timestamp
        # print('VS: time_cell_bounds: {0}'.format(time_cell_bounds))

        # initialize an empty matrix for the space sum and time sum on each cell
        # for n cell boundaries on an axis, there are n-1 cells on that axis
        spaces_sum_mat = np.zeros((len(space_cell_bounds) - 1, (len(time_cell_bounds) - 1)))
        times_sum_mat = np.zeros((len(space_cell_bounds) - 1, (len(time_cell_bounds) - 1)))
        av_times_sum_mat = np.zeros((len(space_cell_bounds) - 1, (len(time_cell_bounds) - 1)))

        real_space_cell_bounds = space_cell_bounds
        # space_cell_bounds = np.unique(list(space_cell_bounds) + list(section_bounds))

        # iterate over every vehicle
        # assuming there is only one replication
        if len(self.replications) > 1:
            raise Exception('Current true state genereator only supports one replication in each sqlite,' +
                            'which is highly recommended for memory efficiency')
        else:
            replication = self.replications[0]

        for vehicle in self.__organized_data[replication]:

            sys.stdout.write('\r')
            sys.stdout.write('Status: processing vehicle {0}'.format(vehicle))
            sys.stdout.flush()

            # get timestamps, sections and distances in arrays
            timestamps = np.array(list(self.__organized_data[replication][vehicle].keys()))
            sections = np.array([self.__organized_data[replication][vehicle][timestamp][1] for
                                 timestamp in self.__organized_data[replication][vehicle]])
            distances = np.array([self.__organized_data[replication][vehicle][timestamp][2] for
                                  timestamp in self.__organized_data[replication][vehicle]])

            # get idxs of the sections that match the sections on the main grid
            # adjust section distances to be absolute to the beginning of main grid
            # extract main absolute distances and timestamps
            main_idxs = []
            for section in self.__main_grid_order:
                idxs = np.array(np.where(sections == section)[0]).tolist()
                up_sect_len = self.__get_up_sect_len(section)
                distances[idxs] = distances[idxs] + up_sect_len
                timestamps[idxs] = timestamps[idxs] - self.__start_timestamp
                main_idxs.extend(idxs)
            main_distances = distances[[main_idxs]]
            main_timestamps = timestamps[[main_idxs]]

            # get rid of duplicate distances and timestamps
            dup_idxs = [idx for idx, item in enumerate(main_distances) if item in main_distances[:idx]]
            main_distances = np.delete(main_distances, dup_idxs)
            main_timestamps = np.delete(main_timestamps, dup_idxs)

            # print('checking vehicle: {0}'.format(vehicle))
            # print('length of vehicle traj {0}'.format(len(distances)))
            # print('length of main_distances: {0}'.format(len(main_distances)))

            # Make sure the vehicle was on the main freeway
            if len(main_distances) > 1:
                # obtain cell bounds that are relevant to current vehicle
                veh_space_cell_bounds = space_cell_bounds[space_cell_bounds >= main_distances[0] - agg_dist]
                veh_space_cell_bounds = veh_space_cell_bounds[veh_space_cell_bounds <= main_distances[-1] + agg_dist]
                veh_time_cell_bounds = time_cell_bounds[time_cell_bounds >= main_timestamps[0] - agg_sec]
                veh_time_cell_bounds = veh_time_cell_bounds[veh_time_cell_bounds <= main_timestamps[-1] + agg_sec]

                # set a 1st order interpolation/extrapolation for the absoulte distances and timestamps
                order = 1
                # interpolate/extrapolate on space given time
                space_spl = InterpolatedUnivariateSpline(main_timestamps, main_distances, k=order)
                # interpolate/extrapolate on time given space
                time_spl = InterpolatedUnivariateSpline(main_distances, main_timestamps, k=order)
                time_at_space_bounds = self.__interpolate_space_crosses(time_spl, veh_space_cell_bounds)
                space_at_time_bounds = self.__interpolate_time_crosses(space_spl, veh_time_cell_bounds)
                crosses = (np.sort(time_at_space_bounds + space_at_time_bounds, axis=0)).tolist()
                crosses = [x for i, x in enumerate(crosses) if not i or x != crosses[i - 1]]

                # update values on spaces_sum_mat and times_sum_mat
                for i in range(1, len(crosses) - 2):
                    ti, xi = crosses[i][0], crosses[i][1]
                    tf, xf = crosses[i + 1][0], crosses[i + 1][1]
                    if ti >= time_cell_bounds[0] and xi >= real_space_cell_bounds[0] and ti < time_cell_bounds[-1] \
                            and xi < real_space_cell_bounds[-1]:
                        delta_t = tf - ti
                        delta_x = xf - xi
                        t_idx, d_idx = int((ti - time_cell_bounds[0]) / agg_sec), int(
                            (xi - real_space_cell_bounds[0]) / agg_dist)
                        times_sum_mat[d_idx, t_idx] = times_sum_mat[d_idx, t_idx] + delta_t
                        spaces_sum_mat[d_idx, t_idx] = spaces_sum_mat[d_idx, t_idx] + delta_x

                        # add av spent time
                        try:
                            if self.__veh_type[vehicle] == av_type:
                                av_times_sum_mat[d_idx, t_idx] = av_times_sum_mat[d_idx, t_idx] + delta_t
                        except KeyError:
                            print('Warning: no matching veh id {0} in veh_type data'.format(vehicle))
                            continue

        # compute flow, density and speed based on edie's definitions
        spaces_sum_mat = np.matrix(spaces_sum_mat)
        times_sum_mat = np.matrix(times_sum_mat)
        av_times_sum_mat = np.matrix(av_times_sum_mat)
        q_mat = np.true_divide(spaces_sum_mat, cell_area)  # veh/s
        k_mat = np.true_divide(times_sum_mat, cell_area)  # veh/m

        with np.errstate(divide='ignore', invalid='ignore'):
            v_mat = np.true_divide(spaces_sum_mat, times_sum_mat)  # m/s
            v_mat[v_mat == np.inf] = np.nan

            # compute the true fraction of AVs in each cell.
            # when the density is 0, the w wil be set as 0
            w_mat = np.true_divide(av_times_sum_mat, times_sum_mat)
            w_mat[np.isinf(w_mat)] = 0
            w_mat[np.isnan(w_mat)] = 0


        # save the true states on the appropriate files
        true_density_file = truestate_file_prefix + '_density.txt'.format(agg_sec, agg_dist)
        true_w_file = truestate_file_prefix + '_w.txt'.format(agg_sec, agg_dist)
        true_speed_file = truestate_file_prefix + '_speed.txt'.format(agg_sec, agg_dist)

        # fill in the nan values, using no update
        dim_space, dim_time = v_mat.shape
        for x in range(0, dim_space):
            for t in range(0, dim_time):
                if np.isnan(v_mat[x, t]):
                    # replace the nan value by not updating
                    j = t - 1
                    while j != -1 and np.isnan(v_mat[x, j]):
                        j -= 1

                    if j != -1:
                        # if found a value that is not nan
                        v_mat[x, t] = v_mat[x, j]

        # save speed to mph
        np.savetxt(true_speed_file, (3600.0/1609.0)*v_mat.T, delimiter=',')
        # save density to veh/mile
        np.savetxt(true_density_file, 1609.0*k_mat.T, delimiter=',')
        np.savetxt(true_w_file, w_mat.T, delimiter=',')

    def __interpolate_space_crosses(self, time_spl, space_cell_bounds):

        """
        This function interpolates on space given a time spline
        :param time_spl: time spline that takes space
        :param space_cell_bounds: space cell bounds
        :return vector of tuples time_space_cross
        """

        time_space_cross = []
        for space_bound in space_cell_bounds:
            time_space_cross.append(
                [float('{0:.2f}'.format(float(time_spl(space_bound)))), float('{0:.2f}'.format(space_bound))])
        return time_space_cross

    def __interpolate_time_crosses(self, space_spl, time_cell_bounds):

        """
        This function interpolates on time given a space spline
        :param space_spl: space spline that takes time
        :param time_cell_bounds: time cell bounds
        :return vector of tuples time_space_cross
        """

        time_space_cross = []
        for time_bound in time_cell_bounds:
            time_space_cross.append(
                [float('{0:.2f}'.format(time_bound)), float('{0:.2f}'.format(float(space_spl(time_bound))))])
        return time_space_cross

    def __get_up_sect_len(self, section):

        """
        This function returns the cumulative lengths of the main grid
        sections that are upstream of the section given
        :param section: section id on the main grid
        :return: float up_sect_len
        """

        up_sect_len = 0.0
        for main_sect in self.__main_grid_order:
            if section == main_sect:
                return up_sect_len
            else:
                up_sect_len = up_sect_len + self.__sections[main_sect]['length']

        raise Exception('Error: The main grid does not contain the section {0}'.format(section))

    @staticmethod
    def __sensordict2string(sensor_id, sensor_att):
        """
        This function converts the sensor key-value store to a string in the following format.
        Entries are separated by ; The entry name and value are separated by :, first entry is id:
        :param sensor_id: the sensor id
        :param sensor_att: key-value store of the sensors.
                general_keys: id, type, section, distance...
                default: Dict, first copied from default parameters and then overwritten by custom values. Use this dict for parameters
                custom: Dict, the set of parameters that have been written to default.
        :return: a string in the above defined format
        """

        line = []
        # append the first id entry
        entry = ':'.join(['id', sensor_id])
        line.append(entry)

        # append other attribute entries
        for key in sensor_att.keys():

            if key != 'id':
                # Do Not repeat id entry

                if key == 'custom':
                    # skip the custom entries since they have been written to default sets.
                    continue
                elif key == 'default':
                    # add all parameters in default dict
                    for para in sensor_att['default']:
                        entry = ':'.join([para, str(sensor_att['default'][para])])
                        line.append(entry)
                else:
                    # other keys, such as type, distance...
                    entry = ':'.join([key, str(sensor_att[key])])
                    line.append(entry)

        # join all entries and return
        return ';'.join(line)
