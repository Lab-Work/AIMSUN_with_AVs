__author__ = 'Carlos and Yanning'
"""
This script contains functions requried to automate AIMSUN.
"""

import sys
import os
from os.path import exists
import time
from datetime import datetime
import csv
import random
from collections import OrderedDict

from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyANGAimsun import *

SITEPACKAGES = "C:\\Python27\\Lib\\site-packages"
sys.path.append(SITEPACKAGES)
import numpy as np
import matplotlib.pyplot as plt

# SITEPACKAGES = "C:\\Anaconda2\\Lib\\site-packages"
# sys.path.append(SITEPACKAGES)
# from scipy import interpolate


# Files used in this file
# parsFilePath = 'c:/tmp/pars.txt'
# flowsFilePath = 'c:/tmp/flows.txt'
# turnsFilePath = 'c:/tmp/turns.txt'

# ====================================================================================================
# Uncomment the following are just to remove the error messages

# def QString():
#     pass
# def QVariant():
#     pass
# def GKScheduleDemandItem():
#     pass
# GKTimeDuration = None
# GKVehicle = None
# GKSystem = None
# GKSection = None
# GKExperiment = None
# GKVehicleReactionTimes = None
# def GAimsunSimulator():
#     pass
# GKSimulationTask = None
# GKReplication = None
# GGetramModule = None
# GKTimeSerieIndex = None
# GK = None
# GKColumnIds = None
# Qt = None
# QTime = None


# ====================================================================================================
# The following variables need to be shared by functions (init, and update)
cmd_logger = None
opt_logger = None
opt_logger_file = None
# TODO: Windows is not properly configured yet. Hence could not plot figure.
# Figure handle is to plot the optimization steps (fig, ax)
fig_handle = None

# Define the columns for the files to be read
# pars.txt
INDEX_PARS_STATE = 0
INDEX_PARS_VEH = 1
INDEX_PARS_FROMTIME = 2
INDEX_PARS_DUR = 3
# flows.txt
INDEX_FLOWS_STATE = 0
INDEX_FLOWS_SECID = 1
INDEX_FLOWS_INFLOW = 2
# turns.txt
INDEX_TURNS_STATE = 0
INDEX_TURNS_FROM = 1
INDEX_TURNS_TO = 2
INDEX_TURNS_PERC = 3

KMH2MPH = 0.62137

_debug = False
_show_detector_data = False


# ====================================================================================================
# --------------------------- Functions for reading demand or data from files ------------------------
def read_demand_from_file(model, demand_name, pars_file, flows_file, turns_file,
                          main_entrance_id, main_entrance_discount_ratio):
    """
    Read the demand from files to create traffic states.
    :param model: GK Model object
    :param demand_name: create the demand for this specific one
    :param pars_file: the file for pars
    :param flows_file: the file for flows
    :param turns_file: the file for turns
    :param main_entrance_id: the main entrance id
    :param main_entrance_discount_ratio: the discount ratio, e.g., 1.0 no change. 1.1, 10% higher
    :return:
    """

    # Set up a Traffic Demand and add it to the model
    if demand_name is not None:
        demand = model.getCatalog().findByName(QString(demand_name))
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

    # Read pars.txt into a dictionary
    pars_dict = __readParsFile(pars_file)
    flows_dict = __readFlowsFile(flows_file)
    turns_dict = __readTurnsFile(turns_file)

    for state_name in pars_dict.keys():
        state = __createState(model, state_name, pars_dict, flows_dict, turns_dict, main_entrance_id,
                              main_entrance_discount_ratio)

        if _debug:
            print_cmd(
                'state from {0}, duration:{1}\n'.format(state.getFrom().toString(), state.getDuration().toString()))

            print_cmd('state entrance flow {0}'.format(state.getEntranceFlow(model.getCatalog().find(int(330)), None)))

            print_cmd('state turn 330->(340, 341): ({0},{1})'.format(
                state.getTurningPercentage(model.getCatalog().find(int(330)),
                                           model.getCatalog().find(int(340)), None),
                state.getTurningPercentage(model.getCatalog().find(int(330)),
                                           model.getCatalog().find(int(341)), None)))

        schedule = __createScheduleItem(state)

        demand.addToSchedule(schedule)

    # append the demand to the demands folder
    # folder = __getDemandFolder(model)
    # folder.append(demand)

    return demand


def load_demand_from_ang(model, demand_name):
    """
    Load demand from the ang file with demand name
    THIS IS ONLY FOR DEBUGGING
    :param model:
    :param demand_name:
    :return:
    """
    # find the demand from the model
    demand = model.getCatalog().findByName(QString(demand_name))
    if demand is None or not demand.isA(QString("GKTrafficDemand")):
        print_cmd('Error: no traffic demand named {0}\n'.format(demand_name))
        return None

    print_cmd('Loaded demand named {0}\n'.format(demand_name))

    # demand.removeSchedule()
    #
    # # find state
    # stateType = model.getType("GKTrafficState")
    # for state in model.getCatalog().getObjectsByType(stateType).itervalues():
    #     print_cmd('loading state {0}'.format(state.getName()))
    #
    #     schedule = __createScheduleItem(state)
    #     demand.addToSchedule(schedule)

    return demand


def read_validation_data(start_time_str, end_time_str, name_strs, valid_file_path):
    """
    This code is to read validation data from csv file
    data format should be as follows: Make SURE there is Header
    Date/Time (UTC-06:00) Central Time (US & Canada),RadarSpeed (MPH),RadarVehiclesCount
    4/26/2015 15:55,,
    4/26/2015 16:00,72.23863137,51
    4/26/2015 16:05,71.23257753,91
    Note: to read data for a simulation from 03:00 to 04:00 (AIMSUN aggragate data at 03:05 + 00:05...)
    need to read 12 lines with start_time and end_time 03:00~to 04:00 (Carlos shifted the time 5 min earlier)

    :param start_time_str: "4/26/2015 16:00", string
    :param end_time_str:"4/26/2015 16:00", string; if start and end both None, the read all rows
    :param name_strs: a list of detector names
    :param valid_file_path: '', then current folder same as the script
    :return: dict: validation_data[detector_name] = np.array([[speed],[count]])
    """
    dict_valid = OrderedDict()

    # if *_time_str is None, then assume the entire file is the validation data
    if start_time_str is None or end_time_str is None:

        for name in name_strs:

            # first list speed, second list count
            dict_valid[name] = [[], []]

            file_name = valid_file_path + name + '.csv'
            f_handle = open(file_name, 'r')

            data_set = csv.reader(f_handle, delimiter=',')
            # skip header
            next(data_set, None)

            for row in data_set:
                dict_valid[name][0].append(float(row[1]))
                dict_valid[name][1].append(float(row[2]))

            f_handle.close()

    # Otherwise on read data in a specific time period
    else:
        dt_format = '%m/%d/%Y %H:%M'
        t_start = datetime.strptime(start_time_str, dt_format)
        t_end = datetime.strptime(end_time_str, dt_format)

        for name in name_strs:
            # first list speed, second list count
            dict_valid[name] = [[], []]

            file_name = valid_file_path + name + '.csv'
            f_handle = open(file_name, 'r')

            data_set = csv.reader(f_handle, delimiter=',')
            # skip header
            next(data_set, None)

            for row in data_set:

                cur_dt = datetime.strptime(row[0], dt_format)

                if t_start <= cur_dt < t_end:
                    # no need to save the time
                    dict_valid[name][0].append(float(row[1]))
                    dict_valid[name][1].append(float(row[2]))

            f_handle.close()

            # print 'Loaded validation data from time {0} to time {1}, [[speed/mph],[count/5min]]: {2}'.format(
            # start_time_str, end_time_str, dict_valid[name])

    return dict_valid
    # check if correctly read
    # for key in dict_valid:
    # print_cmd('key {0}: {1}'.format(key, dict_valid[key])


# =================================================================================================
# ---------------------- Functions for setting up the simulation ---------------------------------

def setup_scenario(model, scenario_name, demand, det_interval="00:05:00"):
    """
    Set up the scenario and connect scenario with demand
    :param model: GK Model
    :param scenario_name: string, the scenario name, must exist in ang file, or create a new one
    :param demand: demand object, The demand for this scenario
    :return:
    """
    print_cmd('\nSetting up scenario...')

    scenario = model.getCatalog().findByName(QString(scenario_name))
    if scenario is None or not scenario.isA(QString("GKScenario")):
        scenario = GKSystem.getSystem().newObject("GKScenario", model, -1, True)
        scenario.setName(QString(scenario_name))
        print_cmd('Error: no traffic scenario named {0}. Creating new one...\n'.format(scenario_name))

    scenario.setDemand(demand)

    # append the state to the state folder
    # folder = __getScenariosFolder(model)
    # folder.append(scenario)

    # set parameters here
    # parameters are set in the ScenarioInput data class
    paras = scenario.getInputData()

    # set the detection and statistical intervals as 5 min
    det_interval = det_interval
    paras.setDetectionInterval(GKTimeDuration.fromString(QString(det_interval)))
    paras.setStatisticalInterval(GKTimeDuration.fromString(QString(det_interval)))

    print_cmd('---- Detection interval is set as : {0}\n'.format(det_interval))

    # TODO: the following parameters does not need to be set
    # scenario.setDataValueByID( GKGenericScenario.weekdayAtt, QVariant('Monday') )
    # scenario.setDataValueByID( GKGenericScenario.seasonAtt, QVariant( 'Summer' ) )
    # scenario.setDataValueByID( GKGenericScenario.weatherAtt, QVariant( 'Sunny' ) )
    # scenario.setDataValueByID( GKGenericScenario.eventAtt, QVariant( 'Fair' ) )
    # print_cmd('scenario: {0}'.format(scenario.getDataValueByID( GKGenericScenario.seasonAtt )[0].toString() )
    # scenario.setValueForVariable(QString('seasonAtt'), QString('Spring'))
    # print_cmd('setup_scenario: '.format(scenario.getValueForVariable( QString('seasonAtt') ) )

    # var_dict = scenario.getVariables()
    # for key in var_dict:
    #     print_cmd('key {0}: {1}'.format(key, var_dict[key])
    # print_cmd('setup_scenario: get_variables: {0}'.format()

    return scenario


def setup_experiment(model, experiment_name, scenario):
    """
    This function sets up the experiment under scenario
    :param model: GK Model
    :param experiment_name: Name of the expeirment
    :param scenario:
    :return:
    """

    print_cmd('\nSetting up experiment...\n')

    experiment = model.getCatalog().findByName(QString(experiment_name))
    if experiment is None or not experiment.isA(QString("GKExperiment")):
        experiment = GKSystem.getSystem().newObject("GKExperiment", model, -1, True)
        print_cmd('ERROR: No traffic experiment named {0}. Creating a new one...\n'.format(experiment_name))

        # attach the new experiment to folder
        folder = __getScenariosFolder(model)
        folder.append(experiment)

    experiment.setScenario(scenario)

    return experiment


# Deprecated function
def setup_experiment_I80_EB(model, experiment_name, scenario, preset_paras_file):
    """
    This function sets upt the experiment with parameters
    :param model:
    :param experiment_name:
    :param scenario:
    :param preset_paras_file: the preset parameters, Better be set in ang file
    :return:
    """

    print_cmd('\nSetting up experiment...\n')

    experiment = model.getCatalog().findByName(QString(experiment_name))
    if experiment is None or not experiment.isA(QString("GKExperiment")):
        experiment = GKSystem.getSystem().newObject("GKExperiment", model, -1, True)
        print_cmd('ERROR: No traffic experiment named {0}. Creating a new one...\n'.format(experiment_name))

        # attach the new experiment to folder
        folder = __getScenariosFolder(model)
        folder.append(experiment)

    experiment.setScenario(scenario)

    # ==================================================================================
    # TODO: some parameters still needs to be set in the .ang file which is easier, including:
    # TODO: the speed limit, simulation step, strategy plan...

    # read preset paras, which contains the initial values
    # Here the preset parameters is limited to a few parameters
    preset_paras = read_preset_paras(preset_paras_file)

    # ==================================================================================
    # vehicle class parameters
    car_type = model.getCatalog().find(53)
    if not car_type.isA(QString("GKVehicle")):
        print_cmd('Error: Car type is not correct: demand may not correctly loaded\n')
        return None

    # unique truck id is 56
    truck_type = model.getCatalog().find(56)
    if not truck_type.isA(QString("GKVehicle")):
        print_cmd('Error: Truck type is not correct: demand may not correctly loaded\n')
        return None

    print_cmd('Setting preset parameters:')

    for key in preset_paras.keys():

        # key format: car_maxSpeed
        para_name = key.split('_')
        para_value = preset_paras[key]
        if para_name[0] == 'car':
            # car paras
            if para_name[1] == 'maxSpeed':
                car_type.setDataValueByID(GKVehicle.maxSpeedMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.maxSpeedDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.maxSpeedMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.maxSpeedMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'speedAcceptance':
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'maxAccel':
                car_type.setDataValueByID(GKVehicle.maxAccelMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.maxAccelDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.maxAccelMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.maxAccelMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'reactionTime':
                # [reaction_time, reaction_stop, reaction_light, reaction_prob]
                car_react = GKVehicleReactionTimes(para_value[0], para_value[1],
                                                   para_value[2], para_value[3])

                car_type.setVariableReactionTimes([car_react])
                experiment.setVariableReactionTimesMicro(car_type, [car_react])
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'minDist':
                car_type.setDataValueByID(GKVehicle.minDistMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.minDistDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.minDistMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.minDistMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'sensitivityFactor':
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            else:
                print_cmd('\n---- ERROR: Could not recognize preset parameter entry {0}: {1}\n'.format(key, para_value))


        elif key.split('_')[0] == 'truck':
            # truck paras
            if para_name[1] == 'maxSpeed':
                truck_type.setDataValueByID(GKVehicle.maxSpeedMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'speedAcceptance':
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'maxAccel':
                truck_type.setDataValueByID(GKVehicle.maxAccelMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.maxAccelDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.maxAccelMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.maxAccelMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'reactionTime':
                # [reaction_time, reaction_stop, reaction_light, reaction_prob]
                truck_react = GKVehicleReactionTimes(para_value[0], para_value[1],
                                                     para_value[2], para_value[3])

                truck_type.setVariableReactionTimes([truck_react])
                experiment.setVariableReactionTimesMicro(truck_type, [truck_react])
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'minDist':
                truck_type.setDataValueByID(GKVehicle.minDistMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.minDistDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.minDistMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.minDistMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'sensitivityFactor':
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            else:
                print_cmd('\n---- ERROR: Could not recognize preset parameter entry {0}: {1}\n'.format(key, para_value))

        else:
            print_cmd('\nERROR: Could not recognize preset parameter entry {0}\n '.format(key))

    return experiment


def setup_replication(model, experiment, num_rep, seed_list):
    """
    This function sets up the replications
    :param model: GK Model
    :param experiment: experiment
    :param num_rep: number of replications
    :param seed_list: the seed for each replication
    :return: the ave_result replication
    """

    print_cmd('\nSetting up replications...')

    if experiment != None and experiment.isA("GKExperiment") \
            and experiment.getSimulatorEngine() == GKExperiment.eMicro:

        # add replications here
        replication_list = experiment.getReplications()

        # ===============================
        # create new replications
        if len(replication_list) == 0:
            # create replications
            for i in range(0, num_rep):
                replication = GKSystem.getSystem().newObject("GKReplication", model, -1, True)
                replication.setExperiment(experiment)
                replication_list.append(replication)

                if seed_list is not None:
                    replication.setRandomSeed(seed_list[i])
                print_cmd('---- Created replication {0} with seed {1}'.format(replication.getId(),
                                                                              replication.getRandomSeed()))
        else:
            # show replcations:
            print_cmd('---- Reloading {0} replications: {1} \n'.format(len(replication_list),
                                                                       [replication.getId() for replication in
                                                                        replication_list]))

        # create the average experiment result
        avg_result = GKSystem.getSystem().newObject("GKExperimentResult", model)
        avg_result.setName('average_result')
        print_cmd('Created new average replication: {0}'.format(avg_result.getName()))
        # print_cmd('Total number of replications is: {0}',format(len(experiment.getReplications()))

        # set the experiment of this result object
        avg_result.setExperiment(experiment)
        # add replcations to the average
        for replication in replication_list:
            avg_result.addReplication(replication)
            print_cmd('---- Added replication {0} to {1}'.format(replication.getId(), avg_result.getName()))

        # compute the average; add to the experiment.
        experiment.addReplication(avg_result)

        return avg_result


def set_random_seed(avg_result):
    """
    This function reset the seed for avg_result replications
    :param avg_result: the avg_result replication
    :return:
    """
    replication_list = avg_result.getReplications()

    print_cmd('\nResetting seeds for replications:')

    i = 0
    for replication in replication_list:
        replication.setRandomSeed(random.randint(0, 10000))
        print_cmd('----Reset replication {0} with seed {1}'.format(replication.getId(),
                                                                   replication.getRandomSeed()))
        # replication.setRandomSeed(int(seed_list[i]))
        i += 1


def create_simulator(model):
    """
    Create a simulator
    :param model:
    :return:
    """
    simulator = GAimsunSimulator()
    simulator.setModel(model)

    return simulator


def set_new_paras(model, experiment, paras):
    """
    This function sets up the parameters for experiment
    :param model:
    :param experiment:
    :param paras: a dict
    :return:
    """
    # Unique car id is 53
    car_type = model.getCatalog().find(53)
    if not car_type.isA(QString("GKVehicle")):
        print_cmd('Error: Car type is not correct: demand may not correctly loaded\n')
        print_cmd(type(car_type))
        # debug_catalog(model)
        return None

    # unique truck id is 56
    truck_type = model.getCatalog().find(56)
    if not truck_type.isA(QString("GKVehicle")):
        print_cmd('Error: Truck type is not correct: demand may not correctly loaded\n')
        return None

    # note the parameter should be in the same format as the preset parameters
    # even if we are only calibrating the mean
    for key in paras.keys():

        para_name = key.split('_')
        para_value = paras[key]

        if para_name[0] == 'car':
            # car paras
            # calibrate speedAcceptanceMean and speedAcceptanceDev for freeflow
            if para_name[1] == 'speedAcceptance':
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.speedAcceptanceMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'minHeadway':
                car_type.setDataValueByID(GKVehicle.minimunHeadwayMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.minimunHeadwayDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.minimunHeadwayMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.minimunHeadwayMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            # calibrate maxAccelMean for congflow
            elif para_name[1] == 'maxAccel':
                car_type.setDataValueByID(GKVehicle.maxAccelMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.maxAccelDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.maxAccelMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.maxAccelMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            # calibrate reaction_time for congflow
            elif para_name[1] == 'reactionTime':
                # [reaction_time, reaction_stop, reaction_light, reaction_prob]
                car_react = GKVehicleReactionTimes(para_value[0], para_value[1],
                                                   para_value[2], para_value[3])

                car_type.setVariableReactionTimes([car_react])
                experiment.setVariableReactionTimesMicro(car_type, [car_react])
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            # calibrate minDistMean for congflow
            elif para_name[1] == 'minDist':
                car_type.setDataValueByID(GKVehicle.minDistMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.minDistDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.minDistMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.minDistMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            # calibrate sensitivityFactorMean = min = max for congflow
            elif para_name[1] == 'sensitivityFactor':
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.sensitivityFactorMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            # This should already been preset as unrealistically high such that speed can be fully
            # controlled by the speed acceptance
            elif para_name[1] == 'maxSpeed':
                car_type.setDataValueByID(GKVehicle.maxSpeedMean, QVariant(para_value[0]))
                car_type.setDataValueByID(GKVehicle.maxSpeedDev, QVariant(para_value[1]))
                car_type.setDataValueByID(GKVehicle.maxSpeedMin, QVariant(para_value[2]))
                car_type.setDataValueByID(GKVehicle.maxSpeedMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            else:
                print_cmd('\n---- ERROR: Could not recognize preset parameter entry {0}: {1}\n'.format(key, para_value))


        elif para_name[0] == 'truck':
            # truck paras
            if para_name[1] == 'speedAcceptance':
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.speedAcceptanceMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'minHeadway':
                truck_type.setDataValueByID(GKVehicle.minimunHeadwayMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.minimunHeadwayDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.minimunHeadwayMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.minimunHeadwayMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'maxAccel':
                truck_type.setDataValueByID(GKVehicle.maxAccelMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.maxAccelDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.maxAccelMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.maxAccelMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'reactionTime':
                # [reaction_time, reaction_stop, reaction_light, reaction_prob]
                truck_react = GKVehicleReactionTimes(para_value[0], para_value[1],
                                                     para_value[2], para_value[3])

                truck_type.setVariableReactionTimes([truck_react])
                experiment.setVariableReactionTimesMicro(truck_type, [truck_react])
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'minDist':
                truck_type.setDataValueByID(GKVehicle.minDistMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.minDistDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.minDistMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.minDistMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'sensitivityFactor':
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.sensitivityFactorMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            elif para_name[1] == 'maxSpeed':
                truck_type.setDataValueByID(GKVehicle.maxSpeedMean, QVariant(para_value[0]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedDev, QVariant(para_value[1]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedMin, QVariant(para_value[2]))
                truck_type.setDataValueByID(GKVehicle.maxSpeedMax, QVariant(para_value[3]))
                print_cmd('---- set {0}: {1}'.format(key, para_value))

            else:
                print_cmd('\n---- ERROR: Could not recognize preset parameter entry {0}: {1}\n'.format(key, para_value))

        elif para_name[0] == 'main':
            # TODO: if we want to add ramp flows as decision variables,
            # TODO: we need to write the corresponding ramp adjust functions
            adjust_I80_demand(model, [key, para_value])

        else:
            print_cmd('\nERROR: Could not recognize preset parameter entry {0}\n '.format(key))


# Deprecated
def adjust_I80_demand(model, para):
    """
    DEPRECATED: a quick hack to adjust the demand
    :param model:
    :param para:
    :return:
    """
    para_name = para[0]

    print_cmd('Setting flows: {0}'.format(para))

    # convert to 30 states
    flow_ratio = []
    for i in range(0, len(para[1])):
        flow_ratio.append(para[1][i])
        flow_ratio.append(para[1][i])

    if len(flow_ratio) != 30:
        print_cmd('\nERROR: adjust_I80_demand now only support 15 flow variables.\n')

    state_type = model.getType("GKTrafficState")

    if para_name.split('_')[0] == 'main':

        main_entrance = __findSection(model, 21216)
        truck_ratio = 0.27

        main_count_from_EB3 = [159, 181, 183, 176, 181,
                               180, 175, 206, 187, 195,
                               174, 176, 185, 173, 160,
                               167, 159, 123, 174, 196,
                               168, 132, 149, 173, 143,
                               147, 157, 128, 130, 161]
        # change the input ratio (0.9~1.1) to flow veh/hr
        main_flow = []
        for i in range(0, len(flow_ratio)):
            main_flow.append(flow_ratio[i] * 12.0 * main_count_from_EB3[i])

        for state in model.getCatalog().getObjectsByType(state_type).itervalues():

            # set the car state and the order must be correct
            if state.getVehicle().getId() == 53:
                # find which state is this
                state_name = str(state.getName())
                # in this format: cong_state#_car/truck
                middle_name = state_name.split('_')[1]
                # print_cmd('middle_name: {0}'.format(middle_name))
                state_id = int(middle_name[5:])
                # print_cmd('Set the car state {0} with with new flow {1}'.format(state_name, main_flow[state_id-1]*(1-truck_ratio)))
                state.setEntranceFlow(main_entrance, None, float(main_flow[state_id - 1] * (1 - truck_ratio)))

            # set the truck state and the order must be correct
            elif state.getVehicle().getId() == 56:
                # find which state is this
                state_name = str(state.getName())
                # in this format: cong_state#_car/truck
                middle_name = state_name.split('_')[1]
                # print_cmd('middle_name: {0}'.format(middle_name))
                state_id = int(middle_name[5:])
                # print_cmd('Set the truck state {0} with new flow {1}'.format(state_name, main_flow[state_id-1]*truck_ratio))
                state.setEntranceFlow(main_entrance, None, float(main_flow[state_id - 1] * truck_ratio))



    elif para_name.split('_')[0] == 'onramp':

        onramp_4 = __findSection(model, 1357)


# ---------------------- Functions for running simulation ---------------------------------
def simulate_rep_from_paras(model, experiment, paras,
                            simulator, avg_result, plugin,
                            valid_data, det_weight,
                            name):
    """
    This is the function that need to be used for simulate and return the simulation result.
    It can be used in two ways:
    - Generate validation data if valid_data is None
    - Simulate and output the objective function value
    :param model: GK Model
    :param experiment: the GK Experiment
    :param paras: parameters to be simulated
    :param simulator: the simulator
    :param avg_result: the avg_result (a list of replications to simulation)
    :param plugin: the AIMSUN plugin for computing the average
    :param valid_data: validation data; if None, then will simply simulate and generate the validation
                        data. The obj_value in this case will be [0, 0]
    :param det_weight: the weight for detectors for computing the obj
    :param name: a string used for printing the status
    :return: [obj_value, sim_data];
            obj_value: [RMSE_speed, RMSE_count]
            sim_data: dict; [det_name] = [[speeds], [counts]]
    """
    if valid_data is not None:
        print_cmd('\n------------Simulating with {0} paras-------------------'.format(name))
    else:
        print_cmd('\n------------Generating validation data------------------')

    # set new parameters
    set_new_paras(model, experiment, paras)

    __simulate_experiment(simulator, avg_result)

    sim_data = __get_averaged_detector_data(model, avg_result, plugin)

    if valid_data is not None:
        obj_value = __evaluate_obj_val_RMSE(sim_data, valid_data, det_weight)
    else:
        obj_value = [0, 0]

    return [obj_value, sim_data]


# ------------------- Functions for interfacing optimization solver -----------------------
def read_optquest_paras(parafile):
    """
    This function reads the new parameters.
    Note: MODIFY set_new_paras if the keys changed.
    :param parafile: the parameter file, each line:
        parameter_name,value1,value2,value3
    :return: dict. [parameter_name]:[value1, value2, value3]
    """
    paras = OrderedDict()

    wait_time = 0
    timeout = 0
    while not exists(parafile):
        time.sleep(0.1)
        wait_time += 1
        timeout += 1
        if wait_time >= 10:  # sleep 1 second
            print_cmd('Waiting for paras...')
            wait_time = 0

        if timeout >= 200:  # 20 s
            # assume finished optimization
            return None

    if exists(parafile):
        paras = __paras_reader(parafile)

    # delete the file once read
    os.remove(parafile)

    # print_cmd('Have read paras:\n')
    # for key in paras.keys():
    #     print_cmd('---- {0}: {1}'.format(key, paras[key]))

    return paras


def read_optquest_solution(solutionfile):
    """
    This function reads the final solution from optquest.
    The solver will stop at its max iteration.
    :param solutionfile: same format as the parafile
    :return: dict. same as read_optquest_paras
    """
    # Have not finished optimization
    if not exists(solutionfile):
        return None

    # a solution has been converged
    else:

        solution = __paras_reader(solutionfile)

        # delete the file once read
        # os.remove(solutionfile)

        print_cmd('Have read solution:\n')
        for key in solution.keys():
            print_cmd('---- {0}: {1}'.format(key, solution[key]))

        return solution


# Deprecated
def read_preset_paras(presetparasfile):
    """
    Read preset parameters. NO LONGER NEEDED. SIMPLY either set in AIMSUN ang. OR use the default
    :param presetparasfile:
    :return:
    """

    if not exists(presetparasfile):
        print_cmd('\nWARNING: No preset paras set \n ---- could not find file {0}\n'.format(presetparasfile))
        return None

    else:
        preset_paras = __paras_reader(presetparasfile)

        print_cmd('Reading the preset paras...\n')
        for key in preset_paras.keys():
            pass
            # print_cmd('---- {0}: {1}'.format(key, preset_paras[key]))

        return preset_paras


def write_simval(simval, simvalfile):
    """
    This function writes the simulation result to file
    :param simval: double, the optimal value
    :param simvalfile: the file string.
    :return:
    """
    f = open(simvalfile, 'w')
    f.write(str.format("{0:.16f}", simval))
    f.close()
    print_cmd('Wrote objective value: {0}'.format(simval))

    return 0


# ------------------- Functions for handling previously obtained obj values -----------------------
def load_previous_opt_solutions(opt_step_file):
    """
    To save the optimization cost, if a value point has been evaluated in the simulation, it will be
    logged in txt file.
    A new optimization will start by loading previously explored value points. If a point has been
    previously evaluated, then it will NOT be re-simulated to save time.
    :param opt_step_file:
    :return: [paras_list, obj_list]
    """
    if opt_step_file is None:
        return None

    f = open(opt_step_file, 'r')

    paras_list = []
    obj_value_list = []

    for line in f:

        paras = OrderedDict()

        line = line.strip()
        items = line.split(',')

        # parse each line. first item is the counter, skip
        for i in range(1, len(items)):

            # a key
            if not __isNumber(items[i]):
                if items[i] == 'RMS':
                    # get the objective value and continue
                    obj_val = [float(items[i + 1]), float(items[i + 2])]
                    break
                else:
                    # register key
                    key_name = items[i]
                    paras[key_name] = []

            else:
                # a number, append to the latest key
                paras[key_name].append(float(items[i]))

        # print_cmd('loaded paras: {0}'.format(paras))

        paras_list.append(paras)
        obj_value_list.append(obj_val)

    f.close()

    return [paras_list, obj_value_list]


def try_get_obj_from_previous_solutions(solution_list, new_para):
    """
    Check if the new_para has been previously simulated.
    :param solution_list: the paras_list from load_previous_opt_solutions
    :param new_para: the new parameters
    :return:
    """

    # if not solution list
    if solution_list is None:
        return None

    for i in range(0, len(solution_list[0])):

        if new_para == solution_list[0][i]:
            return solution_list[1][i]

    # if none found
    return None


# ------------------- Functions for handling optimization logger -----------------------
# The following functions handles the logging data.
# this function logs the optimization step
# iteration_counter, car_para1, car_para2..., truck_para1, truck_para2, RMS_speed, RMS_count
def start_opt_logger(path_str, start_time):
    global opt_logger, opt_logger_file

    file_name = start_time.strftime("%Y%m%d_%H%M%S") + '_opt_log'
    opt_logger_file = open(path_str + file_name + '.csv', 'wb')
    opt_logger = csv.writer(opt_logger_file)


def log_opt_step(opt_solution, iteration_counter, paras, RMS):
    list = []
    list.append(iteration_counter)

    # first append key, then parameters
    for key in paras.keys():
        list.append(key)
        for item in paras[key]:
            list.append(item)

    # Last two are reserved for the RMS result
    list.append('RMS')
    list.append(RMS[0])
    list.append(RMS[1])

    opt_solution.append(list)

    # write in file
    if opt_logger is not None:
        opt_logger.writerow(list)


# save logging of the optimization data
def stop_opt_log():
    opt_logger_file.close()


# save the optimization result; and the data generated
# save logging of the optimization data
def save_solution_data(solution, data, path_name, start_time, name):
    file_name = start_time.strftime("%Y%m%d_%H%M%S") + '_sol_'
    f = open(path_name + file_name + name + '.csv', 'wb')
    writer = csv.writer(f)
    # the solution: key1, value, key2, value
    if solution is not None:
        list = []
        for key in solution.keys():
            list.append(key)
            for item in solution[key]:
                list.append(str(item))
        writer.writerow(list)
    else:
        # just to save the validation data. true parameters are unknown
        writer.writerow(['Data saved with unknown parameters: {0}'.format(name)])

    for key in data:
        tmp_line = []
        tmp_line.append(key)
        # write speed
        for item in data[key][0]:
            tmp_line.append(item)
        writer.writerow(tmp_line)

        tmp_line = []
        tmp_line.append(key)
        # write count
        for item in data[key][1]:
            tmp_line.append(item)
        writer.writerow(tmp_line)
        # writer.writerow([str(key), str(data[key][0]), str(data[key][1])])

    f.close()


# start logger for the cmd output
def start_cmd_logger(path_str, start_time):
    global cmd_logger

    file_name = start_time.strftime("%Y%m%d_%H%M%S") + '_cmd_log'
    cmd_logger = open(path_str + file_name + '.txt', 'wb+')

    # return cmd_logger


# command logger header
# para_list =[ (name, paras) ]
def cmd_logger_header(description,
                      g_num_iter, g_num_rep, seed_list,
                      obj_func,
                      det_for_validation,
                      main_entrance_id, main_entrance_discount,
                      paras_list):
    print_cmd('Calibration experiment description:\n ---- {0}\n'.format(description))
    print_cmd('Calibration Configuration:')
    print_cmd('---- Objective function is:           Minimize    {0}xRMS_speed + {1}xRMS_count'.format(obj_func[0],
                                                                                                       obj_func[1]))
    print_cmd('---- Detectors used for validation:   {0}'.format(det_for_validation))
    print_cmd('---- Number of iterations:            {0}'.format(g_num_iter))
    print_cmd('---- Main Entrance is:                {0}'.format(main_entrance_id))
    print_cmd('---- Main Entrance flow = VerMac EB3 x{0}'.format(main_entrance_discount))
    print_cmd('---- Number of replications {0}, with seeds: {1}\n'.format(g_num_rep, seed_list))

    print_cmd('\nParameters:')
    for para_tup in paras_list:
        print_cmd('-- {0} paras:'.format((para_tup[0])))
        for key in para_tup[1].keys():
            print_cmd('---- {0}: {1}'.format(key, para_tup[1][key]))


# print out on the cmd, and save the cmd output to a file
# If log_file is None, will only print
def print_cmd(line_str):
    if cmd_logger is None:
        print(line_str)
    else:
        # save every time
        # Hence even if the file is not correctly closed, those lines are still saved.
        cmd_logger.write(line_str + '\n')
        print(line_str)


# stop logger for the cmd output
def stop_cmd_logger():
    cmd_logger.close()


# print out result
def print_results(paras_list, obj_val_list):
    print_cmd('\n\n===========================================================================')
    print_cmd('====================Calibration Finished===================================')
    # Forget about beautiful printout. Just log information
    print_cmd('Parameters: \n')
    for para in paras_list:
        print_cmd('---- {0}_paras:'.format(para[0]))
        for key in para[1].keys():
            print_cmd('-------- {0}:    {1}'.format(key, para[1][key]))

    print_cmd('\nObjective values: \n')
    for obj_val in obj_val_list:
        print_cmd('{0} objective value:\n---- RMS_speed: {1}      \n---- RMS_count: {2}'.format(obj_val[0],
                                                                                                obj_val[1][0],
                                                                                                obj_val[1][1]))


# This function plots the optimization steps.
# input: the opt_logger as we update in every iteration,
#        the obj_ratio (1,0) (speed, count), and the obj_val using default parameters used as a baseline
# TODO: this function somehow fails due to matplotlib issue in windows
def plot_opt_steps(opt_solution, obj_ratio, obj_default_val):
    global fig_handle

    # compute the objective values to be plotted
    steps = []
    obj_values = []
    for paras in opt_solution:
        steps.append(float(paras[0]))
        obj_values.append(float(paras[-2]) * obj_ratio[0] + float(paras[-1] * obj_ratio[1]))

    default_baseline = float(obj_default_val[0]) * obj_ratio[0] + float(obj_default_val[1]) * obj_ratio[1]

    if fig_handle is None:
        fig_handle, = plt.plot(obj_values)
        fig_handle.set_xlabel('Iteration step')
        fig_handle.set_ylabel('Objective value')
        fig_handle.set_title('Optimization progress (default val {0})'.format(default_baseline))
    else:
        # update and plot the figure
        # fig_handle.set_xdata(steps)
        fig_handle.set_ydata(obj_values)

    plt.draw()
    plt.show()


# just print out the objective value
# obj_list is a list of [(name, value)] would like to print out and compare
def print_opt_steps(opt_solution, obj_ratio, obj_list):
    # compute the objective values to be plotted
    steps = []
    obj_values = []
    for paras in opt_solution:
        steps.append(float(paras[0]))
        obj_values.append(float(paras[-2]) * obj_ratio[0] + float(paras[-1] * obj_ratio[1]))

    print_cmd('\nOpt Steps:   {0}'.format(obj_values))
    print_cmd('Opt optimal: {0}'.format(np.min(np.array(obj_values))))
    for item in obj_list:
        compare_baseline = float(item[1][0]) * obj_ratio[0] + float(item[1][1]) * obj_ratio[1]
        print_cmd('Opt {0}: {1}'.format(item[0], compare_baseline))


# adjust the demand data
# paras['ramp_on1_car']
# no longer used
def adjust_ramps_I80_EB_full(model, paras):
    state_type = model.getType("GKTrafficState")

    # print 'paras: {0}'.format(paras)

    # originally 500 veh/hr
    onramp_3_1 = __findSection(model, 21192)  # [0, 500] veh/hr
    onramp_3_2 = __findSection(model, 21201)

    # first off ramp at junction 3
    diverge_3_1_main = __findSection(model, 3412)
    diverge_3_1_to = __findSection(model, 3399)
    diverge_3_1_off = __findSection(model, 343)  # [0,2]

    # second off ramp at junction 3
    diverge_3_2_main = __findSection(model, 3400)
    diverge_3_2_to = __findSection(model, 3401)
    diverge_3_2_off = __findSection(model, 1039)

    # onramp of 4
    onramp_4 = __findSection(model, 1357)

    # originally 3%
    diverge_4_main = __findSection(model, 40962)
    diverge_4_to = __findSection(model, 3248)
    diverge_4_off = __findSection(model, 1501)  # [0, 20]

    for state in model.getCatalog().getObjectsByType(state_type).itervalues():
        # print_cmd('state.getVehicle(): {0}'.format(state.getVehicle().getId() ))
        if state.getVehicle().getId() == 53:
            # ramp3, first off and on
            state.setTurningPercentage(diverge_3_1_main, diverge_3_1_off, None, float(paras['ramp3'][0]))
            state.setTurningPercentage(diverge_3_1_main, diverge_3_1_to, None, 100 - float(paras['ramp3'][0]))
            state.setEntranceFlow(onramp_3_1, None, float(paras['ramp3'][1]))

            # ramp3, second off
            state.setTurningPercentage(diverge_3_2_main, diverge_3_2_off, None, float(paras['ramp3'][2]))
            state.setTurningPercentage(diverge_3_2_main, diverge_3_2_to, None, 100 - float(paras['ramp3'][2]))

            # ramp4, off and on
            state.setTurningPercentage(diverge_4_main, diverge_4_off, None, float(paras['ramp4'][0]))
            state.setTurningPercentage(diverge_4_main, diverge_4_to, None, 100 - float(paras['ramp4'][0]))
            state.setEntranceFlow(onramp_4, None, float(paras['ramp4'][1]))

            # main entrance flow
            # state.setEntranceFlow


# add a noise model for the validation data.
# noise_mode is speed_noise_model[det] = [bias, normal_distr_dev]
def add_noise_to_data(data, speed_noise_model, count_noise_model):
    new_data = OrderedDict()

    for det in data.keys():

        speed = data[det][0]
        count = data[det][1]

        new_speed = []
        new_count = []

        for i in range(0, len(speed)):
            new_value = speed[i] + speed_noise_model[det][0] + np.random.normal(0, speed_noise_model[det][1], 1)
            new_speed.append(np.max([0, new_value]))
            new_value = count[i] + count_noise_model[det][0] + np.random.normal(0, count_noise_model[det][1], 1)
            new_count.append(np.max([0, new_value]))

        new_data[det] = [new_speed, new_count]

    return new_data


# ====================================================================================================
# Utility functions. Do not call externally
# ====================================================================================================

# -------------------------- Utility functions for interfacing AIMSUN -------------------------------
def __readParsFile(parsFilePath):
    """
    Read parameter files to create traffic states
    :param parsFilePath: the file name with full path.
        Each row of file: state name, car type and name, state start time, state duration
           e.g: cong_state1_car,53 Car,15:30:00,00:05:00
    :return: pars_dict: [state name] = [vehicle type, state start time, duration], all strings
    """
    pars_dict = OrderedDict()
    pars_file = open(parsFilePath, 'r')
    while True:
        line = pars_file.readline()
        if not (bool(line)):
            break
        line = line.strip()
        items = line.split(',')
        tmp_key = items[INDEX_PARS_STATE].strip()
        if tmp_key not in pars_dict.keys():
            pars_dict[tmp_key] = list()
        pars_dict[tmp_key].append(
            (items[INDEX_PARS_VEH].strip(), items[INDEX_PARS_FROMTIME].strip(), items[INDEX_PARS_DUR].strip()))
    pars_file.close()
    return pars_dict


def __readFlowsFile(flowsFilePath):
    """
    Read flows file to create traffic states
    :param flowsFilePath: the file name with full path.
        Each row of file: state name, section id, inflow (veh/hr)
                   e.g: cong_state3_car,30109,50.0
    :return: flows_dict: [state name] = [section_id, inflow], all strings
    """
    flows_dict = OrderedDict()
    flows_file = open(flowsFilePath, 'r')
    while True:
        line = flows_file.readline()
        if not (bool(line)):
            break
        line = line.strip()
        items = line.split(',')
        tmp_key = items[INDEX_FLOWS_STATE].strip()
        if tmp_key not in flows_dict.keys():
            flows_dict[tmp_key] = list()
        flows_dict[tmp_key].append((items[INDEX_FLOWS_SECID].strip(), items[INDEX_FLOWS_INFLOW].strip()))
    flows_file.close()
    return flows_dict


def __readTurnsFile(turnsFilePath):
    """
    Read turns file to create traffic states
    :param turnsFilePath: the file name with full path.
        Each row of file: state name, from section_id, to section_id, percent
           e.g: cong_state1_car,21217,23587,2.0
    :return: turns_dict: [state name] = [from_sec_id, to_sec_id, percent]
    """
    turns_dict = OrderedDict()
    turns_file = open(turnsFilePath, 'r')
    while True:
        line = turns_file.readline()
        if not (bool(line)):
            break
        line = line.strip()
        items = line.split(',')
        tmp_key = items[INDEX_TURNS_STATE].strip()
        if tmp_key not in turns_dict.keys():
            turns_dict[tmp_key] = list()
        turns_dict[tmp_key].append(
            (items[INDEX_TURNS_FROM].strip(), items[INDEX_TURNS_TO].strip(), items[INDEX_TURNS_PERC].strip()))
    turns_file.close()
    # print '\n\nturns_dict:{0}\n\n'.format(turns_dict)
    return turns_dict


def __createState(model, state_name, pars_dict, flows_dict, turns_dict, main_entrance_id, main_entrance_discount_ratio):
    """
    This function creates the traffic states from the dict read from files
    :param model: GK Model object
    :param state_name: the statename to be created
    :param pars_dict: the dicts created by files
    :param flows_dict: the dicts created by files
    :param turns_dict: the dicts created by files
    :param main_entrance_id: the inflow to be discounted. Set as None if do not want any discount
    :param main_entrance_discount_ratio: set as 1 if not discount
    :return: return the State object
    """
    # create new state and set parameters
    state = GKSystem.getSystem().newObject("GKTrafficState", model)
    state.setName(state_name)
    it = QTime.fromString((pars_dict[state_name])[0][1], Qt.ISODate)
    duration = (pars_dict[state_name])[0][2]
    state.setInterval(it, GKTimeDuration.fromString(duration))

    if _debug is True:
        print_cmd('AIMSUNFUNCTIONS: state from {0} duration {1}'.format(it.toString(), duration))

    # set the vehicle for this state
    vehicleString = str((pars_dict[state_name])[0][0]).split()
    vehId = int(vehicleString[0])  # make sure this is correct
    vehName = vehicleString[1]
    vehicle = state.getModel().getCatalog().find(vehId)
    if vehicle is None:
        # is wont work since the name is not unique, UserClass has object called car
        vehicle = state.getModel().getCatalog().findByName(vehName)
    state.setVehicle(vehicle)

    # set the inflow of the state
    for entrance in range(0, len(flows_dict[state_name])):
        # print (flows_dict[state_name])[entrance][0]
        fromSection = __findSection(model, (flows_dict[state_name])[entrance][0])

        # discount the main entrance flow
        if fromSection.getId() == main_entrance_id:
            state.setEntranceFlow(fromSection, None,
                                  float((flows_dict[state_name])[entrance][1]) * main_entrance_discount_ratio)
        else:
            state.setEntranceFlow(fromSection, None,
                                  float((flows_dict[state_name])[entrance][1]))

    # set the turn percentage of the state
    if state_name in turns_dict.keys():
        for turn in range(0, len(turns_dict[state_name])):
            # print_cmd('For state {0}, has turns: {1}'.format(state_name, turns_dict[state_name]))
            fromSection = __findSection(model, (turns_dict[state_name])[turn][0])
            toSection = __findSection(model, (turns_dict[state_name])[turn][1])
            state.setTurningPercentage(fromSection, toSection, None, float((turns_dict[state_name])[turn][2]))
    else:
        print_cmd('No Turn information for state {0}'.format(state_name))

    # for testing the aimsun automation
    if _debug:
        print_cmd('AIMSUNFUNCTION: state turn 330->(340, 341): ({0},{1})'.format(
            state.getTurningPercentage(model.getCatalog().find(int(330)),
                                       model.getCatalog().find(int(340)), None),
            state.getTurningPercentage(model.getCatalog().find(int(330)),
                                       model.getCatalog().find(int(341)), None)))

    # append the state to the state folder
    folder = __getStateFolder(model)
    folder.append(state)

    return state


def __findSection(model, entry):
    """
    This function returns a section object
    :param model: the GKModel
    :param entry: string, e.g., '330'
    :return: return GKSection object of the entry id
    """
    section = model.getCatalog().find(int(entry))
    if section.isA(QString("GKSection")) is False:
        section = None
    return section


def __getStateFolder(model):
    """
    This function returns the folder object for the traffic states
    :param model: the GK Model
    :return:
    """
    folderName = "GKModel::trafficStates"
    folder = model.getCreateRootFolder().findFolder(folderName)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folderName)

    # print_cmd('__getStateFolder: type: {0}'.format(type(folder))
    # print_cmd('__getStateFolder: name: {0}'.format(folder.getName())
    return folder


def __getDemandFolder(model):
    """
    This function returns the demand folder
    :param model: GK model
    :return: Folder object
    """
    folderName = "GKModel::trafficDemand"
    folder = model.getCreateRootFolder().findFolder(folderName)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folderName)
    return folder


def __getScenariosFolder(model):
    """
    This function returns the folder object for the scenarios
    :param model: GK Model
    :return: return the folder
    """
    folderName = "GKModel::top::scenarios"
    folder = model.getCreateRootFolder().findFolder(folderName)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folderName)
    return folder


def __createScheduleItem(state):
    """
    Before the states can be added to the demand, each state must be added to a schedule item before being added to demand.
    NOTE: states contains the flow, turn percentage, and vehicle type. However, to add the state to the demand, we need to
    assign the state to a scheduleDemandItem. The simulator uses the fromTime and duration of the scheduleDemandItem, NOT
    the fromTime and duration set in the state!
    :param state: the state object
    :return: GKScheduleDemandItem object: associated with the state
    """
    schedule = GKScheduleDemandItem()
    schedule.setTrafficDemandItem(state)

    if _debug:
        print_cmd('schedule.state.duration {0}'.format(schedule.getTrafficDemandItem().getDuration().toString()))

    hr = state.getFrom().hour()
    minute = state.getFrom().minute()
    sec = state.getFrom().second()
    schedule.setFrom(3600 * hr + 60 * minute + sec)

    if _debug:
        print_cmd('state.getDuration().toSeconds(): {0}'.format(state.getDuration().toSeconds()[0]))
        print_cmd(type(state.getDuration().toSeconds()[0]))

    schedule.setDuration(state.getDuration().toSeconds()[0])

    return schedule


def __simulate_experiment(simulator, avg_result):
    """
    This function simulates the replications in an experiment.
    :param simulator: the created simulator from create_simulator
    :param avg_result: the replication from setup_replication
    :return:
    """
    print_cmd('\nReset replications...')

    # first reset replications
    avg_result.resetReplications()

    # add replications to simulator
    replication_list = avg_result.getReplications()

    for replication in replication_list:
        simulation_task = GKSimulationTask(replication, GKReplication.eBatch, "", "", True)  # Other approach
        simulator.addSimulationTask(simulation_task)
        # print_cmd('Added replication {0} to simulator with status {1}. '.format(replication.getId(),
        #                                                                        replication.getSimulationStatus())
        # print_cmd('pending {0}; done {1}; discarded {2}; loaded {3}'.format(GKGenericExperiment.ePending,
        #                                                                    GKGenericExperiment.eDone,
        #                                                                    GKGenericExperiment.eDiscarded,
        #                                                                    GKGenericExperiment.eLoaded)

    # simulate model
    if not simulator.isBusy():
        print_cmd('Simulating...\n')
        sim_status = simulator.simulate()
    else:
        print_cmd('Simulator is busy\n')

    # make sure correctly simulated
    if sim_status is True:
        print_cmd('Simulation finished\n')
    else:
        print_cmd('ERROR: Simulation failed\n')

        # simulator.postSimulate()


def simulate_experiment(simulator, avg_result):
    """
    This function simulates the replications in an experiment.
    :param simulator: the created simulator from create_simulator
    :param avg_result: the replication from setup_replication
    :return:
    """
    print_cmd('\nReset replications...')

    # first reset replications
    avg_result.resetReplications()

    # add replications to simulator
    replication_list = avg_result.getReplications()

    for replication in replication_list:
        simulation_task = GKSimulationTask(replication, GKReplication.eBatch, "", "", True)  # Other approach
        simulator.addSimulationTask(simulation_task)
        print_cmd('Added replication {0} to simulator with status {1}. '.format(replication.getId(),
                                                                                replication.getSimulationStatus()))
        print_cmd('pending {0}; done {1}; discarded {2}; loaded {3}'.format(GKGenericExperiment.ePending,
                                                                            GKGenericExperiment.eDone,
                                                                            GKGenericExperiment.eDiscarded,
                                                                            GKGenericExperiment.eLoaded))

    # simulate model
    if not simulator.isBusy():
        print_cmd('Simulating...\n')
        sim_status = simulator.simulate()
    else:
        print_cmd('Simulator is busy\n')

    # make sure correctly simulated
    if sim_status is True:
        print_cmd('Simulation finished\n')
    else:
        print_cmd('ERROR: Simulation failed\n')


def __read_detector_data(model, data_origin):
    """
    This function reads the detector data from data_origin (a average_result type)
    :param model: GK Model
    :param data_origin: a list of replications:
            [a replication or the average_result(which is a subtype of GKReplication)]
    :return: a dict, avg_data[detector_name] = [[speed mph],[count/5min]]
    """
    avg_data = OrderedDict()

    det_type = model.getType("GKDetector")
    # read data for each replication and then the average
    for replication in data_origin:

        print_cmd('\nReading Replication data: {0}'.format(replication.getName()))

        # get the column id
        speedColumn = det_type.getColumn(GK.BuildContents(GKColumnIds.eSpeed, replication, None))
        countColumn = det_type.getColumn(GK.BuildContents(GKColumnIds.eCount, replication, None))

        # read each detector
        for det in model.getCatalog().getObjectsByType(det_type).itervalues():

            det_name = str(det.getName())
            # print_cmd('----Reading Detector {0}...'.format(det_name))

            # add to dict
            # speed (mph), count
            avg_data[det_name] = [[], []]

            speedData = det.getDataValueTS(speedColumn)
            countData = det.getDataValueTS(countColumn)

            if countData.size() == 0 or speedData.size() == 0 or countData.size() != speedData.size():
                print_cmd('ERROR: Detector {0} has no data available'.format(det_name))
            else:
                # print_cmd('----size of data is: {0}'.format(countData.size()))
                # the speed data returned from AIMSUN is in km/h; 1 km/h = 0.62137 mph when AIMSUN is specified in metric
                for interval in range(countData.size()):
                    avg_data[det_name][0].append(speedData.getValue(GKTimeSerieIndex(interval))[0] * KMH2MPH)
                    avg_data[det_name][1].append(countData.getValue(GKTimeSerieIndex(interval))[0])

                    if _show_detector_data:
                        print_cmd('--------interval {0}: speed {1}; count {2}'.format(interval,
                                                                                      avg_data[det_name][0][-1],
                                                                                      avg_data[det_name][1][-1]))
                        # print_cmd('----Detector {0} data:{1}'.format(det.getName(), avg_data[det.getName()])

    return avg_data


def __get_averaged_detector_data(model, avg_result, plugin):
    """
    This function computes the average data from the simulation
    :param model: GK Model
    :param avg_result: avg_result replication
    :param plugin: the plugin from AIMSUN for computing the average
    :return: a dict, avg_data[detector_name] = [[speed (mph)],[count per detection cycle]]
    """
    # compute the result
    calculate_status = plugin.calculateResult(avg_result)

    if calculate_status == GGetramModule.eOKCalculateResult:
        print_cmd('Retrieving average data finished.')
    elif calculate_status == GGetramModule.eFailCalculateResult:
        # at 5th iteration failed.
        print_cmd('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print_cmd('$$$$$$$$$$$ ERROR: Retrieving average data failed.')
        print_cmd('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    elif calculate_status == GGetramModule.eSimRequiredCalculateResult:
        print_cmd('$$$$$$$$$$$ ERROR: Retrieving average data failed. Simulation required.')

    # Not sure if this line is needed
    plugin.readResult(avg_result)

    # read the detector data out
    avg_data = __read_detector_data(model, [avg_result])

    return avg_data


def extract_detector_data(model, avg_result, plugin):
    """
    This function computes the average data from the simulation
    :param model: GK Model
    :param avg_result: avg_result replication
    :param plugin: the plugin from AIMSUN for computing the average
    :return: a dict, avg_data[detector_name] = [[speed (mph)],[count per detection cycle]]
    """
    # compute the result
    calculate_status = plugin.calculateResult(avg_result)

    if calculate_status == GGetramModule.eOKCalculateResult:
        print_cmd('Retrieving average data finished.')
    elif calculate_status == GGetramModule.eFailCalculateResult:
        # at 5th iteration failed.
        print_cmd('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print_cmd('$$$$$$$$$$$ ERROR: Retrieving average data failed.')
        print_cmd('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    elif calculate_status == GGetramModule.eSimRequiredCalculateResult:
        print_cmd('$$$$$$$$$$$ ERROR: Retrieving average data failed. Simulation required.')

    # Not sure if this line is needed
    plugin.readResult(avg_result)

    # read the detector data out
    avg_data = __read_detector_data(model, [avg_result])

    return avg_data


# evaluate the objective function
# this function compares the averaged simulation results with the validation data
# output one single valued measure, here we use RMSE
# the avg_data must have the same structure as the valid_dict: valid[det_name] = [[speed],[count]]
def __evaluate_obj_val_RMSE(avg_data, valid_dict, det_weight):
    """
    This function evaluate the objective function using RMSE
    :param avg_data: the averaged simulation data speed, count
    :param valid_dict: the validation data
                valid_dict[det_name] = [[speeds], [counts]]
    :param det_weight: the weight assigned for each detector
    :return: [RMSE_speed, RMSE_count]
    """
    rms_speed = 0
    rms_count = 0

    print_cmd('\nEvaluating objective value...')
    # print_cmd('avg_data keys: {0}'.format(avg_data.keys())
    # print_cmd('valid_data keys: {0}'.format(valid_dict.keys())

    # adds up the error for speed and count for each detector
    # Note the detector in the main entrance (EB3) is not used
    for key in valid_dict:
        valid_speed = np.array(valid_dict[key][0])
        valid_count = np.array(valid_dict[key][1])
        avg_speed = np.array(avg_data[key][0])
        avg_count = np.array(avg_data[key][1])

        # the following is added to deal with np.nan values
        # print 'before: {0}'.format(valid_speed-avg_speed)
        tmp_array = np.power(valid_speed - avg_speed, 2)
        # print 'after: {0}'.format(tmp_array)

        rms_speed += det_weight[key] * np.sqrt(np.nansum(tmp_array) / len(valid_speed))

        tmp_array = np.power(valid_count - avg_count, 2)
        rms_count += np.sqrt(np.nansum(tmp_array) / len(valid_count))

    print_cmd('Evaluated objective: (RMS_speed, RMS_flow): ({0}, {1})'.format(rms_speed, rms_count))

    return (rms_speed, rms_count)


# -------------------------- Other utility functions -------------------------------
def __isNumber(s):
    """
    This function tests if string s is a number
    :param s: string
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def __paras_reader(parafile):
    """
    A universal paras reader assuming the file exist.
    :param parafile: the file to read
    :return: a dict, key-value
    """
    paras = OrderedDict()

    f = open(parafile, 'r')

    for line in f:
        line = line.strip()
        items = line.split(',')

        # first item is the key
        paras[items[0]] = []

        for i in range(1, len(items)):
            paras[items[0]].append(float(items[i]))

    f.close()

    return paras




# ====================================================================================================
# Test/Debug functions
# ====================================================================================================









