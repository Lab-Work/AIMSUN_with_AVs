## Folders:
The organization of the source code and data is described as follows:
### /src: 
All the source code related to AIMSUN simulation. 

### /aimsun_files: 
All the aimsun files (.ang format) that can be used to generate all the data.

### /demand_data: 
All the demand data specified in each aimsun simulation file.

###  /detector_data: 
All the detector data generated from the simulation.

### /raw_det_data: 
All raw detector data extracted from the sqlite database after simulation. 

### /traj_data: 
The trajectory data extracted from the sqlite database after simulation.

### /true_states: 
The true states (speed and density) files generated from each trajectory data file for each simulation.

## How to run simulations:
The folloing describes the steps for generating the simualtion results. 
1. Run [run_aimsun.bat](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/run_aimsun.bat) will run [./src/aimsun_api_sim.py](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/src/aimsun_api_sim.py) which will 
	- generate all scinarios (with different penetration rates and seeds)
	- generate the demand file for each scenario, which is saved in [./demand_data folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/demand_data)
	- generate the AIMSUN ang files in [./aimsun_files](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/aimsun_files) for each scenarios and seed
	- simulate the ang files, which will create a sqlite database file in [./aimsun_files](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/aimsun_files)
	Make sure the paths are correct in [./src/aimsun_api_sim.py](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/src/aimsun_api_sim.py) file.
2. Run "./src/generate_true_states.py", which will 
	- extract the raw detector data from sqlite database and save in [./raw_det_data](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/raw_det_data), and then
	- parse the raw_det_data files and format them to clean detector datas in [./detector_data folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/detector_data). 
3. Extract the trajectory data from sqlite database and save in [./traj_data folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/traj_data).
4. Run [./src/generate_true_states.py](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/src/generate_true_states.py) to generate the true states for each scenario and save in [./true_states folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/true_states).

## How to calibrate the fundamental diagram parameters:
1. cd to [./FD_calbration folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/FD_calibration)
2. Run the aimsun files in [./FD_calibration/aimsun_files](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/FD_calibration/aimsun_files), which will generate the sqlite data base that contains the detector data
3. Run [./FD_calibration/calibrate_FD.py](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/FD_calibration/calibrate_FD.py) file will 
	- extract the detector data from the sqlite files, which will be aseved in [./FD_calibration/detector_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/FD_calibration/detector_data)
	- calibrate the FD and the calibrated FDs will be saved in [./FD_calibration/calibrated_FDs/ folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/FD_calibration/calibrated_FDs)


## The final outputs from the simulation which will be used for estimation are:
1. The detector data in [./detector_data folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/detector_data)
2. The true states data in [./true_states folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/true_states)
3. The calibrated fundamental diagrams in [./FD_calibration/calibrated_FDs folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/FD_calibration/calibrated_FDs)
