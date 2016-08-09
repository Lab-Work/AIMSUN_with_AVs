# Estimating mixed human and automated traffic flows in AIMSUN using the particle filter
Ren Wang, Yanning Li, Daniel Work, Auguest 9th, 2016

##1) Overview
This repository contains all the source code and data to replicate the work for the article *Comparing traffic state estimators for mixed human and automated traffic flows* submitted to *Transportation Research Part C: Emerging technologies*. A preprint can be found [here](https://www.dropbox.com/s/lmbzgdx6r8bnr4u/WangLiWork2016.pdf?dl=0).

This work simulated mixed traffic in AIMSUN and compared the first and second order traffic model in the particle filter framework. The simulation setup is as follows:
1. Simulated a three-mile freeway, with four detectors at one mile spacing: PM0, PM1, PM2, PM3.
2. Simulated one hour of traffic containing the formation and dissipation of traffic congestion.
3. Simulated five scenarios, where the penetration rates of the inflow are sampled from uniform distributions U(0,0), U(0,0.25), U(0,0.5), U(0,0.75), U(0,1). 

##2) License

This software is licensed under the *University of Illinois/NCSA Open Source License*:

**Copyright (c) 2016 The Board of Trustees of the University of Illinois. All rights reserved**

**Developed by: Department of Civil and Environmental Engineering University of Illinois at Urbana-Champaign**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. Neither the names of the Department of Civil and Environmental Engineering, the University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

##3) Folders
This repository is organized as follow:

### [./configuration_file.txt](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/configuration_file.txt)
This file contains the parameters of the AIMSUN simulation.

### [./run_aimsun.bat](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/run_aimsun.bat) 
This file (only executable in Windows) automatically runs the AIMSUN simulation scenarios as specified in the configuration file. Edit this file to make sure the paths are correct.

### [./FD_calibration/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration)
This folder contains all the files needed for calibration of the first and second order fundamental diagram. Specifically, it contains three subfolders:
- [./FD_calibration/aimsun_files/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration/aimsun_files). This folder contains the AIMSUN simulation files. The simulation database in sqlite can be downloaded from this [link](https://uofi.box.com/s/ldlolkbtwoloff9y9p8ap4glaqix7ufg).
- [./FD_calibration/detector_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration/detector_data). This folder contains the detector data of the simulation extracted from the simulation sqlite database (Table MIDETEC).
- [./FD_calibration/calibrated_FDs/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration/calibrated_FDs). This folder contains the calibrated FDs.

### [./Simulation/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation)
This folder contains all the files for the simulation.
- [./Simulation/topology.txt](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/Simulation/topology.txt). This file is used for extracting the true states.
- [./Simulation/aimsun_files/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/aimsun_files). This folder contains the simulation file for AIMSUN. The simulation database in sqlite can be downloaded from this [link](https://uofi.box.com/s/yzbi3a7hyzihg00oam8uhf3qa79gzpf0). 
- [./Simulation/demand_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/demand_data). This folder contains the demand data for the simulation. 
- [./Simulation/raw_det_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/raw_det_data). This folder contains the raw detector data extracted from the sqlite database (table MIDETEC). The data can be downladed from this [link](https://uofi.box.com/s/9u0pm39ttqc85660ulj3c0dqrmqk0y56).
- [./Simulation/detector_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/detector_data). This folder contains the cleaned detector data used for the estimation. 
- [./Simulation/traj_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/traj_data). This folder contains the trajectory data extracted from the sqlite database (table MIVEHDETAILEDTRAJECTORY). The trajectory data is used for obtaining the true density and property states. The data can be downloaded from this [link](https://uofi.app.box.com/files/0/f/10806898433/traj_data). 
- [./Simulation/true_states/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/true_states). This folder contains the trues states for all simulation scenarios. 

### [./src/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/src). 
This folder contains all the code for calibration of the fundamental diagram and the automated simulation. See next section for how to run the code. 

### [./Estimation/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation). 
This folder contains the code and data for the estimation. 
- ./Estimation/*.py. Those source files are used to run the estimators. See next section for how to use the code.
- [./Estimation/DATA/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation/DATA). This folder contains the measurement and true data in the correct format requried for the estimators. They are converted from the data in the ./Simualtion/ folder. 
- [./Estimation/TrafficModel/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation/TrafficModel). This folder contains the traffic model for the particle filter.
- [./Estimation/FilterFunctions/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation/FilterFunctions). This folder contains the source code for the particle filter.
- [./Estimation/Result/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation/Result). This folder contains the estimation results, including the visualization figures. The complete results can be downladed from this [link](https://uofi.box.com/s/z9y73m12lryazmddqhvsz8zglc2sm6tr).

##4) Run the code
The code mainly consists of three components: [Calibration of the fundamental diagram (FD)](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration); [Simulation in AIMSUN](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation); [Estimation](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Estimation). Python 2.7 is used to develop and run the code. 

### Calibration of the fundamental diagram
The calibration of the fundamental diagram consists of the following steps:

1. Specify the microscopic parameters and tune the inflow to generate congestion in the AIMSUN files in [./FD_calibration/aimsun_files](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration/aimsun_files). Run the simulations to obtain the sqlite database.
2. `cd` to ./src/ folder, and run `python calibrate_FD.py`, which will extract the raw detector data from the sqlite database, clean the data, and calibrate the fundamental diagrams using the data. 

### Automated simulation of scenarios
Ausomated simulation of the scenarios in AIMSUN requries the following steps:

1. Properly configure the [configuration file](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/configuration_file.txt), e.g., number and seeds of replications, scenraios, etc.
2. Double click [./run_aimsun.bat](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/run_aimsun.bat). Make sure the paths are correct. This script calls the aimsun API [./src/aimsun\_api\_sim.py](https://github.com/Lab-Work/AIMSUN_with_AVs/blob/master/src/aimsun_api_sim.py) to automate the simualtion. 
3. `cd` to ./src/ folder, and run `python extract_sim_detector_data.py`, which extracts the detector measurements in the simulation scenarios. The extracted detector data will be saved in [./Simulation/detector_data/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/detector_data) which will be used for the estimation. The detector data `sim_sce0_seed12424_PM0.txt` file contains the data for scenario `0` simulated with random seed `12424` for detector `PM0`, with four columns: timestamps(s), penetration of AVs (0~1), flow (veh/hr), speed (mph).
4. `cd` to ./src/ folder, and run `python generate_true_states.py`, which will generate the true states for all scenarios. The true states are saved in [./Simulation/true_states/](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/true_states), which will be used to compute the estimation error. The file `truestate_5s179m_sce0_seed12424_density.txt` contains the true density (veh/mile) on a discretized grid with resolution `5s179m` for scenario `0` simulated with random seed `12424`. Each file is a matrix with num\_steps(row) x num\_cells(col). The first row and first column entry is the time step 0 and space cell 0. 

### Estimation using particle filter
`cd` to the ./Estimation/ folder and run the follows:

1. Run `python generate_measurement.py` to convert the detector datas to the format used in the estimators.
2. Run `python generate_truestate.py` to convert the true states to the format used in the estimators for computing the error.
3. Run `python estimation_v1.py` to run the estimators for the first and second order model.
4. Run `python plot_results.py` to visualize the estimation results and the true states. 
5. Run `python compute_error.py` to compute and visualize the estimation error for the first and second order models. 
6. (Optional) Run `python prediction_v1.py` to run the forward traffic prediction for the first and second order mdoel without using the measurement data.






