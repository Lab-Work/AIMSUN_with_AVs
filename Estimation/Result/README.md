# AIMSUN_with_AVs
This repository includes the source code for simulating traffic in AIMSUN with autonomous vehicles.

This repository includes scenarios, where the variation of AV penetrations ranges from 0%, to 100%.

## The network:
1. 3 miles road, with four detectors at one mile spacing: PM0, PM1, PM2, PM3.
2. Simulated 1 hour.
3. Simulated a range of AV penetration rate. 10 replications (labeled by seed) each.


## The detector data:
In each scenario **sce** with **seed**, the detector data for detectors, e.g. **PM0** can be found in [detector\_data](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/Simulation/detector\_data) folder with the file name *sim_sce#_seed#_PM0.csv*. 

Each file contains *timestamps* (s), *speed*(**mph**), and count

## The fundamental diagrams:
1st and 2nd order fundamental diagrams can be found in this [folder](https://github.com/Lab-Work/AIMSUN_with_AVs/tree/master/FD_calibration/calibrated_FDs).

## The true states:
True states can be found in this folder (to be updated).


