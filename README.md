# Estimating mixed human and automated traffic flows in AIMSUN using the particle filter
Ren Wang, Yanning Li, Daniel Work, Auguest 9th, 2016

## 1) Overview
This repository contains all the source code and data to replicate the work for the article *Comparing traffic state estimators for mixed human and automated traffic flows* pubilshed in *Transportation Research Part C: Emerging technologies*. The article can be downloaded in [link](http://www.sciencedirect.com/science/article/pii/S0968090X17300517).

This work simulated mixed traffic in AIMSUN and compared the first and second order traffic model traffic estimators in the particle filter framework. The numerical experiment setup is described as follows:
1. Simulated a three-mile two-lane one-direction freeway, with four detectors at one mile spacing: PM0, PM1, PM2, PM3. The detectors PM0 and PM3 were located at the entrance and exit of the freeway.
2. Simulated one hour of traffic containing the formation and dissipation of traffic congestion.
3. Simulated five scenarios, where the penetration rates of the inflow are sampled from uniform distributions U(0,0), U(0,0.25), U(0,0.5), U(0,0.75), U(0,1). For each scenario, ten replications with different seeds were conducted to reduce the effect of stochasticity of the microsimulation. For each replication, ten runs of PF estimators for both the first order and the second order models were performed to account for the stochasticity of the PF. 

## 2) License

This software is licensed under the *University of Illinois/NCSA Open Source License*:

**Copyright (c) 2016 The Board of Trustees of the University of Illinois. All rights reserved**

**Developed by: Department of Civil and Environmental Engineering University of Illinois at Urbana-Champaign**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. Neither the names of the Department of Civil and Environmental Engineering, the University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

## 3) Folders
This repository is organized as follow:
### ./Simulation
This folder contains all the source code and data needed to run the simulations.

### ./Estimation
This folder contains the source code and data for the traffic estimation using the particle filter. 

Please see the readme file in each directory for how to run the simulation and estimators.







