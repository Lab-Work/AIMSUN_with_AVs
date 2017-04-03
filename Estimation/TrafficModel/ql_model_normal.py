# -*- coding: utf-8 -*-
"""
Created in June 2016
s
@author: Ren


The model is coded as a second order models, when w is specified as constant, the model performs the same as a first order traffic models.

"""

from numpy import *

'''
fdpNB = rhoc_1, rhoc_10, rhoc_20, rhoc_30, rhoc_40, rhoc_50, rhoc_60,\
        rhoc_70, rhoc_80, rhoc_90, rhoc_99, rhom_all, vmax_all, beta,

'''

########################################################################################################################################################

def ctm_2ql(rhoVector, wVector, fdpNB, Lambda, bdl, bdr, modelNoiseMean, modelNoiseStd, inflow = -1.0, outflow = -1.0):
                     
## attach boundary conditions
        rhoLeftGhost = bdl[0]
        wLeftGhost = bdl[1]
        rhoRightGhost = bdr[0]
        wRightGhost = bdr[1]

## forward prediction

        rhoWVector = rhoVector*wVector        

        rhoVectorUpdated = zeros(len(rhoVector))
        rhoWVectorUpdated = zeros(len(rhoVector))
        wVectorUpdated = zeros(len(rhoVector))

        for i in range(len(rhoVector)):
                if i==0:
                        if inflow != -1.0:
                                fluxIn = inflow
                        else:
                                fluxIn = flux_2ql(rhoLeftGhost, rhoVector[i], wLeftGhost, wVector[i], fdpNB)
                                
                        fluxOut = flux_2ql(rhoVector[i], rhoVector[i+1], wVector[i], wVector[i+1], fdpNB)

                        fluxWIn = wLeftGhost*fluxIn
                        fluxWOut = wVector[i]*fluxOut

                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]

                elif i<int(len(rhoVector)-1):
                        fluxIn = flux_2ql(rhoVector[i-1], rhoVector[i], wVector[i-1], wVector[i], fdpNB)
                        
                        fluxOut = flux_2ql(rhoVector[i], rhoVector[i+1], wVector[i], wVector[i+1], fdpNB)
                                                           
                        fluxWIn = wVector[i-1]*fluxIn
                        fluxWOut = wVector[i]*fluxOut
                        
                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]

                else:
                        fluxIn = flux_2ql(rhoVector[i-1], rhoVector[i], wVector[i-1], wVector[i], fdpNB)
                                  
                        if outflow != -1.0:
                                fluxOut = outflow
                        else:
                                fluxOut = flux_2ql(rhoVector[i], rhoRightGhost, wVector[i], wRightGhost, fdpNB)

                        fluxWIn = wVector[i-1]*fluxIn
                        fluxWOut = wVector[i]*fluxOut
                        
                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]
        return (rhoVectorUpdated+random.normal(modelNoiseMean, modelNoiseStd, len(rhoVector)), wVectorUpdated)

########################################################################################################################################################

def fd_parameter(w, fdpNB):
    
## compute the fundamental diagram parameter given model
## O(1)

        rhomTilde = fdpNB[-1]
        vmax = fdpNB[-2]
        rhom = fdpNB[-3]
        pr = w

        if w < 1.0/100:
            rhoc = fdpNB[0]
        elif 1.0/100 <= w < 10.0/100:
            rhoc_min = fdpNB[0] 
            rhoc_max = fdpNB[1] 
            pr_min = 1.0/100
            pr_max = 10.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 10.0/100 <= w < 20.0/100:
            rhoc_min = fdpNB[1] 
            rhoc_max = fdpNB[2] 
            pr_min = 10.0/100
            pr_max = 20.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 20.0/100 <= w < 30.0/100:
            rhoc_min = fdpNB[2] 
            rhoc_max = fdpNB[3] 
            pr_min = 20.0/100
            pr_max = 30.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 30.0/100 <= w < 40.0/100:
            rhoc_min = fdpNB[3] 
            rhoc_max = fdpNB[4] 
            pr_min = 30.0/100
            pr_max = 40.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 40.0/100 <= w < 50.0/100:
            rhoc_min = fdpNB[4] 
            rhoc_max = fdpNB[5] 
            pr_min = 40.0/100
            pr_max = 50.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 50.0/100 <= w < 60.0/100:
            rhoc_min = fdpNB[5] 
            rhoc_max = fdpNB[6] 
            pr_min = 50.0/100
            pr_max = 60.0/100
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 60.0/100 <= w < 70.0/100:
            rhoc_min = fdpNB[6] 
            rhoc_max = fdpNB[7] 
            pr_min = 60.0/100
            pr_max = 70.0/100            
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 70.0/100 <= w < 80.0/100:
            rhoc_min = fdpNB[7] 
            rhoc_max = fdpNB[8] 
            pr_min = 70.0/100
            pr_max = 80.0/100            
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 80.0/100 <= w < 90.0/100:
            rhoc_min = fdpNB[8] 
            rhoc_max = fdpNB[9] 
            pr_min = 80.0/100
            pr_max = 90.0/100            
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 90.0/100 <= w < 99.0/100:
            rhoc_min = fdpNB[9] 
            rhoc_max = fdpNB[10] 
            pr_min = 90.0/100
            pr_max = 99.0/100            
            rhoc = (rhoc_max-rhoc_min)/(pr_max-pr_min)*(pr-pr_min)+rhoc_min
        elif 99.0/100 <= w:
            rhoc = fdpNB[10] 
            
        return rhoc, rhom, vmax, rhomTilde     

########################################################################################################################################################

def sending_2ql(rho, w, fdpNB):

## sending function of the 2CTM
## O(1)

        rhoc, rhom, vmax, rhomTilde = fd_parameter(w, fdpNB)
                
        if rhoc == 0 or rho < 0:
                qSend = 0.0
        elif rho < rhoc:
                qSend = rho*vmax*(1-rho/rhomTilde)
        else:
                qSend = rhoc*vmax*(1-rhoc/rhomTilde)
        return qSend
     
     
########################################################################################################################################################

def vel_2ql_scalar(rho, w, fdpNB):

## velocity function for a cell
## O(1)

        rhoc, rhom, vmax, rhomTilde = fd_parameter(w, fdpNB)

        if rhoc == 0 or rho>rhom:
                v = 0.0
        elif rho <= 0:
                v = vmax
        elif rho <= rhoc:
                v = vmax*(1-rho/rhomTilde)
        else:
                v = vmax*rhoc*(rhom-rho)*(rhomTilde-rhoc)/(rho*rhomTilde*(rhom-rhoc))
        return v

########################################################################################################################################################

def vel_2ql(rhoVector, wVector, fdpNB):
                         
## velocity function for a link
## O(n)

        vVector = zeros(len(rhoVector))
        for i in range(len(rhoVector)):
                w = wVector[i]
                rhoc, rhom, vmax, rhomTilde = fd_parameter(w, fdpNB)
                rho = rhoVector[i]
                
                if rhoc == 0 or rho > rhom:
                        v = 0.0
                elif rho <= 0:
                        v = vmax
                elif rho < rhoc:
                        v = vmax*(1-rho/rhomTilde)
                else:
                        v = vmax*rhoc*(rhom-rho)*(rhomTilde-rhoc)/(rho*rhomTilde*(rhom-rhoc))                    
                vVector[i] = v
        return vVector
        
########################################################################################################################################################        
        
def middle_state_2ql(rho, wUS, wDS, fdpNB):
  
## solve the middle state   
## O(1)                      
                         
        # compute the downstream velocity
        vDS = vel_2ql_scalar(rho, wDS, fdpNB)
        
        # compute fd parameters using the upstream property wUS
        rhoc, rhom, vmax, rhomTilde = fd_parameter(wUS, fdpNB)

        # compute the critical velocity
        vc = vmax*(1-rhoc/rhomTilde)
        # compute rhoMiddle
        if vDS <= vc:
                rhoMiddle = vmax*rhoc*rhom*(rhomTilde-rhoc)/(vDS*rhomTilde*(rhom-rhoc)+rhoc*vmax*(rhomTilde-rhoc))
        elif vDS > vc:
                rhoMiddle = rhomTilde*(1-vDS/vmax)
                
        return rhoMiddle 
        
########################################################################################################################################################

def receiving_2ql(rho, wUS, wDS, fdpNB):

## receive function 
## O(1)

        rhoMiddle = middle_state_2ql(rho, wUS, wDS, fdpNB)
        
        rhoc, rhom, vmax, rhomTilde = fd_parameter(wUS, fdpNB)

        if rhoc == 0 or rhoMiddle > rhom:
                qReceive = 0.0
        elif rhoMiddle < rhoc:
                qReceive = rhoc*vmax*(1-rhoc/rhomTilde)
        else:
                qReceive = vmax*rhoc*(rhom-rhoMiddle)*(rhomTilde-rhoc)/(rhomTilde*(rhom-rhoc))
        return qReceive
        
########################################################################################################################################################

def flux_2ql(rhoUS, rhoDS, wUS, wDS, fdpNB):
                         
        fluxSend = sending_2ql(rhoUS, wUS, fdpNB)
        
        fluxReceive = receiving_2ql(rhoDS, wUS, wDS, fdpNB)
        
        flux = minimum(fluxSend, fluxReceive)

        return flux
        
        


#########################################################################################################################################################


if __name__ == '__main__':
    def plot_density(data):
        plt.rc('xtick',labelsize=30)
        plt.rc('ytick',labelsize=30)
        plt.imshow(data,aspect='auto',origin='lower',interpolation='nearest')
        plt.ylabel('Time Step',fontsize=30)
        plt.clim(0.0, 560)
        plt.xlabel('Cell Number',fontsize=30)
        plt.colorbar()
        plt.show()
        plt.clf()

    def plot_speed(data):    
        plt.rc('xtick',labelsize=30)
        plt.rc('ytick',labelsize=30)
        cmap=plt.cm.jet_r
        plt.imshow(data,aspect='auto',origin='lower', cmap = cmap, interpolation='nearest')
        plt.ylabel('Time Step',fontsize=30)
        plt.clim(0.0, 80)
        plt.xlabel('Cell Number',fontsize=30)
        plt.colorbar()
        plt.show()
        plt.clf()



    # discretization
    dt=5.0/3600
    dx=0.1
    Lambda=dt/dx
    length = 4.0
    cellNumber = floor(length/dx)    
    

    rhoc_1 = 72.48
    rhoc_10 = 76.62 
    rhoc_20 = 83.46
    rhoc_30 = 90.22
    rhoc_40 = 96.83
    rhoc_50 = 109.29
    rhoc_60 = 120.55
    rhoc_70 = 133.88
    rhoc_80 = 151.52
    rhoc_90 = 187.2
    rhoc_99 = 234.37
    rhom_all = 644
    vmax_all = 72.81
    beta = 775.56
 
    fdpNB = rhoc_1, rhoc_10, rhoc_20, rhoc_30, rhoc_40, rhoc_50, rhoc_60,\
            rhoc_70, rhoc_80, rhoc_90, rhoc_99, rhom_all, vmax_all, beta, 
    
    modelNoiseMean = 0
    modelNoiseStd = 0.1

    # boundary
    bdl=[305,0.9,4]
    bdr=[105,0.1,4]

    # init    
    rhoVector = 10*ones(cellNumber)
#    rhoVector[0] = 1000.0
#    rhoVector[6] = 140.0
#    rhoVector[7] = 200.0
    
    wVector = 0.5*ones(cellNumber)
    wVector[0] = 0.2
    wVector[1] = 0.6
    wVector[2] = 0.8

    

    timeStep=100
    
    
    ##  state
    estimatedState = zeros((timeStep, cellNumber))
    estimatedSpeed = zeros((timeStep, cellNumber))
    state = zeros(cellNumber)


    
    for i in range(int(timeStep)):

            rhoVector, wVector = ctm_2ql(rhoVector, wVector, fdpNB, Lambda, bdl, bdr, modelNoiseMean, modelNoiseStd, inflow = -1.0, outflow = -1.0)

            estimatedSpeed[i] = vel_2ql(rhoVector, wVector, fdpNB)

            estimatedState[i] = rhoVector.copy()
            
    
    plot_density(estimatedState)
#    plot_speed(estimatedSpeed)
