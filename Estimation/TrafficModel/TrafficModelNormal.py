# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 12:46:28 2015

@author: Ren

Traffic models with three different fundamental diagrams are coded. 

quadratic linear (1st or 2nd)
quadratic quadratic (1st or 2nd) 
linear quadratic (1st only, w must be constant)

The models are coded as second order models, when w is specified as constant, the models perform as first order traffic models.
For the linear quadratic case, w must be constant. 
"""

from numpy import *

'''

fdpNB = rhoc1NB, rhoc2NB, rhom1NB, rhom2NB, vmaxNB, rhomTildeNB
fdpBL5 = rhoc1L5B1, rhoc2L5B1, vmaxL5B1, rhomTildeL5B1, \
         rhoc1L5B2, rhoc2L5B2, vmaxL5B2, rhomTildeL5B2, \
         rhoc1L5B3, rhoc2L5B3, vmaxL5B3, rhomTildeL5B3, \
         rhoc1L5B4, rhoc2L5B4, vmaxL5B4, rhomTildeL5B4 
         
fdpBL4 = rhoc1L4B1, rhoc2L4B1, vmaxL4B1, rhomTildeL4B1, \
         rhoc1L4B2, rhoc2L4B2, vmaxL4B2, rhomTildeL4B2, \
         rhoc1L4B3, rhoc2L4B3, vmaxL4B3, rhomTildeL4B3 

         
fdpBL3 = rhoc1L3B1, rhoc2L3B1, vmaxL3B1, rhomTildeL3B1, \
         rhoc1L3B2, rhoc2L3B2, vmaxL3B2, rhomTildeL3B2

fdpBL2 = rhoc1L2B1, rhoc2L2B1, vmaxL2B1, rhomTildeL2B1
'''

########################################################################################################################################################

def ctm_2ql_incident(rhoVector, wVector, modelVector, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2, \
                     Lambda, bdl, bdr, laneNumber, modelNoiseMean, modelNoiseStd, inflow = -1.0, outflow = -1.0):
                     
## attach boundary conditions
        rhoLeftGhost = bdl[0]
        wLeftGhost = bdl[1]
        modelLeftGhost = bdl[2]
        rhoRightGhost = bdr[0]
        wRightGhost = bdr[1]
        modelRightGhost = bdr[2]                                

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
                                fluxIn = flux_2ql_incident(rhoLeftGhost, rhoVector[i], wLeftGhost, wVector[i], modelLeftGhost, modelVector[i],\
                                                           laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
                                

                        fluxOut = flux_2ql_incident(rhoVector[i], rhoVector[i+1], wVector[i], wVector[i+1], modelVector[i], modelVector[i+1],\
                                                    laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)


                        fluxWIn = wLeftGhost*fluxIn
                        fluxWOut = wVector[i]*fluxOut

                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]

                elif i<int(len(rhoVector)-1):
                        fluxIn = flux_2ql_incident(rhoVector[i-1], rhoVector[i], wVector[i-1], wVector[i], modelVector[i-1], modelVector[i],\
                                                   laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)

                        
                        fluxOut = flux_2ql_incident(rhoVector[i], rhoVector[i+1], wVector[i], wVector[i+1], modelVector[i], modelVector[i+1],\
                                                    laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
                                                           
                        fluxWIn = wVector[i-1]*fluxIn
                        fluxWOut = wVector[i]*fluxOut
                        
                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]

                else:
                        fluxIn = flux_2ql_incident(rhoVector[i-1], rhoVector[i], wVector[i-1], wVector[i], modelVector[i-1], modelVector[i],\
                                                   laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
                                  
                        if outflow != -1.0:
                                fluxOut = outflow
                        else:
                                fluxOut = flux_2ql_incident(rhoVector[i], rhoRightGhost, wVector[i], wRightGhost, modelVector[i], modelRightGhost,\
                                                            laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)

                        fluxWIn = wVector[i-1]*fluxIn
                        fluxWOut = wVector[i]*fluxOut
                        
                        rhoVectorUpdated[i] = rhoVector[i] + Lambda*(fluxIn - fluxOut)
                        rhoWVectorUpdated[i] = rhoWVector[i] + Lambda*(fluxWIn - fluxWOut)
                        wVectorUpdated[i] =  rhoWVectorUpdated[i]/rhoVectorUpdated[i]
        return (rhoVectorUpdated+random.normal(modelNoiseMean, modelNoiseStd, len(rhoVector)), wVectorUpdated)

########################################################################################################################################################

def fd_parameter(w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):
    
## compute the fundamental diagram parameter given model and laneNumber
## O(1)

        if laneNumber == 5:
            fdpB = fdpBL5
        elif laneNumber == 4:
            fdpB = fdpBL4
        elif laneNumber == 3:
            fdpB = fdpBL3
        elif laneNumber == 2:
            fdpB = fdpBL2
        elif laneNumber == 1:
            pass
##            print 'undefined model in the system'

        if model == laneNumber or model == 0:
                rhoc = model*(fdpNB[0]*fdpNB[1]/(fdpNB[1]*w+fdpNB[0]*(1-w)))
                rhom = model*(fdpNB[2]*fdpNB[3]/(fdpNB[3]*w+fdpNB[2]*(1-w)))
                vmax = fdpNB[4]
                rhomTilde = model*fdpNB[5]
        elif model == (laneNumber - 1):
                rhoc = model*(fdpB[4*(1-1)+0]*fdpB[4*(1-1)+1]/(fdpB[4*(1-1)+1]*w+fdpB[4*(1-1)+0]*(1-w)))
                rhom = model*(fdpNB[2]*fdpNB[3]/(fdpNB[3]*w+fdpNB[2]*(1-w)))
                vmax = fdpB[4*(1-1)+2]
                rhomTilde = model*fdpB[4*(1-1)+3]
        elif model == (laneNumber - 2):
                rhoc = model*(fdpB[4*(2-1)+0]*fdpB[4*(2-1)+1]/(fdpB[4*(2-1)+1]*w+fdpB[4*(2-1)+0]*(1-w)))
                rhom = model*(fdpNB[2]*fdpNB[3]/(fdpNB[3]*w+fdpNB[2]*(1-w)))
                vmax = fdpB[4*(2-1)+2]
                rhomTilde = model*fdpB[4*(2-1)+3]
        elif model == (laneNumber - 3):
                rhoc = model*(fdpB[4*(3-1)+0]*fdpB[4*(3-1)+1]/(fdpB[4*(3-1)+1]*w+fdpB[4*(3-1)+0]*(1-w)))
                rhom = model*(fdpNB[2]*fdpNB[3]/(fdpNB[3]*w+fdpNB[2]*(1-w)))
                vmax = fdpB[4*(3-1)+2]
                rhomTilde = model*fdpB[4*(3-1)+3]
        elif model == (laneNumber - 4):
                rhoc = model*(fdpB[4*(4-1)+0]*fdpB[4*(4-1)+1]/(fdpB[4*(4-1)+1]*w+fdpB[4*(4-1)+0]*(1-w)))
                rhom = model*(fdpNB[2]*fdpNB[3]/(fdpNB[3]*w+fdpNB[2]*(1-w)))
                vmax = fdpB[4*(4-1)+2]
                rhomTilde = model*fdpB[4*(4-1)+3]
        else:
                print 'undefined model in the system'
        return rhoc, rhom, vmax, rhomTilde     

########################################################################################################################################################

def sending_2ql_incident(rho, w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):

## sending function of the 2CTM
## O(1)

        rhoc, rhom, vmax, rhomTilde = fd_parameter(w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
                
        if rhoc == 0 or rho < 0:
                qSend = 0.0
        elif rho < rhoc:
                qSend = rho*vmax*(1-rho/rhomTilde)
        else:
                qSend = rhoc*vmax*(1-rhoc/rhomTilde)
        return qSend
     
     
########################################################################################################################################################

def vel_2ql_incident_scalar(rho, w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):

## velocity function for a cell
## O(1)

        rhoc, rhom, vmax, rhomTilde = fd_parameter(w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)

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

def vel_2ql_incident(rhoVector, wVector, modelVector, laneNumberVector, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):
                         
## velocity function for a link
## O(n)

        vVector = zeros(len(rhoVector))
        for i in range(len(rhoVector)):
                w = wVector[i]
                model = modelVector[i]
                laneNumber = laneNumberVector[i]
                rhoc, rhom, vmax, rhomTilde = fd_parameter(w, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
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
        
def middle_state_2ql(rho, wUS, wDS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):
  
## solve the middle state   
## O(1)                      
                         
        if model == 0:
                rhoMiddle = 10000
        else:
                # compute the downstream velocity
                vDS = vel_2ql_incident_scalar(rho, wDS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
                
                # compute fd parameters using the upstream property wUS
                rhoc, rhom, vmax, rhomTilde = fd_parameter(wUS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
                # compute the critical velocity
                vc = vmax*(1-rhoc/rhomTilde)
                # compute rhoMiddle
                if vDS <= vc:
                        rhoMiddle = vmax*rhoc*rhom*(rhomTilde-rhoc)/(vDS*rhomTilde*(rhom-rhoc)+rhoc*vmax*(rhomTilde-rhoc))
                elif vDS > vc:
                        rhoMiddle = rhomTilde*(1-vDS/vmax)
        return rhoMiddle 
        
########################################################################################################################################################

def receiving_2ql_incident(rho, wUS, wDS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):

## receive function 
## O(1)

        rhoMiddle = middle_state_2ql(rho, wUS, wDS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
        rhoc, rhom, vmax, rhomTilde = fd_parameter(wUS, model, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)

        if rhoc == 0 or rhoMiddle > rhom:
                qReceive = 0.0
        elif rhoMiddle < rhoc:
                qReceive = rhoc*vmax*(1-rhoc/rhomTilde)
        else:
                qReceive = vmax*rhoc*(rhom-rhoMiddle)*(rhomTilde-rhoc)/(rhomTilde*(rhom-rhoc))
        return qReceive
        
########################################################################################################################################################

def flux_2ql_incident(rhoUS, rhoDS, wUS, wDS, modelUS, modelDS, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):
                         
        fluxSend = sending_2ql_incident(rhoUS, wUS, modelUS, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
        fluxReceive = receiving_2ql_incident(rhoDS, wUS, wDS, modelDS, laneNumber, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
        flux = minimum(fluxSend, fluxReceive)

        return flux
        
        
def flux_2ql_incident_junction(rhoUS, rhoDS, wUS, wDS, modelUS, modelDS, laneNumberUS, laneNumberDS, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2):
                         
        fluxSend = sending_2ql_incident(rhoUS, wUS, modelUS, laneNumberUS, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
        fluxReceive = receiving_2ql_incident(rhoDS, wUS, wDS, modelDS, laneNumberDS, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
        
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
    
    rhoc1Ramp = 52.0
    rhoc2Ramp = 48.0
    rhom1Ramp = 110.0
    rhom2Ramp = 90.0
    vmaxRamp = 40.0
    rhomTildeRamp = 10000.0
    
    rhoc1NB = 28.0
    rhoc2NB = 20.0
    rhom1NB = 110.0
    rhom2NB = 90.0
    vmaxNB = 71.0
    rhomTildeNB = 10000.0
    
    rhoc1L5B1 = 78.0
    rhoc2L5B1 = 74.0
    vmaxL5B1 = 18.0
    rhomTildeL5B1 = 10000.0
    rhoc1L5B2 = 65.0
    rhoc2L5B2 = 61.0
    vmaxL5B2 = 18.0
    rhomTildeL5B2 = 10000.0
    rhoc1L5B3 = 49.0
    rhoc2L5B3 = 45.0
    vmaxL5B3 = 18.0
    rhomTildeL5B3 = 10000.0
    rhoc1L5B4 = 49.0
    rhoc2L5B4 = 45.0
    vmaxL5B4 = 18.0
    rhomTildeL5B4 = 10000.0
             
    rhoc1L4B1 = 75.0
    rhoc2L4B1 = 71.0
    vmaxL4B1 = 18.0 
    rhomTildeL4B1 = 10000.0
    rhoc1L4B2 = 49.0
    rhoc2L4B2 = 45.0
    vmaxL4B2 = 18.0
    rhomTildeL4B2 = 10000.0
    rhoc1L4B3 = 51.0
    rhoc2L4B3 = 47.0
    vmaxL4B3 = 18.0 
    rhomTildeL4B3 = 10000.0
    
    rhoc1L3B1 = 71.0
    rhoc2L3B1 = 67.0
    vmaxL3B1 = 18.0 
    rhomTildeL3B1 = 10000.0
    rhoc1L3B2 = 50.0
    rhoc2L3B2 = 46.0
    vmaxL3B2 = 18.0
    rhomTildeL3B2 = 10000.0
    
    rhoc1L2B1 = 68.0
    rhoc2L2B1 = 64.0
    vmaxL2B1 = 18.0
    rhomTildeL2B1 = 10000.0
    
    
    fdpRamp = rhoc1Ramp, rhoc2Ramp, rhom1Ramp, rhom2Ramp, vmaxRamp, rhomTildeRamp    
    fdpNB = rhoc1NB, rhoc2NB, rhom1NB, rhom2NB, vmaxNB, rhomTildeNB
    fdpBL5 = rhoc1L5B1, rhoc2L5B1, vmaxL5B1, rhomTildeL5B1, \
             rhoc1L5B2, rhoc2L5B2, vmaxL5B2, rhomTildeL5B2, \
             rhoc1L5B3, rhoc2L5B3, vmaxL5B3, rhomTildeL5B3, \
             rhoc1L5B4, rhoc2L5B4, vmaxL5B4, rhomTildeL5B4 
             
    fdpBL4 = rhoc1L4B1, rhoc2L4B1, vmaxL4B1, rhomTildeL4B1, \
             rhoc1L4B2, rhoc2L4B2, vmaxL4B2, rhomTildeL4B2, \
             rhoc1L4B3, rhoc2L4B3, vmaxL4B3, rhomTildeL4B3 
    
             
    fdpBL3 = rhoc1L3B1, rhoc2L3B1, vmaxL3B1, rhomTildeL3B1, \
             rhoc1L3B2, rhoc2L3B2, vmaxL3B2, rhomTildeL3B2
    
    fdpBL2 = rhoc1L2B1, rhoc2L2B1, vmaxL2B1, rhomTildeL2B1
    
    modelNoiseMean = 0
    modelNoiseStd = 0.1

    # boundary
    bdl=[105,0.1,4]
    bdr=[105,0.1,4]

    # init    
    rhoVector = 10*ones(cellNumber)
    rhoVector[0] = 1000.0
    rhoVector[6] = 140.0
    rhoVector[7] = 200.0
    
    wVector = 0.5*ones(cellNumber)
    wVector[0] = 0.2
    wVector[1] = 0.6
    wVector[2] = 0.8
    modelVector = 4.0* ones(cellNumber)
    
    laneNumberVector = 4.0* ones(cellNumber)
    laneNumber = 4.0

    timeStep=100
    
    
    ##  state
    estimatedState = zeros((timeStep, cellNumber))
    estimatedSpeed = zeros((timeStep, cellNumber))
    state = zeros(cellNumber)
    model = laneNumber*ones(cellNumber)
    model[4]=4
    
    for i in range(int(timeStep)):
            if i==60:
                    modelVector[4] = 2
            if i==120:
                    modelVector[4] = 4
            rhoVector, wVector = ctm_2ql_incident(rhoVector, wVector, modelVector, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2, \
                     Lambda, bdl, bdr, laneNumber, modelNoiseMean, modelNoiseStd, inflow = -1.0, outflow = -1.0)

            estimatedSpeed[i] = vel_2ql_incident(rhoVector, wVector, modelVector, laneNumberVector, fdpNB, fdpBL5, fdpBL4, fdpBL3, fdpBL2)
            estimatedState[i] = rhoVector.copy()
            
    
    plot_density(estimatedState)
    plot_speed(estimatedSpeed)
