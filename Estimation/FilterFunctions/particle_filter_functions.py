import matplotlib.pyplot as plt
from numpy import *
from random import choice
from copy import deepcopy

def init_weight(sample):
## particle weight update function
    weight = 1.0/sample*ones(sample)
    return weight

def regime_transition_nb_normal(model, sample, pTran, laneNumber, cellNumber):
## Regime transition, assuming no incident at the two boundary cells, one incident at a time.
    for i in range(sample):
            if min(model[i]) == laneNumber:
                    if random.random() > pTran:
                            incidentPosition=choice(range(2,int(cellNumber-2)))
                            model[i][incidentPosition]=choice(range(0,laneNumber))
            else:
                    if random.random()>pTran:
                            model[i]=laneNumber*ones(cellNumber)
    return model

def regime_transition_normal(model, sample, pTran, laneNumber, cellNumber):
## Regime transition, one incident at a time.
    for i in range(sample):
            if min(model[i]) == laneNumber:
                    if random.random() > pTran:
                            incidentPosition=choice(range(int(cellNumber)))
                            model[i][incidentPosition]=choice(range(0,laneNumber))
            else:
                    if random.random()>pTran:
                            model[i]=laneNumber*ones(cellNumber)
    return model

def regime_transition_nb_second(model, sample, pTran, pStay, pClear, laneNumber, cellNumber):
## Regime transition, assuming no incident at the two boundary cells, a second incident may occur on the upsteam of the existing incident 
    for i in range(sample):
            if min(model[i]) == laneNumber:
                    if random.random() > pTran:
                            incidentPosition=choice(range(2,int(cellNumber-2)))
                            model[i][incidentPosition]=choice(range(0,laneNumber))
            else:
                incidentNumber=0
                for j in range(len(model[i])):
                    if model[i][j] != laneNumber:
                        incidentNumber = incidentNumber+1
                        if incidentNumber == 1:
                            incidentLocationOne = j
                            incidentSeverityOne = model[i][j]
                        elif incidentNumber == 2:
                            incidentLocationTwo = j
                            incidentSeverityTwo = model[i][j]
                        else:
                            'something is wrong with the incident transition model, please check'
                if incidentNumber == 1:
                    if random.random() > pStay:
                        if random.random() > pClear:
                            try:
                                incidentPositionSecond = choice(range(2,int(incidentLocationOne)))
                                incidentSeverityList = range(int(laneNumber))
                                indexRemove = incidentSeverityList.index(incidentSeverityOne)
                                del incidentSeverityList[indexRemove]
                                incidentSeveritySecond = choice(incidentSeverityList)
                            except IndexError:
                                incidentPositionSecond = 2
                                incidentSeverityList = range(int(laneNumber))
                                indexRemove = incidentSeverityList.index(incidentSeverityOne)
                                del incidentSeverityList[indexRemove]
                                incidentSeveritySecond = choice(incidentSeverityList)
                            model[i][incidentPositionSecond] = incidentSeveritySecond
                        else:
                            model[i] = laneNumber*ones(cellNumber)
                elif incidentNumber == 2:
                    if random.random() >pStay:
                            incidentLocationClear = choice([incidentLocationOne, incidentLocationTwo])
                            model[i][incidentLocationClear] = laneNumber
    return model



def build_operatorH(cellNumber, densityMea, speedMea):
##  Construct the linear operator for the nonlinear observation equation
##  H is m by 2n, m is the number of measurements, n is the number of cells
    mea=hstack((densityMea,speedMea))
    meaExist= mea!=0
    meaNumber=int(sum(meaExist))
    H=zeros((meaNumber,int(2*cellNumber)))
    if meaNumber==0:
        pass
#        print 'there is no measurements at current time step'
    else:
        k=0
        for i in range(int(2*cellNumber)):
            if mea[i] != 0:
                H[k,i]=1
                k=k+1
    return H

def compute_likelihood(cellDensity, cellSpeed, cellNumber, densityMea, speedMea, densityMeaMean, densityMeaStd, speedMeaMean, speedMeaStd):
## Calculate the likelihood
    H=matrix(build_operatorH(cellNumber,densityMea, speedMea))
    allMea = hstack((densityMea, speedMea))
    estimatedDensitySpeed = hstack((cellDensity, cellSpeed))
    diff = (matrix(H)*matrix(allMea).T-matrix(H)*matrix(estimatedDensitySpeed).T)
    modelLikelihood= 1.0 
    for i in range(0,int(sum(H))):
        if i<sum(densityMea !=0):
            modelLikelihood=modelLikelihood*1.0/(densityMeaStd*sqrt(2*pi))*exp(-(diff[i]-densityMeaMean)*(diff[i]-densityMeaMean)/(2*densityMeaStd*densityMeaStd))
        else:
            modelLikelihood=modelLikelihood*1.0/(speedMeaStd*sqrt(2*pi))*exp(-(diff[i]-speedMeaMean)*(diff[i]-speedMeaMean)/(2*speedMeaStd*speedMeaStd))
    return modelLikelihood

def compute_particle_likelihood(cellDensity, cellSpeed, cellNumber, sample, densityMea, speedMea, densityMeaMean, densityMeaStd, speedMeaMean, speedMeaStd):
## Calculate the likelihood of each particle
    likelihood = zeros(int(sample))
    H=matrix(build_operatorH(cellNumber,densityMea, speedMea))
    for j in range(sample):
        allMea = hstack((densityMea, speedMea))
        estimatedDensitySpeed = hstack((cellDensity[j], cellSpeed[j]))
        diff = (matrix(H)*matrix(allMea).T-matrix(H)*matrix(estimatedDensitySpeed).T)
        modelLikelihood= 1.0 
        for i in range(0,int(sum(H))):
            if i<sum(densityMea !=0):
                modelLikelihood=modelLikelihood*1.0/(densityMeaStd*sqrt(2*pi))*exp(-(diff[i]-densityMeaMean)*(diff[i]-densityMeaMean)/(2*densityMeaStd*densityMeaStd))
            else:
                modelLikelihood=modelLikelihood*1.0/(speedMeaStd*sqrt(2*pi))*exp(-(diff[i]-speedMeaMean)*(diff[i]-speedMeaMean)/(2*speedMeaStd*speedMeaStd))
        likelihood[j] = modelLikelihood
    likelihood = likelihood/sum(likelihood)
    return likelihood

def update_weight(likelihood, weight):
## particle weight update function 
    weight=likelihood*weight
    weight=weight/sum(weight)
    return weight

def resampling(state, model, sample, cellNumber, weight):
    Cum=cumsum(weight)
    stateCopy = state.copy()
    modelCopy = model.copy()
    
    step=1.0/sample
    i=1
    u1=random.uniform(0,step)
    for j in range(sample):
        uj = u1+step*(j-1)
        while uj>Cum[i]:
            i=i+1
        state[j]=stateCopy[i]
        model[j]=modelCopy[i]
    return (state, model)

def resampling_three(state, w, model, sample, cellNumber, weight):
    Cum=cumsum(weight)
    stateCopy = state.copy()
    wCopy = w.copy()
    modelCopy = model.copy()
    
    step=1.0/sample
    i=1
    u1=random.uniform(0,step)
    for j in range(sample):
        uj = u1+step*(j-1)
        while uj>Cum[i]:
            i=i+1
        state[j]=stateCopy[i]
        model[j]=modelCopy[i]
        w[j]=wCopy[i]
    return (state, w, model)    
    

def resampling_state(state, sample, cellNumber, weight):
    Cum=cumsum(weight)
    stateCopy = state.copy()
    
    step=1.0/sample
    i=1
    u1=random.uniform(0,step)
    for j in range(sample):
        uj = u1+step*(j-1)
        while uj>Cum[i]:
            i=i+1
        state[j]=stateCopy[i]
    return state


def compute_sample_match(state):
    totalSample = len(state[:,0])
    counter = 1
    for i in range(1,totalSample-1):
        if array_equal(state[i],state[i-1]):
            pass
        else:
            counter = counter+1
    return counter








    
    
    















