#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
import sys

class emClustering():
    def __init__(self,n):
        '''
        @topic: Declare Initial Parameters
        @Parameters:
            1. n = no of clusters
        '''
        self.n = n

    def inputData(self,dataset_name):
        '''
        @topic: Input and preprocess the data
        @parameters:
            dataset_name: name of the dataset
        '''
        df = pd.read_csv(dataset_name,header=None)
        train_data = df.values
        train_X = train_data[:,0:4]
        return train_X.astype(float),train_data[:,4]

    def initializeCluster(self,dataset,noOfClusters):
        '''
        @topic: Initialize clusters n/k,n/k,... k. The last cluster may have a smaller number of elements
        @Parameters:
            dataset: full dataset
            noOfClusters: No of Clusters dataset to be divided into
        '''
        szData = len(dataset[:,0])
        szCluster = math.ceil(szData/noOfClusters)
        clusterAssign = np.zeros(szData)
        for i in range(szData):
            startId = (i-1)*szCluster
            endId = min(startId + szCluster + 1,szData)
            clusterAssign[startId:endId] = i
        return clusterAssign.astype(int)

    def emInitialization(self,dataset,clusterAssign):
        '''
        @topic: Calculates means, stddev and P(Ci) for each of the cluster of n dimension
        @Parameters:
            dataset: full dataset
            clusterAssign: Cluster Assignment matrix indicating which element belong to which cluster
        '''
        if len(dataset[:,0]) == len(clusterAssign):
            clusterId = np.unique(clusterAssign)
            meanMatrix = np.zeros((len(clusterId),len(dataset[0,:])))
            varMatrix = []
            pOfCi = np.zeros(len(clusterId))
            pOfCi[:] = 1/len(clusterId)
            for i in clusterId:
                dataPts = np.where(clusterAssign == i)
                meanMatrix[i-1,:] = np.mean(dataset[dataPts,:],axis=1)
                #varMatrix.append(np.cov(np.transpose(dataset[dataPts])))
                ## Neglecting cross dimensional dependency, calc only diagonal terms
                ##
                weight = np.ones(len(dataPts[0]))
                axisIn = 0
                thisClusterData = dataset[dataPts,:]
                varMatrix.append(self.calcDiagCovMat(thisClusterData[0],weight))
        return meanMatrix,varMatrix,pOfCi


    def calcDiagCovMat(self,dataset,weight):
        '''
        @topic: Calculates covariances neglecting inter-dimension dependency
        @Parameters:
            1. dataset = full dataset
            2. weight = weight of each of the parameters
        '''
        #thisVarLine = np.var(dataset,axis=1)
        ## Creating fx for weighted variance
        #print("dataset: " + str(len(dataset[:,0])) + " x " + str(len(dataset[0,:])))
        thisVarLine = np.average((dataset-np.average(dataset,axis = 0))**2,axis = 0,weights = weight)
        sz = len(thisVarLine)
        varianceMatrix = np.zeros((sz,sz))
        for i in range(sz):
            varianceMatrix[i,i] = thisVarLine[i]
        return varianceMatrix

    def calcNormalProbabilityDensity(self,x,mean,coVarMat):
        '''
        @topic: Calculates normal distribution probabilty density for given x
        @Parameters:
            1. x = co-ordinates of the point for calculating prob density at
            2. mean = mean of the distribution
            3. coVarMat =  Covariance Matrix of the distribution
        '''
        d = len(coVarMat[0,:])  # dimensions of data
        det_coVar = np.linalg.det(coVarMat)
        x_mu = x-mean
        invCoVar = np.linalg.inv(coVarMat)
        denominator = math.pow(2*np.pi,d/2)*math.pow(det_coVar,0.5)

        exponentTerm =  math.exp(-0.5*np.matmul(np.matmul(np.transpose(x_mu),invCoVar),x_mu))
        probDensity = exponentTerm/denominator
        return probDensity

    def emExpectation(self,dataset,clusterAssign,mean,coVarMat,pOfCi):
        '''
         @topic: Calculates the Wi for each clusters: Posterior probability P(Ci/xj) using eq 13.9 Pg 382 zaki = P(Ci). P(xi/Ci)/P(xi)
         @Parameters:
            1. dataset = full dataset
            2. clusterAssign = n x 1 array indicating the assignment of each of n point in p clusters
            3. mean = mean of the distribution
            4. coVarMat =  Covariance Matrix of the distribution
            5. pOfCi = Probability of Cluster Ci for given data
        '''
        Wij = np.zeros((len(dataset[:,0]),len(np.unique(clusterAssign))))
        for i in range(len(Wij[:,0])): # run for each point

            # Calculate sigma[(f(x).P(Ca))] suming the term for each cluster for this point
            pXnormalProb = np.zeros(len(Wij[0,:]))
            for j in range(len(Wij[0,:])):
                pXnormalProb[j] = pOfCi[j]*self.calcNormalProbabilityDensity(dataset[i,:],mean[j,:],coVarMat[j])
            denominator = sum(pXnormalProb)
            Wij[i] = pXnormalProb*(1/denominator)
        return Wij


    def emMaximization(self,dataset,Wij,mean,coVar,pOfCi):
        '''
         @topic: Calculates new Means, CoVariance and P(Ci) based on the cluster assignment in Expectation step
         @Parameters:
            1. dataset = full dataset
            2. Wij = Expectation array obtained from Expectation Step
        '''
        noOfClusters = len(mean[:,0])
        newMean = np.zeros((len(mean[:,0]),len(mean[0,:])))
        newCoVar = []
        newPOfCi = np.zeros(len(pOfCi))
        for i in range(noOfClusters):
            newMean[i,:] = np.average(dataset,axis = 0, weights = Wij[:,i])
            newCoVar.append(self.calcDiagCovMat(dataset,Wij[:,i]))
        newPOfCi = np.average(Wij,axis=0)

        return newMean,newCoVar,newPOfCi

    def findEuclideanDist(self,newVect,oldVect):
        '''
         @topic: Calculates Euclidean Distance between  two given vectors newVect and oldVect
        '''    
        if len(newVect) == len(oldVect):
            diff = newVect-oldVect
            dist = 0
            for i in range(len(diff)):
                dist = dist + diff[i]**2
            eucDist = math.sqrt(dist)
        else:
            print("Size Mis-match between input vectors in calculating Euclidean Distance")
            eucDist = None
        return eucDist

    def normSortedMean(self,thisMatrix):
        '''
         @topic: Calculates norms of matrix and sorts them in ascending order and returns the sortIds for the norm sort
        '''
        #normVal = np.zeros(len(thisMatrix[0,:]))
        normVal = np.linalg.norm(thisMatrix,axis = 1)
        sortId = np.argsort(normVal)
        return sortId

    def calcPurity(self,predicted,actual):
        '''
         @topic: Calculates the purity score for for the prediction based on actual classification of the cluster
         @Parameters:
            1. predicted = predicted array of cluster assignment
            2. actual = known array of cluster assignment 
        '''
        sz = len(actual)
        if sz == len(predicted):
            # Get the unique list of clusters 
            predictedList = np.unique(predicted)
            actualList = np.unique(actual)
            purity = 0
            thisIntersect = np.zeros(len(actualList))
            for i in range(len(predictedList)):
                # Find points for each of the clusters in the predicted list
                subActual = actual[predicted == predictedList[i]]
                for j in range(len(actualList)):
                    # Find number of points that match the actual cluster name
                    thisIntersect[j] = len(subActual[subActual == actualList[j]])
                purity = purity + max(thisIntersect)
            purity = purity/sz
        else:
            print("Size Mis-match between actual and predicted value in calcPurity")
            purity = None
        return purity



if __name__ == '__main__':
    n = 3  # No of clusters
    dataset_name = 'iris.data'

    if len(sys.argv) > 1 and len(sys.argv) < 4:
        dataset_name = sys.argv[1]
        n = int(sys.argv[2])

    # Class Instantiation
    em = emClustering(n)

    # import the data from the file
    [trainData,trueClassification] = em.inputData(dataset_name)

    # Initialize the cluster
    clusterAssign = em.initializeCluster(trainData,n)

    # Calculate the mean for each cluster
    [clusterWiseMean,clusterWiseVar,pOfCi] = em.emInitialization(trainData,clusterAssign)

    #print(clusterWiseMean)
    #print(clusterWiseVar)
    #print(pOfCi)

    # Initialize EM Iteration until error is epsilon < 0.00001
    epsilon = 1
    mean = clusterWiseMean
    coVar = clusterWiseVar
    pOfCi = pOfCi
    counter = 0
    while epsilon > 0.00001:
        counter = counter + 1
        '''
        ## Expectation Step (Calculating posterier probability of cluster Ci given point xj)
        #Wi matrix i.e. P(Ci)
        '''
        Wij = em.emExpectation(trainData,clusterAssign,clusterWiseMean,clusterWiseVar,pOfCi)
        #print(Wij)


        '''
        ## Maximization Step
        Assuming that all the posterior probability values or weights
        #wij = P(Ci|xj) are known, the maximization step, computes re-estimating μi, sigma2 and P(Ci).
        '''
        [newMean,newCoVar,newPOfCi] = em.emMaximization(trainData,Wij,mean,coVar,pOfCi)
        #print("\n")
        #print(newMean)
        #print(newPOfCi)


        # Find sum of Euclidian distance between old and new mean
        epsilon = 0
        for i in range(n):
            epsilon = epsilon + em.findEuclideanDist(newMean[i,:],mean[i,:])
        #print("Old Mean: " + str(mean))
        #print("New Mean: " + str(newMean))

        #print("Iteration No: " + str(counter) + ". Current convergence error: " + str(dist))

        mean = newMean
        coVar = newCoVar
        pOfCi = newPOfCi

    # Output printing for submit to canvas

    # a: The final mean for each cluster
    sortId = em.normSortedMean(mean)
    sortedMean = mean[sortId,:]
    print("Mean:")
    for i in mean:
        print(i,end=",")
    print("\n")
    #print(np.reshape(sortedMean,len(sortedMean[0,:])*len(sortedMean[:,0])))

    '''
    # b: The final covariance matrix for each cluster
    print("Covariance Matrices: ")
    for i in range(len(sortId)):
        print(coVar[i])
        print("\n")
    
    # c: Iteration count required to converge
    print("Iteration Count=" + str(counter))
    print("\n")
    #print(Wij)
    #print(np.argmax(Wij,axis = 1))
    '''
    
    # D: Print Cluster Membership
    print("Cluster Membership:")
    clusterAssign = np.argmax(Wij,axis = 1)
    sz = np.zeros(n)
    ix = 0
    for i in sortId:
        pointId = (clusterAssign==i)
        clusterPt = trainData[pointId]
        sz[ix] = int(len(clusterPt))
        for j in trainData[pointId]:
            print(j, end=",")
        ix = ix+1
        print("\n")

    # E: Final size of each cluster.
    print("Size",end=":")
    for i in sz:
        print(int(i),end=" ")
    print("\n")


    # Calculate Purity to check the correctness of the classification 
    trueClusterNames = np.unique(trueClassification)
    trueClusterSize = len(trueClusterNames)
    trueClusterAssign = np.zeros(len(trueClassification))
    j = 0
    for i in trueClusterNames:
        trueClusterAssign[trueClassification == i] = j
        j = j+1

    purityVal = em.calcPurity(clusterAssign,trueClusterAssign)
    print("Purity:" + str(purityVal))

# Additional Comments:
# The correctness of the algorithm can also be checked using python inbuild EM Clustering Algorithm and test for verify equal