# -*- coding: utf-8 -*-

"""

Created on Tue Aug 24 11:56:26 2021



@author: matoujak

"""



# -*- coding: utf-8 -*-

"""

  Standard/DWC Point-Mass Filter with moving grid. The grid is based on UKF

  predictive meand and var. Works for arbitrary number of dimensions

  author: Jakub Matousek



References:

PMF:

J. Kralovec and M. Simandl (2004):

Filtering, prediction and smoothing with poin-mass aproach

IFAC Automatic Control in Aerospace, p. 375 - 380, Saint-Petersburg, Russia.





DWC:

J. Dunik, O. Straka, J. Matousek and M. Brandner,

"Accurate Density-Weighted Convolution for Point-Mass Filter and Predictor,"

in IEEE Transactions on Aerospace and Electronic Systems,

doi: 10.1109/TAES.2021.3079568.



J. Duni­k, O. Straka and J. Matousek,

"Reliable Convolution in Point-Mass Filter for a Class of Nonlinear Models,"

2020 IEEE 23rd International Conference on Information Fusion (FUSION),

doi: 10.23919/FUSION45008.2020.9190218.



FFT PMF (not present in the python script):



Matousek, J., Dunik, J., Brandner, M., Park, C. G., & Choe, Y. (2023).

Efficient point mass predictor for continuous and discrete models with

linear dynamics. arXiv preprint arXiv:2302.13827. 

(IFAC WC 2023 waiting for publication)



Matoušek, J., Duník, J., & Brandner, M. (2023). 

Design of Efficient Point-Mass Filter with Illustration in 

Terrain Aided Navigation. arXiv preprint arXiv:2303.05100.

(FUSION 2023 waiting for publication)



(Nonlinear dynamics - not included in this code)

J. Duník, J. Matoušek and O. Straka, 

"Design of Efficient Point-Mass Filter for Linear and Nonlinear Dynamic Models,"

in IEEE Control Systems Letters, vol. 7, pp. 2005-2010, 2023,

doi: 10.1109/LCSYS.2023.3283555.



@author: matoujak@kky.zcu.cz

"""



import time

import numpy as np

import numpy.matlib

from numpy.linalg import inv

import matplotlib.pyplot as plt

from functions import ukfUpdate,gridCreation,pmfMeas,pmfUpdateSTD,pmfUpdateDWC,pmfUpdateDiagDWC, gridCreationFFT, measPdfPrepFFT, pmfUpdateFFT



MC = 100

RMSE = np.zeros((MC,1))



for mc in range(MC):

    # Parameters and system simulation

    

    # System parameters

    nx = 2 # state dimension

    nz = 1 # measurement dimension

    kf = 100 # final time

    

    Q = np.array([[0.1, 0.05], [0.05, 0.1]]) # system noise

    invQ = inv(Q)

    

    R = np.atleast_2d(0.025)

    #R = np.array([[0.1, 0.05], [0.05, 0.1]]) # system noise



    invR = inv(R)

    

    # PMF parameters

    pred = 'FFT' # 'STD','DWC', 'FFT'

    Npa = 21 # number of points per axis, for FFT must be ODD!!!!

    N = np.power(Npa,nx) # number of points - total

    sFactor = 4 # scaling factor (number of sigmas covered by the grid)

    kappa = 0 # UKF parameter

    

    

    # Initial condition - Gaussian

    meanX0 = np.array([20, 5]) # mean value

    

    varX0 = np.array([[0.1, 0], [0, 0.1]]) # variance

    

    x = np.zeros((nx,kf))

    x[:,0] = np.random.multivariate_normal(meanX0, varX0, 1) #initial estimate

    

    

    

    ffunct = lambda x,w,k: np.array([[0.9, 0], [0, 1]])@x + w # state equation

    hfunct = lambda x,v,k: np.arctan((x[1,:]-np.sin(k))/(x[0,:]-np.cos(k))) + v # measurement equation

    #hfunct = lambda x,v,k: x + v # measurement equation

    

    

    

    F = np.array([[0.9, 0], [0, 1]])

    F_lin = lambda x: np.array([[0.9, 0], [0, 1]]) # F, for dwc, can be linearized from nonlinear eq

    

    

    # System simulation

    w = np.random.multivariate_normal(np.zeros(nx), Q, kf-1).T # system noise

    

    v = np.random.multivariate_normal(np.zeros(nz), R, kf).T # measurement noise

    

    for i in range(kf-1):

        # trajectory generation

        x[:,i+1] = ffunct(x[:,i:i+1],w[:,i:i+1],i+1).T 

    # measurement generation

    z = hfunct(x,v,np.arange(kf));

    

    ## PMF init

    

    start_time = time.time()

    

    # Initial grid

    predGridDelta = np.zeros((nx,kf+1))

    predMean = np.zeros((nx,kf+1))

    GridDelta = np.zeros((nx,kf+1))

    

    if pred == 'STD' or pred == 'DWC':

        [predGrid, predGridDelta[:,0]] = gridCreation(np.vstack(meanX0),varX0,sFactor,nx,Npa);

    else:

        [predGrid, gridDimOld, predGridDelta[:,0]] = gridCreationFFT(np.vstack(meanX0),varX0,sFactor,nx,Npa);

    

    

    meanX0 = np.vstack(meanX0)

    pom = predGrid-np.matlib.repmat(meanX0,1,N)

    denominator = np.sqrt((2*np.pi)**nx)*np.linalg.det(varX0)

    pompom = np.sum(-0.5*np.multiply(pom.T@inv(varX0),pom.T),1) #elementwise multiplication

    pomexp = np.exp(pompom)

    predDensityProb = pomexp/denominator # Adding probabilities to points

    

    measMean = np.zeros((nx,kf))

    predMean[:,0] = meanX0.T

    measVar = np.zeros((kf,len(varX0),len(varX0))) 

    

    

    # Auxiliary variables 

    predDenDenomV = np.sqrt((2*np.pi)**nx*np.linalg.det(R)) #Denominator for convolution in measurement step

    predDenDenomW = np.sqrt((2*np.pi)**nx*np.linalg.det(Q)) #Denominator for convolution in predictive step

    

    

    for k in range(kf): 

        

        # plt.figure()

        # plt.plot(predDensityProb.flatten())

    

        ## Measurement update

        measPdf = pmfMeas(predGrid,nz,k,z[:,k],invR,predDenDenomV,predDensityProb,predGridDelta[:,k],hfunct)

        

        # Mean and variance

        measMean[:,k] =  np.hstack(predGrid @ measPdf * np.prod(predGridDelta[:,k]))  # Measurement update mean

    

        

        covariance = np.zeros((nx,nx))

        xMinMu = predGrid -  np.matlib.repmat(measMean[:,k],np.size(predGrid,1),1).T

        

        for ind in range(0,N):

            covariance = covariance + np.outer(measPdf[ind] * xMinMu[:,ind] , xMinMu[:,ind])    

        measVar[k,:,:] = covariance * np.prod(predGridDelta[:,k]); # Measurement update covariance    

        

        measGrid = predGrid;    

            

        ## Time update 

        # UKF prediction for grid placement

       

        [xp_aux,Pp_aux] = ukfUpdate(measVar[k,:,:],nx,kappa,measMean[:,k],ffunct,k,Q)

    

    

        [predGrid, predGridDelta[:,k+1]] = gridCreation(xp_aux,Pp_aux,sFactor,nx,Npa)

        # Predictive density

        predDensityProb = np.zeros((N,N))

        

        if pred == 'STD': # Standard PMF       

            predDensityProb = pmfUpdateSTD(measGrid,measPdf,predGridDelta,ffunct,predGrid,nx,k,invQ,predDenDenomW,N)

        elif pred == 'DWC': #DWC for diag Q and diag F fow now just for 2D

            invF = inv(F_lin(measMean[:,k]))

            Qa = invF*Q*invF.T

            cNorm = 1/np.sqrt(((2*np.pi)**nx)*np.linalg.det(Q))

            cNorm2 = np.sqrt(((2*np.pi)**nx)*np.linalg.det(Qa))

            cnormHere = cNorm*cNorm2

            invQa = inv(Qa)

            if (np.diag(np.diag(Q)) == Q).all() and  (np.diag(np.diag(F_lin(measMean[:,k]))) == F_lin(measMean[:,k-1])).all():

                fINV = 1/(np.prod(np.diag(F_lin(measMean[:,k]))))

                s2 = np.sqrt(2)

                normDiagDWC = fINV*np.prod(np.diag(np.sqrt((2*np.pi)**nx*Q))) # normalization constant

                predDensityProb = pmfUpdateDiagDWC(measGrid,N,predGridDelta,F_lin(predGrid, measMean[:,k]),predGrid,Q,s2,measPdf,normDiagDWC,k)        

            else: # General DWC

                predDensityProb = pmfUpdateDWC(invF,predGrid,measGrid,predGridDelta,Qa,cnormHere,measPdf,N,k)

        elif pred == 'FFT':

            # Measurement pdf interpolation to compensate forced grid

            # movement

            [measPdf,gridDimOld,GridDelta[:,k],measGridNew] = measPdfPrepFFT(measPdf,gridDimOld,xp_aux,Pp_aux,F,sFactor,nx,Npa,k);

            # Time Update

            [predDensityProb,predGrid,predGridDelta] = pmfUpdateFFT(F,measPdf,measGridNew,GridDelta,k,Npa,invQ,predDenDenomW,nx);

         

    end_time = time.time()

    

    execution_time = (end_time - start_time)/kf 

    

    RMSE[mc] = np.sqrt(np.mean(np.power(x-measMean,2)))

    

    

    

print(execution_time)             

## Plot

plt.figure()

plt.plot(x[0,:],'b')

plt.plot(measMean[0,:],'-.b')

plt.plot(x[1,:],'r')

plt.plot(measMean[1,:],'-.r')

print('RMSE:',np.mean(RMSE))





  

    



















