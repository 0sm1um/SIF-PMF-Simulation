#!/usr/bin/env python

"""
==============================================
3 - Non-linear models: unscented Kalman filter
==============================================
"""
# Some general imports and initialise time
import numpy as np

from datetime import datetime, timedelta


from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity


from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.detection import Detection

from stonesoup.updater.kalman import SIFKalmanUpdater

from stonesoup.predictor.kalman import SIFKalmanPredictor
from stonesoup.predictor.kalman import UnscentedKalmanPredictor

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from stonesoup.updater.kalman import UnscentedKalmanUpdater

from stonesoup.smoother.kalman import  UnscentedKalmanSmoother, SIFKalmanSmoother

# np.random.seed(42) 

MC = 30

rmseSIFmc = np.zeros((4,MC))
rmseUKFmc = np.zeros((4,MC))

rmseSIFmcS = np.zeros((4,MC))
rmseUKFmcS = np.zeros((4,MC))

start_time = datetime.now().replace(microsecond=0)

# %%
for mc in range(MC):


    # %%
    # Create ground truth
    # ^^^^^^^^^^^^^^^^^^^
    #

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([50, 1, 1, 1], timestamp=timesteps[0])])


    for k in range(1, 21):
        timesteps.append(start_time+timedelta(seconds=k))
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))
    
    
    state = np.zeros((4,21)) 
    for ind in range(21):
        state[:,ind] = truth[ind].state_vector.ravel()
    # %%
    # Simulate the measurement
    # ^^^^^^^^^^^^^^^^^^^^^^^^
    #
    
    # Sensor position
    sensor_x = 50
    sensor_y = 0
    
    # Make noisy measurement (with bearing variance = 0.2 degrees).
    measurement_model = CartesianToBearingRange(ndim_state=4,
                                                mapping=(0, 2),
                                                noise_covar=np.diag([np.radians(0.2), 1])*0.1,
                                                translation_offset=np.array([[sensor_x], [sensor_y]]))
    
    # %%
    
    
    # Make sensor that produces the noisy measurements.
    measurements = []
    z = np.array([])
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        z = np.append(z, measurement)
        measurements.append(Detection(measurement, timestamp=state.timestamp,
                                      measurement_model=measurement_model))
    
    kwargs = {"alpha": 0.5, "beta": 2, "kappa": 1}
    
    predictor = UnscentedKalmanPredictor(transition_model, **kwargs)
    
    unscented_updater = UnscentedKalmanUpdater(measurement_model, **kwargs)  # Keep alpha as default = 0.5
    
    
    predictorSIF = SIFKalmanPredictor(transition_model)
    
    updaterSIF = SIFKalmanUpdater(measurement_model)
    
    
    smootherUKF = UnscentedKalmanSmoother(transition_model)
    smootherSIF = SIFKalmanSmoother(transition_model)

    # %%
    # Run the Unscented Kalman Filter
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Create a prior
    
    prior = GaussianState([[50], [1], [1], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    priorSIF = GaussianState([[50], [1], [1], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    
    # %%
    # Populate the track
    
    
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)
        post = unscented_updater.update(hypothesis)
        track.append(post)
        prior = track[-1]
        
    trackSIF = Track()
    for measurement in measurements:
        predictionSIF = predictorSIF.predict(priorSIF, timestamp=measurement.timestamp)
        hypothesisSIF = SingleHypothesis(predictionSIF, measurement)
        postSIF = updaterSIF.update(hypothesisSIF)
        trackSIF.append(postSIF)
        priorSIF = trackSIF[-1]
        
        
    sifSmooth = smootherSIF.smooth(trackSIF)
    ukfSmooth = smootherUKF.smooth(track)

    
    
    rmseSIF = np.zeros((4,len(track.states)))
    rmseUKF = np.zeros((4,len(track.states)))
    truthState = np.zeros((4,len(track.states)))
    rmseSIFs = np.zeros((4,len(track.states)))
    rmseUKFs = np.zeros((4,len(track.states)))
    
    for ind in range(0,len(track.states)):
    
        a = track.states[ind].state_vector
        b = truth.states[ind].state_vector
        
        a = a.ravel()
        b = b.ravel()
        
        truthState[:,ind] = b
    
        rmseUKF[:,ind] = np.power((a-b),2)
        
        a = trackSIF.states[ind].state_vector
        a = a.ravel()
    
        rmseSIF[:,ind] = np.power((a-b),2)
        
        
        a = sifSmooth.states[ind].state_vector
        a = a.ravel()
        
        rmseSIFs[:,ind] = np.power((a-b),2)
        
        a = ukfSmooth.states[ind].state_vector
        a = a.ravel()
        
        rmseUKFs[:,ind] = np.power((a-b),2)        
        
        
    rmseSIFmc[:,mc] = np.sqrt(np.mean(rmseSIF,1))
    rmseUKFmc[:,mc] = np.sqrt(np.mean(rmseUKF,1))
    
    rmseSIFmcS[:,mc] = np.sqrt(np.mean(rmseSIFs,1))
    rmseUKFmcS[:,mc] = np.sqrt(np.mean(rmseUKFs,1))
    
    
print("--------SIF--------\n")  
print("Filtering", (np.mean(rmseSIFmc,1)))
print("Smoothed", (np.mean(rmseSIFmcS,1)))
print("--------UKF--------\n")
print("Filtering", (np.mean(rmseUKFmc,1)))
print("Smoothed", (np.mean(rmseUKFmcS,1)))


    
    
