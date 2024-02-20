import datetime
import numpy as np
import matplotlib.pyplot as plt
import datetime

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection

from stonesoup.types.array import StateVector, CovarianceMatrix, StateVectors
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import State, GaussianState, EnsembleState
from stonesoup.types.track import Track

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.predictor.kalman import SIFKalmanPredictor, UnscentedKalmanPredictor

from stonesoup.updater.kalman import SIFKalmanUpdater, UnscentedKalmanUpdater

from stonesoup.smoother.kalman import  UnscentedKalmanSmoother, SIFKalmanSmoother


class simulator():
        
    def __init__(self, transition_model, measurement_model):
        """
        Parameters
        ----------
        transition_model : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        measurement_model : :class:`~.Predictor`
            The Updater to be used.
        """
        self.transition_model= transition_model
        self.measurement_model = measurement_model
    
    def _generate_ground_truth(self, prior, time_span):
        
        time_interval = time_span[1]-time_span[0]
        ground_truth = GroundTruthPath([prior])
        for k in range(1, len(time_span)):
            ground_truth.append(GroundTruthState(
                self.transition_model.function(ground_truth[k-1], noise=True, time_interval=time_interval),
                timestamp=time_span[k-1]))
        return ground_truth
    
    def _simulate_measurements(self, ground_truth):
        #Simulate Measurements
        measurements = []
        for state in ground_truth:
            measurement = self.measurement_model.function(state, noise=True)
            measurements.append(Detection(measurement,
                                          timestamp=state.timestamp,
                                          measurement_model=self.measurement_model))
        return measurements
    
    def simulate_track(self, predictor, updater, initial_state, prior, time_span):
        """
        Parameters
        ----------
        predictor : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        updater : :class:`~.Predictor`
            The Updater to be used.
        ground_truth : :class:`~.GroundTruthPath`
            StateMutableSequence type object used to store ground truth.
        initial_state : :class:`~.State`
            Initial state for the ground truth system. This MUST be a State,
            not a State subclass, like GaussianState or EnsembleState.
        prior : :class:`~.GaussianState` or :class:`~.EnsembleState`
            Initial state prediction of tracking algorithm.

        Returns
        -------
        track : :class:`~.Track`
            The Stone Soup track object which contains the list of updated 
            state predictions made by the tracking algorithm employed.
        """
        
        #Simulate Measurements
        ground_truth = self._generate_ground_truth(initial_state, time_span)
        measurements = self._simulate_measurements(ground_truth)
        
        #Initialize Loop Variables
        track = Track()
        for measurement in measurements:
            prediction = predictor.predict(prior, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
            posterior = updater.update(hypothesis)
            track.append(posterior)
            prior = track[-1]
        return ground_truth, track


class multimodelsimulator():
        
    def __init__(self, transition_models, measurement_models):
        """
        Parameters
        ----------
        transition_model : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        measurement_model : :class:`~.Predictor`
            The Updater to be used.
        """
        self.transition_models = transition_models
        self.measurement_models = measurement_models
    
    def _generate_ground_truth(self, initial_state, time_span):
        
        time_interval = time_span[1]-time_span[0]
        ground_truth = GroundTruthPath([initial_state])
        for k in range(0, len(time_span)-1):
            ground_truth.append(GroundTruthState(
                self.transition_models[k].function(ground_truth[k], noise=True, time_interval=time_interval),
                timestamp=time_span[k]))
        return ground_truth

    def _simulate_measurements(self, ground_truth):
        #Simulate Measurements
        measurements = []
        for k in range(0, len(ground_truth)):
            measurement = self.measurement_models[k].function(ground_truth.states[k], noise=True)
            measurements.append(Detection(measurement,
                                          timestamp=ground_truth.states[k].timestamp,
                                          measurement_model=self.measurement_models[k]))
        return measurements
    
    def _simulate_track(self, predictor, updater, initial_state, prior, time_span):
        """

        Parameters
        ----------
        predictor : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        updater : :class:`~.Predictor`
            The Updater to be used.
        ground_truth : :class:`~.GroundTruthPath`
            StateMutableSequence type object used to store ground truth.
        initial_state : :class:`~.State`
            Initial state for the ground truth system. This MUST be a State,
            not a State subclass, like GaussianState or EnsembleState.
        prior : :class:`~.GaussianState` or :class:`~.EnsembleState`
            Initial state prediction of tracking algorithm.

        Returns
        -------
        track : :class:`~.Track`
            The Stone Soup track object which contains the list of updated 
            state predictions made by the tracking algorithm employed.
        """
        
        #Simulate Measurements
        ground_truth = self._generate_ground_truth(initial_state, time_span)
        measurements = self._simulate_measurements(ground_truth)
        
        #Initialize Loop Variables
        track = Track()
        k=0
        for measurement in measurements:
            predictor.transition_model = self.transition_models[k]
            updater.measurement_model = self.measurement_models[k]
            prediction = predictor.predict(prior, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
            posterior = updater.update(hypothesis)
            track.append(posterior)
            prior = track[-1]
            k=k+1
        return ground_truth, track

def calc_RMSE(ground_truth_list, track_list):
    """This function computes the RMSE of filter with respect to time.
    It accepts lists of ground truth paths, and tracks. It returns an instance
    of `StateVectors` in which columns contain Vectors of the RMSE at a given time"""
    if len(ground_truth_list) != len(track_list):
        print(len(ground_truth_list))
        print(len(track_list))
        return NotImplemented
    residual = np.zeros([ground_truth_list[0].states[0].ndim,len(ground_truth_list[0].states)])
    for instance in range(len(ground_truth_list)):
        ground_truth_states = StateVectors([e.state_vector for e in ground_truth_list[instance].states])
        tracked_states = StateVectors([e.state_vector for e in track_list[instance].states])
        residual = (tracked_states - ground_truth_states)**2 + residual
    RMSE = np.sqrt(residual/len(ground_truth_list))
    return RMSE

def plot_RMSE(RMSE,time_span):
    '''This function accepts an instance of `StateVectors` as an argument, and
    plots each of its root mean square errors as a function of time for each of
    its dimensions.

    Parameters
    ----------
    RMSE : StateVectors
        DESCRIPTION.
    filter_name : string
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure()
    
    x=RMSE[0]
    vx=RMSE[1]
    y=RMSE[2]
    vy=RMSE[3]
    t=time_span

    fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

    ax[0][0].plot(t, x)
    '''
    ax[0][0].title('$RMSE$ of {} versus $time$'.format(filter_name))
    ax[0][0].xlabel('$Time (in s)$')
    ax[0][0].ylabel('$RMSE of x$')
    '''
    ax[0][1].plot(t, vx)
    '''
    ax[1][0].title('$RMSE$ versus $time$')
    ax[1][0].xlabel('$Time (in s)$')
    ax[1][0].ylabel('$RMSE of vx$')
    '''
    ax[1][0].plot(t, y)
    '''
    ax[2][0].xlabel('$Time (in s)$')
    ax[2][0].ylabel('$RMSE of y$')
    '''
    ax[1][1].plot(t, vy)
    '''
    ax[0][1].xlabel('$Time (in s)$')
    ax[0][1].ylabel('$RMSE of vy$')
    '''
    

def plot_position(states,time_span):
    '''This function accepts an instance of `StateVectors` as an argument, and
    plots each of its root mean square errors as a function of time for each of
    its dimensions.

    Parameters
    ----------
    RMSE : StateVectors
        DESCRIPTION.
    filter_name : string
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x = []
    vx=[]
    y=[]
    vy=[]
    for t in states:
        x.append(states.state.state_vector[0])
        vx.append(states.state.state_vector[1])
        y.append(states.state.state_vector[2])
        vy.append(states.state.state_vector[3])
    
    x = np.array([x])
    vx = np.array([vx])
    y = np.array([y])
    vy = np.array([vy])
    
    t=np.array([time_span]).reshape(np.shape(x))

    fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

    ax[0][0].plot(t, x)
    '''
    ax[0][0].title('$RMSE$ of {} versus $time$'.format(filter_name))
    ax[0][0].xlabel('$Time (in s)$')
    ax[0][0].ylabel('$RMSE of x$')
    '''
    ax[0][1].plot(t, vx)
    '''
    ax[1][0].title('$RMSE$ versus $time$')
    ax[1][0].xlabel('$Time (in s)$')
    ax[1][0].ylabel('$RMSE of vx$')
    '''
    ax[1][0].plot(t, y)
    '''
    ax[2][0].xlabel('$Time (in s)$')
    ax[2][0].ylabel('$RMSE of y$')
    '''
    ax[1][1].plot(t, vy)
    '''
    ax[0][1].xlabel('$Time (in s)$')
    ax[0][1].ylabel('$RMSE of vy$')
    '''
    
"""
    HERE IS WHERE THE SIMULATIONS START
"""

i = 30
num_vectors = i*5

"""
    Here, we get our initial variables for simulation. For this, we are just
    using a time span of 60 time instances spaced one second apart.
"""

timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
tMax = 60
dt = 1
tRange = tMax // dt
plot_time_span = np.array([dt*i for i in range(tRange)])

time_span = np.array([timestamp + datetime.timedelta(seconds=dt*i) for i in range(tRange)])

monte_carlo_iterations = 10


"""
Here we instantiate our transition and measurement models. These are the 
same models used in the StoneSoup Kalman Filter examples.
"""

q_x = 0.05
q_y = 0.05
sensor_x = 50  # Placing the sensor off-centre
sensor_y = 0

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), 
                                                          ConstantVelocity(q_y)])
measurement_model = CartesianToBearingRange(
ndim_state=4,
mapping=(0, 2),
noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
# bearing and 1 metre in range
translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
# sensor in cartesian.
)


"""
Here we instantiate the simulator with our transition and measurement model.
This class is capable of generating sets of ground truth points, and simulate
measurements for our recursive filters.
"""

nonlinear_simulator = simulator(transition_model=transition_model,
                      measurement_model=measurement_model)

"""
Finally, before running our monte carlo simulation, we need to instantiate
all the algorithm predictor/updaters we wish to run
"""

SIFpredictor = SIFKalmanPredictor(transition_model)
SIFupdater = SIFKalmanUpdater(measurement_model)

UKFpredictor = SIFKalmanPredictor(transition_model)
UKFupdater = SIFKalmanUpdater(measurement_model)

smootherUKF = UnscentedKalmanSmoother(transition_model)
smootherSIF = SIFKalmanSmoother(transition_model)


"""
Now, we will provide initial states for the ground truth, and the value we 
initialize our algorithms with. It is of course optional to initialize the 
algorithms with the same value as the ground truth, and I would actually 
encourage you not to do this.
"""

initial_ground_truth = GroundTruthState([50, 1, 1, 1], timestamp=time_span[0])

UKFprior = GaussianState(state_vector=StateVector([50, 1, 1, 1]),
              covar = CovarianceMatrix(np.diag(np.array([1.5, 0.5, 1.5, 0.5]))**2),
              timestamp = time_span[0])
SIFprior = GaussianState(state_vector=StateVector([50, 1, 1, 1]),
              covar = CovarianceMatrix(np.diag(np.array([1.5, 0.5, 1.5, 0.5]))**2),
              timestamp = time_span[0])

"""
Finally, we run our Simulations many times. By populating a list with tracks
of simulation instances, we can compute Root Mean Squared by averaging the 
difference between the ground truth and our algorithm's predictions across 
many simulation runs, hence it is a Monte Carlo approach.
"""

SIF_monte_carlo_runs = []
UKF_monte_carlo_runs = []
for i in range(monte_carlo_iterations):
    print(i+1)
    UKF_monte_carlo_runs.append(nonlinear_simulator.simulate_track(predictor = UKFpredictor, 
                                        updater = UKFupdater, 
                                        initial_state = initial_ground_truth,
                                        prior = UKFprior,
                                        time_span=time_span))
    SIF_monte_carlo_runs.append(nonlinear_simulator.simulate_track(predictor = SIFpredictor, 
                                        updater = SIFupdater,
                                        initial_state = initial_ground_truth,
                                        prior = SIFprior,
                                        time_span=time_span))

"""
Now we have three lists of M tracks for each algorithm we wish to evaluate.
To compute RMSE, I am just going to call our purpose made function.

Note, its arguments are for the lists of ground truth paths, and for their
associated simulated tracks. For our variables, the 0th and 1st indicies 
correspond to these.
"""

RMSE_UKF = calc_RMSE(UKF_monte_carlo_runs[0],UKF_monte_carlo_runs[1])
RMSE_SIF = calc_RMSE(SIF_monte_carlo_runs[0],SIF_monte_carlo_runs[1])

RMSE_LIST = [RMSE_UKF, RMSE_SIF]
#RMSE_LIST = [RMSE_EnKF]

"""
Now that we have our RMSE data, it is a matter of plotting it. For the paper,
the authors collaborated remotley, and so the above results were saved by one
author and sent to the other, where they were plotted.

The code below was written to automate the packaging of simulations run with
varying ensemble sizes. Change ensemble size at the start of the script, run
it, and a file is saved with the appropriate name.
"""

for RMSE in RMSE_LIST:
    plot_list = plot_RMSE(RMSE, plot_time_span)
    
