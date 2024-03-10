#!/usr/bin/env python

Comparing Multiple Trackers On Manoeuvring Targets
This example shows how multiple trackers can be compared against each other using the same set of detections.

This notebook creates groundtruth with manoeuvring motion, generates detections using a sensor, and attempts to track the groundtruth using the Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), the Particle Filter (PF), and the Extended Sliding Innovation Filter (ESIF). Each of these trackers assumes a constant velocity transition model. The trackers are compared against each other using distance-error metrics, with the capability of displaying other metrics for the user to explore. The aim of this example is to display the effectiveness of different trackers when tasked with following a manoeuvring target with non-linear detections.

Layout
The layout of this example is as follows:

The ground truth is created using multiple transition models
The non-linear detections are generated once per time step using a bearing-range sensor
Each tracker is initialised and run on the detections
The results are plotted, and tracking metrics displayed for the user to explore
1) Create Groundtruth
Firstly, we initialise the ground truth states:

import numpy as np
import datetime
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import State, GaussianState

start_time = datetime.datetime.now()
np.random.seed(2)
initial_state_mean = StateVector([[0], [1], [0], [1]])
initial_state_covariance = CovarianceMatrix(np.diag([2, 0.5, 2, 0.5]))
timestep_size = datetime.timedelta(seconds=1)
number_steps = 50
initial_state = GaussianState(initial_state_mean, initial_state_covariance)
Next, we initialise the transition models used to generate the ground truth. Here, we say that the targets will mostly go straight ahead with a constant velocity, but will sometimes turn left or right. This is implemented using the :class:~.SwitchMultiTargetGroundTruthSimulator.

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

# initialise the transition models the ground truth can use
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])
Now we have initialised everything, we can generate the ground truth. Here we use a Single Target simulator.

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState
# generate truths
ground_truth_gen = SingleTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_steps,
)
2) Generate detections using a bearing-range sensor
The next step is to create a sensor and use it to generate detections from the targets. The sensor we use in this example is a radar with imperfect measurements in bearing-range space.

First we initialise the radar:

from stonesoup.sensor.radar import RadarBearingRange

# Create the sensor
sensor = RadarBearingRange(
    ndim_state=4,
    position_mapping=[0, 2],  # Detecting x and y
    noise_covar=np.diag([np.radians(1), 1]),  # Radar doesn't take perfect measurements 0.2, 0.2
    clutter_model=None,  # Can add clutter model in future if desired
)
Now we place the sensor into the simulation:

from stonesoup.platform import FixedPlatform
platform = FixedPlatform(State(StateVector([40, 0, -20, 0])), position_mapping=[0, 2], # 40
                         sensors=[sensor])
Now we run the sensor and create detections:

from itertools import tee
from stonesoup.simulator.platform import PlatformDetectionSimulator

detector = PlatformDetectionSimulator(ground_truth_gen, platforms=[platform])
detector, *detectors = tee(detector, 6)
# Enables multiple trackers to run on the same detections
We put the detections and ground truths into sets so that we can plot them:

detections = set()
ground_truth = set()

for time, dets in detector:
    detections |= dets
    ground_truth |= ground_truth_gen.groundtruth_paths
And now we plot the ground truth and detections:

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(ground_truth, [0, 2])
plotter.plot_measurements(detections, [0, 2])
plotter.plot_sensors(sensor)
plotter.fig
3) Initialise and run each tracker on the detections
With the detections now generated, our focus turns to creating and running the trackers. This section of the notebook is quite long because each tracker requires an initiator, deleter, detector, data associator, and updater. However, all of these things are standard stonesoup building blocks.

Firstly, we approximate the transition model of the target. Here we assume a constant velocity model, which will be wrong due to the fact that we designed the targets to sometimes turn left or right. We do this to test how effectively each tracking algorithm can perform against target behaviour that doesn't move exactly as predicted.

transition_model_estimate = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5),
                                                                   ConstantVelocity(0.5)])
# Tracking algorithm incorrectly estimates the type of path the truth takes
Next, we initialise the predictors, updaters, hypothesisers, data associators, and deleter. The particle filter requires a resampler as part of its updater. Note that the ESIF is a slight extension of the EKF and uses an EKF predictor.

from stonesoup.predictor.kalman import ExtendedKalmanPredictor, UnscentedKalmanPredictor, SIFKalmanPredictor
from stonesoup.predictor.particle import ParticlePredictor
# introduce the predictors
predictor_EKF = ExtendedKalmanPredictor(transition_model_estimate)
predictor_UKF = UnscentedKalmanPredictor(transition_model_estimate)
predictor_SIF = SIFKalmanPredictor(transition_model_estimate)
predictor_PF = ParticlePredictor(transition_model_estimate)

# ######################################################################

from stonesoup.resampler.particle import ESSResampler
resampler = ESSResampler()

# ######################################################################

from stonesoup.updater.kalman import ExtendedKalmanUpdater, UnscentedKalmanUpdater, SIFKalmanUpdater
from stonesoup.updater.particle import ParticleUpdater
# introduce the updaters

updater_EKF = ExtendedKalmanUpdater(sensor)
updater_UKF = UnscentedKalmanUpdater(sensor)
updater_SIF = SIFKalmanUpdater(sensor)
updater_PF = ParticleUpdater(measurement_model=None, resampler=resampler)

# ######################################################################

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
# introduce the hypothesisers

hypothesiser_EKF = DistanceHypothesiser(predictor_EKF, updater_EKF,
                                        measure=Mahalanobis(), missed_distance=4)
hypothesiser_UKF = DistanceHypothesiser(predictor_UKF, updater_UKF,
                                        measure=Mahalanobis(), missed_distance=4)
hypothesiser_SIF = DistanceHypothesiser(predictor_SIF, updater_SIF,
                                      measure=Mahalanobis(), missed_distance=4)
hypothesiser_PF = DistanceHypothesiser(predictor_PF, updater_PF,
                                      measure=Mahalanobis(), missed_distance=4)

# ######################################################################

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
# introduce the data associators

data_associator_EKF = GNNWith2DAssignment(hypothesiser_EKF)
data_associator_UKF = GNNWith2DAssignment(hypothesiser_UKF)
data_associator_SIF = GNNWith2DAssignment(hypothesiser_SIF)
data_associator_PF = GNNWith2DAssignment(hypothesiser_PF)

# ######################################################################

from stonesoup.deleter.time import UpdateTimeDeleter
# create a deleter
deleter = UpdateTimeDeleter(datetime.timedelta(seconds=5), delete_last_pred=True)
Now we introduce the initial predictors which will be used in the data associator in the track initiators:

init_transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.05), ConstantVelocity(0.05)))
init_predictor_EKF = ExtendedKalmanPredictor(init_transition_model)
init_predictor_UKF = UnscentedKalmanPredictor(init_transition_model)
init_predictor_SIF = SIFKalmanPredictor(init_transition_model)
init_predictor_PF = ParticlePredictor(init_transition_model)
The final step before running the trackers is to create the initiators:

from stonesoup.initiator.simple import MultiMeasurementInitiator

initiator_EKF = MultiMeasurementInitiator(
    GaussianState(
        np.array([[20], [0], [10], [0]]),  # Prior State
        np.diag([1, 1, 1, 1])),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(init_predictor_EKF, updater_EKF, Mahalanobis(), missed_distance=5)),
    updater=updater_EKF,
    min_points=2
)

# ######################################################################

initiator_UKF = MultiMeasurementInitiator(
    GaussianState(
        np.array([[20], [0], [10], [0]]),  # Prior State
        np.diag([1, 1, 1, 1])),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(init_predictor_UKF, updater_UKF, Mahalanobis(), missed_distance=5)),
    updater=updater_UKF,
    min_points=2
)

############################################################################

initiator_SIF = MultiMeasurementInitiator(
    GaussianState(
        np.array([[20], [0], [10], [0]]),  # Prior State
        np.diag([1, 1, 1, 1])),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(init_predictor_SIF, updater_SIF, Mahalanobis(), missed_distance=5)),
    updater=updater_SIF,
    min_points=2
)
The initiator for the particle filter works differently, so is shown below for clarity:

from stonesoup.initiator.simple import GaussianParticleInitiator
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import SimpleMeasurementInitiator

prior_state = GaussianState(
    StateVector([20, 0, 10, 0]),
    np.diag([1, 1, 1, 1]) ** 2)

initiator_Part = SimpleMeasurementInitiator(prior_state, measurement_model=None,
                                            skip_non_reversible=True)
number_particles = 5000 # 2500
initiator_PF = GaussianParticleInitiator(number_particles=number_particles,
                                        initiator=initiator_Part,
                                        use_fixed_covar=False)
Now we run the trackers and store the tracks in sets for plotting:

from stonesoup.tracker.simple import MultiTargetTracker

kalman_tracker_EKF = MultiTargetTracker(  # Runs the tracker
    initiator=initiator_EKF,
    deleter=deleter,
    detector=detectors[0],
    data_associator=data_associator_EKF,
    updater=updater_EKF,
)

tracks_EKF = set()
for step, (time, current_tracks) in enumerate(kalman_tracker_EKF, 1):
    tracks_EKF.update(current_tracks)

    # Stores the tracks in a set for plotting

# #######################################################################

kalman_tracker_UKF = MultiTargetTracker(
    initiator=initiator_UKF,
    deleter=deleter,
    detector=detectors[1],
    data_associator=data_associator_UKF,
    updater=updater_UKF,
)

tracks_UKF = set()
for step, (time, current_tracks) in enumerate(kalman_tracker_UKF, 1):
    tracks_UKF.update(current_tracks)

# ##########################################################################

kalman_tracker_SIF = MultiTargetTracker(
    initiator=initiator_SIF,
    deleter=deleter,
    detector=detectors[2],
    data_associator=data_associator_SIF,
    updater=updater_SIF,
)

tracks_SIF = set()
for step, (time, current_tracks) in enumerate(kalman_tracker_SIF, 1):
    tracks_SIF.update(current_tracks)
# ##########################################################################

tracker_PF = MultiTargetTracker(
    initiator=initiator_PF,
    deleter=deleter,
    detector=detectors[3],
    data_associator=data_associator_PF,
    updater=updater_PF,
)

tracks_PF = set()
for step, (time, current_tracks) in enumerate(tracker_PF, 1):
    tracks_PF.update(current_tracks)
Finally, we plot the results:

plotter.plot_tracks(tracks_EKF, [0, 2], track_label="EKF", line=dict(color="orange"),
                    uncertainty=False)
plotter.plot_tracks(tracks_UKF, [0, 2], track_label="UKF", line=dict(color="blue"),
                    uncertainty=False)
plotter.plot_tracks(tracks_SIF, [0, 2], track_label="SIF", line=dict(color="red"),
                    uncertainty=False)
plotter.plot_tracks(tracks_PF, [0, 2], track_label="PF", line=dict(color="brown"),
                    uncertainty=False)
plotter.fig
4) Calculate and display metrics to show effectiveness of different tracking algorithms
The final part of this example is to calculate metrics that can determine how well each tracking algorithm followed the target. None will be perfect due to the sensor measurement noise and error in data association where multiple tracks meet, but some will perform better than others.

This section of the example follows code from the metrics example, which is also used in the sensor management tutorials. More complete documentation can be found there.

Firstly, we calculate the Optimal Sub-Pattern Assignment (OSPA) distance at each time step for each tracker. This is a measure of how far the calculated tracks are from the ground truth. We first initialise the metrics before plotting:

from stonesoup.metricgenerator.ospametric import OSPAMetric

ospa_generator = OSPAMetric(c=40, p=1)

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean

siap_generator = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)))

from stonesoup.dataassociator.tracktotrack import TrackToTruth

associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric

uncertainty_generator = SumofCovarianceNormsMetric()
Now we initialise the metric managers for each tracker:

from stonesoup.metricgenerator.manager import SimpleManager

metric_manager_EKF = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                   associator=associator)
metric_manager_EKF.add_data(ground_truth, tracks_EKF)
metrics_EKF = metric_manager_EKF.generate_metrics()

# ##################################################

metric_manager_UKF = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                   associator=associator)
metric_manager_UKF.add_data(ground_truth, tracks_UKF)
metrics_UKF = metric_manager_UKF.generate_metrics()

# ##################################################

metric_manager_SIF = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                  associator=associator)
metric_manager_SIF.add_data(ground_truth, tracks_SIF)
metrics_SIF = metric_manager_SIF.generate_metrics()

# ##################################################

metric_manager_PF = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                  associator=associator)
metric_manager_PF.add_data(ground_truth, tracks_PF)
metrics_PF = metric_manager_PF.generate_metrics()
Now we can plot the OSPA distance for each tracker:

from plotly.subplots import make_subplots

ospa_metric_EKF = metrics_EKF['OSPA distances']
ospa_metric_UKF = metrics_UKF['OSPA distances']
ospa_metric_SIF = metrics_SIF['OSPA distances']
ospa_metric_PF = metrics_PF['OSPA distances']

fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.040,
    subplot_titles=['Tracker distance error from target - lower is better'])
fig.add_scatter(
    x=[i.timestamp for i in ospa_metric_EKF.value],
    y=[i.value for i in ospa_metric_EKF.value],
    name='EKF',
    legendgroup="orange",
    yaxis='y',
    row=1,
    col=1,
    showlegend=True,
    line_color='orange',
)
fig.add_scatter(
    x=[i.timestamp for i in ospa_metric_SIF.value],
    y=[i.value for i in ospa_metric_SIF.value],
    name='SIF',
    legendgroup="red",
    yaxis='y',
    row=1,
    col=1,
    showlegend=True,
    line_color='red',
)
fig.add_scatter(
    x=[i.timestamp for i in ospa_metric_UKF.value],
    y=[i.value for i in ospa_metric_UKF.value],
    name='UKF',
    legendgroup="blue",
    yaxis='y',
    row=1,
    col=1,
    showlegend=True,
    line_color='blue',
)
fig.add_scatter(
    x=[i.timestamp for i in ospa_metric_PF.value],
    y=[i.value for i in ospa_metric_PF.value],
    name='Particle',
    legendgroup="brown",
    yaxis='y',
    row=1,
    col=1,
    showlegend=True,
    line_color='brown',
)
It can be seen that the EKF, UKF, and Particle Filter all behave very similarly, whereas the ESIF has very poor relative performance. A singular performance metric is calculated from this by summing the OSPA value over all timesteps:

# sum up distance error from ground truth over all timestamps
ospa_EKF_total = sum([ospa_metric_EKF.value[i].value for i in range(0,
                                                                    len(ospa_metric_EKF.value))])
ospa_UKF_total = sum([ospa_metric_UKF.value[i].value for i in range(0,
                                                                    len(ospa_metric_UKF.value))])
ospa_SIF_total = sum([ospa_metric_SIF.value[i].value for i in range(0,
                                                                    len(ospa_metric_SIF.value))])
ospa_PF_total = sum([ospa_metric_PF.value[i].value for i in range(0,
                                                                len(ospa_metric_PF.value))])

print("OSPA total value for EKF is ", f'{ospa_EKF_total:.3f}')
print("OSPA total value for UKF is ", f'{ospa_UKF_total:.3f}')
print("OSPA total value for SIF is ", f'{ospa_SIF_total:.3f}')
print("OSPA total value for PF is ", f'{ospa_PF_total:.3f}')
OSPA total value for EKF is  164.953
OSPA total value for UKF is  165.525
OSPA total value for SIF is  165.544
OSPA total value for PF is  164.005
Monte Carlo Generation

monte_carlo_iterations = 50

for i in range(monte_carlo_iterations):
    # Generate Ground Truth
    detections = set()
    ground_truth = set()
    for time, dets in detector:
        detections |= dets
        ground_truth |= ground_truth_gen.groundtruth_paths
    # Run Tracks
    tracks_UKF = set()
    for step, (time, current_tracks) in enumerate(kalman_tracker_UKF, 1):
        tracks_UKF.update(current_tracks)
    tracks_PF = set()
    for step, (time, current_tracks) in enumerate(tracker_PF, 1):
        tracks_PF.update(current_tracks)
 