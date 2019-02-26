# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..reader import GroundTruthReader
from ..types.detection import TrueDetection, Clutter
from ..types.groundtruth import GroundTruthPath, GroundTruthState
from ..types.numeric import Probability
from ..types.state import GaussianState, State
from .base import DetectionSimulator, GroundTruthSimulator
from stonesoup.buffered_generator import BufferedGenerator


class SingleTargetGroundTruthSimulator(GroundTruthSimulator):
    """Target simulator that produces a single target"""
    transition_model = Property(
        TransitionModel, doc="Transition Model used as propagator for track.")
    initial_state = Property(
        State,
        doc="Initial state to use to generate ground truth")
    timestep = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=1),
        doc="Time step between each state. Default one second.")
    number_steps = Property(
        int, default=100, doc="Number of time steps to run for")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        time = self.initial_state.timestamp or datetime.datetime.now()

        gttrack = GroundTruthPath([
            GroundTruthState(self.initial_state.state_vector, timestamp=time)])
        yield time, {gttrack}

        for _ in range(self.number_steps - 1):
            time += self.timestep
            # Move track forward
            trans_state_vector = self.transition_model.function(
                gttrack[-1].state_vector,
                time_interval=self.timestep)
            gttrack.append(GroundTruthState(
                trans_state_vector, timestamp=time))
            yield time, {gttrack}


class MultiTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the birth rate
    and death probability."""
    initial_state = Property(
        GaussianState,
        doc="Initial state to use to generate states")
    birth_rate = Property(
        float, default=1.0, doc="Rate at which tracks are born. Expected "
        "number of occurrences (λ) in Poisson distribution. Default 1.0.")
    death_probability = Property(
        Probability, default=0.1,
        doc="Probability of track dying in each time step. Default 0.1.")

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        groundtruth_paths = set()
        time = self.initial_state.timestamp or datetime.datetime.now()

        for _ in range(self.number_steps):
            # Random drop tracks
            groundtruth_paths.difference_update(
                gttrack
                for gttrack in groundtruth_paths.copy()
                if np.random.rand() <= self.death_probability)

            # Move tracks forward
            for gttrack in groundtruth_paths.copy():
                trans_state_vector = self.transition_model.function(
                    gttrack[-1].state_vector,
                    time_interval=self.timestep)
                gttrack.append(GroundTruthState(
                    trans_state_vector, timestamp=time))
            # Random create
            for _ in range(np.random.poisson(self.birth_rate)):
                gttrack = GroundTruthPath()
                gttrack.append(GroundTruthState(
                    self.initial_state.state_vector +
                    self.initial_state.covar @
                    np.random.randn(self.initial_state.ndim, 1),
                    timestamp=time))
                groundtruth_paths.add(gttrack)

            yield time, groundtruth_paths
            time += self.timestep


class SimpleDetectionSimulator(DetectionSimulator):
    """A simple detection simulator.

    Parameters
    ----------
    groundtruth : GroundTruthReader
        Source of ground truth tracks used to generate detections for.
    measurement_model : MeasurementModel
        Measurement model used in generating detections.
    """
    groundtruth = Property(GroundTruthReader)
    measurement_model = Property(MeasurementModel)
    meas_range = Property(np.ndarray)
    detection_probability = Property(Probability, default=0.9)
    clutter_rate = Property(float, default=2.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_detections = set()
        self.clutter_detections = set()

    @property
    def clutter_spatial_density(self):
        """returns the clutter spatial density of the measurement space - num
        clutter detections per unit volume per timestep"""
        return self.clutter_rate/np.prod(np.diff(self.meas_range))

    def in_state_space(self, detection):
        """
        Checks if a measurement is in the state space
        """
        is_valid = True
        for dim in range(self.meas_range.ndim):
            if not self.meas_range[dim][0] <= detection.state_vector[dim] \
                                            <= self.meas_range[dim][-1]:
                is_valid = False
        return is_valid

    def detections_gen(self):
        for time, tracks in self.groundtruth:
            self.real_detections.clear()
            self.clutter_detections.clear()
            self.number_of_active_targets = 0
            for track in tracks:
                detection = Detection(
                    self.measurement_model.function(
                        track[-1].state_vector),
                    timestamp=track[-1].timestamp)
                detection.clutter = False
                if self.in_state_space(detection):
                    self.number_of_active_targets += 1
                    if np.random.rand() < self.detection_probability:
                        self.real_detections.add(detection)

            # generate clutter
            for _ in range(np.random.poisson(self.clutter_rate)):
                detection = Clutter(
                    np.random.rand(self.measurement_model.ndim_meas, 1) *
                    np.diff(self.meas_range) + self.meas_range[:, :1],
                    timestamp=time)
                if self.in_state_space(detection):
                    self.clutter_detections.add(detection)

            yield time, self.real_detections | self.clutter_detections
