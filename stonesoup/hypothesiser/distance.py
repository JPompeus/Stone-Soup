# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from .. import measures as Measure
from ..types import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..predictor import Predictor
from ..types.multihypothesis import \
    MultipleHypothesis
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.detection import MissedDetection
from ..updater import Updater


class DistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`Measure` class.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    measure = Property(
        Measure,
        doc="Measure class used to calculate the distance between two states.")
    missed_distance = Property(
        float,
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")
    include_all = Property(
        bool,
        default=False,
        doc="If `True`, hypotheses beyond missed distance will be returned. "
            "Default `False`")

    def hypothesise(self, track, detections, timestamp):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a
        MultipleHypothesis object with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated distance measure..

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleDistanceHypothesis` objects

        """
        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        measurement_prediction = self.updater.predict_measurement(
            prediction)

        # Missed detection hypothesis with distance as 'missed_distance'
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                self.missed_distance,
                measurement_prediction))

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction
            prediction = self.predictor.predict(
                track.state, timestamp=detection.timestamp)

            # Compute measurement prediction and distance measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            distance = self.measure(measurement_prediction, detection)

            if self.include_all or distance < self.missed_distance:
                # True detection hypothesis
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        detection,
                        distance,
                        measurement_prediction))

        return sorted(hypotheses, reverse=True)


class GaussianMixtureHypothesier(Hypothesiser):
    """Gaussian Mixture Prediction Hypothesiser based on an underlying Hypothesiser

    Generates a list of :class:`MultipleHypothesis`, where each MultipleHypothesis in the list contains SingleHypotheses
            pertaining to an individual component-detection hypothesis
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    hypothesiser = Property(
        Hypothesiser,
        doc="Underlying Hypothesiser used to generate component-detection hypotheses")   
    order_by_detection = Property(
        bool,
        default=false
        doc="Flag to order the :class:`MultipleHypothesis` list by detection or component")    

    def hypothesise(self, predict_state, detections, timestamp):
        """Form hypotheses for associations between Detections and Gaussian
        Mixture components.

        Parameters
        ----------
        predict_state : :class:`list`
            List of :class:`WeightedGaussianState` components
            representing the predicted state of the space
        detections : list of :class:`Detection`
            Retrieved measurements
        timestamp : datetime
            time of the detections/predicted state

        Returns
        -------
        list of :class:`MultipleHypothesis`
            each MultipleHypothesis in the list contains SingleHypotheses
            pertaining to the same Detection
        """

        hypotheses = list()
        reordered_hypotheses = list()
        for component in predict_state:
            component_hypotheses = self.hypothesiser.hypothesise(component, detections, timestamp)
            # Reorder list of MultipleHypothesis so that they are ordered by detection, not component
            if self.order_by_detection:
                for detection in detections:

            if len(component_hypotheses) > 0:
                hypotheses.append(MultipleHypothesis(component_hypotheses))
                            
        return hypotheses
