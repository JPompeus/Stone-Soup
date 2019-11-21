# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..predictor import Predictor
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..types.state import TaggedWeightedGaussianState
from ..types.track import Track
from ..updater import Updater


class GaussianMixtureHypothesiser(Hypothesiser):
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
        default=False,
        doc="Flag to order the :class:`MultipleHypothesis` list by detection or component")  
    prob_survival = Property(
        Probability,
        default=1,
        doc="Probability of a component surviving until the next timestep")    
    birth_component = Property(
        TaggedWeightedGaussianState,
        default=None,
        doc="""The birth component. The weight is equal to the mean of the
        expected number of births per timestep (Poission distributed)""")
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
            Time of the detections/predicted state

        Returns
        -------
        list of :class:`MultipleHypothesis`
            Each MultipleHypothesis in the list contains SingleHypotheses
            pertaining to the same Gaussian component unless order_by_detection is
            true, then they pertain to the same Detection.
        """

        hypotheses = list()
        for component in predict_state:
            # Get hypotheses for that component for all measurements
            component.weight *= self.prob_survival
            component_hypotheses = self.hypothesiser.hypothesise(component, detections, timestamp)
            # Create Multiple Hypothesis and add to list
            if len(component_hypotheses) > 0:
                hypotheses.append(MultipleHypothesis(component_hypotheses))

        # Reorder list of MultipleHypothesis so that they are ordered by detection, not component
        if self.order_by_detection:
            single_hypothesis_list = list()
            # Retrieve all single hypotheses
            for multiple_hypothesis in hypotheses:
                for single_hypothesis in multiple_hypothesis:
                    single_hypothesis_list.append(single_hypothesis)
            
            reordered_hypotheses = list()
            # Get miss detected components
            miss_detections_hypothesis = MultipleHypothesis([x for x in single_hypothesis_list if isinstance(x.measurement, MissedDetection)])
            for detection in detections:
                 # Create multiple hypothesis per detection
                indices \
                    = [x for x in range(len(single_hypothesis_list)) if single_hypothesis_list[x].measurement == detection]
                detection_multiple_hypothesis = MultipleHypothesis(list(map(single_hypothesis_list.__getitem__, indices)))
                # Add to new list
                reordered_hypotheses.append(detection_multiple_hypothesis)
            # Add miss detected hypothesis to end
            reordered_hypotheses.append(miss_detections_hypothesis)
            # Assign reordered list to original list
            hypotheses = reordered_hypotheses

        return hypotheses
