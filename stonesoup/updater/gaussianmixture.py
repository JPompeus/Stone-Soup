# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache
from scipy.stats import multivariate_normal
import uuid

from ..base import Property
from .base import Updater
from .kalman import KalmanUpdater
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate
from ..types.numeric import Probability
from ..types import TaggedWeightedGaussianState



class GaussianMixtureUpdater(Updater):
    r"""
    """
    updater = Property(
        KalmanUpdater,
        default=None,
        doc="Underlying updater used to perform the single target Kalman Update.")

    prob_of_detection = Property(
        Probability,
        default=0.9,
        doc="""The probability that an exisiting target is detected
                at each timestep""")
    clutter_spatial_density = Property(
        float,
        default=1e-10,
        doc="""The clutter intensity at a point in the state point.""")
    

    def update(self, hypotheses):
        """
        Updates the current components in a
        :state:`GaussianMixtureState` by applying the underlying :class:`KalmanUpdater` to each component with the supplied measurements.

        Parameters
        ==========
        hypotheses : list of :class:`MultipleHypothesis`
            Measurements obtained at time :math:`k+1`

        Returns
        =======
        updated_components : :state:`GMPHDTargetTracker`
            GMPHD Tracker with updated components at time :math:`k+1`
        """
        updated_components = []
        # Loop over all measurements
        for i in range(len(hypotheses)-1):
            updated_measurement_components = []
            # Initialise weight sum for measurement to clutter intensity
            weight_sum = self.clutter_spatial_density
            # For every valid single hypothesis, update that component with
            # measurements and calculate new weight
            for j in range(len(hypotheses[i])):
                measurement_prediction = \
                    hypotheses[i][j].measurement_prediction
                measurement = hypotheses[i][j].measurement
                prediction = hypotheses[i][j].prediction
                # Calculate new weight and add to weight sum
                q = multivariate_normal.pdf(
                    measurement.state_vector.flatten(),
                    mean=measurement_prediction.mean.flatten(),
                    cov=measurement_prediction.covar
                )
                new_weight = self.prob_of_detection*prediction.weight*q
                weight_sum += new_weight
                # Perform single target Kalman Update
                temp_updated_component = self.updater.update(hypotheses[i][j])
                updated_component = TaggedWeightedGaussianState(
                    tag=prediction.tag,
                    weight=new_weight,
                    state_vector=temp_updated_component.mean,
                    covar=temp_updated_component.covar,
                    timestamp=temp_updated_component.timestamp
                )
                # Assign new tag if spawned from birth component
                if updated_component.tag == 0:
                    updated_component.tag = uuid.uuid4()
                # Add updated component to mixture
                updated_measurement_components.append(updated_component)
            for component in updated_measurement_components:
                component.weight /= weight_sum
                updated_components.append(component)
        for missed_detected_hypotheses in hypotheses[-1]:
            # Add all active components except birth component back into
            # mixture
            if not component.tag == 0:
                component =  TaggedWeightedGaussianState(
                    tag=missed_detected_hypotheses.prediction.tag,
                    weight=missed_detected_hypotheses*(1-self.prob_of_detection),
                    state_vector=missed_detected_hypotheses.prediction.mean,
                    covar=missed_detected_hypotheses.prediction.covar,
                    timestamp=temp_updated_component.timestamp)
                updated_components.append(component)
        # Return updated components
        return updated_components
            
