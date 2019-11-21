# -*- coding: utf-8 -*-
import scipy
from scipy.stats import multivariate_normal

from .gaussianmixture import GaussianMixtureMultiTargetTracker
from ..types import TaggedWeightedGaussianState


class GMPHDTargetTracker(GaussianMixtureMultiTargetTracker):
    """A implementation of the Gaussian Mixture
    Probability Hypothesis Density (GM-PHD) multi-target filter

    References
    ----------

    .. [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """
    def tracks_gen(self):
        tracks = set()
        for time, detections in self.detector:
            # Add birth component
            self.birth_component.timestamp = time
            self.gaussian_mixture.append(self.birth_component)
            # Perform GM-PHD prediction and generate hypotheses
            hypotheses = self.hypothesiser.hypothesise(
                                self.gaussian_mixture.components,
                                detections,
                                time
                                )
            # Perform GM-PHD Update
            self.gaussian_mixture.components = self.updater.update(hypotheses)
            # Reduce mixture - Pruning and Merging
            self.gaussian_mixture.components = \
                self.reducer.reduce(self.gaussian_mixture.components)
            # Update the tracks
            tracks = self.tracks()
            yield time, tracks, self.estimated_number_of_targets