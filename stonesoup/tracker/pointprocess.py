# -*- coding: utf-8 -*-
from .gaussianmixture import GaussianMixtureMultiTargetTracker
from ..types import TaggedWeightedGaussianState
from ..base import Property


class GMPHDTargetTracker(GaussianMixtureMultiTargetTracker):
    """
    """


        
class GMLCCTargetTracker(GaussianMixtureMultiTargetTracker):
    """A implementation of the Gaussian Mixture
    Probability Hypothesis Density (GM-PHD) multi-target filter

    References
    ----------

    .. [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """
    