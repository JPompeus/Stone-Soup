# -*- coding: utf-8 -*-
from operator import attrgetter
import datetime

import numpy as np

from ..distance import DistanceHypothesiser
from ..gaussianmixture import GaussianMixtureHypothesier

from ...types.detection import Detection
from ...types.state import GaussianState, WeightedGaussianState
from ...types.track import Track
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.mixture import GaussianMixtureState
from ... import measures


def test_gm_ordered_by_measurement(predictor, updater):

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixtureState(
        [WeightedGaussianState(
            np.array([[0.3]]), np.array([[1]]), timestamp, 0.4),
            WeightedGaussianState(
                np.array([[5]]), np.array([[0.5]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}
    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=20)
    hypothesiser = GaussianMixtureHypothesier(predictor, updater,
                                                     hypothesiser=hypothesiser,
                                                     order_by_detection=True)

    hypotheses = hypothesiser.hypothesise(gaussian_mixture,
                                          detections, timestamp)

    # There are 4 hypotheses - 2 each associated with detection1/detection2
    assert all(isinstance(multi_hyp, MultipleHypothesis)
               for multi_hyp in hypotheses)
    assert all(isinstance(hyp, SingleHypothesis)
               for multi_hyp in hypotheses for hyp in multi_hyp)
    assert len(hypotheses) == 2
    assert len(hypotheses[0]) == 2
    assert len(hypotheses[1]) == 2

    # each SingleHypothesis has a distance attribute
    assert all(hyp.distance >= 0
               for multi_hyp in hypotheses for hyp in multi_hyp)

    # sanity-check the values returned by the hypothesiser
    assert hypotheses[0][0].distance < 10
    assert hypotheses[0][1].distance > 0
    assert hypotheses[1][0].distance > 0
    assert hypotheses[1][1].distance < 10

    # Check the measurements are the same
    assert hypotheses[0][0].measurement.state_vector == np.array([[1]])
    assert hypotheses[0][1].measurement.state_vector == np.array([[1]])
    assert hypotheses[1][0].measurement.state_vector == np.array([[6.2]])
    assert hypotheses[1][1].measurement.state_vector == np.array([[6.2]])

def test_gm_ordered_by_component(predictor, updater):

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixtureState(
        [WeightedGaussianState(
            np.array([[0.3]]), np.array([[1]]), timestamp, 0.4),
            WeightedGaussianState(
                np.array([[5]]), np.array([[0.5]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}
    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=20)
    hypothesiser = GaussianMixtureHypothesier(predictor, updater,
                                                     hypothesiser=hypothesiser,
                                                     order_by_detection=False)

    hypotheses = hypothesiser.hypothesise(gaussian_mixture,
                                          detections, timestamp)

    # There are 4 hypotheses - 2 each associated with detection1/detection2
    assert all(isinstance(multi_hyp, MultipleHypothesis)
               for multi_hyp in hypotheses)
    assert all(isinstance(hyp, SingleHypothesis)
               for multi_hyp in hypotheses for hyp in multi_hyp)
    assert len(hypotheses) == 2
    assert len(hypotheses[0]) == 2
    assert len(hypotheses[1]) == 2

    # each SingleHypothesis has a distance attribute
    assert all(hyp.distance >= 0
               for multi_hyp in hypotheses for hyp in multi_hyp)

    # sanity-check the values returned by the hypothesiser
    assert hypotheses[0][0].distance < 10
    assert hypotheses[0][1].distance > 0
    assert hypotheses[1][0].distance > 0
    assert hypotheses[1][1].distance < 10

    # Check the components are the same
    assert hypotheses[0][0].prediction.state_vector == np.array([[1.3]])
    assert hypotheses[0][1].prediction.state_vector == np.array([[1.3]])
    assert hypotheses[1][0].prediction.state_vector == np.array([[6]])
    assert hypotheses[1][1].prediction.state_vector == np.array([[6]])
