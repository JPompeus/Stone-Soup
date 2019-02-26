from .base import Tracker
from ..base import Property
from ..reader import DetectionReader
from ..predictor import Predictor
from ..types import TaggedWeightedGaussianState, GaussianMixtureState
from ..updater import Updater
from ..hypothesiser import GMMahalanobisDistanceHypothesiser
from ..dataassociator.neighbour import GaussianMixtureAssociator
from ..mixturereducer import GaussianMixtureReducer


class GaussianMixtureMultiTargetTracker(Tracker):
    """
    Base class for Gaussian Mixture style implmentations of
    point process derived filters
    """

    detector = Property(
        DetectionReader,
        default=None,
        doc="Detector used to generate detection objects.")
    predictor = Property(
        Predictor,
        default=None,
        doc="Predictor used to predict the objects to their new state.")
    updater = Property(
        Updater,
        default=None,
        doc="Updater used to update the objects to their new state.")
    gaussian_mixture = Property(
        GaussianMixtureState,
        default=None,
        doc="""Gaussian Mixture modelling the
                intensity over the target state space.""")
    data_associator = Property(
        GaussianMixtureAssociator,
        default=None,
        doc="Association algorithm to pair predictions to detections")

    reducer = Property(
        GaussianMixtureReducer,
        default=None,
        doc="Reducer used to reduce the number of components in the mixture.")
    prob_of_detection = Property(
        float,
        default=0.9,
        doc="""The probability that an exisiting target is detected
                at each timestep""")
    prob_of_survival = Property(
        float,
        default=0.999,
        doc="""The probability that an exisiting target surivives
                until the next timestep""")
    clutter_spatial_density = Property(
        float,
        default=1e-10,
        doc="""The clutter intensity at a point in the state point.""")
    birth_component = Property(
        TaggedWeightedGaussianState,
        default=None,
        doc="""The birth component. The weight is equal to the mean of the
        expected number of births per timestep (Poission distributed)""")

    def __init__(self, *args, **kwargs):
        if ("association_threshold" in kwargs):
            association_threshold = kwargs.pop("association_threshold")
        else:
            association_threshold = 16

        if ("merge_threshold" in kwargs):
            merge_threshold = kwargs.pop("merge_threshold")
        else:
            merge_threshold = 16

        if ("prune_threshold" in kwargs):
            prune_threshold = kwargs.pop("prune_threshold")
        else:
            prune_threshold = 1e-5

        if ("components" in kwargs):
            components = kwargs.pop("components")
        else:
            components = []

        super().__init__(*args, **kwargs)
        self.gaussian_mixture = GaussianMixtureState(
            components=components
        )
        # Create Hypothesiser and Associator

        hypothesiser = GMMahalanobisDistanceHypothesiser(
            self.predictor, self.updater, association_threshold)

        self.data_associator = GaussianMixtureAssociator(hypothesiser)
        # Create reducer
        self.reducer = GaussianMixtureReducer(
                            prune_threshold=prune_threshold,
                            merge_threshold=merge_threshold
                            )

    def predict(self, timestamp, control_input=None, **kwargs):
        """
        Predicts the current components in the
        :state:`GaussianMixtureState` according to the filter's prediction
        step.

        Parameters
        ==========
        self : :state:`GaussianMixtureMultiTargetTracker`
            Current GM Multi Target Tracker at time :math:`k`

        Returns
        =======
        self : :state:`GaussianMixtureMultiTargetTracker`
            GM Multi Target Tracker with predicted components to time
            :math:`k | k+1`

        Note
        ======
        This is an iteration over a list (:class:`GaussianMixtureState`).
        It predicts each component :math:`i` according to the underlying
        :class:`Predictor` class
        """
        raise NotImplementedError

    def update(self, measurements, timestamp, **kwargs):
        """
        Updates the current components in the
        :state:`GaussianMixtureState` according to the GM-PHD update
        step with the supplied measurements.

        Parameters
        ==========
        self : :state:`GaussianMixtureMultiTargetTracker`
            GM Multi Target Tracker with predicted components to time
            :math:`k | k+1`
        measurements : list
            Measurements obtained at time :math:`k+1`

        Returns
        =======
        self : :state:`GaussianMixtureMultiTargetTracker`
            GM Multi Target Tracker with updated components at time :math:`k+1`
        """
        raise NotImplementedError

    def tracks(self):
        pass

    def tracks_gen(self):
        pass

    @property
    def estimated_number_of_targets(self):
        """
        The number of hypothesised targets.
        """
        if self.gaussian_mixture:
            estimated_number_of_targets = sum(component.weight for component in
                                              self.gaussian_mixture)
        else:
            estimated_number_of_targets = 0
        return estimated_number_of_targets
