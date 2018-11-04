from abc import ABCMeta, abstractmethod

from contracts import contract
from six import with_metaclass

from duckietown_serialization_ds1 import Serializable
from duckietown_world.geo import PlacedObject
from duckietown_world.seqs import Sequence

__all__ = [
    'RuleEvaluationContext',
    'RuleEvaluationResult',
    'Rule',
    'evaluate_rules',
]


class RuleEvaluationContext(object):

    def __init__(self, interval, world, ego_name, lane_pose_seq):
        self._interval = interval
        self._world = world
        self._ego_name = ego_name
        self._lane_pose_seq = lane_pose_seq

    @contract(returns=Sequence)
    def get_interval(self):
        """ Returns the interval over which to evaluate the rule. """
        return self._interval

    @contract(returns=PlacedObject)
    def get_world(self):
        """ Returns the world object. """
        return self._world

    @contract(returns='seq(string)')
    def get_ego_name(self):
        """ Returns the name of the ego-vehicle
            as an object in the hierarchy """
        return self._ego_name

    @contract(returns=Sequence)
    def get_lane_pose_seq(self):
        """ Returns the lane pose result sequence.
            At each timestamp a possibly empty dict of index -> LanePoseResult """
        return self._lane_pose_seq


class EvaluatedMetric(Serializable):
    def __init__(self, total, incremental, description, cumulative):
        self.total = total
        self.incremental = incremental
        self.cumulative = cumulative
        self.description = description


class RuleEvaluationResult(object):

    def __init__(self):
        self.metrics = {}

    @contract(name='tuple,seq(string)', total='float|int', incremental=Sequence)
    def set_metric(self, name, total, incremental=None, description=None, cumulative=None):
        assert isinstance(name, tuple)
        name = tuple(name)
        self.metrics[name] = EvaluatedMetric(total, incremental, description, cumulative=cumulative)


class Rule(with_metaclass(ABCMeta)):

    @abstractmethod
    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        """ Evaluates the rule in this context.

            Must make at least one call to

                result.set_violation()
        """


def evaluate_rules(poses_sequence, interval, world, ego_name):
    from duckietown_world.world_duckietown import create_lane_highlight
    lane_pose_seq = create_lane_highlight(poses_sequence, world)
    from duckietown_world.rules import DeviationFromCenterLine
    from duckietown_world.rules import InDrivableLane
    from duckietown_world.rules import DeviationHeading
    rules = {
        'deviation-heading': DeviationHeading(),
        'in-drivable-lane': InDrivableLane(),
        'deviation-center-line': DeviationFromCenterLine(),
    }

    context = RuleEvaluationContext(interval, world, ego_name, lane_pose_seq)

    evaluated = {}
    for name, rule in rules.items():
        result = RuleEvaluationResult()
        rule.evaluate(context, result)
        evaluated[name] = result
        # for k, v in result.metrics.items():
        #     kk = (name,) + k
        #     metrics[kk] = result
    return evaluated
