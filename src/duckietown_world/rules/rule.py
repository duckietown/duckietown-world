from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from contracts import contract, check_isinstance
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

    def __init__(self, interval, world, ego_name, lane_pose_seq, pose_seq):
        self._interval = interval
        self._world = world
        self._ego_name = ego_name
        self._lane_pose_seq = lane_pose_seq
        self._pose_seq = pose_seq

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

    @contract(returns=Sequence)
    def get_ego_pose_global(self):
        """ Returns the global pose of the vehicle. """
        return self._pose_seq


class EvaluatedMetric(Serializable):
    def __init__(self, total, incremental, description, cumulative, title):
        self.total = float(total)
        self.title = title
        self.incremental = incremental
        self.cumulative = cumulative
        self.description = description

    def __repr__(self):
        return 'EvaluatedMetric(%s, %s)' % (self.title, self.total)


class RuleEvaluationResult(object):

    def __init__(self, rule):
        self.metrics = OrderedDict()
        self.rule = rule

    @contract(name='tuple,seq(string)', total='float|int', incremental=Sequence)
    def set_metric(self, name, total, title=None, incremental=None, description=None, cumulative=None):
        check_isinstance(name, tuple)
        self.metrics[name] = EvaluatedMetric(total=total, incremental=incremental,
                                             title=title, description=description, cumulative=cumulative)

    def __repr__(self):
        return 'RuleEvaluationResult(%s, %s)' % (self.rule, self.metrics)


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
    from duckietown_world.rules import DrivenLength
    from duckietown_world.rules import DrivenLengthConsecutive
    rules = OrderedDict()
    rules['deviation-heading'] = DeviationHeading()
    rules['in-drivable-lane'] = InDrivableLane()
    rules['deviation-center-line'] = DeviationFromCenterLine()
    rules['driving-distance'] = DrivenLength()
    rules['driving-distance-consecutive'] = DrivenLengthConsecutive()

    context = RuleEvaluationContext(interval, world, ego_name=ego_name,
                                    lane_pose_seq=lane_pose_seq, pose_seq=poses_sequence)

    evaluated = OrderedDict()
    for name, rule in rules.items():
        result = RuleEvaluationResult(rule)
        rule.evaluate(context, result)
        evaluated[name] = result
    return evaluated


def make_timeseries(evaluated):
    timeseries = OrderedDict()
    for k, rer in evaluated.items():
        from duckietown_world.rules import RuleEvaluationResult
        from duckietown_world.svg_drawing.misc import TimeseriesPlot
        assert isinstance(rer, RuleEvaluationResult)

        for km, evaluated_metric in rer.metrics.items():
            assert isinstance(evaluated_metric, EvaluatedMetric)
            sequences = {}
            if evaluated_metric.incremental:
                sequences['incremental'] = evaluated_metric.incremental
            if evaluated_metric.cumulative:
                sequences['cumulative'] = evaluated_metric.cumulative

            kk = "/".join((k,) + km)
            # title = evaluated_metric.title + ( '(%s)' % evaluated_metric.title if km else "")
            title = evaluated_metric.title
            timeseries[kk] = TimeseriesPlot(title, evaluated_metric.description, sequences)
    # print(list(timeseries))
    return timeseries
