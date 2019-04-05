from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import *

from duckietown_serialization_ds1 import Serializable

from contracts import check_isinstance
from duckietown_world import LanePose
from duckietown_world.geo import PlacedObject, SE2Transform
from duckietown_world.seqs import SampledSequence
from duckietown_world.seqs.tsequence import Timestamp
from duckietown_world.svg_drawing.misc import TimeseriesPlot

__all__ = [
    'RuleEvaluationContext',
    'RuleEvaluationResult',
    'Rule',
    'evaluate_rules',
]


@dataclass
class RuleEvaluationContext:
    interval: SampledSequence[Timestamp]
    world: PlacedObject
    ego_name: str
    lane_pose_seq: SampledSequence[LanePose]
    pose_seq: SampledSequence[SE2Transform]

    def get_interval(self) -> SampledSequence[Timestamp]:
        """ Returns the interval over which to evaluate the rule. """
        return self.interval

    def get_world(self) -> PlacedObject:
        """ Returns the world object. """
        return self.world

    def get_ego_name(self) -> str:
        """ Returns the name of the ego-vehicle
            as an object in the hierarchy """
        return self.ego_name

    def get_lane_pose_seq(self) -> SampledSequence[LanePose]:
        """ Returns the lane pose result sequence.
            At each timestamp a possibly empty dict of index -> LanePoseResult """
        return self.lane_pose_seq

    def get_ego_pose_global(self) -> SampledSequence[SE2Transform]:
        """ Returns the global pose of the vehicle. """
        return self.pose_seq


class EvaluatedMetric(Serializable):
    total: float
    incremental: float
    cumulative: float
    description: str
    title: str

    def __init__(self, total, incremental, description, cumulative, title):
        self.total = float(total)
        self.title = title
        self.incremental = incremental
        self.cumulative = cumulative
        self.description = description

    def __repr__(self):
        return 'EvaluatedMetric(%s, %s)' % (self.title, self.total)


class RuleEvaluationResult:
    metrics: Dict[str, EvaluatedMetric]
    rule: 'Rule'

    def __init__(self, rule):
        self.metrics = OrderedDict()
        self.rule = rule

    # @contract(name='tuple,seq(string)', total='float|int', incremental=Sequence)
    def set_metric(self,
                   name: Tuple[str, ...],
                   total: Union[float, int],
                   title: Optional[str] = None,
                   description: Optional[str] = None,
                   incremental: Optional[SampledSequence] = None,

                   cumulative: Optional[SampledSequence] = None):
        check_isinstance(name, tuple)
        self.metrics[name] = EvaluatedMetric(total=total,
                                             incremental=incremental,
                                             title=title,
                                             description=description,
                                             cumulative=cumulative)

    def __repr__(self):
        return 'RuleEvaluationResult(%s, %s)' % (self.rule, self.metrics)


class Rule(metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        """ Evaluates the rule in this context.

            Must make at least one call to

                result.set_violation()
        """


def evaluate_rules(poses_sequence,
                   interval: SampledSequence[Timestamp],
                   world: PlacedObject,
                   ego_name: str) -> Dict[
    str, RuleEvaluationResult]:
    from duckietown_world.world_duckietown import create_lane_highlight
    lane_pose_seq = create_lane_highlight(poses_sequence, world)
    from duckietown_world.rules import DeviationFromCenterLine
    from duckietown_world.rules import InDrivableLane
    from duckietown_world.rules import DeviationHeading
    from duckietown_world.rules import DrivenLength
    from duckietown_world.rules import DrivenLengthConsecutive
    from duckietown_world.rules import SurvivalTime

    rules = OrderedDict()
    rules['deviation-heading'] = DeviationHeading()
    rules['in-drivable-lane'] = InDrivableLane()
    rules['deviation-center-line'] = DeviationFromCenterLine()
    rules['driving-distance'] = DrivenLength()
    rules['driving-distance-consecutive'] = DrivenLengthConsecutive()
    rules['survival_time'] = SurvivalTime()

    context = RuleEvaluationContext(interval=interval, world=world, ego_name=ego_name,
                                    lane_pose_seq=lane_pose_seq, pose_seq=poses_sequence)

    evaluated = OrderedDict()
    for name, rule in rules.items():
        result = RuleEvaluationResult(rule)
        rule.evaluate(context, result)
        evaluated[name] = result
    return evaluated


def make_timeseries(evaluated) -> Dict[str, 'TimeseriesPlot']:
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
            timeseries[kk] = TimeseriesPlot(title or kk, evaluated_metric.description, sequences)

    return timeseries
