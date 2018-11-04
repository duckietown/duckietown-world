import textwrap

import numpy as np
from contracts import contract

from duckietown_world.seqs import SampledSequence, UndefinedAtTime, iterate_with_dt
from duckietown_world.world_duckietown import LanePose, GetLanePoseResult
from .rule import Rule, RuleEvaluationContext, RuleEvaluationResult

__all__ = [
    'InDrivableLane',
    'DeviationFromCenterLine',
    'DeviationHeading',
]


def integrate(sequence):
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v0 = _.v0
        dt = _.dt
        total += v0 * dt

        timestamps.append(_.t0)
        values.append(total)

    return SampledSequence(timestamps, values)


class DeviationFromCenterLine(Rule):

    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        assert isinstance(result, RuleEvaluationResult)
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 0
            else:
                if name2lpr:
                    first = name2lpr[sorted(name2lpr)[0]]
                    assert isinstance(first, GetLanePoseResult)

                    lp = first.lane_pose
                    assert isinstance(lp, LanePose)

                    d = lp.distance_from_center
                else:
                    # no lp
                    d = 0

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence(timestamps, values)
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]
        title = "[Rule] Deviation from center line"
        description = textwrap.dedent("""\
            This metric describes the amount of deviation from the center line.
        """)
        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)


class DeviationHeading(Rule):
    # def get_name_UI(self):
    #     return '[Rule] Heading deviation'
    #
    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        assert isinstance(result, RuleEvaluationResult)
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 0
            else:
                if name2lpr:
                    first = name2lpr[sorted(name2lpr)[0]]
                    assert isinstance(first, GetLanePoseResult)

                    lp = first.lane_pose
                    assert isinstance(lp, LanePose)

                    d = np.abs(lp.relative_heading)
                else:
                    # no lp
                    d = 0

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence(timestamps, values)
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]
        # result.set_metric((), dtot, sequence, description, cumulative=cumulative)
        title = "[Rule] Deviation from lane direction"
        description = textwrap.dedent("""\
            This metric describes the amount of deviation from the relative heading.
        """)
        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)


class InDrivableLane(Rule):

    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        assert isinstance(result, RuleEvaluationResult)
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 1
            else:
                if name2lpr:
                    d = 0
                else:
                    # no lp
                    d = 1

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence(timestamps, values)
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]

        title = "[Rule] Drivable areas"
        description = textwrap.dedent("""\
            This metric computes whether the robot was in a drivable area.
            
            Note that we check that the robot is in the lane in a correct heading 
            (up to 90deg deviation from the lane direction). 
        """)

        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)
