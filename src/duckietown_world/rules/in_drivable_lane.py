import textwrap

import numpy as np
from contracts import contract

from duckietown_world.seqs import SampledSequence, UndefinedAtTime, iterate_with_dt
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.world_duckietown import LanePose, GetLanePoseResult
from duckietown_world.world_duckietown.tile import relative_pose
from .rule import Rule, RuleEvaluationContext, RuleEvaluationResult

__all__ = [
    'InDrivableLane',
    'DeviationFromCenterLine',
    'DeviationHeading',
    'DrivenLength',
    'DrivenLengthConsecutive',
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
        title = "Deviation from center line"
        description = textwrap.dedent("""\
            This metric describes the amount of deviation from the center line.
        """)
        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)


class DeviationHeading(Rule):

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
        title = "Deviation from lane direction"
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

        title = "Drivable areas"
        description = textwrap.dedent("""\
            This metric computes whether the robot was in a drivable area.
            
            Note that we check that the robot is in the lane in a correct heading 
            (up to 90deg deviation from the lane direction). 
        """)

        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)


import geometry as geo


class DrivenLength(Rule):

    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        assert isinstance(result, RuleEvaluationResult)
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()
        ego_pose_sequence = context.get_ego_pose_global()

        driven_any_builder = SampledSequenceBuilder()
        driven_lanedir_builder = SampledSequenceBuilder()

        for idt in iterate_with_dt(interval):
            t0, t1 = idt.v0, idt.v1  # not v
            try:
                name2lpr = lane_pose_seq.at(t0)

                p0 = ego_pose_sequence.at(t0).as_SE2()
                p1 = ego_pose_sequence.at(t1).as_SE2()
            except UndefinedAtTime:
                dr_any = dr_lanedir = 0.0

            else:
                prel = relative_pose(p0, p1)
                translation, _ = geo.translation_angle_from_SE2(prel)
                dr_any = np.linalg.norm(translation)

                if name2lpr:

                    ds = []
                    for k, lpr in name2lpr.items():
                        assert isinstance(lpr, GetLanePoseResult)
                        c0 = lpr.center_point
                        ctas = geo.translation_angle_scale_from_E2(c0.asmatrix2d().m)
                        c0_ = geo.SE2_from_translation_angle(ctas.translation, ctas.angle)
                        prelc0 = relative_pose(c0_, p1)
                        tas = geo.translation_angle_scale_from_E2(prelc0)

                        # otherwise this lane should not be reported
                        # assert tas.translation[0] >= 0, tas
                        ds.append(tas.translation[0])

                    dr_lanedir = max(ds)
                else:
                    # no lp
                    dr_lanedir = 0.0

            driven_any_builder.add(t0, dr_any)
            driven_lanedir_builder.add(t0, dr_lanedir)

        driven_any_incremental = driven_any_builder.as_sequence()
        driven_any_cumulative = integrate(driven_any_incremental)

        title = "Distance"
        description = textwrap.dedent("""\
            This metric computes how far the robot drove.
        """)

        result.set_metric(name=('driven_any',), total=driven_any_cumulative.values[-1],
                          incremental=driven_any_incremental,
                          title=title, description=description, cumulative=driven_any_cumulative)
        title = "Lane distance"

        driven_lanedir_incremental = driven_lanedir_builder.as_sequence()
        driven_lanedir_cumulative = integrate(driven_lanedir_incremental)

        description = textwrap.dedent("""\
            This metric computes how far the robot drove
            **in the direction of the lane**.
        """)
        result.set_metric(name=('driven_lanedir',), total=driven_lanedir_cumulative.values[-1],
                          incremental=driven_lanedir_incremental,
                          title=title, description=description, cumulative=driven_lanedir_cumulative)


class DrivenLengthConsecutive(Rule):

    @contract(context=RuleEvaluationContext, result=RuleEvaluationResult)
    def evaluate(self, context, result):
        assert isinstance(result, RuleEvaluationResult)
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()
        ego_pose_sequence = context.get_ego_pose_global()

        timestamps = []
        driven_any = []
        driven_lanedir = []

        tile_fqn2lane_fqn = {}
        for idt in iterate_with_dt(interval):
            t0, t1 = idt.v0, idt.v1  # not v
            try:
                name2lpr = lane_pose_seq.at(t0)

                p0 = ego_pose_sequence.at(t0).as_SE2()
                p1 = ego_pose_sequence.at(t1).as_SE2()
            except UndefinedAtTime:
                dr_any = dr_lanedir = 0.0

            else:
                prel = relative_pose(p0, p1)
                translation, _ = geo.translation_angle_from_SE2(prel)
                dr_any = np.linalg.norm(translation)

                if name2lpr:

                    ds = []
                    for k, lpr in name2lpr.items():
                        if lpr.tile_fqn in tile_fqn2lane_fqn:
                            if lpr.lane_segment_fqn != tile_fqn2lane_fqn[lpr.tile_fqn]:
                                # msg = 'Backwards detected'
                                # print(msg)
                                continue

                        tile_fqn2lane_fqn[lpr.tile_fqn] = lpr.lane_segment_fqn

                        assert isinstance(lpr, GetLanePoseResult)
                        c0 = lpr.center_point
                        ctas = geo.translation_angle_scale_from_E2(c0.asmatrix2d().m)
                        c0_ = geo.SE2_from_translation_angle(ctas.translation, ctas.angle)
                        prelc0 = relative_pose(c0_, p1)
                        tas = geo.translation_angle_scale_from_E2(prelc0)

                        # otherwise this lane should not be reported
                        # assert tas.translation[0] >= 0, tas
                        ds.append(tas.translation[0])

                    dr_lanedir = max(ds) if ds else 0.0
                else:
                    # no lp
                    dr_lanedir = 0.0

            driven_any.append(dr_any)
            driven_lanedir.append(dr_lanedir)
            timestamps.append(t0)

        # driven_any_incremental = SampledSequence(timestamps, driven_any)
        # driven_any_cumulative = integrate(driven_any_incremental)

        title = "Consecutive lane distance"

        driven_lanedir_incremental = SampledSequence(timestamps, driven_lanedir)
        driven_lanedir_cumulative = integrate(driven_lanedir_incremental)

        description = textwrap.dedent("""\
            This metric computes how far the robot drove
            **in the direction of the correct lane**,
            discounting whenever it was driven
            in the wrong direction with respect to the start.
        """)
        result.set_metric(name=('driven_lanedir_consec',), total=driven_lanedir_cumulative.values[-1],
                          incremental=driven_lanedir_incremental,
                          title=title, description=description, cumulative=driven_lanedir_cumulative)
