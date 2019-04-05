import textwrap

import numpy as np

import geometry as geo
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
    'SurvivalTime',

]


def integrate(sequence: SampledSequence[float]) -> SampledSequence[float]:
    """ Integrates with respect to time.
        That is, it multiplies the value with the Delta T. """
    if not sequence:
        msg = 'Cannot integrate empty sequence.'
        raise ValueError(msg)
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v0 = _.v0
        dt = _.dt
        total += v0 * dt

        timestamps.append(_.t0)
        values.append(total)

    return SampledSequence[float](timestamps, values)


def accumulate(sequence: SampledSequence[float]) -> SampledSequence[float]:
    """ Integrates with respect to time.
        Sums the values along the horizontal. """
    total = 0.0
    timestamps = []
    values = []
    for t, v in sequence:
        total += v

        timestamps.append(t)
        values.append(total)

    return SampledSequence[float](timestamps, values)


from typing import cast


class SurvivalTime(Rule):

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        lane_pose_seq = context.get_lane_pose_seq()
        if len(lane_pose_seq) < 1:
            raise ValueError(lane_pose_seq)

        title = "Survival time"
        description = "Length of the episode."

        incremental = lane_pose_seq.transform_values(lambda _: 1.0)
        cumulative = integrate(incremental)
        total = cumulative.values[-1]

        result.set_metric(name=(),
                          title=title,
                          description=description,
                          total=total,
                          incremental=incremental,
                          cumulative=cumulative)


class DeviationFromCenterLine(Rule):

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = cast(SampledSequence, context.get_interval())
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 0.0
            else:
                if name2lpr:
                    first = name2lpr[sorted(name2lpr)[0]]
                    assert isinstance(first, GetLanePoseResult)

                    lp = first.lane_pose
                    assert isinstance(lp, LanePose)

                    d = lp.distance_from_center
                else:
                    # no lp
                    d = 0.0

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence[float](timestamps, values)

        if len(sequence) <= 1:
            cumulative = 0
            dtot = 0
        else:
            cumulative = integrate(sequence)
            dtot = cumulative.values[-1]

        title = "Deviation from center line"
        description = textwrap.dedent("""\
            This metric describes the amount of deviation from the center line.
        """)
        result.set_metric(name=(), total=dtot, incremental=sequence,
                          title=title, description=description, cumulative=cumulative)


class DeviationHeading(Rule):

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = cast(SampledSequence, context.get_interval())
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 0.0
            else:
                if name2lpr:
                    first = name2lpr[sorted(name2lpr)[0]]
                    assert isinstance(first, GetLanePoseResult)

                    lp = first.lane_pose
                    assert isinstance(lp, LanePose)

                    d = np.abs(lp.relative_heading)
                else:
                    # no lp
                    d = 0.0

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence[float](timestamps, values)
        if len(sequence) <= 1:
            cumulative = 0.0
            dtot = 0.0
        else:
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

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        interval = cast(SampledSequence, context.get_interval())
        lane_pose_seq = context.get_lane_pose_seq()

        timestamps = []
        values = []

        for i, timestamp in interval:
            try:
                name2lpr = lane_pose_seq.at(timestamp)
            except UndefinedAtTime:
                d = 1.0
            else:
                if name2lpr:
                    d = 0.0
                else:
                    # no lp
                    d = 1.0

            values.append(d)
            timestamps.append(timestamp)

        sequence = SampledSequence[float](timestamps, values)
        if len(sequence) <= 1:
            cumulative = 0
            dtot = 0
        else:
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


class DrivenLength(Rule):

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        interval = context.get_interval()
        lane_pose_seq = context.get_lane_pose_seq()
        ego_pose_sequence = context.get_ego_pose_global()

        driven_any_builder = SampledSequenceBuilder[float]()
        driven_lanedir_builder = SampledSequenceBuilder[float]()

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

        driven_any_cumulative = accumulate(driven_any_incremental)

        if len(driven_any_incremental) <= 1:
            total = 0
        else:
            total = driven_any_cumulative.values[-1]

        title = "Distance"
        description = textwrap.dedent("""\
            This metric computes how far the robot drove.
        """)

        result.set_metric(name=('driven_any',), total=total,
                          incremental=driven_any_incremental,
                          title=title, description=description, cumulative=driven_any_cumulative)
        title = "Lane distance"

        driven_lanedir_incremental = driven_lanedir_builder.as_sequence()
        driven_lanedir_cumulative = accumulate(driven_lanedir_incremental)

        if len(driven_lanedir_incremental) <= 1:
            total = 0
        else:
            total = driven_lanedir_cumulative.values[-1]

        description = textwrap.dedent("""\
            This metric computes how far the robot drove
            **in the direction of the lane**.
        """)
        result.set_metric(name=('driven_lanedir',), total=total,
                          incremental=driven_lanedir_incremental,
                          title=title, description=description, cumulative=driven_lanedir_cumulative)


class DrivenLengthConsecutive(Rule):

    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
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

        title = "Consecutive lane distance"

        driven_lanedir_incremental = SampledSequence[float](timestamps, driven_lanedir)
        driven_lanedir_cumulative = accumulate(driven_lanedir_incremental)

        if len(driven_lanedir_incremental) <= 1:
            total = 0
        else:
            total = driven_lanedir_cumulative.values[-1]
        description = textwrap.dedent("""\
            This metric computes how far the robot drove **in the direction of the correct lane**,
            discounting whenever it was driven in the wrong direction with respect to the start.
        """)
        result.set_metric(name=('driven_lanedir_consec',), total=total,
                          incremental=driven_lanedir_incremental,
                          title=title, description=description, cumulative=driven_lanedir_cumulative)
