# coding=utf-8

import os

import numpy as np
from comptests import comptest, run_module_tests, get_comptests_output_dir
from numpy.testing import assert_almost_equal

import geometry as geo
from duckietown_world import LaneSegment, RectangularArea, PlacedObject, SE2Transform
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric, make_timeseries
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.differential_drive_dynamics import DifferentialDriveDynamicsParameters, \
    WheelVelocityCommands
from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.world_duckietown.lane_segment import get_distance_two
from duckietown_world.world_duckietown.map_loading import load_map
from duckietown_world.world_duckietown.tile_template import load_tile_types


def same_point(a, b):
    assert_almost_equal(a.p, b.p)
    assert_almost_equal(a.theta, b.theta)


@comptest
def lane_pose2():
    templates = load_tile_types()

    for name, ls in templates.items():
        if not isinstance(ls, LaneSegment):
            continue

        print(ls.get_lane_lengths())
        length = ls.get_lane_length()

        # first pose is the first control point
        lp0 = ls.lane_pose(along_lane=0.0, relative_heading=0.0, lateral=0.0)
        t0 = ls.SE2Transform_from_lane_pose(lp0)

        p0 = ls.control_points[0]
        same_point(p0, t0)

        # at 1 pose it is the second control point
        lp1 = ls.lane_pose(along_lane=length, relative_heading=0.0, lateral=0.0)
        t1 = ls.SE2Transform_from_lane_pose(lp1)

        p1 = ls.control_points[-1]
        same_point(p1, t1)

        l1 = length * 0.2
        l2 = length * 0.3
        d = l2 - l1


@comptest
def lane_pose3():
    templates = load_tile_types()

    for name, ls in templates.items():
        if not isinstance(ls, LaneSegment):
            continue

        length = ls.get_lane_length()

        # first pose is the first control point
        l1 = length * 0.3
        l2 = length * 0.4
        d = l2 - l1

        lp1 = ls.lane_pose(along_lane=l1, relative_heading=0.0, lateral=0.0)
        lp2 = ls.lane_pose(along_lane=l2, relative_heading=0.0, lateral=0.0)

        q1 = ls.SE2Transform_from_lane_pose(lp1).as_SE2()
        q2 = ls.SE2Transform_from_lane_pose(lp2).as_SE2()

        d2 = get_distance_two(q1, q2)

        assert_almost_equal(d, d2, decimal=3)


@comptest
def lane_pose4():
    templates = load_tile_types()

    for name, ls in templates.items():
        if not isinstance(ls, LaneSegment):
            continue
        print(name)

        lp1 = ls.lane_pose_random()
        # print('lp1: %s' % lp1)
        q1 = ls.SE2Transform_from_lane_pose(lp1)
        # print('q1: %s' % q1)
        lp2 = ls.lane_pose_from_SE2Transform(q1, tol=0.001)

        # print('lp2: %s' % lp2)
        assert_almost_equal(lp1.along_lane, lp2.along_lane, decimal=3)
        assert_almost_equal(lp1.lateral, lp2.lateral, decimal=3)
        assert_almost_equal(lp1.relative_heading, lp2.relative_heading, decimal=3)


@comptest
def center_point1():
    outdir = get_comptests_output_dir()
    templates = load_tile_types()

    for k, v in templates.items():
        if isinstance(v, LaneSegment):

            area = RectangularArea((-2, -2), (3, 3))
            dest = os.path.join(outdir, k)

            N = len(v.control_points)
            betas = list(np.linspace(-2, N + 1, 20))
            betas.extend(range(N))
            betas = sorted(betas)
            transforms = []
            for timestamp in betas:
                beta = timestamp
                p = v.center_point(beta)
                # print('%s: %s' % (beta, geo.SE2.friendly(p)))

                transform = SE2Transform.from_SE2(p)
                transforms.append(transform)

            c = SampledSequence(betas, transforms)
            v.set_object('a', PlacedObject(), ground_truth=c)
            draw_static(v, dest, area=area)


@comptest
def lane_pose_test1():
    outdir = get_comptests_output_dir()

    # load one of the maps (see the list using dt-world-draw-maps)
    dw = load_map('udem1')

    v = 5

    # define a SampledSequence with timestamp, command
    commands_sequence = SampledSequence.from_iterator([
        # we set these velocities at 1.0
        (1.0, WheelVelocityCommands(0.1 * v, 0.1 * v)),
        # at 2.0 we switch and so on
        (2.0, WheelVelocityCommands(0.1 * v, 0.4 * v)),
        (4.0, WheelVelocityCommands(0.1 * v, 0.4 * v)),
        (5.0, WheelVelocityCommands(0.1 * v, 0.2 * v)),
        (6.0, WheelVelocityCommands(0.1 * v, 0.1 * v)),
    ])

    # we upsample the sequence by 5
    commands_sequence = commands_sequence.upsample(5)

    ## Simulate the dynamics of the vehicle
    # start from q0
    q0 = geo.SE2_from_translation_angle([1.8, 0.7], 0)

    # instantiate the class that represents the dynamics
    dynamics = reasonable_duckiebot()
    # this function integrates the dynamics
    poses_sequence = get_robot_trajectory(dynamics, q0, commands_sequence)


    #################
    # Visualization and rule evaluation

    # Creates an object 'duckiebot'
    ego_name = 'duckiebot'
    db = DB18() # class that gives the appearance

    # convert from SE2 to SE2Transform representation
    transforms_sequence = poses_sequence.transform_values(SE2Transform.from_SE2)
    # puts the object in the world with a certain "ground_truth" constraint
    dw.set_object(ego_name, db, ground_truth=transforms_sequence)

    # Rule evaluation (do not touch)
    interval = SampledSequence.from_iterator(enumerate(commands_sequence.timestamps))
    evaluated = evaluate_rules(poses_sequence=transforms_sequence,
                                interval=interval, world=dw, ego_name=ego_name)
    timeseries = make_timeseries(evaluated)
    # Drawing
    area = RectangularArea((0, 0), (3, 3))
    draw_static(dw, outdir, area=area, timeseries=timeseries)


def integrate_commands(s0, commands_sequence):
    states = [s0]
    timestamps = commands_sequence.timestamps
    t0 = timestamps[0]
    yield t0, s0
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        commands = commands_sequence.values[i]
        s_prev = states[-1]
        s_next = s_prev.integrate(dt, commands)
        states.append(s_next)
        t = timestamps[i + 1]
        yield t, s_next


def get_robot_trajectory(factory, q0, commands_sequence):
    assert isinstance(commands_sequence, SampledSequence)
    t0 = commands_sequence.timestamps[0]

    # initialize trajectory
    c0 = q0, geo.se2.zero()
    s0 = factory.initialize(c0=c0, t0=t0)

    states_sequence = SampledSequence.from_iterator(integrate_commands(s0, commands_sequence))
    f = lambda _: _.TSE2_from_state()[0]
    poses_sequence = states_sequence.transform_values(f)
    return poses_sequence


def reasonable_duckiebot():
    radius = 0.1
    radius_left = radius
    radius_right = radius
    wheel_distance = 0.5
    dddp = DifferentialDriveDynamicsParameters(radius_left=radius_left, radius_right=radius_right,
                                               wheel_distance=wheel_distance)
    return dddp


if __name__ == '__main__':
    run_module_tests()
