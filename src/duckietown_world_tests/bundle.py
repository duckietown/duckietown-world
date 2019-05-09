import os
from typing import *

import numpy as np

import geometry as geo
from comptests import comptest, run_module_tests, get_comptests_output_dir
from duckietown_world import PWMCommands, SampledSequence, draw_static, \
    SE2Transform, DB18, construct_map, iterate_with_dt
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.types import TSE2v, se2v
from duckietown_world.world_duckietown.utils import get_velocities_from_sequence


def get_bundle(N: int, times: List[float]):
    tries = {}

    d = 0.1  # deviation from average
    b = 0.5  # average pwm
    choices = [-d + b, b, b + d]
    for i in range(N):
        commands_ssb = SampledSequenceBuilder[PWMCommands]()
        for t in times:
            u_left = np.random.choice(choices)
            u_right = np.random.choice(choices)
            u = PWMCommands(motor_left=u_left, motor_right=u_right)
            commands_ssb.add(t, u)
        commands = commands_ssb.as_sequence()
        tries[str(i)] = commands
    return tries


@comptest
def test_bundle():
    parameters = get_DB18_nominal(delay=0)

    # initial configuration
    init_pose = np.array([0, 0.8])
    init_vel = np.array([0, 0])

    q0 = geo.SE2_from_R2(init_pose)
    v0 = geo.se2_from_linear_angular(init_vel, 0)
    N = 60
    times = list(np.linspace(0, 4, 30))

    times_begin = list(np.linspace(0, 1, 30))
    times = sorted(set(times) | set(times_begin))
    commands_bundle = get_bundle(N, times)
    trajs_bundle = {}
    for id_try, commands in commands_bundle.items():
        seq = integrate_dynamics2(parameters, q0, v0, commands)

        trajs_bundle[id_try] = seq

    outdir = os.path.join(get_comptests_output_dir(), 'together')
    visualize(commands_bundle, trajs_bundle, outdir)


def visualize(commands_bundle, trajs_bundle, outdir):
    root = get_simple_map()
    timeseries = {}
    for id_try, commands in commands_bundle.items():
        traj = trajs_bundle[id_try]
        ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))

        robot_name = str(id_try)
        root.set_object(robot_name, DB18(), ground_truth=ground_truth)
        poses = traj.transform_values(lambda t: t[0])
        velocities = get_velocities_from_sequence(poses)
        linear = velocities.transform_values(linear_from_se2)
        angular = velocities.transform_values(angular_from_se2)
        # print(linear.values)
        # print(angular.values)

        if len(timeseries) < 5:
            sequences = {}
            sequences['motor_left'] = commands.transform_values(lambda _: _.motor_left)
            sequences['motor_right'] = commands.transform_values(lambda _: _.motor_right)
            plots = TimeseriesPlot(f'{id_try} - PWM commands', 'pwm_commands', sequences)
            timeseries[f'{id_try} - commands'] = plots

            sequences = {}
            sequences['linear_velocity'] = linear
            sequences['angular_velocity'] = angular
            plots = TimeseriesPlot(f'{id_try} - Velocities', 'velocities', sequences)
            timeseries[f'{id_try} - velocities'] = plots

        interval = SampledSequence.from_iterator(enumerate(poses.timestamps))
        transforms_sequence = poses.transform_values(SE2Transform.from_SE2)
        evaluated = evaluate_rules(poses_sequence=transforms_sequence,
                                   interval=interval,
                                   world=root,
                                   ego_name=robot_name)
        for name, rule_evaluation_result in evaluated.items():
            for metric_name, evaluated_metric in rule_evaluation_result.metrics.items():
                assert isinstance(evaluated_metric, EvaluatedMetric)
                print(name, metric_name, evaluated_metric.total)

    draw_static(root, outdir, timeseries=timeseries)

    # interval = SampledSequence.from_iterator(enumerate(log.pose.timestamps))


#     evaluated = evaluate_rules(poses_sequence=log.pose,
#                                interval=interval,
#                                world=duckietown_env,
#                                ego_name=robot_main)
#     evaluated.update(log0.render_time)

def get_simple_map():
    map_data_yaml = """

       tiles:
       - [floor/W, floor/W, floor/W, floor/W, floor/W] 
       - [straight/W, straight/W, straight/W, straight/W, straight/N]
       - [floor/W,floor/W, floor/W, floor/W, floor/W]
       
       tile_size: 0.61
       """

    import yaml

    map_data = yaml.load(map_data_yaml)

    root = construct_map(map_data)
    return root


def linear_from_se2(x: se2v) -> float:
    linear, angular = geo.linear_angular_from_se2(x)
    return linear[0]


def angular_from_se2(x: se2v) -> float:
    _, angular = geo.linear_angular_from_se2(x)
    return angular


def integrate_dynamics2(factory, q0, v0,
                        commands: SampledSequence) \
        -> SampledSequence[TSE2v]:
    # starting time
    c0 = q0, v0
    state = factory.initialize(c0=c0, t0=commands.timestamps[0])
    ssb = SampledSequenceBuilder[TSE2v]()
    # ssb.add(t0, state.TSE2_from_state())
    for it in iterate_with_dt(commands):
        ssb.add(it.t0, state.TSE2_from_state())
        state = state.integrate(it.dt, it.v0)

    return ssb.as_sequence()


if __name__ == '__main__':
    run_module_tests()
