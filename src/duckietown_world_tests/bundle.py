import os

import numpy as np
from typing import *
import geometry as geo
from comptests import comptest, run_module_tests, get_comptests_output_dir
from duckietown_world import PWMCommands, SampledSequence, draw_static, \
    SE2Transform, DB18, construct_map, iterate_with_dt
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.types import TSE2v, se2v
from duckietown_world.world_duckietown.utils import get_velocities_from_sequence
from duckietown_world.rules import evaluate_rules, get_scores, RuleEvaluationResult
from duckietown_world.optimization import LexicographicSemiorderTracker, \
    LexicographicTracker, ProductOrderTracker


def get_bundle(N: int, times: List[float]):
    tries = {
    }

    d = 0.1
    b = 0.5
    # TODO remove after testing
    np.random.RandomState(seed=0)
    for i in range(N):
        commands_ssb = SampledSequenceBuilder[PWMCommands]()
        for t in times:
            u_left = np.random.choice([-d + b, b, b + d])
            u_right = np.random.choice([-d + b, b, b + d])
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
    N = 5

    commands_bundle = get_bundle(N, list(np.linspace(0, 4, 30)))
    trajs_bundle = {}

    for id_try, commands in commands_bundle.items():
        seq = integrate_dynamics2(parameters, q0, v0, commands)

        trajs_bundle[id_try] = seq

    outdir = os.path.join(get_comptests_output_dir(), 'together')
    visualize(commands_bundle, trajs_bundle, outdir)


def visualize(commands_bundle, trajs_bundle, outdir):
    root = get_simple_map()
    timeseries = {}

    rules_list = ['Deviation from center line', 'Drivable areas']
    optimal_traj_tracker1 = LexicographicSemiorderTracker(rules_list)
    optimal_traj_tracker2 = LexicographicTracker(rules_list)
    optimal_traj_tracker3 = ProductOrderTracker(rules_list)

    for id_try, commands in commands_bundle.items():
        traj = trajs_bundle[id_try]
        c = commands
        a = c.values
        b = a[0]
        ground_truth = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        ego_name = f'Duckiebot{id_try}'
        root.set_object(ego_name, DB18(), ground_truth=ground_truth)
        poses = traj.transform_values(lambda t: t[0])
        velocities = get_velocities_from_sequence(poses)
        # TODO what's the difference with below
        # v_test = traj.transform_values(lambda t: t[1])
        linear = velocities.transform_values(linear_from_se2)
        angular = velocities.transform_values(angular_from_se2)

        poses_sequence = traj.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        interval = SampledSequence.from_iterator(enumerate(commands.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence, interval=interval, world=root, ego_name=ego_name)

        print('Rules for Duckiebot' + str(id_try))
        scores = get_scores(evaluated)
        for k, score in scores.items():
            if k in rules_list:
                print(k, '=', score)

        optimal_traj_tracker1.digest_traj(ego_name, scores)
        optimal_traj_tracker2.digest_traj(ego_name, scores)
        optimal_traj_tracker3.digest_traj(ego_name, scores)

        # print("linear values for traj" + str(id_try))
        # print(linear.values)
        # print(angular.values)
        # print(commands.timestamps[-1])
        # print("commands for traj" + str(id_try))
        # print(commands)

        for key, rer in evaluated.items():
            assert isinstance(rer, RuleEvaluationResult)

            for km, evaluated_metric in rer.metrics.items():
                sequences = {}
                sequences[evaluated_metric.title] = evaluated_metric.cumulative
                plots = TimeseriesPlot(f'{ego_name} - {evaluated_metric.title}', evaluated_metric.title, sequences)
                timeseries[f'{ego_name} - {evaluated_metric.title}'] = plots

        sequences = {}
        sequences['motor_left'] = commands.transform_values(lambda _: _.motor_left)
        sequences['motor_right'] = commands.transform_values(lambda _: _.motor_right)
        plots = TimeseriesPlot(f'{id_try} - PWM commands', 'pwm_commands', sequences)
        timeseries[f'{ego_name} - commands'] = plots

        sequences = {}
        sequences['linear_velocity'] = linear
        sequences['angular_velocity'] = angular
        plots = TimeseriesPlot(f'{id_try} - Velocities', 'velocities', sequences)
        timeseries[f'{ego_name} - velocities'] = plots

    draw_static(root, outdir, timeseries=timeseries)

    display_optima(optimal_traj_tracker1)
    display_optima(optimal_traj_tracker2)
    display_optima(optimal_traj_tracker3)


def display_optima(optimal_traj_tracker):
    optima = optimal_traj_tracker.get_optimal_trajs()
    print('Optimal Trajectories for ', type(optimal_traj_tracker), ':')
    for item in optima:
        print(item)
        for rule in optimal_traj_tracker.rules:
            print(rule, '=', optima[item][rule])


def get_simple_map():
    map_data_yaml = """

       tiles:
       - [floor/W,floor/W, floor/W, floor/W, floor/W] 
       - [straight/W   , straight/W   , straight/W, straight/W, straight/W]
       - [floor/W,floor/W, floor/W, floor/W, floor/W]
       tile_size: 0.61
       """

    import yaml

    map_data = yaml.load(map_data_yaml)

    root = construct_map(map_data)
    return root


def linear_from_se2(x: se2v) -> float:
    linear, _ = geo.linear_angular_from_se2(x)
    # FIXME why index 0 and not all?
    return linear[0]


def angular_from_se2(x: se2v) -> float:
    _, angular = geo.linear_angular_from_se2(x)
    return angular


def integrate_dynamics2(factory, q0, v0, commands: SampledSequence) -> SampledSequence[TSE2v]:
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
