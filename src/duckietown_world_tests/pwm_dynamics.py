import os

import numpy as np

import geometry as geo
from comptests import comptest, run_module_tests, get_comptests_output_dir
from duckietown_world import (
    PWMCommands,
    SampledSequence,
    draw_static,
    SE2Transform,
    DB18,
    construct_map,
)
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.types import TSE2v, se2v
from duckietown_world.world_duckietown.utils import get_velocities_from_sequence


@comptest
def test_pwm1():
    parameters = get_DB18_nominal(delay=0)

    # initial configuration
    init_pose = np.array([0, 0.8])
    init_vel = np.array([0, 0])

    q0 = geo.SE2_from_R2(init_pose)
    v0 = geo.se2_from_linear_angular(init_vel, 0)
    tries = {
        "straight_50": (PWMCommands(+0.5, 0.5)),
        "straight_max": (PWMCommands(+1.0, +1.0)),
        "straight_over_max": (PWMCommands(+1.5, +1.5)),
        "pure_left": (PWMCommands(motor_left=-0.5, motor_right=+0.5)),
        "pure_right": (PWMCommands(motor_left=+0.5, motor_right=-0.5)),
        "slight-forward-left": (PWMCommands(motor_left=0, motor_right=0.25)),
        "faster-forward-right": (PWMCommands(motor_left=0.5, motor_right=0)),
        # 'slight-right': (PWMCommands(-0.1, 0.1)),
    }
    dt = 0.03
    t_max = 10

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

    timeseries = {}
    for id_try, commands in tries.items():
        seq = integrate_dynamics(parameters, q0, v0, dt, t_max, commands)

        ground_truth = seq.transform_values(lambda t: SE2Transform.from_SE2(t[0]))
        poses = seq.transform_values(lambda t: t[0])
        velocities = get_velocities_from_sequence(poses)
        linear = velocities.transform_values(linear_from_se2)
        angular = velocities.transform_values(angular_from_se2)
        # print(linear.values)
        # print(angular.values)
        root.set_object(id_try, DB18(), ground_truth=ground_truth)

        sequences = {}
        sequences["motor_left"] = seq.transform_values(lambda _: commands.motor_left)
        sequences["motor_right"] = seq.transform_values(lambda _: commands.motor_right)
        plots = TimeseriesPlot(f"{id_try} - PWM commands", "pwm_commands", sequences)
        timeseries[f"{id_try} - commands"] = plots

        sequences = {}
        sequences["linear_velocity"] = linear
        sequences["angular_velocity"] = angular
        plots = TimeseriesPlot(f"{id_try} - Velocities", "velocities", sequences)
        timeseries[f"{id_try} - velocities"] = plots

    outdir = os.path.join(get_comptests_output_dir(), "together")
    draw_static(root, outdir, timeseries=timeseries)


def linear_from_se2(x: se2v) -> float:
    linear, _ = geo.linear_angular_from_se2(x)
    return linear[0]


def angular_from_se2(x: se2v) -> float:
    _, angular = geo.linear_angular_from_se2(x)
    return angular


def integrate_dynamics(
    factory, q0, v0, dt, t_max, fixed_commands
) -> SampledSequence[TSE2v]:
    # starting time
    t0 = 0
    c0 = q0, v0
    state = factory.initialize(c0=c0, t0=t0)
    ssb = SampledSequenceBuilder[TSE2v]()
    ssb.add(t0, state.TSE2_from_state())
    n = int(t_max / dt)
    for i in range(n):
        state = state.integrate(dt, fixed_commands)
        t = t0 + (i + 1) * dt
        ssb.add(t, state.TSE2_from_state())
    return ssb.as_sequence()


if __name__ == "__main__":
    run_module_tests()
