import os

import geometry as geo
import numpy as np
import yaml
from comptests import comptest, get_comptests_output_dir, run_module_tests
from zuper_commons.types import check_isinstance

from duckietown_world import (
    construct_map,
    DB18,
    draw_static,
    PlatformDynamics,
    PWMCommands,
    SampledSequence,
    SampledSequenceBuilder,
    SE2Transform,
)
from duckietown_world.svg_drawing.misc import TimeseriesPlot
from duckietown_world.world_duckietown.pwm_dynamics import DynamicModel, get_DB18_nominal
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

    map_data = yaml.load(map_data_yaml)

    root = construct_map(map_data)

    timeseries = {}
    for id_try, commands in tries.items():
        seq = integrate_dynamics(parameters, q0, v0, dt, t_max, commands)

        # return getattr(x, 'axis_left_ticks', None)

        ground_truth = seq.transform_values(get_SE2transform)
        poses = seq.transform_values(get_pose)
        velocities = get_velocities_from_sequence(poses)
        linear = velocities.transform_values(linear_from_se2)
        angular = velocities.transform_values(angular_from_se2)

        odo_left = seq.transform_values(odometry_left)
        odo_right = seq.transform_values(odometry_right)
        root.set_object(id_try, DB18(), ground_truth=ground_truth)

        sequences = {}
        sequences["motor_left"] = seq.transform_values(lambda _: commands.motor_left)
        sequences["motor_right"] = seq.transform_values(lambda _: commands.motor_right)
        plots = TimeseriesPlot(f"PWM commands", "pwm_commands", sequences)
        timeseries[f"{id_try}/commands"] = plots

        sequences = {}
        sequences["linear_velocity"] = linear
        sequences["angular_velocity"] = angular
        plots = TimeseriesPlot(f"Velocities", "velocities", sequences)
        timeseries[f"{id_try}/velocities"] = plots

        sequences = {}
        sequences["theta"] = poses.transform_values(lambda x: geo.translation_angle_from_SE2(x)[1])

        plots = TimeseriesPlot(f"Pose", "theta", sequences)
        timeseries[f"{id_try}/pose"] = plots

        sequences = {}
        sequences["odo_left"] = odo_left
        sequences["odo_right"] = odo_right
        plots = TimeseriesPlot(f"Odometry", "odometry", sequences)
        timeseries[f"{id_try}/odometry"] = plots

    outdir = os.path.join(get_comptests_output_dir(), "together")
    draw_static(root, outdir, timeseries=timeseries)


def get_SE2transform(x: PlatformDynamics) -> SE2Transform:
    check_isinstance(x, PlatformDynamics)
    q, v = x.TSE2_from_state()
    return SE2Transform.from_SE2(q)


def get_pose(x: PlatformDynamics) -> geo.SE2value:
    check_isinstance(x, PlatformDynamics)
    q, v = x.TSE2_from_state()
    return q


def odometry_left(x: DynamicModel) -> float:
    check_isinstance(x, DynamicModel)
    return x.axis_left_obs_rad


def odometry_right(x: DynamicModel) -> float:
    check_isinstance(x, PlatformDynamics)
    return x.axis_right_obs_rad


def linear_from_se2(x: geo.se2value) -> float:
    linear, _ = geo.linear_angular_from_se2(x)
    return linear[0]


def angular_from_se2(x: geo.se2value) -> float:
    _, angular = geo.linear_angular_from_se2(x)
    return angular


def integrate_dynamics(
    factory, q0, v0, dt: float, t_max: float, fixed_commands
) -> SampledSequence[PlatformDynamics]:
    # starting time
    t0 = 0
    c0 = q0, v0
    state = factory.initialize(c0=c0, t0=t0)
    ssb: SampledSequenceBuilder[PlatformDynamics] = SampledSequenceBuilder[PlatformDynamics]()
    ssb.add(t0, state)
    n = int(t_max / dt)
    for i in range(n):
        state = state.integrate(dt, fixed_commands)
        t = t0 + (i + 1) * dt
        ssb.add(t, state)
    return ssb.as_sequence()


if __name__ == "__main__":
    run_module_tests()
