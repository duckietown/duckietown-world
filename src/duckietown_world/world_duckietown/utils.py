import geometry as geo
import numpy as np
from geometry import SE2, SE2value

from ..svg_drawing import TimeseriesPlot
from ..seqs import SampledSequence
from ..seqs import SampledSequenceBuilder

from geometry import se2value, linear_angular_from_se2
from typing import Dict

__all__ = [
    "get_velocities_from_sequence",
    "velocity_from_poses",
    "relative_pose",
    "timeseries_robot_velocity",
]


def get_velocities_from_sequence(s: SampledSequence[SE2value]) -> SampledSequence[SE2value]:
    ssb = SampledSequenceBuilder[SE2value]()
    ssb.add(s.get_start(), geo.se2.zero())
    for i in range(1, len(s)):
        t0 = s.timestamps[i - 1]
        t1 = s.timestamps[i]
        q0 = s.values[i - 1]
        q1 = s.values[i]
        v = velocity_from_poses(t0, q0, t1, q1)
        if i == 0:
            ssb.add(t0, v)
        ssb.add(t1, v)
    return ssb.as_sequence()


def velocity_from_poses(t1: float, q1: SE2value, t2: float, q2: SE2value) -> SE2value:
    delta = t2 - t1
    if not delta > 0:
        msg = f"invalid delta {delta}"
        raise ValueError(msg)

    x = SE2.multiply(SE2.inverse(q1), q2)
    xt = SE2.algebra_from_group(x)
    v = xt / delta
    return v


def relative_pose(base: SE2value, pose: SE2value) -> SE2value:
    assert isinstance(base, np.ndarray), base
    assert isinstance(pose, np.ndarray), pose
    return np.dot(np.linalg.inv(base), pose)


def timeseries_robot_velocity(log_velocity: SampledSequence[se2value]) -> Dict[str, TimeseriesPlot]:
    timeseries = {}
    sequences = {}

    # logger.info(log_velocity)

    def speed(x: se2value) -> float:
        l, omega_ = linear_angular_from_se2(x)
        return l[0]

    def omega(x: se2value) -> float:
        l, omega_ = linear_angular_from_se2(x)
        return np.rad2deg(omega_)

    def lateral(x: se2value) -> float:
        l, omega_ = linear_angular_from_se2(x)
        return l[1]

    sequences["angular"] = log_velocity.transform_values(omega, float)
    sequences["longitudinal"] = log_velocity.transform_values(speed, float)
    sequences["lateral"] = log_velocity.transform_values(lateral, float)
    # logger.info("linear speed: %s" % sequences["linear_speed"])
    # logger.info("angular velocity: %s" % sequences["angular_velocity"])
    long_description = """

Velocities in body frame. Given in m/s and deg/s.

    """
    timeseries["velocity"] = TimeseriesPlot("Velocities in body frame", long_description, sequences)
    return timeseries
