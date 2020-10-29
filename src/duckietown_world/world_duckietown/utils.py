import geometry as geo
import numpy as np
from geometry import SE2, SE2value

from duckietown_world import SampledSequence
from duckietown_world.seqs.tsequence import SampledSequenceBuilder


def get_velocities_from_sequence(s: SampledSequence[SE2value]) -> SampledSequence[SE2value]:
    ssb = SampledSequenceBuilder[SE2value]()
    ssb.add(0, geo.se2.zero())
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
