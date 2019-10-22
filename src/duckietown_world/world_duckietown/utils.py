import geometry as geo
from duckietown_world import SampledSequence
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from duckietown_world.world_duckietown.types import SE2v, se2v

from geometry import SE2


def get_velocities_from_sequence(s: SampledSequence[SE2v]) -> SampledSequence[se2v]:
    ssb = SampledSequenceBuilder[se2v]()
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


def velocity_from_poses(t1: float, q1: SE2v, t2: float, q2: SE2v) -> se2v:
    delta = t2 - t1
    if not delta > 0:
        raise ValueError("invalid sequence")

    x = SE2.multiply(SE2.inverse(q1), q2)
    xt = SE2.algebra_from_group(x)
    v = xt / delta
    return v
