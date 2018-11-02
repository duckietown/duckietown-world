# coding=utf-8
import numpy as np
from contracts import contract

import geometry as geo

__all__ = [
    'SE2_interpolate',
    'SE2_apply_R2',
]


@contract(q0='SE2', q1='SE2', alpha='int|(float,finite)')
def SE2_interpolate(q0, q1, alpha):
    alpha = float(alpha)
    v = geo.SE2.algebra_from_group(geo.SE2.multiply(geo.SE2.inverse(q0), q1))
    vi = v * alpha
    q = np.dot(q0, geo.SE2.group_from_algebra(vi))
    return q


@contract(q='SE2', p='R2')
def SE2_apply_R2(q, p):
    pp = [p[0], p[1], 1]
    res = np.dot(q, pp)
    return res[:2]
