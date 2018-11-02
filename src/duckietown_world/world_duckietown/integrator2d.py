# coding=utf-8
import numpy as np
from contracts import contract

import geometry as geo
from duckietown_serialization_ds1 import Serializable
from .platform_dynamics import PlatformDynamicsFactory, PlatformDynamics

__all__ = ['Integrator2D']


class Integrator2D(PlatformDynamicsFactory, PlatformDynamics, Serializable):

    @classmethod
    @contract(c0='TSE2')
    def initialize(cls, c0, t0=0, seed=None):
        q, v = c0
        p, theta = geo.translation_angle_from_SE2(q)
        v2d, _ = geo.linear_angular_from_se2(v)
        return Integrator2D(p, v2d, t0)

    @contract(p0='seq[2]')
    def __init__(self, p0, v0, t0):
        self.t0 = t0
        self.v0 = v0
        self.p0 = p0

    def integrate(self, dt, commands):
        dt = float(dt)
        commands = np.array(commands, np.float64)
        p1 = self.p0 + dt * commands
        t1 = self.t0 + dt
        v1 = dt * commands
        return Integrator2D(p1, v1, t1)

    @contract(returns='TSE2')
    def TSE2_from_state(self):
        q = geo.SE2_from_R2(self.p0)
        v = geo.se2.zero()
        return q, v
