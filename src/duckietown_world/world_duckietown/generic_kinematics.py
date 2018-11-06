# coding=utf-8
import geometry as geo
from duckietown_serialization_ds1 import Serializable
from .platform_dynamics import PlatformDynamicsFactory, PlatformDynamics

__all__ = [
    'GenericKinematicsSE2',

]

# noinspection PyUnresolvedReferences
from geometry.poses import *

from contracts import contract


class GenericKinematicsSE2(PlatformDynamicsFactory, PlatformDynamics, Serializable):
    """
        Any dynamics on SE(2)

        Commands = velocities in se(2)
    """

    @classmethod
    @contract(c0='TSE2')
    def initialize(cls, c0, t0=0, seed=None):
        return GenericKinematicsSE2(c0, t0)

    @contract(c0='TSE2')
    def __init__(self, c0, t0):
        # start at q0, v0
        q0, v0 = c0
        geo.SE2.belongs(q0)
        geo.se2.belongs(v0)
        self.t0 = t0
        self.v0 = v0
        self.q0 = q0

    def integrate(self, dt, commands):
        """ commands = velocity in body frame """
        # convert to float
        dt = float(dt)
        # the commands must belong to se(2)
        geo.se2.belongs(commands)
        v = commands
        # suppose we hold v for dt, which pose are we going to?
        diff = geo.SE2.group_from_algebra(dt * v) # exponential map
        # compute the absolute new pose; applying diff from q0
        q1 = geo.SE2.multiply(self.q0, diff)
        # the new configuration
        c1 = q1, v
        # the new time
        t1 = self.t0 + dt
        # return the new state
        return GenericKinematicsSE2(c1, t1)

    @contract(returns='TSE2')
    def TSE2_from_state(self):
        return self.q0, self.v0
