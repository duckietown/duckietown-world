# coding=utf-8
import numpy as np
from contracts import contract

import geometry as geo
from duckietown_serialization_ds1 import Serializable
from .platform_dynamics import PlatformDynamicsFactory, PlatformDynamics

__all__ = ['Integrator2D']


class Integrator2D(PlatformDynamicsFactory, PlatformDynamics, Serializable):
    """
        This represents the dynamics of a 2D integrator.

        Commands are sequences of 2 numbers / array of two numbers.

    """

    @classmethod
    @contract(c0='TSE2')
    def initialize(cls, c0, t0=0, seed=None):
        """
            This class initializes the dynamics at a given configuration
        """
        # pose, velocity in SE(2), se(2)
        q, v = c0
        # get position p from pose
        p, theta = geo.translation_angle_from_SE2(q)
        # get linear velocity from se(2)
        v2d, _ = geo.linear_angular_from_se2(v)
        # create the integrator2d initial state
        return Integrator2D(p, v2d, t0)

    @contract(p0='seq[2]')
    def __init__(self, p0, v0, t0):
        """
        :param p0: initial point
        :param v0: initial velocity
        :param t0: initial time
        """
        self.t0 = t0
        self.v0 = v0
        self.p0 = p0

    def integrate(self, dt, commands):
        """
            Returns the next state after applying commands for time dt.

            :param dt: for how long to apply the commands
            :param commands: sequences of two numbers
            :return: another Integrator2D
        """
        # convert things to float, array
        dt = float(dt)
        commands = np.array(commands, np.float64)

        # time incremensts by dt
        t1 = self.t0 + dt
        # we integrate commands
        p1 = self.p0 + dt * commands
        # the velocity is the commands
        v1 = commands
        return Integrator2D(p1, v1, t1)

    @contract(returns='TSE2')
    def TSE2_from_state(self):
        """
            For visualization purposes, this function gets a configuration in SE2
            from the internal state.
        """
        # pose
        q = geo.SE2_from_R2(self.p0)
        # velocity
        linear = self.v0
        angular = 0.0
        v = geo.se2_from_linear_angular(linear, angular)
        return q, v
