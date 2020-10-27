# coding=utf-8
from dataclasses import dataclass

import geometry as geo
import numpy as np

from .dynamics_delay import ApplyDelay
from .generic_kinematics import GenericKinematicsSE2
from .platform_dynamics import PlatformDynamicsFactory
from .types import TSE2value

__all__ = ["DynamicModelParameters", "DynamicModel", "PWMCommands", "get_DB18_nominal"]


@dataclass
class PWMCommands:
    """
        PWM commands are floats between -1 and 1.
    """

    motor_left: float
    motor_right: float


class DynamicModelParameters(PlatformDynamicsFactory):
    wheel_radius_left: float
    wheel_radius_right: float
    wheel_distance: float
    encoder_resolution_rad: float

    def __init__(self, u1, u2, u3, w1, w2, w3, uar, ual, war, wal):
        # parameters for autonomous dynamics
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # parameters for forced dynamics
        self.u_alpha_r = uar
        self.u_alpha_l = ual
        self.w_alpha_r = war
        self.w_alpha_l = wal
        R = 0.067 / 2  # 6.7 cm diameter
        D = 0.1  # 10 cm
        self.wheel_radius_left = R
        self.wheel_radius_right = R
        self.wheel_distance = D
        ticks = 180  # XXX
        res = (np.pi * 2) / ticks
        self.encoder_resolution_rad = res

    def initialize(self, c0, t0: float = 0, seed: int = None) -> "DynamicModel":
        return DynamicModel(self, c0, t0, 0.0, 0.0)


def get_DB18_nominal(delay: float) -> PlatformDynamicsFactory:
    # parameters for autonomous dynamics
    u1 = 5
    u2 = 0
    u3 = 0
    w1 = 4
    w2 = 0
    w3 = 0
    # parameters for forced dynamics
    uar = 1.5
    ual = 1.5
    war = 15  # modify this for trim
    wal = 15

    parameters = DynamicModelParameters(
        u1=u1, u2=u2, u3=u3, w1=w1, w2=w2, w3=w3, uar=uar, ual=ual, war=war, wal=wal
    )

    if delay > 0:
        delayed = ApplyDelay(parameters, delay, PWMCommands(0, 0))
        return delayed
    else:
        return parameters


def get_DB18_uncalibrated(delay: float, trim: float = 0) -> PlatformDynamicsFactory:
    # parameters for autonomous dynamics
    u1 = 5
    u2 = 0
    u3 = 0
    w1 = 4
    w2 = 0
    w3 = 0
    # parameters for forced dynamics
    uar = 1.5
    ual = 1.5
    war = 15 * (1.0 + trim)
    wal = 15 * (1.0 - trim)

    parameters = DynamicModelParameters(
        u1=u1, u2=u2, u3=u3, w1=w1, w2=w2, w3=w3, uar=uar, ual=ual, war=war, wal=wal
    )

    if delay > 0:
        delayed = ApplyDelay(parameters, delay, PWMCommands(0, 0))
        return delayed
    else:
        return parameters


class DynamicModel(GenericKinematicsSE2):
    """
        This represents a dynamical formulation of of a differential-drive vehicle.
    """

    parameters: DynamicModelParameters

    axis_left_rad: float
    axis_right_rad: float
    axis_left_obs_rad: float
    axis_right_obs_rad: float

    def __init__(
        self,
        parameters: DynamicModelParameters,
        c0: TSE2value,
        t0: float,
        axis_left_rad: float,
        axis_right_rad: float,
    ):
        self.parameters = parameters
        GenericKinematicsSE2.__init__(self, c0, t0)

        self.axis_left_rad = axis_left_rad
        self.axis_right_rad = axis_right_rad
        left_ticks = int(np.round(axis_left_rad / parameters.encoder_resolution_rad))
        right_ticks = int(np.round(axis_right_rad / parameters.encoder_resolution_rad))
        self.axis_left_obs_rad = left_ticks * parameters.encoder_resolution_rad
        self.axis_right_obs_rad = right_ticks * parameters.encoder_resolution_rad

    @staticmethod
    def model(commands: PWMCommands, parameters: DynamicModelParameters, u=None, w=None):
        """ Returns the second derivative of x"""
        ## Unpack Inputs
        U = np.array([commands.motor_right, commands.motor_left])
        V = U.reshape(U.size, 1)
        V = np.clip(V, -1, +1)
        # parameters for autonomous dynamics
        u1 = parameters.u1
        u2 = parameters.u2
        u3 = parameters.u3
        w1 = parameters.w1
        w2 = parameters.w2
        w3 = parameters.w3
        # parameters for forced dynamics
        u_alpha_r = parameters.u_alpha_r
        u_alpha_l = parameters.u_alpha_l
        w_alpha_r = parameters.w_alpha_r
        w_alpha_l = parameters.w_alpha_l

        ## Calculate Dynamics
        # nonlinear Dynamics - autonomous response
        f_dynamic = np.array([[-u1 * u - u2 * w + u3 * w ** 2], [-w1 * w - w2 * u - w3 * u * w]])  #
        # input Matrix
        B = np.array([[u_alpha_r, u_alpha_l], [w_alpha_r, -w_alpha_l]])  #
        # forced response
        f_forced = np.matmul(B, V)
        # acceleration
        x_dot_dot = f_dynamic + f_forced

        return x_dot_dot

    def integrate(self, dt: float, commands: PWMCommands) -> "DynamicModel":
        # previous velocities (v0)
        linear_angular_prev = geo.linear_angular_from_se2(self.v0)
        linear_prev = linear_angular_prev[0]
        longit_prev = linear_prev[0]
        lateral_prev = linear_prev[1]
        angular_prev = linear_angular_prev[1]

        # predict the acceleration of the vehicle
        x_dot_dot = self.model(commands, self.parameters, u=longit_prev, w=angular_prev)

        # convert the acceleration to velocity by forward euler
        longitudinal = longit_prev + dt * x_dot_dot[0]
        angular = angular_prev + dt * x_dot_dot[1]
        lateral = 0.0

        linear = [longitudinal, lateral]

        # represent this as se(2)
        commands_se2 = geo.se2_from_linear_angular(linear, angular)

        # call the "integrate" function of GenericKinematicsSE2
        s1 = GenericKinematicsSE2.integrate(self, dt, commands_se2)

        # new state
        c1 = s1.q0, s1.v0
        t1 = s1.t0

        # now we compute the axis rotation using the inverse way...
        # forward = both wheels spin positive
        # angular_velocity = wR/d - Wl/d   # if R rotates more, we increase theta
        # linear_velocity = (wR*R_r + Wl*R_l)/2

        # that is
        # [ang, lin ] = [ Rr/d -Rl/d; Rr/2 Rl/2]  [wR wL]
        d = self.parameters.wheel_distance
        Rr = self.parameters.wheel_radius_right
        Rl = self.parameters.wheel_radius_left
        M = np.array([[Rr / d, -Rl / d], [Rr / 2, Rl / 2]])
        anglin = np.array((angular, longitudinal))
        MInv = np.linalg.inv(M)
        wRL = MInv @ anglin
        wR = float(wRL[0, 0])
        wL = float(wRL[1, 0])

        axis_left_rad = self.axis_left_rad + wL * dt
        axis_right_rad = self.axis_right_rad + wR * dt

        return DynamicModel(
            self.parameters, c1, t1, axis_left_rad=axis_left_rad, axis_right_rad=axis_right_rad
        )
