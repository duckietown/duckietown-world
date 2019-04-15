# coding=utf-8

from contracts import check_isinstance
import numpy as np
import geometry as geo
from duckietown_serialization_ds1 import Serializable
from .generic_kinematics import GenericKinematicsSE2
from .platform_dynamics import PlatformDynamicsFactory

__all__ = [
    'WheelVelocityCommands',
    'DifferentialDriveDynamicsParameters',
    'DifferentialDriveDynamics',
]


class WheelVelocityCommands(Serializable):
    '''
        This represents the velocity commands for differential drive
        kinematics.

        These are expressed in rad/s.

        For both wheels, positive = moves forward.

    '''

    def __init__(self, left_wheel_angular_velocity, right_wheel_angular_velocity):
        self.left_wheel_angular_velocity = left_wheel_angular_velocity
        self.right_wheel_angular_velocity = right_wheel_angular_velocity


class DifferentialDriveDynamicsParameters(PlatformDynamicsFactory, Serializable):
    '''
        This class represents the parameters of the ideal differential drive dynamics.

        radius_left, radius_right: wheels radii
        wheel_distance: distance between two wheels

    '''

    def __init__(self, radius_left, radius_right, wheel_distance):
        self.radius_left = radius_left
        self.radius_right = radius_right
        self.wheel_distance = wheel_distance

    def initialize(self, c0, t0=0, seed=None):
        return DifferentialDriveDynamics(self, c0, t0)


class DifferentialDriveDynamics(GenericKinematicsSE2):
    """
        This represents the state of differential drive.

        This is a particular case of GenericKinematicsSE2.

    """

    def __init__(self, parameters, c0, t0):
        """
        :param parameters:  instance of DifferentialDriveDynamicsParameters
        :param c0: initial configuration
        :param t0: initial time
        """
        check_isinstance(parameters, DifferentialDriveDynamicsParameters)
        self.parameters = parameters
        GenericKinematicsSE2.__init__(self, c0, t0)

    def integrate(self, dt, commands):
        """

        :param dt:
        :param commands: an instance of WheelVelocityCommands
        :return:
        """
        check_isinstance(commands, WheelVelocityCommands)

        # Compute the linear velocity for the wheels
        # by multiplying radius times angular velocity
        v_r = self.parameters.radius_right * commands.right_wheel_angular_velocity
        v_l = self.parameters.radius_left * commands.left_wheel_angular_velocity

        # compute the linear, angular velocities for the platform
        # using the differential drive equations
        longitudinal = (v_r + v_l) * 0.5
        angular = (v_r - v_l) / self.parameters.wheel_distance
        lateral = 0.0

        linear = [longitudinal, lateral]

        # represent this as se(2)
        commands_se2 = geo.se2_from_linear_angular(linear, angular)

        # Call the "integrate" function of GenericKinematicsSE2
        s1 = GenericKinematicsSE2.integrate(self, dt, commands_se2)

        # new state
        c1 = s1.q0, s1.v0
        t1 = s1.t0
        return DifferentialDriveDynamics(self.parameters, c1, t1)

"""
Classes DynamicModelParameters and DynamicModel implement a dynamical model of a
differential-drive vehicle. The derivation will be made available in near future.

For questions, please send an email to: ercans@student.ethz.ch
"""

class DynamicModelParameters(PlatformDynamicsFactory, Serializable):
    '''
        This class represents the parameters of the ideal differential drive dynamics.

        radius_left, radius_right: wheels radii
        wheel_distance: distance between two wheels
    '''

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

    def initialize(self, c0, t0=0, seed=None):
        return DifferentialDriveDynamics(self, c0, t0)


class DynamicModel(GenericKinematicsSE2):
    """
        This represents a dynamical formulation of of a differential-drive vehicle.
    """

    def __init__(self, parameters, c0, t0):
        """
        :param parameters:  instance of DifferentialDriveDynamicsParameters
        :param c0: initial configuration
        :param t0: initial time
        """
        check_isinstance(parameters, DynamicModelParameters)
        self.parameters = parameters
        GenericKinematicsSE2.__init__(self, c0, t0)

    @staticmethod
    def model(input, parameters, u=None, w=None):
        ## Unpack Inputs
        U = np.array([input.right_wheel_angular_velocity, input.left_wheel_angular_velocity])
        V = U.reshape(U.size, 1)

        ## Unpack Parameters
        check_isinstance(parameters, DynamicModelParameters)

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
        f_dynamic = np.array([
            [-u1 * u - u2 * w + u3 * w ** 2],
            [-w1 * w - w2 * u - w3 * u * w]
        ])
        # input Matrix
        B = np.array([
            [u_alpha_r, u_alpha_l],
            [w_alpha_r, -w_alpha_l]
        ])
        # forced response
        f_forced = np.matmul(B, V)
        # acceleration
        x_dot_dot = f_dynamic + f_forced

        return x_dot_dot

    def integrate(self, dt, commands):
        """

        :param dt:
        :param commands: an instance of WheelVelocityCommands
        :return:
        """
        check_isinstance(commands, WheelVelocityCommands)

        # previous velocities (v0)
        linear_angular_prev = linear_angular_from_se2(self.v0)
        linear_prev = linear_angular_prev[0]
        longit_prev = linear_prev[0]
        lateral_prev = linear_prev[1]
        angular_prev = linear_angular_prev[1]

        # predict the acceleration of the vehicle
        x_dot_dot = self.model(commands, parameters, u=longit_prev, w=angular_prev)

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

        return DynamicModel(self.parameters, c1, t1)
