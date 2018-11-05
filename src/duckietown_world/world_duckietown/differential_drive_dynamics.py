# coding=utf-8

from contracts import check_isinstance

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
