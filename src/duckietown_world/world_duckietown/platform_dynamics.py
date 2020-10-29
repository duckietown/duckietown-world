# coding=utf-8
from abc import ABCMeta, abstractmethod

from .types import TSE2value

__all__ = ["PlatformDynamicsFactory", "PlatformDynamics"]


class PlatformDynamics(metaclass=ABCMeta):
    """
        This class represents the state of a dynamical system.


            s0 = ...

            s1 = s0.integrate(dt=0.1, commands=[1,1])
            s2 = s0.integrate(dt=0.1, commands=[1,1])


        Each subclass has its own representation of commands.

    """

    @abstractmethod
    def integrate(self, dt: float, commands):
        """
            Returns the result of applying commands for dt.

            :param dt: time interval
            :param commands: class-specific commands
            :return: the next state
        """

    @abstractmethod
    def TSE2_from_state(self) -> TSE2value:
        """ Returns pose, velocity for the state. """


class PlatformDynamicsFactory(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def initialize(cls, c0: TSE2value, t0: float = 0, seed: int = None) -> PlatformDynamics:
        """
            Returns the dynamics initalized at a certain configuration.

            :param c0: configuration in TSE2
            :param t0: time in which to sstart
            :param seed: seed for a possible random number generator

        """
