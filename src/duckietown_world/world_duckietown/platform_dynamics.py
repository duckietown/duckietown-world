# coding=utf-8
from abc import ABCMeta, abstractmethod

from contracts import contract
from six import with_metaclass

__all__ = [
    'PlatformDynamicsFactory',
    'PlatformDynamics',
]


class PlatformDynamics(with_metaclass(ABCMeta)):

    @abstractmethod
    @contract(dt='>0')
    def integrate(self, dt, commands):
        """
            Returns the result of applying commands for dt.

            :param dt: time interval
            :param commands: class-specific commands
            :return: None
        """

    @abstractmethod
    @contract(returns='TSE2')
    def TSE2_from_state(self):
        """ Returns pose, velocity for the state. """


class PlatformDynamicsFactory(with_metaclass(ABCMeta)):

    @classmethod
    @abstractmethod
    @contract(c0='TSE2', returns=PlatformDynamics)
    def initialize(cls, c0, t0=0, seed=None):
        """
            Returns the dynamics initalized at a certain configuration.

            :param c0: configuration in TSE2
            :param t0: time in which to sstart
            :param seed: seed for a possible random number generator

        """
