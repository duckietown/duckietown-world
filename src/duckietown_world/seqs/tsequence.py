# coding=utf-8
from abc import ABCMeta, abstractmethod

from duckietown_serialization import Serializable

__all__ = ['Sequence']


class Sequence(Serializable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def at(self, t):
        pass
