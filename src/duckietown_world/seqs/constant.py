# coding=utf-8
from .tsequence import Sequence
from duckietown_serialization import Serializable

__all__ = ['Constant']


class Constant(Sequence, Serializable):
    def __init__(self, always):
        self.always = always

    def at(self, t):
        return self.always

    def params_to_json_dict(self):
        return dict(always=self.always)
