# coding=utf-8
from .tsequence import Sequence

__all__ = [
    'Constant',
]


class Constant(Sequence):
    def __init__(self, always):
        self.always = always

    def at(self, t):
        return self.always

    def get_sampling_points(self):
        return Sequence.CONTINUOUS

    def get_start(self):
        return None

    def get_end(self):
        return None
