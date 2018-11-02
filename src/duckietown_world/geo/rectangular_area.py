# coding=utf-8
import numpy as np

from duckietown_serialization_ds1 import Serializable
from .region import Region

__all__ = [
    'RectangularArea',
]


class RectangularArea(Serializable, Region):
    def __init__(self, pmin, pmax):
        self.pmin = np.array(pmin, dtype=np.float64)
        self.pmax = np.array(pmax, dtype=np.float64)

        if not np.all(self.pmin < self.pmax):
            msg = 'Invalid area: %s %s' % (pmin, pmax)
            raise ValueError(msg)

    @classmethod
    def join(cls, a, b):
        pmin = np.minimum(a.pmin, b.pmin)
        pmax = np.maximum(a.pmax, b.pmax)
        return RectangularArea(pmin, pmax)

    def contains(self, p):
        pmin, pmax = self.pmin, self.pmax
        return (pmin[0] <= p[0] <= pmax[0]) and (pmin[1] <= p[1] <= pmax[1])
