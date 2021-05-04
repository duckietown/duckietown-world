# coding=utf-8
import random
from typing import Tuple

import math

import numpy as np
from duckietown_serialization_ds1 import Serializable

from .region import Region

__all__ = ["RectangularArea", "sample_in_rect"]


class RectangularArea(Serializable, Region):
    def __init__(self, pmin, pmax):
        self.pmin = np.array(pmin, dtype=np.float64)
        self.pmax = np.array(pmax, dtype=np.float64)

        if not np.all(self.pmin < self.pmax):
            msg = "Invalid area: %s %s" % (pmin, pmax)
            raise ValueError(msg)

    @classmethod
    def join(cls, a, b):
        pmin = np.minimum(a.pmin, b.pmin)
        pmax = np.maximum(a.pmax, b.pmax)
        return RectangularArea(pmin, pmax)

    def contains(self, p, epsilon=0.0):
        pmin, pmax = self.pmin, self.pmax
        return (pmin[0] - epsilon <= p[0] <= pmax[0] + epsilon) and (
            pmin[1] - epsilon <= p[1] <= pmax[1] + epsilon
        )

    def distance(self, p) -> float:
        pmin, pmax = self.pmin, self.pmax
        abs = math.fabs
        if pmin[0] <= p[0] <= pmax[0]:
            d0 = 0.0
        else:
            d0 = min(abs(pmin[0] - p[0]), abs(pmax[0] - p[0]))

        if pmin[1] <= p[1] <= pmax[1]:
            d1 = 0.0
        else:
            d1 = d0 = min(abs(pmin[1] - p[1]), abs(pmax[1] - p[1]))

        return math.hypot(d0, d1)


def sample_in_rect(a: RectangularArea) -> Tuple[float, float]:
    x = random.uniform(a.pmin[0], a.pmax[0])
    y = random.uniform(a.pmin[1], a.pmax[1])

    return float(x), float(y)
