from duckietown_serialization_ds1 import Serializable
import numpy as np


class RectangularArea(Serializable):
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
