# coding=utf-8
import numpy as np
from duckietown_world.geo import Transform

from duckietown_serialization import Serializable

__all__ = ['TileCoords', 'TileRelativeTransform']


class TileCoords(Transform, Serializable):
    def __init__(self, i, j, orientation):
        self.i = i
        self.j = j
        self.orientation = orientation

    def params_to_json_dict(self):
        return dict(i=self.i, j=self.j, orientation=self.orientation)


class TileRelativeTransform(Transform, Serializable):
    def __init__(self, p, z, theta):
        self.p = np.array(p, dtype='float64')
        self.z = z
        self.theta = float(theta)

    def params_to_json_dict(self):
        return dict(p=self.p, z=self.z, theta=self.theta)
