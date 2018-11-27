# coding=utf-8
import numpy as np

import geometry
from duckietown_serialization_ds1 import Serializable
from duckietown_world.geo import Transform, Matrix2D

__all__ = [
    'TileCoords',
    'TileRelativeTransform',
]


class TileCoords(Transform, Serializable):

    def asmatrix2d(self):

        angle0 = {
            'N': 0,
            'E': -np.pi/2,
            'S': np.pi,
            'W': +np.pi/2,

        }[self.orientation] + np.pi/2

        angle = {
                    'N': np.pi / 2,
                    'E': 0,
                    'S': np.pi+np.pi/2,
                    'W': np.pi,

                }[self.orientation]

        assert np.allclose(angle, angle0)

        p = [self.i + 0.5, self.j + 0.5]
        M = geometry.SE2_from_translation_angle(p, angle)
        return Matrix2D(M)

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

    def asmatrix2d(self):
        M = geometry.SE2_from_translation_angle(self.p, self.theta)
        return Matrix2D(M)

    def params_to_json_dict(self):
        return dict(p=self.p, z=self.z, theta=self.theta)
