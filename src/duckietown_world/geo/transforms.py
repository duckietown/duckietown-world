# coding=utf-8
# from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

import numpy as np
from contracts import contract, new_contract
from six import with_metaclass

import geometry as geo
from duckietown_serialization_ds1 import Serializable

__all__ = [
    'TransformSequence',
    'Transform',
    'SE2Transform',
    'Scale2D',
    'Matrix2D',
]


class Transform(with_metaclass(ABCMeta)):

    @abstractmethod
    def asmatrix2d(self):
        pass


new_contract('Transform', Transform)


class TransformSequence(object):

    # @contract(transforms='list[>=1](Transform)')
    def __init__(self, transforms):
        self.transforms = transforms

    def asmatrix2d(self):
        ms = [_.asmatrix2d() for _ in self.transforms]
        result = ms[0].m

        for mi in ms[1:]:
            result = np.dot(result, mi.m)
        return Matrix2D(result)


class SE2Transform(Transform, Serializable):
    @contract(p='seq[2](float|int)')
    def __init__(self, p, theta):
        self.p = np.array(p, dtype='float64')
        self.theta = float(theta)

    def __repr__(self):
        return 'SE2Transform(%s,%s)' % (self.p.tolist(), self.theta)

    @classmethod
    def identity(cls):
        return SE2Transform([0.0, 0.0], 0.0)

    @classmethod
    def from_SE2(cls, q):
        """ From a matrix """
        translation, angle = geo.translation_angle_from_SE2(q)
        return SE2Transform(translation, angle)

    def params_to_json_dict(self):
        return dict(p=self.p, theta=self.theta)

    @classmethod
    def params_from_json_dict(cls, d):
        if d is None:
            d = {}
        p = d.pop('p', [0.0, 0.0])

        if 'theta' in d:
            theta = d.pop('theta')

        elif 'theta_deg' in d:
            theta_deg = d.pop('theta_deg')
            theta = np.deg2rad(theta_deg)
        else:
            theta = 0.0

        return dict(p=p, theta=theta)

    def as_SE2(self):
        import geometry
        M = geometry.SE2_from_translation_angle(self.p, self.theta)
        return M

    def asmatrix2d(self):
        M = self.as_SE2()
        return Matrix2D(M)


new_contract('SE2Transform', SE2Transform)


class Scale2D(Transform, Serializable):
    def __init__(self, scale):
        self.scale = scale

    def asmatrix2d(self):
        S = self.scale
        M = np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]])
        return Matrix2D(M)


class Matrix2D(Transform, Serializable):

    def __init__(self, m):
        self.m = np.array(m, 'float32')
        assert self.m.shape == (3, 3)

    def asmatrix2d(self):
        return self
