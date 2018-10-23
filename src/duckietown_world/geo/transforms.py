# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

import numpy as np
from contracts import contract

from duckietown_serialization_ds1 import Serializable


class Transform(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def asmatrix2d(self):
        pass


class SE2Transform(Transform, Serializable):
    @contract(p='seq(float)')
    def __init__(self, p, theta):
        self.p = np.array(p, dtype='float64')
        self.theta = float(theta)

    def __repr__(self):
        return 'SE2Transform(%s,%s)' % (self.p.tolist(), self.theta)

    @classmethod
    def identity(cls):
        return SE2Transform([0.0, 0.0], 0.0)

    def params_to_json_dict(self):
        return dict(p=self.p, theta=self.theta)

    def asmatrix2d(self):
        import geometry
        M = geometry.SE2_from_translation_angle(self.p, self.theta)
        return Matrix2D(M)


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
