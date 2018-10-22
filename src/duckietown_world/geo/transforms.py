# coding=utf-8
from abc import ABCMeta

import numpy as np
from duckietown_serialization import Serializable

class Transform(object):
    __metaclass__ = ABCMeta


class SE2Transform(Transform, Serializable):
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
