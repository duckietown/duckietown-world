# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import NewType

import numpy as np
from duckietown_serialization_ds1 import Serializable

import geometry as geo
from contracts import contract, new_contract
from duckietown_world.seqs.tsequence import Timestamp

__all__ = ["TransformSequence", "Transform", "SE2Transform", "Scale2D", "Matrix2D"]

SE2value = NewType("SE2value", np.ndarray)


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def asmatrix2d(self) -> "Matrix2D":
        """ """

    def as_SE2(self):
        m = self.asmatrix2d().m
        try:
            geo.SE2.belongs(m)
        except:
            msg = "Cannot convert %s" % type(self).__name__
            msg += "m = %s" % m
            raise Exception(msg)
        return m


new_contract("Transform", Transform)


class TransformSequence(Transform):

    # @contract(transforms='list[>=1](Transform)')
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def asmatrix2d(self):
        ms = [_.asmatrix2d() for _ in self.transforms]
        # print('ms: %s' % ms)
        result = ms[0].m

        for mi in ms[1:]:
            result = np.dot(result, mi.m)
        return Matrix2D(result)

    def __repr__(self):
        return "TransformSequence(%s)" % self.transforms


from duckietown_world.seqs import GenericSequence


class VariableTransformSequence(TransformSequence, GenericSequence[Transform]):  # XXX
    def at(self, t: Timestamp):
        res = []
        for transform in self.transforms:
            if isinstance(transform, GenericSequence):
                transform = transform.at(t)
            res.append(transform)
        return TransformSequence(res)

    def get_end(self):
        return None

    def get_start(self):
        return None

    def get_sampling_points(self):
        pass


class SE2Transform(Transform, Serializable):
    @contract(p="seq[2](float|int)")
    def __init__(self, p, theta):
        self.p = np.array(p, dtype="float64")
        self.theta = float(theta)

    def __repr__(self):
        return "SE2Transform(%s,%s)" % (self.p.tolist(), self.theta)

    @classmethod
    def identity(cls) -> "SE2Transform":
        return SE2Transform([0.0, 0.0], 0.0)

    @classmethod
    def from_SE2(cls, q: SE2value) -> "SE2Transform":
        """ From a matrix """
        translation, angle = geo.translation_angle_from_SE2(q)
        return SE2Transform(translation, angle)

    def params_to_json_dict(self):
        res = {}
        if np.allclose(self.theta, 0):
            pass
        else:
            for a in [-270, -180, -90, -45, +45, 90, 180, 270]:
                if np.allclose(a, np.rad2deg(self.theta)):
                    res["theta_deg"] = a
                    break
            else:
                res["theta"] = self.theta
        if np.allclose(np.linalg.norm(self.p), 0):
            pass
        else:
            res["p"] = self.p
        return res

    @classmethod
    def params_from_json_dict(cls, d):
        if d is None:
            d = {}
        p = d.pop("p", [0.0, 0.0])

        if "theta" in d:
            theta = d.pop("theta")

        elif "theta_deg" in d:
            theta_deg = d.pop("theta_deg")
            theta = np.deg2rad(theta_deg)
        else:
            theta = 0.0

        return dict(p=p, theta=theta)

    def as_SE2(self) -> SE2value:
        import geometry

        M = geometry.SE2_from_translation_angle(self.p, self.theta)
        return M

    def asmatrix2d(self):
        M = self.as_SE2()
        return Matrix2D(M)


new_contract("SE2Transform", SE2Transform)


class Scale2D(Transform, Serializable):
    def __init__(self, scale):
        self.scale = scale

    def asmatrix2d(self):
        S = self.scale
        M = np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]])
        return Matrix2D(M)


class Matrix2D(Transform, Serializable):
    def __init__(self, m):
        self.m = np.array(m, "float32")
        assert self.m.shape == (3, 3)

    def asmatrix2d(self):
        return self
