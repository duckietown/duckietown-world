# coding=utf-8
from __future__ import unicode_literals


__all__ = ['PlacedObject']

from duckietown_serialization_ds1 import Serializable


class SpatialRelation(Serializable):
    SR_TYPE_PRIOR = 'prior'
    SR_TYPE_GT = 'ground_truth'
    SR_TYPE_MEASUREMENT = 'measurement'
    SR_TYPES = [SR_TYPE_PRIOR, SR_TYPE_GT, SR_TYPE_MEASUREMENT]

    def __init__(self, a, transform, b, sr_type):
        # check_isinstance(transform, (SpatialRelation, Sequence))
        if sr_type not in SpatialRelation.SR_TYPES:
            msg = 'Invalid value %s' % sr_type
            raise ValueError(msg)
        self.a = tuple(a)
        self.transform = transform
        self.b = tuple(b)
        self.sr_type = sr_type

    def filter_all(self, f):
        return SpatialRelation(self.a, f(self.transform), self.b, sr_type=self.sr_type)


import copy

#
# class MyDict(UserDict):
#
#     def __getitem__(self, key):
#         try:
#             return UserDict.__getitem__(self, key)
#         except KeyError:
#             msg = 'Cannot find key "{}"; I know {}.'.format(key, sorted(self.data))
#             raise KeyError(msg)


class PlacedObject(Serializable):
    def __init__(self, children=None, spatial_relations=None):
        # name of the frame to Transform object
        if children is None:
            children = {}

        if spatial_relations is None:
            spatial_relations = {}

        self.children = children
        self.spatial_relations = spatial_relations
        # self.children = MyDict(**children)
        # self.spatial_relations = MyDict(**spatial_relations)

    def filter_all(self, f):
        x = copy.deepcopy(self)

        x.children = dict((k, f(v.filter_all(f))) for (k, v) in self.children.items())
        x.spatial_relations = dict((k, f(v.filter_all(f))) for (k, v) in self.spatial_relations.items())
        return x
        # klass = type(self)
        # params = self.params_to_json_dict()
        # return f(klass(**params))

    def get_object_from_fqn(self, fqn):
        if fqn == ():
            return self
        first, rest = fqn[0], fqn[1:]
        if first in self.children:
            return self.children[first].get_object_from_fqn(rest)
        else:
            msg = 'Cannot find child %s in %s' % (first, list(self.children))
            raise KeyError(msg)

    def params_to_json_dict(self):
        res = {}

        if self.children:
            res['children'] = self.children
        if self.spatial_relations:
            res['spatial_relations'] = self.spatial_relations

        return res

    def set_object(self, name, ob, **transforms):
        assert self is not ob
        self.children[name] = ob
        for k, v in transforms.items():
            st = SpatialRelation(a=(), b=(name,), sr_type=k, transform=v)
            i = len(self.spatial_relations)
            self.spatial_relations[i] = st

    # @abstractmethod
    def draw_svg(self, drawing, g):
        pass
        # print('draw_svg not implemented for %s' % type(self).__name__)

    def get_drawing_children(self):
        return sorted(self.children)
