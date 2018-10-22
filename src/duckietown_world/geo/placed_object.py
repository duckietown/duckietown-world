# coding=utf-8
from duckietown_world.seqs import Sequence

__all__ = ['ChildObject', 'PlacedObject']

from duckietown_serialization import Serializable


class ChildObject(Serializable):

    def __init__(self, child, transforms=None):
        self.child = child
        if transforms is None:
            transforms = {}
        self.transforms = transforms
        for k, v in self.transforms.items():
            assert isinstance(v, Sequence), type(v)

    def params_to_json_dict(self):
        res = dict(child=self.child)
        if self.transforms:
            res['transforms'] = self.transforms
        return res


class PlacedObject(Serializable):
    def __init__(self, children=None):
        # name of the frame to Transform object
        if children is None:
            children = {}

        self.children = children

    def get_object_from_fqn(self, fqn):
        if fqn == ():
            return self
        first, rest = fqn[0], fqn[1:]
        if first in self.children:
            return self.children[first].child.get_object_from_fqn(rest)
        else:
            msg = 'Cannot find child %s in %s' % (first, list(self.children))
            raise KeyError(msg)

    def params_to_json_dict(self):
        if self.children:
            return dict(children=self.children)
        else:
            return None

    def set_object(self, name, ob, **transforms):
        assert self is not ob
        po = ChildObject(ob, transforms)
        self.children[name] = po
