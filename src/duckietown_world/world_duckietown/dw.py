# coding=utf-8
from ..geo import PlacedObject

from duckietown_serialization import Serializable

__all__ = ['DuckietownWorld']


class DuckietownWorld(Serializable):
    def __init__(self, root=None):
        if root is None:
            root = PlacedObject()
        self.root = root

    def params_to_json_dict(self):
        return dict(root=self.root)
