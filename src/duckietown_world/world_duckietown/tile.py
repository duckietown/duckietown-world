# coding=utf-8
from duckietown_world.geo import PlacedObject

__all__ = [
    'Tile',
]

class Tile(PlacedObject):
    def __init__(self, kind, drivable, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.kind = kind
        self.drivable = drivable

    def params_to_json_dict(self):
        return dict(kind=self.kind, drivable=self.drivable)

