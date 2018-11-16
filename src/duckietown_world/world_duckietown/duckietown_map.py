# coding=utf-8
from duckietown_world.geo import PlacedObject, SE2Transform

__all__ = [
    'DuckietownMap',
]


class DuckietownMap(PlacedObject):
    def __init__(self, tile_size, *args, **kwargs):
        self.tile_size = tile_size
        PlacedObject.__init__(self, *args, **kwargs)

    def params_to_json_dict(self):
        return dict(tile_size=self.tile_size)

    def se2_from_curpos(self, cur_pos, cur_angle):
        """ Conversion from Duckietown Gym Simulator coordinates z"""
        H = self.children['tilemap'].H
        gx, gy, gz = cur_pos
        p = [gx, (H - 1) * self.tile_size - gz]
        transform = SE2Transform(p, cur_angle)
        return transform

    def get_drawing_children(self):
        children = sorted(self.children)
        children.remove('tilemap')
        return ['tilemap'] + children
