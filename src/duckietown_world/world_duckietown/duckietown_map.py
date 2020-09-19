# coding=utf-8
from typing import cast

from duckietown_world.geo import PlacedObject, SE2Transform

__all__ = ["DuckietownMap"]


class DuckietownMap(PlacedObject):
    tile_size: float

    def __init__(self, tile_size: float, *args, **kwargs):
        self.tile_size = tile_size
        PlacedObject.__init__(self, *args, **kwargs)

    def params_to_json_dict(self):
        return dict(tile_size=self.tile_size)

    def se2_from_curpos(self, cur_pos, cur_angle):
        """ Conversion from Duckietown Gym Simulator coordinates z"""
        from duckietown_world import TileMap

        tilemap: TileMap = cast(TileMap, self.children["tilemap"])
        H = tilemap.H
        gx, gy, gz = cur_pos
        p = [gx, (H - 1) * self.tile_size - gz]
        transform = SE2Transform(p, cur_angle)
        return transform

    def get_drawing_children(self):
        children = sorted(self.children)
        children.remove("tilemap")
        return ["tilemap"] + children
