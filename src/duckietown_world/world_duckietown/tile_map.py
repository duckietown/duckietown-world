# coding=utf-8
import itertools

from ..geo import PlacedObject
from ..seqs.constant import Constant
from ..world_duckietown import TileCoords

__all__ = ['TileMap']


class TileMap(PlacedObject):
    def __init__(self, H, W, tile_size, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.H = H
        self.W = W
        self.tile_size = tile_size
        self.ij2tile = {}

        for i, j in itertools.product(range(H), range(W)):
            tile_name = 'tile-%d-%d' % (i, j)
            if tile_name in self.children:
                self.ij2tile[(i, j)] = self.children[tile_name]

    def params_to_json_dict(self):
        return dict(H=self.H, W=self.W, tile_size=self.tile_size)

    def __getitem__(self, coords):
        return self.ij2tile[coords]

    def add_tile(self, i, j, orientation, tile):
        assert orientation in ['S', 'E', 'N', 'W'], orientation
        self.ij2tile[(i, j)] = tile
        tile_name = 'tile-%d-%d' % (i, j)
        placement = Constant(TileCoords(i, j, orientation))
        self.set_object(tile_name, tile, ground_truth=placement)
