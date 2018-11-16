# coding=utf-8
import itertools

from ..geo import PlacedObject
from ..world_duckietown import TileCoords

__all__ = [
    'TileMap',
]


class TileMap(PlacedObject):
    def __init__(self, H, W, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.H = H
        self.W = W

        self.ij2tile = {}

        for i, j in itertools.product(range(H), range(W)):
            tile_name = 'tile-%d-%d' % (i, j)
            if tile_name in self.children:
                self.ij2tile[(i, j)] = self.children[tile_name]

    def params_to_json_dict(self):
        return dict(H=self.H, W=self.W)

    def __getitem__(self, coords):
        if not coords in self.ij2tile:
            msg = 'Tile "%s" not available' % coords.__repr__()
            raise KeyError(msg)
        return self.ij2tile[coords]

    def add_tile(self, i, j, orientation, tile, can_be_outside=False):
        if not can_be_outside:
            assert 0 <= i < self.H, (i, self.H)
            assert 0 <= j < self.W, (j, self.W)

        assert orientation in ['S', 'E', 'N', 'W'], orientation
        self.ij2tile[(i, j)] = tile
        tile_name = 'tile-%d-%d' % (i, j)
        placement = TileCoords(i, j, orientation)
        self.set_object(tile_name, tile, ground_truth=placement)

    def get_drawing_children(self):
        def key(x):
            if x.startswith('tile'):
                return 0, x
            else:
                return 1, x

        return sorted(self.children, key=key)
