# coding=utf-8
from typing import Tuple

from duckietown_world.world_duckietown.tile_coords import ALLOWED_ORIENTATIONS
from ..geo import PlacedObject
from ..world_duckietown import Tile, TileCoords

__all__ = ["TileMap", "ij_from_tilename", "tilename_from_ij"]


def tilename_from_ij(i: int, j: int):
    tile_name = "tile-%d-%d" % (i, j)
    return tile_name


def ij_from_tilename(tilename: str) -> Tuple[int, int]:
    tokens = tilename.split("-")
    i, j = int(tokens[1]), int(tokens[2])
    return i, j


class TileMap(PlacedObject):
    H: int  # i
    W: int  # j

    def __init__(self, H: int, W: int, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.H = H
        self.W = W

    def params_to_json_dict(self):
        return dict(H=self.H, W=self.W)

    def __getitem__(self, coords: Tuple[int, int]) -> Tile:
        tilename = tilename_from_ij(coords[0], coords[1])
        if not tilename in self.children:
            msg = 'Tile "%s" not available' % coords.__repr__()
            raise KeyError(msg)
        return self.children[coords]

    def add_tile(self, i: int, j: int, orientation: str, tile: Tile, can_be_outside: bool = False):
        if not can_be_outside:
            assert 0 <= i < self.H, (i, self.H)
            assert 0 <= j < self.W, (j, self.W)

        assert orientation in ALLOWED_ORIENTATIONS, orientation
        tile_name = "tile-%d-%d" % (i, j)
        placement = TileCoords(i, j, orientation)
        self.set_object(tile_name, tile, ground_truth=placement)

    def get_drawing_children(self):
        def key(x):
            if x.startswith("tile"):
                return 0, x
            else:
                return 1, x

        return sorted(self.children, key=key)
