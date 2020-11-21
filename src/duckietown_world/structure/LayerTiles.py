from abc import ABC
from typing import Union

from . import AbstractLayer

import duckietown_world.world_duckietown as wd
from duckietown_world.world_duckietown.tile_template import load_tile_types
from duckietown_world.world_duckietown.tile import Tile
from duckietown_world.geo import PlacedObject, SE2Transform, Scale2D


class LayerTiles(AbstractLayer, ABC):
    def __init__(self, data: dict, dm: 'DuckietownMap'):
        super().__init__()
        if "tile_maps" not in dm:
            msg = "must load tile_maps before tiles"
            raise ValueError(msg)

        for name, desc in data.items():
            map_name, tile_name = name.split("/")
            assert map_name in dm.tile_maps
            dm.tile_maps[map_name]["tiles"][name] = desc
        for _, tile_map in dm.tile_maps.items():
            tiles = tile_map["tiles"]
            assert len(tiles) > 1
            A = max(map(lambda t: tiles[t]["j"], tiles)) + 1
            B = max(map(lambda t: tiles[t]["i"], tiles)) + 1
            tm = wd.TileMap(H=B, W=A)  # TODO k-coordinate

            rect_checker = []
            for i in range(A):
                rect_checker.append([0] * B)
            for _, t in tiles.items():
                rect_checker[t["j"]][t["i"]] += 1
            for j in range(A):
                for i in range(B):
                    if rect_checker[j][i] == 0:
                        msg = "missing tile at pose " + str([i, j, 0])
                        raise ValueError(msg)
                    if rect_checker[j][i] >= 2:
                        msg = "duplicated tile at pose " + str([i, j, 0])
                        raise ValueError(msg)

            templates = load_tile_types()

            DEFAULT_ORIENT = "E"
            for tile_name, tile_data in tiles.items():
                kind = tile_data["type"]
                if "orientation" in tile_data:
                    orient = tile_data["orientation"]
                    drivable = True
                else:
                    orient = DEFAULT_ORIENT
                    drivable = (kind == "4way")

                tile = Tile(kind=kind, drivable=drivable)
                if kind in templates:
                    tile.set_object(kind, templates[kind], ground_truth=SE2Transform.identity())

                tm.add_tile(tile_data["i"], (A - 1) - tile_data["j"], orient, tile)
                self._items[tile_name] = tile

            wrapper = PlacedObject()
            wrapper.set_object("tilemap", tm, ground_truth=Scale2D(tile_map["map_object"].tile_size))
            tile_map["map_object"].set_object("tilemap_wrapper", wrapper, ground_truth=tile_map["frame"]["transform"])

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerTiles':
        return LayerTiles(data, dm)
