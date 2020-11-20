from abc import ABC
from typing import Union

from . import AbstractLayer

import duckietown_world.world_duckietown as wd


class LayerTileMaps(AbstractLayer, ABC):
    tile_maps: dict

    def __init__(self, data: dict, **kwargs):
        self.tile_maps = {}
        if "frames" not in kwargs:
            msg = "must load frames before tile_maps"
            raise ValueError(msg)

        for name, desc in data.items():
            dm = wd.DuckietownMap(desc["tile_size"]["x"])  # TODO y-width
            frame = kwargs["frames"].frames.get(name, None)
            if frame is None:
                msg = "not found frame for map " + name
                raise ValueError(msg)
            self.tile_maps[name] = {"map_object": dm, "frame": frame, "tiles": {}}  # TODO: do we need tiles?

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, **kwargs) -> 'LayerTileMaps':
        return LayerTileMaps(data, **kwargs)

    def items(self):
        return self.tile_maps.items()
