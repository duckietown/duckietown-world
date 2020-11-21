from abc import ABC
from typing import Union

from . import AbstractLayer

import duckietown_world.world_duckietown as wd


class LayerTileMaps(AbstractLayer, ABC):
    def __init__(self, data: dict, dm: 'DuckietownMap'):
        super().__init__()
        if "frames" not in dm:
            msg = "must load frames before tile_maps"
            raise ValueError(msg)

        for name, desc in data.items():
            dt_map = wd.DuckietownMap(desc["tile_size"]["x"])  # TODO y-width
            frame = dm.frames[name]
            if frame is None:
                msg = "not found frame for map " + name
                raise ValueError(msg)
            self._items[name] = {"map_object": dt_map, "frame": frame, "tiles": {}}  # TODO: do we need tiles?

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerTileMaps':
        return LayerTileMaps(data, dm)
