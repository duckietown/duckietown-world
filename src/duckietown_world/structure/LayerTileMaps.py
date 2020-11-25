from abc import ABC
from typing import Union

from .AbstractLayer import AbstractLayer

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

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerTileMaps':
        return LayerTileMaps(data, dm)

    def serialize(self) -> dict:
        yaml_dict = {}
        for item_name, item_data in self._items.items():
            s = item_data["map_object"].tile_size
            yaml_dict[item_name] = {"tile_size": {"x": s, "y": s}}
        return {"tile_maps": yaml_dict}
