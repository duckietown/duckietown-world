from abc import ABC
from typing import Union

from .AbstractLayer import AbstractLayer

from duckietown_world.world_duckietown import map_loading


class LayerWatchtowers(AbstractLayer, ABC):
    def __init__(self, data: dict):
        super().__init__()
        for name, desc in data.items():
            kind = "watchtower"
            obj_name = "ob%02d-%s" % (map_loading.obj_idx(), kind)
            desc["kind"] = kind
            obj = map_loading.get_object(desc)
            self._items[name] = {"name": name, "obj_name": obj_name, "obj": obj, "frame": None}

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerWatchtowers':
        return LayerWatchtowers(data)
