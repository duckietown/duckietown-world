from abc import ABC
from typing import Union

from . import AbstractLayer

from duckietown_world.world_duckietown import map_loading


class LayerWatchtowers(AbstractLayer, ABC):
    watchtowers: dict

    def __init__(self, data: dict):
        self.watchtowers = {}
        for name, desc in data.items():
            kind = "watchtower"
            obj_name = "ob%02d-%s" % (map_loading.obj_idx(), kind)
            desc["kind"] = kind
            obj = map_loading.get_object(desc)
            self.watchtowers[obj_name] = {
                "name": name, "obj_name": obj_name, "obj": obj, "frame": None
            }

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, **kwargs) -> 'LayerWatchtowers':
        return LayerWatchtowers(data)

    def items(self):
        return self.watchtowers.items()