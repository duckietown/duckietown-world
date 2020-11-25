from abc import ABC
from typing import Union

from .AbstractLayer import AbstractLayer

from duckietown_world.geo import SE2Transform


class LayerFrames(AbstractLayer, ABC):
    def __init__(self, data: dict):
        super().__init__()
        for name, desc in data.items():
            pose = desc["pose"]
            x = pose.get("x", 0)
            y = pose.get("y", 0)
            theta = pose.get("yaw", 0)
            transform = SE2Transform((x, y), theta)  # TODO z, roll, pitch
            self._items[name] = {"relative_to": desc["relative_to"], "transform": transform, "object": None}

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerFrames':
        return LayerFrames(data)

    def serialize(self) -> dict:
        yaml_dict = {}
        for item_name, item_data in self._items.items():
            t = item_data["transform"]
            pose = {"x": float(t.p[0]), "y": float(t.p[1]), "yaw": t.theta}
            yaml_dict[item_name] = {"relative_to": item_data["relative_to"], "pose": pose}
        return {"frames": yaml_dict}
