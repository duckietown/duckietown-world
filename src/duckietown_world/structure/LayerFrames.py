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

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerFrames':
        return LayerFrames(data)
