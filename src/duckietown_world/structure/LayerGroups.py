from abc import ABC
from typing import Union

from . import AbstractLayer


class LayerGroups(AbstractLayer, ABC):
    groups: dict

    def __init__(self, data: dict):
        self.groups = data

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict, **kwargs) -> 'LayerGroups':
        return LayerGroups(data)
