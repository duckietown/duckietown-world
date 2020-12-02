from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Tuple, Dict

from pprint import pformat

from .base_map import IBaseMap
from .objects import _Object, _Frame, _Tile, _Group, _TileMap, _Watchtower


class AbstractLayer(metaclass=ABCMeta):
    dm: IBaseMap = None

    def __init__(self, dm):
        self.dm = dm

    @classmethod
    @abstractmethod
    def item_type(cls) -> type:
        return _Object

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name) -> Optional[_Object]:
        obj_type = self.item_type()
        return self.dm.get_object(name, obj_type)

    @classmethod
    def deserialize(cls, data: dict, dm) -> Tuple['AbstractLayer', Dict[Tuple[str, type], '_Object']]:
        layer = cls(dm)
        obj_type = cls.item_type()
        layer_items = {(name, obj_type): obj_type(**desc, dm=dm) for name, desc in data.items()}
        return layer, layer_items

    def __str__(self):
        obj_type = self.item_type()
        items = self.dm.get_layer_objects(obj_type)
        return pformat(items)


class LayerGeneral(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Object


class LayerFrames(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Frame


class LayerTileMaps(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _TileMap


class LayerTiles(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Tile


class LayerWatchtowers(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Watchtower


class LayerGroups(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Group
