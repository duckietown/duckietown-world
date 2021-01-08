from dataclasses import dataclass
from typing import Optional, Tuple, List, ItemsView, Dict, Any, Callable, Iterator
from abc import ABCMeta, abstractmethod

from pprint import pformat
from svgwrite.drawing import Drawing as DrawingSVG
from svgwrite.container import Group as GroupSVG


__all__ = ['_Object', '_PlacedObject', '_Pose', '_Frame',
           'ConstructedObject', 'IBaseMap', 'AbstractLayer']


class IBaseMap(metaclass=ABCMeta):
    @abstractmethod
    def get_object_name(self, obj: "_Object") -> str:
        pass

    @abstractmethod
    def get_object(self, name: str, obj_type: type) -> "_Object":
        pass

    @abstractmethod
    def get_object_frame(self, obj: "_Object") -> "_Frame":
        pass

    @abstractmethod
    def get_frame_by_name(self, name: str) -> "_Frame":
        pass

    @abstractmethod
    def get_objects_by_type(self, obj_type: type) -> Dict[Tuple[str, type], "_Object"]:
        pass

    @abstractmethod
    def get_placed_objects(self) -> Dict[Tuple[str, type], "_PlacedObject"]:
        pass

    @abstractmethod
    def get_relative_frames_list(self, frame: "_Frame") -> List["_Frame"]:
        pass

    @abstractmethod
    def apply_operator(self, apply: Callable[["_Object"], "_Object"], exception: type) -> None:
        pass

    @abstractmethod
    def apply_consumer(self, accept: Callable[["_Object"], None], exception: type) -> None:
        pass

    @abstractmethod
    def copy(self) -> "IBaseMap":
        pass


@dataclass
class _Object(metaclass=ABCMeta):
    dm: "IBaseMap" = None

    @abstractmethod
    def dict(self) -> Dict[str, Any]:
        pass

    def copy(self, dm: "IBaseMap") -> "_Object":
        params = self.dict()
        params.update(dm=dm)
        return self.__class__(**params)


class _PlacedObject(_Object, metaclass=ABCMeta):
    @property
    def frame(self) -> "_Frame":
        return self.dm.get_object_frame(self)

    @staticmethod
    def extent_points() -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0.0, 0.1), (0.1, 0.0)

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        pass


@dataclass
class _Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0

    def dict(self) -> Dict[str, Any]:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw}


@dataclass
class _Frame(_Object):
    relative_to: Optional[str] = None
    pose: "_Pose" = _Pose()
    scale: float = 1.0

    def __init__(self, pose: Dict[str, float], relative_to: str = None, dm: "IBaseMap" = None):
        super().__init__(dm=dm)
        self.relative_to = relative_to
        self.pose = _Pose(**pose)
        self.scale = 1.0

    def dict(self) -> Dict[str, Any]:
        return {'relative_to': self.relative_to, 'pose': self.pose.dict()}


class ConstructedObject:
    name: str
    obj: "_Object"
    frame: "_Frame"

    @classmethod
    @abstractmethod
    def object_type(cls) -> type:
        pass

    def __init__(self, name: str, **kwargs):
        self.name = name
        obj_type = self.object_type()
        self.obj = obj_type()
        self.frame = _Frame(kwargs)


class AbstractLayer(metaclass=ABCMeta):
    dm: "IBaseMap" = None

    def __init__(self, dm: "IBaseMap"):
        self.dm = dm

    @classmethod
    @abstractmethod
    def item_type(cls) -> type:
        pass

    def __getattr__(self, name: str) -> Optional["_Object"]:
        return self.__getitem__(name)

    def __getitem__(self, name: str) -> Optional["_Object"]:
        obj_type = self.item_type()
        return self.dm.get_object(name, obj_type)

    @classmethod
    def deserialize(cls, data: Dict[str, Any], dm: "IBaseMap") -> \
            Tuple["AbstractLayer", Dict[Tuple[str, type], "_Object"]]:
        layer = cls(dm)
        obj_type = cls.item_type()
        layer_items = {(name, obj_type): obj_type(**desc, dm=dm) for name, desc in data.items()}
        return layer, layer_items

    def serialize(self) -> Dict[str, Dict[str, Any]]:
        items = self.items()
        return {nm: ob.dict() for (nm, _), ob in items}

    def dict(self) -> Dict[Tuple[str, type], "_Object"]:
        obj_type = self.item_type()
        items = self.dm.get_objects_by_type(obj_type)
        return items

    def items(self) -> ItemsView[Tuple[str, type], "_Object"]:
        return self.dict().items()

    def __iter__(self) -> Iterator[Tuple[Tuple[str, type], "_Object"]]:
        return self.items().__iter__()

    def __str__(self) -> str:
        return pformat(self.dict())

    @classmethod
    def items_to_update(cls, dm: "IBaseMap") -> Dict[Tuple[str, type], "_Object"]:
        return {}
