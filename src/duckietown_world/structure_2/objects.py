from dataclasses import dataclass, field
from typing import Optional, List

from .base_map import IBaseMap


__all__ = ['_Object', '_Pose', '_Frame', '_TileMap', '_Tile', '_Watchtower', '_Group',
           'ConstructedObject', 'Watchtower']


@dataclass
class _Object:
    dm: IBaseMap = None

    @property
    def frame(self):
        return self.dm.get_object_frame(self)


@dataclass
class _Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0


@dataclass
class _Frame(_Object):
    relative_to: Optional[str] = None
    pose: _Pose = _Pose()

    def __init__(self, pose, relative_to=None, dm=None):
        super().__init__(dm)
        self.relative_to = relative_to
        self.pose = _Pose(**pose)


@dataclass
class _TileMap(_Object):
    x: float = 0
    y: float = 0

    def __init__(self, tile_size, dm=None):
        super().__init__(dm)
        self.x = tile_size.get('x', 0)
        self.y = tile_size.get('x', self.x)


@dataclass
class _Tile(_Object):
    i: int = 0
    j: int = 0
    type: str = 'floor'
    orientation: Optional[str] = None


@dataclass
class _Watchtower(_Object):
    configuration: str = 'WT18'


@dataclass
class _Group(_Object):
    description: str = ""
    members: List[str] = field(default_factory=list)


class ConstructedObject:
    name: str
    obj: _Object
    frame: _Frame


class Watchtower(ConstructedObject):
    def __init__(self, name, **kwargs):
        self.name = name
        self.obj = _Watchtower()
        self.frame = _Frame(kwargs)
