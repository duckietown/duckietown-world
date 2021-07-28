from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

from svgwrite.drawing import Drawing as DrawingSVG
from svgwrite.container import Group as GroupSVG

from duckietown_world.world_duckietown.tile import Tile as old_tile

from .bases import _Object, _PlacedObject, ConstructedObject, IBaseMap

__all__ = ['_TileMap', '_Tile', '_Watchtower', '_Group', '_Citizen', '_TrafficSign', '_GroundTag', "_VehicleTag",
           'Watchtower', 'Citizen', 'TrafficSign', 'TrafficSign', 'Tile', '_Vehicle', "Vehicle", "VehicleTag",
           "Decoration", "_Decoration"]


@dataclass
class _TileMap(_PlacedObject):
    x: float = 0
    y: float = 0

    def __init__(self, tile_size: Dict[str, float], dm: "IBaseMap" = None):
        super().__init__(dm)
        self.x = tile_size.get('x', 0)
        self.y = tile_size.get('x', self.x)

    def dict(self) -> Dict[str, Any]:
        return {'tile_size': {'x': self.x, 'y': self.y}}


@dataclass
class _Tile(_PlacedObject):
    i: int = 0
    j: int = 0
    type: str = 'floor'
    orientation: Optional[str] = 'S'

    def dict(self) -> Dict[str, Any]:
        return {'i': self.i, 'j': self.j, 'type': self.type, 'orientation': self.orientation}

    def extent_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0.0, 0.5), (0.5, 0.0)

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        print(self.type)
        old_tile(kind=self.type, drivable=False).draw_svg(drawing=drawing, g=g)  # TODO: transfer implementation


@dataclass
class _Watchtower(_PlacedObject):
    configuration: str = 'WT18'
    id: Union[None, str] = None

    def dict(self) -> Dict[str, Any]:
        return {'configuration': self.configuration, 'id': self.id}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill='brown', stroke='black', stroke_width=0.01)
        g.add(rect)


@dataclass
class _Citizen(_PlacedObject):
    color: str = 'yellow'

    def dict(self) -> Dict[str, Any]:
        return {'color': self.color}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill=self.color, stroke='black', stroke_width=0.01)
        g.add(rect)


@dataclass
class _Group(_Object):
    description: str = ""
    members: List[str] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        return {'description': self.description, 'members': self.members}


@dataclass
class _TrafficSign(_PlacedObject):
    id: int = 0
    type: str = 'stop'
    family: str = '36h11'

    def dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'type': self.type, 'family': self.family}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill='black', stroke='black', stroke_width=0.01)
        g.add(rect)


@dataclass
class _GroundTag(_PlacedObject):
    size: int = 0.15
    family: str = '36h11'
    id: int = 300

    def dict(self) -> Dict[str, Any]:
        return {'size': self.size, 'id': self.id, 'family': self.family}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill='black', stroke='red', stroke_width=0.01)
        g.add(rect)


@dataclass
class _VehicleTag(_GroundTag):
    pass


@dataclass
class _Vehicle(_PlacedObject):
    configuration: str = "DB19"
    id: Optional[str] = ""
    color: str = ""

    def dict(self) -> Dict[str, Any]:
        return {"configuration": self.configuration, "id": self.id, "color": self.color}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill='red', stroke='yellow', stroke_width=0.01)
        g.add(rect)


@dataclass
class _Camera(_Object):
    camera_matrix: List[float] = None  # TODO:  type check need
    distortion_parameters: List[float] = None
    width: int = 0
    height: int = 0
    framerate: int = 0

    def dict(self) -> Dict[str, Any]:
        return {"camera_matrix": self.camera_matrix, "distortion_parameters": self.distortion_parameters,
                "width": self.width, "height": self.height, "framerate": self.framerate
                }


@dataclass
class _Environment(_Object):
    date: str = ""
    location: str = ""
    weather: str = ""

    def dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "location": self.location,
            "weather": self.weather
        }


@dataclass
class _Decoration(_PlacedObject):
    type: str = "tree"
    colors: dict = field(default_factory={"trunk": "brown", "foliage": "green"})

    def dict(self) -> Dict[str, Any]:
        return {"type": self.type, "colors": self.colors}

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        s = 0.1
        rect = drawing.rect(insert=(-s / 2, -s / 2), size=(s, s), fill=self.colors["trunk"],
                            stroke=self.colors["foliage"], stroke_width=0.01)
        g.add(rect)


class Camera(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Camera


class Vehicle(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Vehicle


class Watchtower(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Watchtower


class Citizen(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Citizen


class TrafficSign(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _TrafficSign


class GroundTag(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _GroundTag


class VehicleTag(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Vehicle


class Tile(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Tile


class Group(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Group


class Decoration(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Decoration


class Environment(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Environment

