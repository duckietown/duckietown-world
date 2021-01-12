from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from svgwrite.drawing import Drawing as DrawingSVG
from svgwrite.container import Group as GroupSVG

from duckietown_world.world_duckietown.tile import Tile

from .bases import _Object, _PlacedObject, ConstructedObject, IBaseMap


__all__ = ['_TileMap', '_Tile', '_Watchtower', '_Group', '_Citizen',
           'Watchtower']


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
    orientation: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        return {'i': self.i, 'j': self.j, 'type': self.type, 'orientation': self.orientation}

    def extent_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0.0, 0.5), (0.5, 0.0)

    def draw_svg(self, drawing: "DrawingSVG", g: "GroupSVG") -> None:
        Tile(kind=self.type, drivable=False).draw_svg(drawing=drawing, g=g)  # TODO: transfer implementation


@dataclass
class _Watchtower(_PlacedObject):
    configuration: str = 'WT18'

    def dict(self) -> Dict[str, Any]:
        return {'configuration': self.configuration}

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


class Watchtower(ConstructedObject):
    @classmethod
    def object_type(cls) -> type:
        return _Watchtower
