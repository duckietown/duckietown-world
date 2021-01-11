from abc import ABC
from typing import Tuple, Dict
from re import search
import numpy as np

from .bases import _Object, _Frame, IBaseMap, AbstractLayer
from .objects import _Tile, _Group, _TileMap, _Watchtower, _Citizen


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

    @classmethod
    def items_to_update(cls, dm: "IBaseMap") -> Dict[Tuple[str, type], "_Object"]:
        scaled_frames = {}
        tile_maps = dm.get_objects_by_type(_TileMap)
        for (nm, _), ob in tile_maps.items():
            frame = dm.get_object_frame(ob)
            scaled_frame = frame.copy(dm)
            assert isinstance(ob, _TileMap)
            scaled_frame.scale = ob.x
            scaled_frames[(nm, _Frame)] = scaled_frame
        return scaled_frames


class LayerTiles(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Tile

    @classmethod
    def items_to_update(cls, dm: "IBaseMap") -> Dict[Tuple[str, type], "_Object"]:
        tile_frames = {}
        tiles = dm.get_objects_by_type(_Tile)
        for (nm, _), ob in tiles.items():
            frame = dm.get_object_frame(ob)
            if frame is None:
                s = search(r'(.*)/tile_(\d)_(\d)$', nm)
                try:
                    parent_nm, i, j = s.group(1), s.group(2), s.group(3)
                except AttributeError:
                    raise ValueError('Cannot parse tile name: %s' % nm)
                x = float(i) + 0.5
                y = float(j) + 0.5
                assert isinstance(ob, _Tile)
                orientation = ob.orientation if ob.orientation is not None else 'E'
                yaw = {'E': 0, 'N': np.pi * 0.5, 'W': np.pi, 'S': np.pi * 1.5}[orientation]
                tile_frames[(nm, _Frame)] = _Frame({'x': x, 'y': y, 'yaw': yaw}, relative_to=parent_nm, dm=dm)

        # invert y axes
        if tile_frames:
            w = max([ob.pose.y for _, ob in tile_frames.items()]) - 0.5
            for _, ob in tile_frames.items():
                ob.pose.y = w - (ob.pose.y - 0.5) + 0.5

        return tile_frames


class LayerWatchtowers(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Watchtower


class LayerGroups(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Group


class LayerCitizens(AbstractLayer, ABC):
    @classmethod
    def item_type(cls) -> type:
        return _Citizen
