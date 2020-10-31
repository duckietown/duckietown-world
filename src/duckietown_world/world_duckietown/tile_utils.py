from dataclasses import dataclass
from typing import cast, Dict, Set, Tuple

import numpy as np
from geometry import SE2value
from networkx import connected_components
from zuper_commons.types import ZException, ZValueError

from duckietown_world import get_skeleton_graph, TileMap
from duckietown_world.geo.measurements_utils import (
    iterate_by_class,
    IterateByTestResult,
)
from duckietown_world.geo.placed_object import get_child_transform
from duckietown_world.geo.transforms import Transform
from duckietown_world.world_duckietown.tile_map import ij_from_tilename
from .duckietown_map import DuckietownMap
from .tile import Tile, translation_from_O3
from .tile_coords import TileCoords
from .utils import relative_pose


class NotInMap(ZException):
    pass


def get_tile_at_point(dw: DuckietownMap, q: SE2value) -> TileCoords:
    if not isinstance(q, np.ndarray):
        raise TypeError(type(q))
    l = list(iterate_by_class(dw, Tile))
    if not l:
        msg = "No tiles"
        raise ZValueError(msg, dw=dw)

    distances = {}
    for it in l:
        assert isinstance(it, IterateByTestResult), it
        assert isinstance(it.object, Tile), it.object
        tile = it.object
        tile_transform = it.transform_sequence
        for _ in tile_transform.transforms:
            if isinstance(_, TileCoords):
                tile_coords = _
                break
        else:
            msg = "Could not find tile coords in %s" % tile_transform
            assert False, msg

        # print('tile_transform: %s' % tile_transform.asmatrix2d().m)
        tile_relative_pose = relative_pose(tile_transform.asmatrix2d().m, q)
        p = translation_from_O3(tile_relative_pose)
        footprint = tile.get_footprint()
        d = footprint.distance(p)
        distances[it.fqn] = d
        # print(f'tile_relative_pose: {tile_coords} {p} {d}')
        # if d > 0.001:
        if tile.get_footprint().contains(p, 0.001):
            return tile_coords
    raise NotInMap(q=q, distances=distances)


@dataclass
class ComponentFootprint:
    # tile_names: List[str]
    tile_coords: Set[Tuple[int, int]]


def get_map_components(m: DuckietownMap) -> Dict[str, ComponentFootprint]:
    from duckietown_world.world_duckietown.segmentify import MeetingPoint

    sk2 = get_skeleton_graph(m)
    G0 = sk2.G0
    c = list(connected_components(G0.to_undirected()))
    res = {}
    for i, nodes in enumerate(c):
        s = "comp%d" % i
        tiles: Set[Tuple[int, int]] = set()
        for n in nodes:
            mp: MeetingPoint = G0.nodes[n]["meeting_point"]
            tiles.add(mp.from_tile)
            tiles.add(mp.into_tile)
        res[s] = ComponentFootprint(tiles)
    return res


#
# def compute_extent_of_tiles()
#     # compute region for tiles
#     tilemap =
#     extents = set()
#     for ij in tiles:
def contained(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
    return (a[0] <= b[0] <= c[0]) and (a[1] <= b[1] <= c[1])


def get_interest_map(m: DuckietownMap, q: SE2value) -> DuckietownMap:
    """ Returns the map of interest given a pose. """
    components = get_map_components(m)
    r = get_tile_at_point(m, q)
    tile = r.i, r.j
    for component_id, component in components.items():
        if tile in component.tile_coords:
            break
    else:
        msg = "Cannot find any component for this point."
        raise ZValueError(msg, tile=tile, q=q, components=components)

    tiles: Set[Tuple[int, int]] = component.tile_coords
    tmin: Tuple[int, int] = min(tiles)
    tmax: Tuple[int, int] = max(tiles)
    M = 1
    tmin = (tmin[0] - M, tmin[1] - M)
    tmax = (tmax[0] + M, tmax[1] + M)

    print(f"tmin tmax {tmin} {tmax}")
    children2 = {}
    for name, child in m.children.items():
        if name == "tilemap":
            tilemap = cast(TileMap, child)
            tiles2 = {}
            for tilename, tile in tilemap.children.items():
                coords = ij_from_tilename(tilename)
                if contained(tmin, coords, tmax):
                    tiles2[tilename] = tile

            sr2 = {}
            for sr_id, sr in list(tilemap.spatial_relations.items()):
                first = sr.b[0]
                if first in tiles2:
                    sr2[sr_id] = sr
            tilemap2 = TileMap(H=tilemap.H, W=tilemap.W, children=tiles2, spatial_relations=sr2)
            children2[name] = tilemap2
        else:
            t: Transform = get_child_transform(m, name)
            try:
                tile_for_child = get_tile_at_point(m, t.as_SE2())
            except NotInMap:
                print(f"cutting {name}")
                continue
            c = tile_for_child.i, tile_for_child.j
            if contained(tmin, c, tmax):
                print(f"keep {name} {c}")
                children2[name] = child
            else:
                print(f"cutting {name}")
    # print(debug_print(children2))
    sr2 = {}
    for sr_id, sr in list(m.spatial_relations.items()):
        first = sr.b[0]
        if first in children2:
            sr2[sr_id] = sr
    return DuckietownMap(tile_size=m.tile_size, children=children2, spatial_relations=sr2)
