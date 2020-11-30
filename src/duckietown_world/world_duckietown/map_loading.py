# coding=utf-8
import itertools
import os
import traceback
from typing import List

import numpy as np
import oyaml as yaml
from zuper_commons.fs import FilePath
from zuper_commons.types import ZKeyError, ZValueError

from duckietown_serialization_ds1 import Serializable
from . import logger
from .duckiebot import DB18
from .duckietown_map import DuckietownMap
from .old_map_format import MapFormat1Object
from .other_objects import (
    Barrier,
    Building,
    Bus,
    Cone,
    Duckie,
    GenericObject,
    House,
    SIGNS,
    Tree,
    Truck,
)
from .tags_db import FloorTag
from .tile import Tile
from .tile_map import TileMap
from .tile_template import load_tile_types
from .traffic_light import TrafficLight
from ..geo import Scale2D, SE2Transform
from ..geo.measurements_utils import iterate_by_class

__all__ = [
    "create_map",
    "construct_map",
    "load_map",
    "get_texture_file",
    "_get_map_yaml",
]

from ..resources import get_data_resources, get_maps_dir


def create_map(H: int = 3, W: int = 3) -> TileMap:
    tile_map = TileMap(H=H, W=W)

    for i, j in itertools.product(range(H), range(W)):
        tile = Tile(kind="mykind", drivable=True)
        tile_map.add_tile(i, j, "N", tile)

    return tile_map


def _get_map_yaml(map_name: str) -> str:
    maps_dir = get_maps_dir()
    fn = os.path.join(maps_dir, map_name + ".yaml")
    if not os.path.exists(fn):
        msg = f"Could not find file {fn}"
        raise ValueError(msg)
    with open(fn) as _:
        data = _.read()
    return data


def load_map(map_name: str) -> DuckietownMap:
    logger.info(f"loading map {map_name}")
    data = _get_map_yaml(map_name)
    yaml_data = yaml.load(data, Loader=yaml.SafeLoader)

    return construct_map(yaml_data)


def construct_map(yaml_data: dict) -> DuckietownMap:
    tile_size = yaml_data["tile_size"]
    dm = DuckietownMap(tile_size)
    tiles = yaml_data["tiles"]
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    A = len(tiles)
    B = len(tiles[0])
    tm = TileMap(H=B, W=A)

    templates = load_tile_types()
    for a, row in enumerate(tiles):
        if len(row) != B:
            msg = "each row of tiles must have the same length"
            raise ValueError(msg)

        # For each tile in this row
        for b, tile in enumerate(row):
            tile = tile.strip()

            if tile == "empty":
                continue

            DEFAULT_ORIENT = "E"  # = no rotation
            if "/" in tile:
                kind, orient = tile.split("/")
                kind = kind.strip()
                orient = orient.strip()

                drivable = True
            elif "4" in tile:
                kind = "4way"
                # angle = 2
                orient = DEFAULT_ORIENT
                drivable = True
            else:
                kind = tile
                # angle = 0
                orient = DEFAULT_ORIENT
                drivable = False

            tile = Tile(kind=kind, drivable=drivable)
            if kind in templates:
                tile.set_object(kind, templates[kind], ground_truth=SE2Transform.identity())
            else:
                pass
                # msg = 'Could not find %r in %s' % (kind, templates)
                # logger.debug(msg)

            tm.add_tile(b, (A - 1) - a, orient, tile)

    def go(obj_name0: str, desc0: MapFormat1Object):
        obj = get_object(desc0)
        transform = get_transform(desc0, tm.W, tile_size)
        dm.set_object(obj_name0, obj, ground_truth=transform)

    objects = yaml_data.get("objects", [])
    if isinstance(objects, list):
        for obj_idx, desc in enumerate(objects):
            kind = desc["kind"]
            obj_name = f"ob{obj_idx:02d}-{kind}"
            go(obj_name, desc)
    elif isinstance(objects, dict):
        for obj_name, desc in objects.items():
            go(obj_name, desc)
    else:
        raise ValueError(objects)

    for it in list(iterate_by_class(tm, Tile)):
        ob = it.object
        if "slots" in ob.children:
            slots = ob.children["slots"]
            for k, v in list(slots.children.items()):
                if not v.children:
                    slots.remove_object(k)
            if not slots.children:
                ob.remove_object("slots")

    dm.set_object("tilemap", tm, ground_truth=Scale2D(tile_size))
    return dm


def get_object(desc: MapFormat1Object):
    kind = desc["kind"]

    attrs = {}
    if "tag" in desc:
        tag_desc = desc["tag"]
        # tag_id = tag_desc.get('tag_id')
        # size = tag_desc.get('size', DEFAULT_TAG_SIZE)
        # family = tag_desc.get('family', DEFAULT_FAMILY)
        attrs["tag"] = Serializable.from_json_dict(tag_desc)

    kind2klass = {
        "trafficlight": TrafficLight,
        "duckie": Duckie,
        "cone": Cone,
        "barrier": Barrier,
        "building": Building,
        "duckiebot": DB18,
        "tree": Tree,
        "house": House,
        "bus": Bus,
        "truck": Truck,
        "floor_tag": FloorTag,
    }
    kind2klass.update(SIGNS)
    if kind in kind2klass:
        klass = kind2klass[kind]
        try:
            obj = klass(**attrs)
        except TypeError as e:
            msg = "Could not initialize %s with attrs %s:\n%s" % (
                klass.__name__,
                attrs,
                traceback.format_exc(),
            )
            raise Exception(msg) from e

    else:
        logger.debug("Do not know special kind %s" % kind)
        obj = GenericObject(kind=kind)
    return obj


def get_transform(desc: MapFormat1Object, W: int, tile_size: float) -> SE2Transform:
    rotate_deg = desc.get("rotate", 0)
    rotate = np.deg2rad(rotate_deg)
    if "pos" in desc:

        pos = desc["pos"]
        x = float(pos[0]) * tile_size
        # account for non-righthanded
        y = float(W - 1 - pos[1]) * tile_size
        # account for non-righthanded
        rotate = -rotate
        transform = SE2Transform([x, y], rotate)
        return transform

    elif "pose" in desc:
        # noinspection PyTypedDict
        pose = Serializable.from_json_dict(desc["pose"])
        return pose
    elif "place" in desc:
        # noinspection PyTypedDict
        place = desc["place"]
        tile_coords = tuple(place["tile"])
        relative = Serializable.from_json_dict(place["relative"])
        p, theta = relative.p, relative.theta
        i, j = tile_coords

        fx = (i + 0.5) * tile_size + p[0]
        fy = (j + 0.5) * tile_size + p[1]
        transform = SE2Transform([fx, fy], theta)
        # logger.info(tile_coords=tile_coords, tile_size=tile_size, transform=transform)
        return transform

    elif "attach" in desc:
        # noinspection PyTypedDict
        attach = desc["attach"]
        tile_coords = tuple(attach["tile"])
        slot = str(attach["slot"])

        x, y = get_xy_slot(slot)
        i, j = tile_coords

        u, v = (x + i) * tile_size, (y + j) * tile_size
        transform = SE2Transform([u, v], rotate)

        q = transform.as_SE2()

        return SE2Transform.from_SE2(q)
    else:
        msg = "Cannot find positiong"
        raise ZValueError(msg, desc=desc)


def get_xy_slot(i):
    # tile_offset
    # to = 0.20
    # # tile_curb
    # tc = 0.05
    to = 0.09  # [m] the longest of two distances from the center of the tag to the edge
    tc = 0.035  # [m] the shortest of two distances from the center of the tag to the edge

    positions = {
        0: (+to, +tc),
        1: (+tc, +to),
        2: (-tc, +to),
        3: (-to, +tc),
        4: (-to, -tc),
        5: (-tc, -to),
        6: (+tc, -to),
        7: (+to, -tc),
    }
    x, y = positions[int(i)]
    return x, y


def get_texture_file(tex_name: str) -> List[FilePath]:
    if tex_name.endswith(".png"):
        logger.warn(f"do not provide extension: {tex_name}")
        tex_name = tex_name.replace(".png", "")
    if tex_name.endswith(".jpg"):
        logger.warn(f"do not provide extension: {tex_name}")
        tex_name = tex_name.replace(".jpg", "")

    resources, res2 = get_data_resources()
    res: List[FilePath] = []
    tried = []

    suffixes = ["", "_1", "_2", "_3", "_4"]
    extensions = [".jpg", ".png", ".tga", ".bmp"]
    for s, e in itertools.product(suffixes, extensions):
        basename = f"{tex_name}{s}{e}"

        for v in res2:
            if v.endswith(basename):
                res.append(v)
        #
        #
        # if basename in resources:
        #     return resources[basename]
        tried.append(basename)

    if not res:
        msg = f"Could not find any texture for {tex_name}"
        raise ZKeyError(msg, tried=tried)
    return res
