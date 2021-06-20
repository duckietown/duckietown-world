import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from typing import Dict, List, NewType, Union

__all__ = ["MapFormat1", "MapFormat1Object", "MapFormat1Constants"]


class MapFormat1Constants:
    ObjectKind = NewType("ObjectKind", str)

    KIND_DUCKIE = ObjectKind("duckie")
    KIND_DUCKIEBOT = ObjectKind("duckiebot")
    KIND_TRAFFICLIGHT = ObjectKind("trafficlight")
    KIND_CHECKERBOARD = ObjectKind("checkerboard")


class MapFormat1Object(TypedDict, total=False):
    kind: MapFormat1Constants.ObjectKind

    pos: List[float]
    rotate: float
    optional: bool
    color: str
    static: bool
    height: float
    scale: float

    tag: object


#
#
# class MapFormat2Object(TypedDict, total=False):
#     kind: MapFormat1Constants.ObjectKind
#
#     pos: List[float]
#     rotate: float
#     optional: bool
#     color: str
#     static: bool
#     height: float
#     scale: float
#     pose: object
#     attache: object


class MapFormat1(TypedDict, total=False):
    tiles: List[List[str]]

    tile_size: float
    start_tile: List[int]
    # [[0.480, 0, 0.2925], 0]
    start_pose: List[Union[List[Union[float, int]], Union[float, int]]]
    objects: Dict[str, MapFormat1Object]
