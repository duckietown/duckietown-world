import random
from typing import cast

import geometry as geo
import numpy as np

from duckietown_world import (
    iterate_by_class,
    IterateByTestResult,
    LanePose,
    LaneSegment,
    PlacedObject,
    SE2Transform,
    Tile,
)

__all__ = ["sample_good_starting_pose"]


# @dataclass
# class SampledPose:
#     # pose
#     q: np.ndarray
#     # pose as an SE2Transform
#     qt: SE2Transform
#     # corresponding  tile
#     tile: Any


def sample_good_starting_pose(m: PlacedObject, only_straight: bool, along_lane: float) -> geo.SE2value:
    """ Samples a good starting pose on a straight lane """
    choices = list(iterate_by_class(m, LaneSegment))

    if only_straight:
        choices = [_ for _ in choices if is_straight(_)]

    choice = random.choice(choices)
    ls: LaneSegment = choice.object

    lp: LanePose = ls.lane_pose(along_lane, 0.0, 0.0)
    rel: SE2Transform = ls.SE2Transform_from_lane_pose(lp)

    m1 = choice.transform_sequence.asmatrix2d().m
    m2 = rel.asmatrix2d().m
    g = np.dot(m1, m2)

    t, a, s = geo.translation_angle_scale_from_E2(g)
    g = geo.SE2_from_translation_angle(t, a)

    return g


def is_straight(choice: IterateByTestResult) -> bool:
    segment = cast(LaneSegment, choice.object)
    lane_segment_is_straight = np.allclose(segment.get_lane_length(), 1.0)
    tiles = [_ for _ in choice.parents if isinstance(_, Tile)]
    assert len(tiles) == 1
    tile: Tile = tiles[0]
    tile_is_straight = tile.kind == "straight"
    return lane_segment_is_straight and tile_is_straight
