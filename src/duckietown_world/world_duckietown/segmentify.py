from collections import defaultdict
from dataclasses import dataclass
from typing import cast, Dict, Optional, Set, Tuple

import numpy as np
from networkx import DiGraph, MultiDiGraph

import geometry as geo
from duckietown_world.geo import Matrix2D, PlacedObject, SE2Transform
from duckietown_world.geo.measurements_utils import iterate_by_class
from zuper_commons.types import ZException
from .duckietown_map import DuckietownMap
from .lane_segment import LaneSegment
from .tile_coords import TileCoords

import networkx as nx


@dataclass
class SkeletonGraphResult:
    root: PlacedObject
    root2: PlacedObject
    G: MultiDiGraph
    # This is a graph with nodes PointLabels
    G0: DiGraph


__all__ = ["get_skeleton_graph", "SkeletonGraphResult"]

PointLabel = Tuple[float, float, float, float]


@dataclass
class MeetingPoint:
    incoming: Set[str]  # set of lane names
    outcoming: Set[str]  # set of lanes

    connects_to: Set[PointLabel]  # set of points
    point: Optional[SE2Transform]
    into_tile: Optional[Tuple[int, int]]
    from_tile: Optional[Tuple[int, int]]
    #
    # def __init__(self):
    #     self.point = None
    #     self.incoming = set()
    #     self.outcoming = set()
    #     self.connects_to = set()
    #     self.into_tile = None
    #     self.from_tile = None

    # def __repr__(self):
    #     return "MP(%d %d | %s, %s)" % (
    #         len(self.incoming),
    #         len(self.outcoming),
    #         self.incoming,
    #         self.outcoming,
    #     )


def discretize(tran: SE2Transform) -> PointLabel:
    def D(x):
        return np.round(x, decimals=2)

    p, theta = geo.translation_angle_from_SE2(tran.as_SE2())
    return D(p[0]), D(p[1]), D(np.cos(theta)), D(np.sin(theta))


def graph_for_meeting_points(mp: Dict[str, MeetingPoint]) -> DiGraph:
    G = DiGraph()
    for k, p in mp.items():
        G.add_node(k, meeting_point=p)
    for k, p in mp.items():
        for k2 in p.connects_to:
            G.add_edge(k, k2)
    return G


def get_skeleton_graph(po: DuckietownMap) -> SkeletonGraphResult:
    """ Returns a graph with the lane segments of the map """

    root = PlacedObject()

    meeting_points: Dict[str, MeetingPoint] = defaultdict(MeetingPoint)

    for i, it in enumerate(iterate_by_class(po, LaneSegment)):
        lane_segment = cast(LaneSegment, it.object)
        assert isinstance(lane_segment, LaneSegment), lane_segment

        absolute_pose = it.transform_sequence.asmatrix2d()

        lane_segment_transformed = transform_lane_segment(lane_segment, absolute_pose)

        identity = SE2Transform.identity()
        name = "ls%03d" % i
        root.set_object(name, lane_segment_transformed, ground_truth=identity)

        p0 = discretize(lane_segment_transformed.control_points[0])
        p1 = discretize(lane_segment_transformed.control_points[-1])

        if not p0 in meeting_points:
            meeting_points[p0] = MeetingPoint(
                set(), set(), set(), lane_segment_transformed.control_points[0], None, None,
            )
        if not p1 in meeting_points:
            meeting_points[p1] = MeetingPoint(
                set(), set(), set(), lane_segment_transformed.control_points[-1], None, None,
            )

        # meeting_points[p0].point = lane_segment_transformed.control_points[0]
        meeting_points[p0].outcoming.add(name)
        # meeting_points[p1].point = lane_segment_transformed.control_points[-1]
        meeting_points[p1].incoming.add(name)

        meeting_points[p0].connects_to.add(p1)

        tile_coords = [_ for _ in it.transform_sequence.transforms if isinstance(_, TileCoords)]
        if not tile_coords:
            raise ZException(p0=p0, p1=p1, transforms=it.transform_sequence.transforms)
        tile_coord = tile_coords[0]
        ij = tile_coord.i, tile_coord.j
        meeting_points[p0].into_tile = ij
        meeting_points[p1].from_tile = ij

    for k, mp in meeting_points.items():
        if (len(mp.incoming) == 0) or (len(mp.outcoming) == 0):
            msg = "Completeness assumption violated at point %s: %s" % (k, mp)
            raise Exception(msg)

    G0 = graph_for_meeting_points(meeting_points)
    # compress the lanes which are contiguous

    aliases = {}

    created = {}

    def resolve_alias(x):
        return x if x not in aliases else resolve_alias(aliases[x])

    for k, mp in list(meeting_points.items()):
        # continue
        if not (len(mp.incoming) == 1 and len(mp.outcoming) == 1):
            continue

        # not necessary anymore
        meeting_points.pop(k)
        lin_name = list(mp.incoming)[0]
        lout_name = list(mp.outcoming)[0]

        lin_name = resolve_alias(lin_name)
        lout_name = resolve_alias(lout_name)

        # print(' -> %s and %s meet at %s' % (lin_name, lout_name, mp))
        # print('%s and %s meet at %s' % (lin_name, lout_name, k))

        def get(it):
            if it in root.children:
                return root.children[it]
            else:
                return created[it]

        lin = get(lin_name)
        lout = get(lout_name)

        # name = 'alias%s' % (len(aliases))
        # name = '%s-%s' % (lin_name, lout_name)
        name = "L%d" % (len(created))
        width = lin.width

        control_points = lin.control_points + lout.control_points[1:]
        ls = LaneSegment(width=width, control_points=control_points)
        created[name] = ls

        aliases[lin_name] = name
        aliases[lout_name] = name
        # print('new alias %s' % name)
    #
    # print('created: %s' % list(created))
    # print('aliases: %s' % aliases)
    root2 = PlacedObject()
    for k, v in created.items():
        if not k in aliases:
            root2.set_object(k, v, ground_truth=SE2Transform.identity())

    for k, v in root.children.items():
        if not k in aliases:
            root2.set_object(k, v, ground_truth=SE2Transform.identity())

    G = nx.MultiDiGraph()

    k2name = {}
    for i, (k, mp) in enumerate(meeting_points.items()):
        node_name = "P%d" % i
        k2name[k] = node_name

        G.add_node(node_name, point=mp.point)

    ls2start = {}
    ls2end = {}
    for i, (k, mp) in enumerate(meeting_points.items()):
        node_name = k2name[k]

        for l in mp.incoming:
            ls2end[resolve_alias(l)] = node_name
        for l in mp.outcoming:
            ls2start[resolve_alias(l)] = node_name

    # print(ls2start)
    # print(ls2end)

    for l in ls2start:
        n1 = ls2start[l]
        n2 = ls2end[l]
        G.add_edge(n1, n2, lane=l)

    return SkeletonGraphResult(root=root, root2=root2, G=G, G0=G0)


def transform_lane_segment(lane_segment: LaneSegment, transformation: Matrix2D) -> LaneSegment:
    M = transformation.m

    def transform_point(p):
        q = p.as_SE2()
        q2 = np.dot(M, q)
        p2 = SE2Transform.from_SE2(q2)
        return p2

    control_points = list(map(transform_point, lane_segment.control_points))

    det = np.linalg.det(M)
    width = float(lane_segment.width * np.sqrt(det))
    return LaneSegment(control_points=control_points, width=width)
