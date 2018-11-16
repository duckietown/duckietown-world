# coding=utf-8
from collections import namedtuple

import numpy as np
from contracts import contract
from duckietown_world import logger
from duckietown_world.geo import PlacedObject, RectangularArea, TransformSequence, Matrix2D, SE2Transform
from duckietown_world.seqs import SampledSequence
from duckietown_world.svg_drawing import data_encoded_for_src, draw_axes, draw_children
from duckietown_world.svg_drawing.misc import mime_from_fn
from geometry import extract_pieces

__all__ = [
    'Tile',
    'GetLanePoseResult',
    'get_lane_poses',
    'create_lane_highlight',
]

GetLanePoseResult = namedtuple('GetLanePoseResult',
                               'tile tile_fqn tile_transform tile_relative_pose '
                               'lane_segment lane_segment_fqn lane_pose '
                               'lane_segment_relative_pose tile_coords '
                               'lane_segment_transform '
                               'center_point')


class SignSlot(PlacedObject):
    """ Represents a slot where you can put a sign. """
    L = 0.1

    def get_footprint(self):
        L = SignSlot.L
        return RectangularArea([-L / 2, -L / 2], [L / 2, L / 2])

    def draw_svg(self, drawing, g):
        L = SignSlot.L

        rect = drawing.rect(insert=(-L / 2, -L / 2),
                            size=(L, L),
                            fill="none",
                            # style='opacity:0.4',
                            stroke_width="0.005",
                            stroke="pink", )
        g.add(rect)
        draw_axes(drawing, g, 0.04)


def get_tile_slots():
    LM = 0.5  # half tile
    # tile_offset
    to = 0.20
    # tile_curb
    tc = 0.05

    positions = {
        0: (+ to, + tc),
        1: (+ tc, + to),
        2: (- tc, + to),
        3: (- to, + tc),
        4: (- to, - tc),
        5: (- tc, - to),
        6: (+ tc, - to),
        7: (+ to, - tc),
    }

    po = PlacedObject()
    for i, (x, y) in positions.items():
        name = str(i)
        # if name in self.children:
        #     continue

        sl = SignSlot()
        # theta = np.deg2rad(theta_deg)
        theta = 0
        t = SE2Transform((-LM + x, -LM + y), theta)
        po.set_object(name, sl, ground_truth=t)
    return po


class Tile(PlacedObject):
    def __init__(self, kind, drivable, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.kind = kind
        self.drivable = drivable

        from duckietown_world.world_duckietown.map_loading import get_texture_file

        try:
            self.fn = get_texture_file(kind)
        except KeyError:
            msg = 'Cannot find texture for %s' % kind
            logger.warning(msg)

            self.fn = None
        # if kind in ['asphalt']:
        if not 'slots' in self.children:
            slots = get_tile_slots()
            self.set_object('slots', slots, ground_truth=SE2Transform.identity())

    def _copy(self):
        return type(self)(self.kind, self.drivable,
                          children=dict(**self.children), spatial_relations=dict(**self.spatial_relations))

    def params_to_json_dict(self):
        return dict(kind=self.kind, drivable=self.drivable)

    def get_footprint(self):
        return RectangularArea([-0.5, -0.5], [0.5, 0.5])

    def draw_svg(self, drawing, g):
        T = 0.562 / 0.585
        S = 1.0
        rect = drawing.rect(insert=(-S / 2, -S / 2),
                            size=(S, S),
                            fill='#222',
                            stroke="none", )
        g.add(rect)

        if self.fn:
            # texture = get_jpeg_bytes(self.fn)
            texture = open(self.fn, 'rb').read()
            href = data_encoded_for_src(texture, mime_from_fn(self.fn))
            img = drawing.image(href=href,
                                size=(T, T),
                                insert=(-T / 2, -T / 2),
                                style='transform: rotate(90deg) scaleX(-1)  rotate(-90deg) '
                                )
            img.attribs['class'] = 'tile-textures'
            g.add(img)
        #
        # if draw_directions_lanes:
        #     if self.kind != 'floor':
        #         start = (-0.5, -0.25)
        #         end = (+0, -0.25)
        #         line = drawing.line(start=start, end=end, stroke='blue', stroke_width='0.01')
        #         g.add(line)

        draw_axes(drawing, g)

        draw_children(drawing, self, g)


@contract(q='SE2')
def get_lane_poses(dw, q, tol=0.000001):
    from duckietown_world.geo.measurements_utils import iterate_by_class, IterateByTestResult
    from .lane_segment import LaneSegment
    from duckietown_world import TileCoords

    for it in iterate_by_class(dw, Tile):
        assert isinstance(it, IterateByTestResult), it
        assert isinstance(it.object, Tile), it.object
        tile = it.object
        tile_fqn = it.fqn
        tile_transform = it.transform_sequence
        for _ in tile_transform.transforms:
            if isinstance(_, TileCoords):
                tile_coords = _
                break
        else:
            msg = 'Could not find tile coords in %s' % tile_transform
            assert False, msg
        # print('tile_transform: %s' % tile_transform.asmatrix2d().m)
        tile_relative_pose = relative_pose(tile_transform.asmatrix2d().m, q)
        p = translation_from_O3(tile_relative_pose)
        # print('tile_relative_pose: %s' % tile_relative_pose)
        if not tile.get_footprint().contains(p):
            continue
        nresults = 0
        for it2 in iterate_by_class(tile, LaneSegment):
            lane_segment = it2.object
            lane_segment_fqn = tile_fqn + it2.fqn
            assert isinstance(lane_segment, LaneSegment), lane_segment
            lane_segment_wrt_tile = it2.transform_sequence.asmatrix2d()
            lane_segment_relative_pose = relative_pose(lane_segment_wrt_tile.m, tile_relative_pose)
            lane_segment_transform = TransformSequence(tile_transform.transforms + it2.transform_sequence.transforms)
            lane_pose = lane_segment.lane_pose_from_SE2(lane_segment_relative_pose, tol=tol)

            M = lane_segment_transform.asmatrix2d().m
            center_point = lane_pose.center_point.as_SE2()

            center_point_abs = np.dot(M, center_point)
            center_point_abs_t = Matrix2D(center_point_abs)

            if lane_pose.along_inside and lane_pose.inside and lane_pose.correct_direction:
                yield GetLanePoseResult(tile=tile, tile_fqn=tile_fqn,
                                        tile_transform=tile_transform,
                                        tile_relative_pose=Matrix2D(tile_relative_pose),
                                        lane_segment=lane_segment,
                                        lane_segment_relative_pose=Matrix2D(lane_segment_relative_pose),
                                        lane_pose=lane_pose,
                                        lane_segment_fqn=lane_segment_fqn,
                                        lane_segment_transform=lane_segment_transform,
                                        tile_coords=tile_coords,
                                        center_point=center_point_abs_t)
                nresults += 1

        # if nresults == 0:
        #     msg = 'Could not find any lane in tile %s' % tile_transform
        #     msg += '\ntile_relative_pose: %s' % tile_relative_pose
        #     for it2 in iterate_by_class(tile, LaneSegment):
        #         lane_segment = it2.object
        #         lane_segment_wrt_tile = it2.transform_sequence.as_SE2()
        #         lane_segment_relative_pose = relative_pose(lane_segment_wrt_tile, tile_relative_pose)
        #         lane_pose = lane_segment.lane_pose_from_SE2(lane_segment_relative_pose, tol=tol)
        #
        #         msg += '\n lane_relative: %s' % lane_segment_relative_pose
        #         msg += '\n lane pose: %s' % lane_pose
        #     logger.warning(msg)


@contract(  # pose='O3',
        returns='array[2]')
def translation_from_O3(pose):
    _, t, _, _ = extract_pieces(pose)
    return t


def relative_pose(base, pose):
    assert isinstance(base, np.ndarray), base
    assert isinstance(pose, np.ndarray), pose
    return np.dot(np.linalg.inv(base), pose)


class GetClosestLane(object):
    def __init__(self, dw):
        self.no_matches_for = []
        self.dw = dw

    def __call__(self, transform):
        if isinstance(transform, SE2Transform):
            transform = transform.as_SE2()
        poses = list(get_lane_poses(self.dw, transform))
        # if not poses:
        #     self.no_matches_for.append(transform)
        #     return None

        s = sorted(poses, key=lambda _: np.abs(_.lane_pose.relative_heading))
        res = {}
        for i, _ in enumerate(s):
            res[i] = _

        return res


class Anchor(PlacedObject):
    def _copy(self):
        return self._simplecopy()

    def draw_svg(self, drawing, g):
        draw_axes(drawing, g, klass='anchor-axes')
        c = drawing.circle(center=(0, 0), r=0.03,
                           fill='blue', stroke='black', stroke_width=0.001)
        g.add(c)


def create_lane_highlight(poses_sequence, dw):
    assert isinstance(poses_sequence, SampledSequence)

    def mapi(v):
        if isinstance(v, SE2Transform):
            return v.as_SE2()
        else:
            return v

    poses_sequence = poses_sequence.transform_values(mapi)

    lane_pose_results = poses_sequence.transform_values(GetClosestLane(dw))

    visualization = PlacedObject()
    dw.set_object('visualization', visualization, ground_truth=SE2Transform.identity())
    for i, (timestamp, name2pose) in enumerate(lane_pose_results):
        for name, lane_pose_result in name2pose.items():
            assert isinstance(lane_pose_result, GetLanePoseResult)
            lane_segment = lane_pose_result.lane_segment
            rt = lane_pose_result.lane_segment_transform
            s = SampledSequence([timestamp], [rt])
            visualization.set_object('ls%s-%s-lane' % (i, name), lane_segment, ground_truth=s)
            p = SampledSequence([timestamp], [lane_pose_result.center_point])
            visualization.set_object('ls%s-%s-anchor' % (i, name), Anchor(), ground_truth=p)

    return lane_pose_results
