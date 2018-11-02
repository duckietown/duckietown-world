# coding=utf-8
import base64
import logging
from collections import namedtuple

import numpy as np
from contracts import contract
from six import BytesIO

from duckietown_world import logger
from duckietown_world.geo import PlacedObject, RectangularArea, SE2Transform, TransformSequence
from duckietown_world.utils.memoizing import memoized_reset

__all__ = [
    'Tile',
]
draw_directions_lanes = False


@memoized_reset
def get_jpeg_bytes(fn):
    from PIL import Image
    pl = logging.getLogger('PIL')
    pl.setLevel(logging.ERROR)

    image = Image.open(fn).convert('RGB')

    out = BytesIO()
    image.save(out, format='jpeg')
    return out.getvalue()


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

    def params_to_json_dict(self):
        return dict(kind=self.kind, drivable=self.drivable)

    def get_footprint(self):
        return RectangularArea([-0.5, -0.5], [0.5, 0.5])

    def draw_svg(self, drawing, g):
        # rect = drawing.rect(width=1,height=1,)
        # g.add(rect)
        rect = drawing.rect(insert=(-0.5, -0.5),
                            width=1,
                            height=1,
                            fill="#eee",
                            style='opacity:0.4',
                            stroke_width="0.01",
                            stroke="none", )
        # g.add(rect)

        if self.fn:
            # texture = get_jpeg_bytes(self.fn)
            texture = open(self.fn, 'rb').read()
            href = data_encoded_for_src(texture, 'image/jpeg')
            img = drawing.image(href=href,
                                size=(1, 1),
                                insert=(-0.5, -0.5),
                                style='transform: rotate(90deg) scaleX(-1)  rotate(-90deg) '
                                )
            img.attribs['class'] = 'tile-textures'
            g.add(img)

        if draw_directions_lanes:
            if self.kind != 'floor':
                start = (-0.5, -0.25)
                end = (+0, -0.25)
                line = drawing.line(start=start, end=end, stroke='blue', stroke_width='0.01')
                g.add(line)

        from duckietown_world.world_duckietown.duckiebot import draw_axes
        draw_axes(drawing, g)

        from duckietown_world.svg_drawing.misc import draw_children
        draw_children(drawing, self, g)


def data_encoded_for_src(data, mime):
    """ data =
        ext = png, jpg, ...

        returns "data: ... " sttring
    """
    encoded = base64.b64encode(data).decode()
    link = 'data:%s;base64,%s' % (mime, encoded)
    return link


GetLanePoseResult = namedtuple('GetLanePoseResult',
                               'tile tile_fqn tile_transform tile_relative_pose '
                               'lane_segment lane_segment_fqn lane_pose '
                               'lane_segment_relative_pose tile_coords '
                               'lane_segment_transform '
                               'center_point')


@contract(q='SE2')
def get_lane_poses(dw, q, tol=0.000001):
    from duckietown_world.geo.measurements_utils import iterate_by_class, IterateByTestResult
    from .lane_segment import LaneSegment
    from duckietown_world import TileCoords
    import geometry as geo

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

        tile_relative_pose = relative_pose(tile_transform.as_SE2(), q)
        p, _ = geo.translation_angle_from_SE2(tile_relative_pose)
        if not tile.get_footprint().contains(p):
            continue
        nresults = 0
        for it2 in iterate_by_class(tile, LaneSegment):
            lane_segment = it2.object
            lane_segment_fqn = tile_fqn + it2.fqn
            assert isinstance(lane_segment, LaneSegment), lane_segment
            lane_segment_wrt_tile = it2.transform_sequence.as_SE2()
            lane_segment_relative_pose = relative_pose(lane_segment_wrt_tile, tile_relative_pose)
            lane_segment_transform = TransformSequence(tile_transform.transforms + it2.transform_sequence.transforms)
            lane_pose = lane_segment.lane_pose_from_SE2(lane_segment_relative_pose, tol=tol)

            M = lane_segment_transform.asmatrix2d().m
            center_point = lane_pose.center_point.as_SE2()

            center_point_abs = np.dot(M, center_point)
            center_point_abs_t = SE2Transform.from_SE2(center_point_abs)

            if lane_pose.along_inside and lane_pose.inside and lane_pose.correct_direction:
                yield GetLanePoseResult(tile=tile, tile_fqn=tile_fqn,
                                        tile_transform=tile_transform,
                                        tile_relative_pose=SE2Transform.from_SE2(tile_relative_pose),
                                        lane_segment=lane_segment,
                                        lane_segment_relative_pose=SE2Transform.from_SE2(lane_segment_relative_pose),
                                        lane_pose=lane_pose,
                                        lane_segment_fqn=lane_segment_fqn,
                                        lane_segment_transform=lane_segment_transform,
                                        tile_coords=tile_coords,
                                        center_point=center_point_abs_t)
                nresults += 1
        if nresults == 0:
            msg = 'Could not find any lane in tile %s' % tile_transform
            msg += '\ntile_relative_pose: %s' % tile_relative_pose
            for it2 in iterate_by_class(tile, LaneSegment):
                lane_segment = it2.object
                lane_segment_wrt_tile = it2.transform_sequence.as_SE2()
                lane_segment_relative_pose = relative_pose(lane_segment_wrt_tile, tile_relative_pose)
                lane_pose = lane_segment.lane_pose_from_SE2(lane_segment_relative_pose, tol=tol)

                msg += '\n lane_relative: %s' % lane_segment_relative_pose
                msg += '\n lane pose: %s' % lane_pose
            logger.warning(msg)


def relative_pose(base, pose):
    import geometry as geo
    return geo.SE2.multiply(geo.SE2.inverse(base), pose)
