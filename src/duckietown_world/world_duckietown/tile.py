# coding=utf-8
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Optional, Tuple

import numpy as np
from geometry import extract_pieces, SE2value
from svgwrite.container import Use
from zuper_commons.fs import FilePath, read_bytes_from_file
from zuper_commons.types import ZValueError

from duckietown_world.geo import (
    Matrix2D,
    PlacedObject,
    RectangularArea,
    SE2Transform,
    Transform,
    TransformSequence,
)
from duckietown_world.geo.measurements_utils import (
    iterate_by_class,
    IterateByTestResult,
)
from duckietown_world.seqs import SampledSequence
from duckietown_world.svg_drawing import data_encoded_for_src, draw_axes, draw_children
from duckietown_world.svg_drawing.misc import mime_from_fn
from . import logger
from .lane_segment import LanePose, LaneSegment
from .tile_coords import TileCoords
from .utils import relative_pose

__all__ = [
    "Tile",
    "GetLanePoseResult",
    "get_lane_poses",
    "create_lane_highlight",
    "translation_from_O3",
]


class SignSlot(PlacedObject):
    """ Represents a slot where you can put a sign. """

    L = 0.065 / 0.61

    def get_footprint(self):
        L = SignSlot.L
        return RectangularArea([-L / 2, -L / 2], [L / 2, L / 2])

    def draw_svg(self, drawing, g):
        L = SignSlot.L

        rect = drawing.rect(
            insert=(-L / 2, -L / 2),
            size=(L, L),
            fill="none",
            # style='opacity:0.4',
            stroke_width="0.005",
            stroke="pink",
        )
        g.add(rect)
        draw_axes(drawing, g, 0.04)


def get_tile_slots():
    LM = 0.5  # half tile
    # tile_offset
    to = 0.20
    # tile_curb
    tc = 0.05

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

    po = PlacedObject()
    for i, (x, y) in positions.items():
        name = str(i)
        # if name in self.children:
        #     continue

        sl = SignSlot()
        # theta = np.deg2rad(theta_deg)
        theta = 0
        t = SE2Transform((-LM + x, -LM + y), theta)
        # noinspection PyTypeChecker
        po.set_object(name, sl, ground_truth=t)
    return po


@dataclass
class FancyTextures:
    texture: np.ndarray
    normals: Optional[np.ndarray]
    emissive: Optional[np.ndarray]
    metallic_roughness: Optional[np.ndarray]
    occlusion: Optional[np.ndarray]

    fn_texture: Optional[FilePath] = None
    fn_normals: Optional[FilePath] = None
    fn_emissive: Optional[FilePath] = None
    fn_metallic_roughness: Optional[FilePath] = None
    fn_occlusion: Optional[FilePath] = None

    def write(self, prefix: str, ff: str):
        from ..utils.images import save_rgb_to_jpg
        from ..utils.images import save_rgb_to_png

        if ff == "jpg":
            ext = ".jpg"

            f = save_rgb_to_jpg
        elif ff == "png":
            ext = ".png"
            f = save_rgb_to_png
        else:
            raise ValueError(ff)

        if self.texture is not None:
            self.fn_texture = os.path.join(prefix, f"texture{ext}")
            f(self.texture, self.fn_texture)
        if self.emissive is not None:
            self.fn_emissive = os.path.join(prefix, f"emissive{ext}")
            f(self.emissive, self.fn_emissive)
        if self.normals is not None:
            self.fn_normals = os.path.join(prefix, f"normals{ext}")
            f(self.normals, self.fn_normals)
        if self.metallic_roughness is not None:
            self.fn_metallic_roughness = os.path.join(prefix, f"metallic_roughness{ext}")
            f(self.metallic_roughness, self.fn_metallic_roughness)
        if self.occlusion is not None:
            self.fn_occlusion = os.path.join(prefix, f"occlusion{ext}")
            f(self.occlusion, self.fn_occlusion)


from PIL import Image


@lru_cache(maxsize=None)
def read_rgba(fn: FilePath, resize: int) -> np.ndarray:
    try:
        im = Image.open(fn)
    except Exception as e:
        msg = f"Could not open filename {fn!r}"
        raise ValueError(msg) from e

    im = im.convert("RGBA")
    im = im.resize((resize, resize))
    data = np.array(im)
    if data.ndim != 3:
        raise ZValueError(fn=fn, shape=data.shape)
    # assert data.ndim == 4
    assert data.dtype == np.uint8
    return data


def read_rgb(fn: FilePath, resize: int) -> np.ndarray:
    try:
        im = Image.open(fn)
    except Exception as e:
        msg = f"Could not open filename {fn!r}"
        raise ValueError(msg) from e

    im = im.convert("RGB")
    im = im.resize((resize, resize))
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


# TEX_SIZE = 512


@lru_cache(maxsize=None)
def get_textures_triple(style: str, kind: str, size: int) -> FancyTextures:
    from .map_loading import get_texture_file

    fn_texture = get_texture_file(f"{style}/{kind}")[0]
    try:
        fn_normal = get_texture_file(f"{style}/{kind}-normal")[0]
    except KeyError:
        logger.warn(f"No normal for {style}/{kind}-normal")
        fn_normal = None
    try:
        fn_emissive = get_texture_file(f"{style}/{kind}-emissive")[0]
    except KeyError:
        logger.warn(f"No emissive {style}/{kind}-emissive")
        fn_emissive = None
    try:
        fn_metallic_roughness = get_texture_file(f"{style}/{kind}-metallic_roughness")[0]
    except KeyError:
        logger.warn(f"No metallic_roughness  {style}/{kind}-metallic_roughness")
        fn_metallic_roughness = None
    try:
        fn_occlusion = get_texture_file(f"{style}/{kind}-occlusion")[0]
    except KeyError:
        logger.warn(f"No occlusion for {style}/{kind}-occlusion")
        fn_occlusion = None

    TEX_SIZE = size
    texture = None if fn_texture is None else read_rgba(fn_texture, TEX_SIZE)
    normals = (
        get_straight_normal_map((TEX_SIZE, TEX_SIZE)) if fn_normal is None else read_rgb(fn_normal, TEX_SIZE)
    )
    emissive = (
        get_base_emissive((TEX_SIZE, TEX_SIZE)) if fn_emissive is None else read_rgb(fn_emissive, TEX_SIZE)
    )
    occlusion = (
        get_base_occlusion((TEX_SIZE, TEX_SIZE)) if fn_occlusion is None else read_rgb(fn_occlusion, TEX_SIZE)
    )
    metallic_roughness = (
        get_base_metallic_roughness((TEX_SIZE, TEX_SIZE))
        if fn_metallic_roughness is None
        else read_rgb(fn_metallic_roughness, TEX_SIZE)
    )

    ft = FancyTextures(
        texture=texture,
        normals=normals,
        emissive=emissive,
        metallic_roughness=metallic_roughness,
        occlusion=occlusion,
    )
    # ft.write(f"/tmp/duckietown/dw/textures/original/{style}/{kind}")

    return ft


def get_straight_normal_map(shape: Tuple[int, int]) -> np.ndarray:
    z = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    z[:, :, 0] = 128
    z[:, :, 1] = 128
    z[:, :, 2] = 255
    return z


def get_base_emissive(shape: Tuple[int, int]) -> np.ndarray:
    z = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    return z


def get_base_occlusion(shape: Tuple[int, int]) -> np.ndarray:
    z = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    z.fill(255)
    return z


def get_base_metallic_roughness(shape: Tuple[int, int]) -> np.ndarray:
    z = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    z[:, :, 0] = 255  # ignored
    z[:, :, 1] = 255
    z[:, :, 2] = 255

    return z


@lru_cache(maxsize=None)
def get_floor_textures(style: str, size: int) -> FancyTextures:
    return get_textures_triple(style, "floor", size)


@lru_cache(maxsize=None)
def get_asphalt_textures(style: str, size: int) -> FancyTextures:
    return get_textures_triple(style, "asphalt", size)


@lru_cache(maxsize=None)
def get_tape_textures(style: str, size: int) -> FancyTextures:
    return get_textures_triple(style, "tape", size)


@lru_cache(maxsize=None)
def get_fancy_textures(style: str, tile_kind: str, size: int) -> FancyTextures:
    floor = get_floor_textures(style, size)

    asphalt = get_asphalt_textures(style, size)
    tape = get_tape_textures(style, size)

    base = get_textures_triple(style, tile_kind, size)

    R = base.texture[:, :, 0]
    G = base.texture[:, :, 1]
    B = base.texture[:, :, 2]
    grey = base.texture[:, :, 0] + base.texture[:, :, 1] + base.texture[:, :, 2]
    is_floor = base.texture[:, :, 3] == 0
    is_not_floor = np.logical_not(is_floor)
    is_asphalt = np.logical_and(grey == 0, is_not_floor)

    is_white = np.logical_and(R > 250, np.logical_and(G > 250, B > 250))
    is_red = np.logical_and(R > 250, np.logical_and(G < 5, B < 5))
    is_yellow = np.logical_and(R > 250, np.logical_and(G > 250, B < 5))
    is_tape = np.logical_or(is_red, np.logical_or(is_white, is_yellow))

    texture = base.texture.copy()
    texture[is_floor] = floor.texture[is_floor]

    texture[is_asphalt] = asphalt.texture[is_asphalt]

    if base.normals is None:
        normals = None
    else:
        normals = base.normals.copy()
        if floor.normals is not None:
            normals[is_floor] = floor.normals[is_floor]
        if asphalt.normals is not None:
            normals[is_asphalt] = asphalt.normals[is_asphalt]
        if tape.normals is not None:
            normals[is_tape] = tape.normals[is_tape]

    if base.emissive is None:
        emissive = None
    else:
        emissive = base.emissive.copy()
        if floor.emissive is not None:
            emissive[is_floor] = floor.emissive[is_floor]
        if asphalt.emissive is not None:
            emissive[is_asphalt] = asphalt.emissive[is_asphalt]
        if tape.emissive is not None:
            emissive[is_tape] = tape.emissive[is_tape]

    if base.metallic_roughness is None:
        metallic_roughness = None
    else:
        metallic_roughness = base.metallic_roughness.copy()
        if floor.metallic_roughness is not None:
            metallic_roughness[is_floor] = floor.metallic_roughness[is_floor]
        if asphalt.metallic_roughness is not None:
            metallic_roughness[is_asphalt] = asphalt.metallic_roughness[is_asphalt]
        if tape.metallic_roughness is not None:
            metallic_roughness[is_tape] = tape.metallic_roughness[is_tape]

    if base.occlusion is None:
        occlusion = None
    else:
        occlusion = base.occlusion.copy()
        if floor.occlusion is not None:
            occlusion[is_floor] = floor.occlusion[is_floor]
        if asphalt.occlusion is not None:
            occlusion[is_asphalt] = asphalt.occlusion[is_asphalt]
        if tape.occlusion is not None:
            occlusion[is_tape] = tape.occlusion[is_tape]

    ft = FancyTextures(texture, normals, emissive, metallic_roughness=metallic_roughness, occlusion=occlusion)
    # ft.write(f"/tmp/duckietown/dw/textures-processed/{style}/{tile_kind}")

    return ft


#
#
# def write_textures(ft: FancyTextures, prefix: str):
#


def get_if_exists(style, kind, which: str) -> Optional[FilePath]:
    from .map_loading import get_texture_file

    q = f"tiles-processed/{style}/{kind}/{which}"
    try:
        fn = get_texture_file(q)[0]

    except KeyError:
        logger.warn(f"Could not get {q}")
        return None
    else:
        return fn


class Tile(PlacedObject):
    kind: str
    drivable: bool

    style = "photos"

    fn: Optional[FilePath]
    fn_normal: Optional[FilePath]
    fn_emissive: Optional[FilePath]
    fn_metallic_roughness: Optional[FilePath]

    def __init__(self, kind, drivable, **kwargs):
        # noinspection PyArgumentList
        PlacedObject.__init__(self, **kwargs)
        self.kind = kind
        self.drivable = drivable

        self.fn_emissive = get_if_exists(self.style, kind, "emissive")
        self.fn_normal = get_if_exists(self.style, kind, "normals")
        self.fn = get_if_exists(self.style, kind, "texture")
        self.fn_metallic_roughness = get_if_exists(self.style, kind, "metallic_roughness")
        self.fn_occlusion = get_if_exists(self.style, kind, "occlusion")

        if not "slots" in self.children:
            slots = get_tile_slots()
            # noinspection PyTypeChecker
            self.set_object("slots", slots, ground_truth=SE2Transform.identity())

    def _copy(self):

        return type(self)(
            self.kind,
            self.drivable,
            children=dict(self.children),
            spatial_relations=dict(self.spatial_relations),
        )

    def params_to_json_dict(self):
        return dict(kind=self.kind, drivable=self.drivable)

    def get_footprint(self):
        return RectangularArea([-0.5, -0.5], [0.5, 0.5])

    def draw_svg(self, drawing, g):
        T = 0.562 / 0.585
        T = 1
        S = 1.0
        rect = drawing.rect(insert=(-S / 2, -S / 2), size=(S, S), fill="#0a0", stroke="none")
        g.add(rect)

        if self.fn:
            texture = read_bytes_from_file(self.fn)
            if b"git-lfs" in texture:
                msg = f"The file {self.fn} is a Git LFS pointer. Repo not checked out correctly."
                raise Exception(msg)

            ID = f"texture-{self.kind}"

            for img in drawing.defs.elements:
                if img.attribs.get("id", None) == ID:
                    break
            else:

                href = data_encoded_for_src(texture, mime_from_fn(self.fn))
                img = drawing.image(
                    href=href,
                    size=(T, T),
                    insert=(-T / 2, -T / 2),
                    # style=" ",
                    style="transform: rotate(0deg) scaleX(-1)  rotate(-90deg) ",
                )
                img.attribs["class"] = "tile-textures"

                img.attribs["id"] = ID
                drawing.defs.add(img)

            use = Use(f"#{ID}")
            g.add(use)
        #
        # if draw_directions_lanes:
        #     if self.kind != 'floor':
        #         start = (-0.5, -0.25)
        #         end = (+0, -0.25)
        #         line = drawing.line(start=start, end=end, stroke='blue', stroke_width='0.01')
        #         g.add(line)

        draw_axes(drawing, g)

        draw_children(drawing, self, g)


@dataclass
class GetLanePoseResult:
    tile: Tile
    tile_fqn: Tuple[str, ...]
    tile_transform: TransformSequence
    tile_relative_pose: Matrix2D
    lane_segment: LaneSegment
    lane_segment_fqn: Tuple[str, ...]
    lane_pose: LanePose
    lane_segment_relative_pose: Matrix2D
    tile_coords: TileCoords
    lane_segment_transform: TransformSequence
    center_point: Matrix2D


def get_lane_poses(dw: PlacedObject, q: SE2value, tol: float = 0.000001) -> Iterator[GetLanePoseResult]:
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
            msg = "Could not find tile coords in %s" % tile_transform
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
            lane_segment_transform = TransformSequence(
                tile_transform.transforms + it2.transform_sequence.transforms
            )
            lane_pose = lane_segment.lane_pose_from_SE2(lane_segment_relative_pose, tol=tol)

            M = lane_segment_transform.asmatrix2d().m
            center_point = lane_pose.center_point.as_SE2()

            center_point_abs = np.dot(M, center_point)
            center_point_abs_t = Matrix2D(center_point_abs)

            if lane_pose.along_inside and lane_pose.inside and lane_pose.correct_direction:
                yield GetLanePoseResult(
                    tile=tile,
                    tile_fqn=tile_fqn,
                    tile_transform=tile_transform,
                    tile_relative_pose=Matrix2D(tile_relative_pose),
                    lane_segment=lane_segment,
                    lane_segment_relative_pose=Matrix2D(lane_segment_relative_pose),
                    lane_pose=lane_pose,
                    lane_segment_fqn=lane_segment_fqn,
                    lane_segment_transform=lane_segment_transform,
                    tile_coords=tile_coords,
                    center_point=center_point_abs_t,
                )
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


def translation_from_O3(pose) -> np.ndarray:
    """ Returns a 2x array """
    _, t, _, _ = extract_pieces(pose)
    return t


class GetClosestLane:
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
        draw_axes(drawing, g, klass="anchor-axes")
        c = drawing.circle(center=(0, 0), r=0.03, fill="blue", stroke="black", stroke_width=0.001)
        g.add(c)


def create_lane_highlight(poses_sequence: SampledSequence, dw):
    def mapi(v):
        if isinstance(v, SE2Transform):
            return v.as_SE2()
        else:
            return v

    poses_sequence = poses_sequence.transform_values(mapi, np.ndarray)

    lane_pose_results = poses_sequence.transform_values(GetClosestLane(dw), object)

    if False:
        visualization = PlacedObject()
        dw.set_object("visualization", visualization, ground_truth=SE2Transform.identity())

        # center_points = lane_pose_seq.transform_values(lambda _: 1.0, float)
        for i, (timestamp, name2pose) in enumerate(lane_pose_results):
            for name, lane_pose_result in name2pose.items():
                assert isinstance(lane_pose_result, GetLanePoseResult)
                lane_segment = lane_pose_result.lane_segment
                rt = lane_pose_result.lane_segment_transform
                s = SampledSequence[Transform]([timestamp], [rt])
                visualization.set_object("ls%s-%s-lane" % (i, name), lane_segment, ground_truth=s)
                p = SampledSequence[Transform]([timestamp], [lane_pose_result.center_point])
                visualization.set_object("ls%s-%s-anchor" % (i, name), Anchor(), ground_truth=p)

    return lane_pose_results
