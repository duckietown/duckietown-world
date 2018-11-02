# coding=utf-8
import os

import numpy as np
from comptests import comptest, run_module_tests, get_comptests_output_dir

import geometry as geo
from duckietown_world import SE2Transform, RectangularArea, list_maps
from duckietown_world.svg_drawing import draw_static
from duckietown_world.svg_drawing.draw_maps import draw_map
from duckietown_world.world_duckietown.lane_segment import LaneSegment
from duckietown_world.world_duckietown.map_loading import load_map
from duckietown_world.world_duckietown.tile_template import load_tile_types


@comptest
def svg1():
    outdir = get_comptests_output_dir()
    control_points = [
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([0, 0], 0)),
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([1, 0], 0)),
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([2, -1], np.deg2rad(-90))),
    ]

    width = 0.3

    ls = LaneSegment(width, control_points)

    area = RectangularArea((-1, -2), (3, 2))
    draw_static(ls, outdir, area=area)


@comptest
def svg2():
    outdir = get_comptests_output_dir()
    templates = load_tile_types()

    for k, v in templates.items():
        area = RectangularArea((-1, -1), (1, 1))
        dest = os.path.join(outdir, k)
        draw_static(v, dest, area=area)


@comptest
def maps():
    outdir = get_comptests_output_dir()

    map_names = list_maps()
    print(map_names)

    for map_name in map_names:
        duckietown_map = load_map(map_name)
        out = os.path.join(outdir, map_name)
        draw_map(out, duckietown_map)


if __name__ == '__main__':
    run_module_tests()
