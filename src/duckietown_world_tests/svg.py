import numpy as np
from comptests import comptest, run_module_tests

import geometry as geo
from duckietown_world import SE2Transform, RectangularArea
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.lane_segment import LaneSegment
from duckietown_world.world_duckietown.tile_template import load_tile_types


@comptest
def svg1():
    control_points = [
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([0, 0], 0)),
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([1, 0], 0)),
        SE2Transform.from_SE2(geo.SE2_from_translation_angle([2, -1], np.deg2rad(-90))),
    ]

    width = 0.3

    ls = LaneSegment(width, control_points)

    area = RectangularArea((-1, -2), (3, 2))
    draw_static(ls, 'svg1', area=area)

    pass


@comptest
def svg2():
    templates = load_tile_types()

    for k, v in templates.items():
        area = RectangularArea((-1, -1), (1, 1))
        draw_static(v, 'svg2/%s' % k, area=area)

    pass


if __name__ == '__main__':
    run_module_tests()
