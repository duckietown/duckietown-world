import os
import duckietown_world as dw
import geometry as geo
from duckietown_world.world_duckietown.tile_utils import get_tile_at_point
from duckietown_world.svg_drawing.draw_maps import draw_map
from comptests import comptest, run_module_tests, get_comptests_output_dir


@comptest
def layers_map(map_name="udem1"):
    m: dw.DuckietownMap = dw.load_map_layers(map_name)[0]

    p = geo.SE2_from_translation_angle([1.3, 0.3], 0.2)
    r = get_tile_at_point(m, p)
    assert (r.i, r.j) == (2, 0)

    outdir = get_comptests_output_dir()

    out = os.path.join(outdir, map_name)
    draw_map(out, m)


if __name__ == "__main__":
    run_module_tests()
