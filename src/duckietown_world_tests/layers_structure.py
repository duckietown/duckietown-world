import os
import duckietown_world as dw
import geometry as geo
import duckietown_world.structure as st
from duckietown_world.world_duckietown.tile_utils import get_tile_at_point
from duckietown_world.svg_drawing.draw_maps import draw_map
from comptests import comptest, run_module_tests, get_comptests_output_dir


def placed_object_check(m):
    p = geo.SE2_from_translation_angle([1.3, 0.3], 0.2)
    r = get_tile_at_point(m, p)
    assert (r.i, r.j) == (2, 0)

@comptest
def layers_map():
    map_name = "gd2/udem1"
    dm = st.DuckietownMap.deserialize(map_name)

    m: dw.DuckietownMap = dm.tile_maps.get("map_1")["map_object"]
    placed_object_check(m)

    outdir = get_comptests_output_dir()
    out = os.path.join(outdir, map_name)
    draw_map(out, m)

    print("=========================== GROUPS ===========================")
    import pprint
    pprint.pprint(dict(dm.groups.items()))


if __name__ == "__main__":
    run_module_tests()
