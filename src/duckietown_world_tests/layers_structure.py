import os

from comptests import comptest, run_module_tests, get_comptests_output_dir
from pprint import pprint

from duckietown_world.structure.map_factory import MapFactory
from duckietown_world.structure.objects import Watchtower
from duckietown_world.svg_drawing.draw_maps import draw_map

@comptest
def layers_map():
    map_name = "gd2/udem1"
    dm = MapFactory.load_map(map_name)

    print("================ internal map objects ================")
    pprint(dm._layers)
    print("------------------------------------------------------")
    pprint(dm._items)

    print("==================== items access ====================")
    print(dm.frames)
    print("------------------------------------------------------")
    print(dm.watchtowers['watchtower1/watchtower2'])
    print("------------------------------------------------------")
    print(dm.tile_maps.map_1)
    print("------------------------------------------------------")
    print(dm.tile_maps.map_1.x)
    print("------------------------------------------------------")
    print(dm.watchtowers['watchtower1/watchtower2'].frame)

    print("================== object addition ===================")
    wt = Watchtower("wt1", x=1.2, yaw=0.4)
    dm.add(wt)
    print(dm.frames)
    print("------------------------------------------------------")
    print(dm.watchtowers)

    print("==================== map drawing =====================")
    outdir = get_comptests_output_dir()
    draw_path = os.path.join(outdir, map_name)
    draw_map(draw_path, dm)


if __name__ == "__main__":
    run_module_tests()
