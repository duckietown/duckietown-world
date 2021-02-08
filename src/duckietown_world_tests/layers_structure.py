import os
import sys

from comptests import comptest, run_module_tests, get_comptests_output_dir
from pprint import pprint

from duckietown_world.structure.map_factory import MapFactory
from duckietown_world.structure.duckietown_map import DuckietownMap
from duckietown_world.structure.objects import Watchtower, Tile
from duckietown_world.svg_drawing.draw_maps import draw_map


def print_tiles(dm: DuckietownMap):
    tiles = dm.tiles.only_tiles()
    for row in tiles:
        for tile in row:
            print(tile)

@comptest
def layers_map():
    #map_name = "gd2/udem1"
    map_name = "maps/test_draw_8"
    #map_name = "maps/empty"
    dm = MapFactory.load_map(map_name)

    print("================ internal map objects ================")
    pprint(dm._layers)
    print("------------------------------------------------------")
    pprint(dm._items)

    print("==================== items access ====================")
    print(dm.frames)
    print("------------------------------------------------------")
    print_tiles(dm)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('---- !---- ! -')
    print(dm.frames)
    print(dm.tile_maps['map_1'].x)

    #print(tiles)
    #print("------1111111111111------------------------------------------------")
    #print(tiles[1])
    #print("------------------------------------------------------")
    #print(tiles[3][4])
    #print(dm.watchtowers['watchtower1/watchtower2'])
    print("------------------------------------------------------")
    #print(dm.tile_maps.map_1)
    print("------------------------------------------------------")
    #print(dm.tile_maps.map_1.x)
    print("------------------------------------------------------")
    print(dm.tiles.only_tiles())
    #print(dm.watchtowers['watchtower1/watchtower2'].frame)

    #print("================== object addition ===================")
    #wt = Watchtower("wt1", x=1.2, yaw=0.4)
    #dm.add(wt)
    #print(dm.frames)
    #print("------------------------------------------------------")
    #print(dm.watchtowers)

    print("==================== map drawing =====================")
    outdir = get_comptests_output_dir()
    draw_path = os.path.join(outdir, map_name)
    draw_map(draw_path, dm)


if __name__ == "__main__":
    run_module_tests()
