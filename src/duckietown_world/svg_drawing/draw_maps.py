import argparse
import os
import sys

from duckietown_world.svg_drawing import get_basic_upright, draw_recursive
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap
from duckietown_world.world_duckietown.map_loading import list_gym_maps, load_gym_map


def draw_maps_main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output dir", default='out-draw_maps')
    parsed = parser.parse_args(args)

    output = parsed.output

    map_names = list_gym_maps()
    print(map_names)

    for map_name in map_names:
        duckietown_map = load_gym_map(map_name)
        out = os.path.join(output, map_name)

        draw_map(out, duckietown_map)


def draw_map(output, duckietown_map):
    if not os.path.exists(output):
        os.makedirs(output)
    assert isinstance(duckietown_map, DuckietownMap)
    tilemap = duckietown_map.children['tilemap']
    gh, gw = tilemap.H * duckietown_map.tile_size, tilemap.W * duckietown_map.tile_size
    # gh = int(math.ceil(gh))
    # gw = int(math.ceil(gw))
    B = 640
    pixel_size = (B, B * gw / gh)
    space = (gh, gw)
    drawing, base = get_basic_upright('out.svg', space, pixel_size)

    gmg = drawing.g(id='duckietown_map')
    draw_recursive(drawing, duckietown_map, gmg)
    base.add(gmg)

    fn = os.path.join(output, 'map.svg')
    drawing.filename = fn
    drawing.save(pretty=True)
    print('written to %s' % fn)


if __name__ == '__main__':
    draw_maps_main()
