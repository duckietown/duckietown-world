# coding=utf-8

from comptests import comptest, run_module_tests, get_comptests_output_dir

from duckietown_serialization_ds1 import Serializable
from duckietown_world import list_maps
from duckietown_world.geo import PlacedObject, SE2Transform
from duckietown_world.seqs import Constant
from duckietown_world.world_duckietown import create_map
from duckietown_world.world_duckietown.map_loading import load_map


@comptest
def wb1():
    outdir = get_comptests_output_dir()
    root = PlacedObject()
    tile_map = create_map(H=3, W=3)

    world = PlacedObject()
    root.set_object('world', world)

    placement = Constant(SE2Transform.identity())

    world.set_object('map1', tile_map, ground_truth=placement)

    ego = PlacedObject()
    world_coordinates = Constant(SE2Transform([0, 0], 0))

    world.set_object('ego', ego, ground_truth=world_coordinates)

    d = root.as_json_dict()
    # print(json.dumps(DW.root.as_json_dict(), indent=4))
    # print(yaml.safe_dump(d, default_flow_style=False))
    # print('------')
    r1 = Serializable.from_json_dict(d)
    # print('read: %s' % r1)
    d1 = r1.as_json_dict()
    # print(yaml.safe_dump(d1, default_flow_style=False))
    # assert d == d1


@comptest
def wb2():
    root = PlacedObject()

    for map_name in list_maps():
        tm = load_map(map_name)
        root.set_object(map_name, tm)

    d = root.as_json_dict()
    # print(json.dumps(d, indent=4))
    # print(yaml.safe_dump(d, default_flow_style=False))

    # print('------')
    r1 = Serializable.from_json_dict(d)
    d1 = r1.as_json_dict()
    # print(yaml.safe_dump(d1, default_flow_style=False))
    # assert d == d1


if __name__ == '__main__':
    run_module_tests()
