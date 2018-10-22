# coding=utf-8
import json

import yaml

from comptests import comptest, run_module_tests

from duckietown_world.geo import PlacedObject, SE2Transform, get_meausurements_graph
from duckietown_world.seqs import Constant
from duckietown_world.world_duckietown import DuckietownWorld, create_map, load_gym_maps

from duckietown_serialization import Serializable


@comptest
def wb1():
    DW = DuckietownWorld()
    tile_map = create_map(H=3, W=3)

    world = PlacedObject()
    DW.root.set_object('world', world)

    placement = Constant(SE2Transform.identity())

    world.set_object('map1', tile_map, prior=placement)

    ego = PlacedObject()
    world_coordinates = Constant(SE2Transform([0, 0], 0))

    world.set_object('ego', ego, trajectory=world_coordinates)
    #
    # placement={
    #     'world': world_coordinates,
    #     'tile-1-2': Constant(SE2Transform.identity())
    # })

    r0 = DW.root
    d = r0.as_json_dict()
    print(json.dumps(DW.root.as_json_dict(), indent=4))
    print(yaml.safe_dump(d, default_flow_style=False))
    print('------')
    r1 = Serializable.from_json_dict(d)
    print('read: %s' % r1)
    d1 = r1.as_json_dict()
    print(yaml.safe_dump(d1, default_flow_style=False))
    assert d == d1

    G = get_meausurements_graph(world)


@comptest
def wb2():
    gym_maps = load_gym_maps()
    DW = DuckietownWorld()
    for map_name, tm in gym_maps.items():
        DW.root.set_object(map_name, tm)
    r0 = DW.root
    d = r0.as_json_dict()
    print(json.dumps(DW.root.as_json_dict(), indent=4))
    print(yaml.safe_dump(d, default_flow_style=False))

    print('------')
    r1 = Serializable.from_json_dict(d)
    d1 = r1.as_json_dict()
    print(yaml.safe_dump(d1, default_flow_style=False))
    assert d == d1

    # G = get_meausurements_graph(world)


if __name__ == '__main__':
    run_module_tests()
