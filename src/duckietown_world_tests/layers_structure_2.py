import duckietown_world.structure_2 as st
from comptests import comptest, run_module_tests
from pprint import pprint


@comptest
def layers_map():
    map_name = "gd2/udem1"
    dm = st.DuckietownMap.deserialize(map_name)

    print("================ internal map objects ================")
    pprint(dm._layers)
    print("------------------------------------------------------")
    pprint(dm._items)

    print("==================== items access ====================")
    print(dm.frames)
    print("------------------------------------------------------")
    print(dm.watchtowers['map_1/watchtower1'])
    print("------------------------------------------------------")
    print(dm.tile_maps.map_1)
    print("------------------------------------------------------")
    print(dm.tile_maps.map_1.x)
    print("------------------------------------------------------")
    print(dm.watchtowers['map_1/watchtower1'].frame)

    print("================== object addition ===================")
    wt = st.Watchtower("wt1", x=1.2, yaw=0.4)
    dm.add(wt)
    print(dm.frames)
    print("------------------------------------------------------")
    print(dm.watchtowers)
    print("------------------------------------------------------")
    print(dm.citizens)



if __name__ == "__main__":
    run_module_tests()
