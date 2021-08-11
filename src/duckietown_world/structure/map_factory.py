import os
from typing import Optional, Dict
from collections import defaultdict

from duckietown_world.dt_yaml.dt_yaml import DTYaml

from .duckietown_map import DuckietownMap
from .layers import LayerGeneral, LayerFrames, LayerTileMaps, LayerTiles, LayerWatchtowers, LayerGroups, LayerCitizens,\
    LayerTrafficSigns, LayerGroundTags, LayerVehicles, LayerCameras, LayerDecorations, LayerEnvironment, \
    LayerVehicleTags, LayerLights


def get_existing_map_path(map_name: str) -> str:
    if os.path.isabs(map_name):
        return map_name
    map_path = get_map_path(map_name)
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    return map_path


def get_map_path(map_name: str) -> str:
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    map_path = os.path.join(module_dir, '../data', map_name)
    return map_path


class MapFactory:
    LAYER_KEY_TO_CLASS: Dict[str, type] = defaultdict(default_factory=lambda: LayerGeneral)
    LAYER_KEY_TO_CLASS.update(**{
        'frames': LayerFrames,
        'tile_maps': LayerTileMaps,
        'tiles': LayerTiles,
        'watchtowers': LayerWatchtowers,
        'groups': LayerGroups,
        'citizens': LayerCitizens,
        'ground_tags': LayerGroundTags,
        'traffic_signs': LayerTrafficSigns,
        'vehicles': LayerVehicles,
        'cameras': LayerCameras,
        'decorations': LayerDecorations,
        'environment': LayerEnvironment,
        'vehicle_tags': LayerVehicleTags,
        'lights': LayerLights
    })

    @classmethod
    def load_map(cls, map_name: str) -> Optional["DuckietownMap"]:
        map_path = get_existing_map_path(map_name)

        main_layer_file = os.path.join(map_path, 'main.yaml')
        if not os.path.isfile(main_layer_file):
            return None
        with open(main_layer_file, 'rt') as fin:
            yaml_data = DTYaml.load(map_path, fin)
            return DuckietownMap.deserialize(yaml_data, cls.LAYER_KEY_TO_CLASS)


if __name__ == '__main__':
    from .objects import Watchtower

    map_name = 'gd2/udem1'
    dm = MapFactory.load_map(map_name)
    print(dm.frames)
    print(dm.watchtowers)
    wt = Watchtower('wt1', x=1.2, yaw=0.4)
    dm.add(wt)
    print(dm.frames)
    print(dm.watchtowers)
