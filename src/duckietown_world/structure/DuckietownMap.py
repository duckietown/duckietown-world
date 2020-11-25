from typing import Optional, Dict, List
from collections import defaultdict

import os
import yaml

from .AbstractEntity import AbstractEntity
from .AbstractLayer import AbstractLayer
from .LayerGeneral import LayerGeneral
from .LayerFrames import LayerFrames
from .LayerTileMaps import LayerTileMaps
from .LayerTiles import LayerTiles
from .LayerWatchtowers import LayerWatchtowers
from .LayerGroups import LayerGroups

import duckietown_world as dw
from duckietown_world.yaml_include import YamlIncludeConstructor
from duckietown_world.world_duckietown.tile import Tile
from duckietown_world.geo.measurements_utils import iterate_by_class
from duckietown_world.geo import PlacedObject
from duckietown_world.svg_drawing.draw_maps import draw_map


class DuckietownMap(AbstractEntity):

    _layers: Dict[str, AbstractLayer]
    root: PlacedObject

    LAYER_KEY_TO_CLASS: Dict[str, AbstractLayer]
    LAYER_KEY_TO_CLASS = defaultdict(default_factory=lambda: LayerGeneral)
    LAYER_KEY_TO_CLASS.update(**{
        'frames': LayerFrames,
        'tile_maps': LayerTileMaps,
        'tiles': LayerTiles,
        'watchtowers': LayerWatchtowers,
        'groups': LayerGroups,
    })
    OBJECT_LAYERS: List[str]
    OBJECT_LAYERS = [
        'watchtowers'
    ]
    LAYERS_ORDER: Dict[str, int]
    LAYERS_ORDER = {
        'frames': 0,
        'tile_maps': 1,
        'tiles': 2,
        'default': 100,
        'groups': 200
    }

    def get_order(self, layer):
        return self.LAYERS_ORDER.get(layer, self.LAYERS_ORDER["default"])

    @classmethod
    def deserialize(cls, map_name: str) -> Optional['DuckietownMap']:
        map_path = get_map_path(map_name)
        assert os.path.exists(map_path)

        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=map_path)

        main_layer_file = os.path.join(map_path, "main.yaml")
        try:
            with open(main_layer_file, 'rt') as fin:
                yaml_data = yaml.load(fin, Loader=yaml.FullLoader)
                return DuckietownMap(yaml_data["main"])
        except FileNotFoundError:
            return None

    class YamlLayer:
        name: str

        def __init__(self, name: str):
            self.name = name

        @staticmethod
        def to_yaml(dumper, data):
            return dumper.represent_scalar("!include", data.name, style=None)

    def serialize(self, map_name: str) -> None:
        map_path = get_existing_map_path(map_name)

        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '~')

        def quoted_presenter(dumper, data):
            if len(data.split()) > 1:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(type(None), represent_none)
        yaml.add_representer(str, quoted_presenter)
        yaml.add_representer(DuckietownMap.YamlLayer, DuckietownMap.YamlLayer.to_yaml)

        main_layer_file = os.path.join(map_path, "main.yaml")
        main_layer = {"main": {}}

        for layer_key, layer_content in self._layers.items():
            layer_file_name = "%s.yaml" % layer_key
            main_layer["main"][layer_key] = DuckietownMap.YamlLayer(layer_file_name)

            layer_dict = self._layers[layer_key].serialize()
            layer_file = os.path.join(map_path, layer_file_name)
            with open(layer_file, 'w') as f:
                yaml.dump(layer_dict, f)

        with open(main_layer_file, 'w') as f:
            yaml.dump(main_layer, f)

    def draw(self, path, map_name):
        map_path = get_existing_map_path(path)
        m: dw.DuckietownMap = self.tile_maps[map_name]["map_object"]
        draw_map(map_path, m)

    def __init__(self, yaml_data: dict):
        self._layers = {}
        self.root = PlacedObject()
        data_items = sorted(list(yaml_data.items()), key=lambda p: self.get_order(p[0]))
        for layer_key, layer_content in data_items:
            layer_class = self.LAYER_KEY_TO_CLASS[layer_key]
            self._layers[layer_key] = layer_class.deserialize(layer_content, self)  # TODO assume basic layers exist
        self.bind_objects()
        self.finish_building()

    def bind_objects(self):
        processed_objects = {}
        for name, mp in self.tile_maps.items():
            frame = self.frames.get(name, None)
            if frame is None:
                msg = "not found frame for map " + name
                raise ValueError(msg)
            self.root.set_object(name, mp["map_object"], ground_truth=frame["transform"])
            processed_objects[name] = mp["map_object"]

        objects = {}
        for layer_name, layer_data in self._layers.items():
            if layer_name not in self.OBJECT_LAYERS:
                continue
            for name, desc in layer_data.items():
                frame_name = desc["name"]
                depth = frame_name.count("/")
                if depth not in objects:
                    objects[depth] = []
                objects[depth].append(desc)
        for depth in objects:
            for o in objects[depth]:
                frame = self.frames.get(o["name"], None)
                if frame is None:
                    msg = "not found frame for object " + o["name"]
                    raise ValueError(msg)
                o["frame"] = frame
                frame["object"] = o
        sorted_depths = sorted(list(objects))
        for depth in sorted_depths:
            for o in objects[depth]:
                if depth == 0:
                    parent_object = self.root
                else:
                    parent_frame = o["frame"]["relative_to"]
                    parent_object = processed_objects[parent_frame]
                    if parent_object is None:
                        msg = "not found object for frame " + parent_frame
                        raise ValueError(msg)
                parent_object.set_object(o["obj_name"], o["obj"], ground_truth=o["frame"]["transform"])
                processed_objects[o["name"]] = o["obj"]

    def finish_building(self):
        for _, tile_map in self.tile_maps.items():
            for it in list(
                    iterate_by_class(tile_map["map_object"].children["tilemap_wrapper"].children["tilemap"], Tile)):
                ob = it.object
                if "slots" in ob.children:
                    slots = ob.children["slots"]
                    for k, v in list(slots.children.items()):
                        if not v.children:
                            slots.remove_object(k)
                    if not slots.children:
                        ob.remove_object("slots")

    def __getattr__(self, item):
        return self._layers[item]

    def __iter__(self):
        return self._layers.__iter__()

    def has_layer(self, name: str) -> bool:
        return hasattr(self, name)

    def inner_items(self):
        items = {}
        for layer_name, layer_data in self._layers.items():
            if layer_name != "frames":
                items.update(layer_data.items())
        return items


def get_existing_map_path(map_name):
    map_path = get_map_path(map_name)
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    return map_path


def get_map_path(map_name):
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    map_path = os.path.join(module_dir, "../data", map_name)
    return map_path


if __name__ == '__main__':
    map_name = "gd2/udem1"
    dm = DuckietownMap.deserialize(map_name)
    print(dm.frames)
