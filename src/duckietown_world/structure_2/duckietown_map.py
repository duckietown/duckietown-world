import os
import yaml
from collections import defaultdict

from duckietown_world.yaml_include import DTYamlIncludeConstructor

from .layers import *
from .objects import *


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


class DuckietownMap(IBaseMap, ABC):

    _layers: Dict[str, 'AbstractLayer']
    _items: Dict[Tuple[str, type], '_Object']

    LAYER_KEY_TO_CLASS: Dict[str, 'AbstractLayer']
    LAYER_KEY_TO_CLASS = defaultdict(default_factory=lambda: LayerGeneral)
    LAYER_KEY_TO_CLASS.update(**{
        'frames': LayerFrames,
        'tile_maps': LayerTileMaps,
        'tiles': LayerTiles,
        'watchtowers': LayerWatchtowers,
        'groups': LayerGroups,
        'citizens': LayerCitizens
    })

    def __getattr__(self, item):
        return self._layers[item]

    def __init__(self, yaml_data: dict):
        self._layers = {}
        self._items = {}
        for layer_key, layer_content in yaml_data.items():
            layer_class = self.LAYER_KEY_TO_CLASS[layer_key]
            layer, layer_items = layer_class.deserialize(layer_content, self)
            self._layers[layer_key] = layer
            self._items.update(layer_items)

    def add(self, co: ConstructedObject):
        tp = co.obj.__class__
        self._items[(co.name, tp)] = co.obj
        self._items[(co.name, tp)].dm = self
        self._items[(co.name, _Frame)] = co.frame
        self._items[(co.name, _Frame)].dm = self

    @classmethod
    def deserialize(cls, map_name: str) -> Optional['DuckietownMap']:
        map_path = get_map_path(map_name)
        assert os.path.exists(map_path)

        DTYamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=map_path)

        main_layer_file = os.path.join(map_path, "main.yaml")
        try:
            with open(main_layer_file, 'rt') as fin:
                yaml_data = yaml.load(fin, Loader=yaml.FullLoader)
                return DuckietownMap(yaml_data["main"])
        except FileNotFoundError:
            return None

    def get_object_name(self, obj) -> Optional[str]:
        for (nm, _), ob in self._items.items():
            if obj == ob:
                return nm
        return None

    def get_object_frame(self, obj) -> Optional[_Frame]:
        name = self.get_object_name(obj)
        return self._items.get((name, _Frame), None)

    def get_object(self, name, obj_type) -> Optional[_Object]:
        return self._items.get((name, obj_type), None)

    def get_layer_objects(self, obj_type) -> Dict[str, '_Object']:
        items = {}
        for (nm, tp), ob in self._items.items():
            if obj_type == tp:
                items[nm] = ob
        return items


if __name__ == '__main__':
    map_name = "gd2/udem1"
    dm = DuckietownMap.deserialize(map_name)
    print(dm.frames)
    print(dm.watchtowers)
    wt = Watchtower("wt1", x=1.2, yaw=0.4)
    dm.add(wt)
    print(dm.frames)
    print(dm.watchtowers)

