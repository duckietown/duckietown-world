from typing import Optional, Dict
from collections import defaultdict

import yaml

from .AbstractEntity import AbstractEntity
from .AbstractLayer import AbstractLayer
from .LayerFrames import LayerFrames
from .LayerMapFrame import LayerMapFrame
from .LayerWatchtower import LayerWatchtower
from .LayerGeneral import LayerGeneral


class DuckietownMap(AbstractEntity):

    LAYER_KEY_TO_CLASS: Dict[str, AbstractLayer]
    LAYER_KEY_TO_CLASS = defaultdict(default_factory=lambda: LayerGeneral)
    LAYER_KEY_TO_CLASS.update(**{
        'frames': LayerFrames,
        'map_frame': LayerMapFrame,
        'watchtower': LayerWatchtower,
    })

    @classmethod
    def deserialize(cls, file_path: str) -> Optional['DuckietownMap']:
        try:
            with open(file_path, 'rt') as fin:
                return DuckietownMap(yaml.safe_load(fin))
        except FileNotFoundError:
            return None

    def serialize(self) -> dict:
        pass

    def draw_svg(self):
        for layer in self._layers:
            layer.draw_svg()

    def __init__(self, map: dict):
        self._layers: Dict[str, AbstractLayer]
        self._layers = {}
        for layer_key, layer_content in map.items():
            self._layers[layer_key] = self.LAYER_KEY_TO_CLASS[layer_key].deserialize(layer_content)

    def __getattr__(self, item):
        return self._layers[item]

    def has_layer(self, name: str) -> bool:
        return hasattr(self, name)






if __name__ == '__main__':
    fpath = '/data/maps/my_map/'
    map = DuckietownMap.deserialize(fpath)
    print(map.frames)

    svg = map.draw_svg()