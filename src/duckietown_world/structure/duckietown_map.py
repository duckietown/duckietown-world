from typing import Optional, Any, Tuple, List, Type, Dict, Callable, Iterator
from abc import ABC

from duckietown_world.dt_yaml import DTYaml, DTYamlLayer

from .bases import _Object, _PlacedObject, _Frame, ConstructedObject, IBaseMap, AbstractLayer


class DuckietownMap(IBaseMap, ABC):
    _layers: Dict[str, "AbstractLayer"]
    _items: Dict[Tuple[str, type], "_Object"]
    _layer_classes: Dict[str, type]

    def __init__(self, yaml_data: Dict[str, Dict[str, Any]], layer_classes: Dict[str, type]):
        self._layers = {}
        self._items = {}
        self._layer_classes = layer_classes.copy()
        for layer_key, layer_content in yaml_data.items():
            layer_class = self._layer_classes[layer_key]
            assert issubclass(layer_class, AbstractLayer)
            layer, layer_items = layer_class.deserialize(layer_content, self)
            self._layers[layer_key] = layer
            self._items.update(layer_items)
        for _, layer in self._layers.items():
            new_items = layer.items_to_update(self)
            self._items.update(new_items)

    def __getattr__(self, item: str) -> "AbstractLayer":
        return self._layers[item]

    def add(self, co: ConstructedObject) -> None:
        tp = co.obj.__class__
        assert issubclass(tp, _Object)
        self._items[(co.name, tp)] = co.obj
        self._items[(co.name, tp)].dm = self
        self._items[(co.name, _Frame)] = co.frame
        self._items[(co.name, _Frame)].dm = self

    def copy(self) -> "DuckietownMap":
        yaml_data = self.serialize(self)
        return self.deserialize(yaml_data, self._layer_classes)

    def apply_consumer(self, accept: Callable[["_Object"], None], exception: type) -> None:
        for _, ob in self._items.items():
            try:
                accept(ob)
            except exception:
                pass

    def apply_operator(self, apply: Callable[["_Object"], "_Object"], exception: type) -> None:
        for key, ob in self._items.items():
            try:
                self._items[key] = apply(ob)
            except exception:
                pass

    def get_relative_frames_list(self, frame: "_Frame") -> List["_Frame"]:
        assert frame is not None
        rel_name = frame.relative_to
        if rel_name:
            rel_frame = self.get_frame_by_name(rel_name)
            return self.get_relative_frames_list(rel_frame) + [frame]
        return [frame]

    @staticmethod
    def deserialize(yaml_data: Dict[str, Dict[str, Dict[str, Any]]], layer_classes: Dict[str, type]) -> "DuckietownMap":
        return DuckietownMap(yaml_data['main'], layer_classes)

    @staticmethod
    def serialize(dm: "DuckietownMap") -> Dict[str, Dict[str, Dict[str, Any]]]:
        return {'main': {key: layer.serialize() for key, layer in dm._layers.items()}}

    @staticmethod
    def dump(dm: "DuckietownMap") -> Dict[str, str]:
        layers = DuckietownMap.serialize(dm)['main']
        layers['main'] = {key: DTYamlLayer('%s.yaml' % key) for key in layers}
        return {key: DTYaml.dump(layer) for key, layer in layers.items()}

    def get_object_name(self, obj: "_Object") -> Optional[str]:
        for (nm, _), ob in self._items.items():
            if obj is ob:
                return nm
        return None

    def get_object_frame(self, obj: "_Object") -> Optional["_Frame"]:
        name = self.get_object_name(obj)
        return self._items.get((name, _Frame), None)

    def get_object(self, name: str, obj_type: Type["_Object"]) -> Optional["_Object"]:
        return self._items.get((name, obj_type), None)

    def get_frame_by_name(self, name: str) -> Optional["_Frame"]:
        return self._items.get((name, _Frame), None)

    def get_objects_by_name(self, name: str) -> Dict[Tuple[str, type], "_Object"]:
        return {(nm, tp): ob for (nm, tp), ob in self._items.items() if nm == name}

    def get_objects_by_type(self, obj_type: Type["_Object"]) -> Dict[Tuple[str, type], "_Object"]:
        return {(nm, tp): ob for (nm, tp), ob in self._items.items() if tp == obj_type}

    def get_placed_objects(self) -> Dict[Tuple[str, type], "_Object"]:
        return {key: ob for key, ob in self._items.items() if isinstance(ob, _PlacedObject)}

    def __iter__(self) -> Iterator[Tuple[str, type]]:
        return self._items.__iter__()
