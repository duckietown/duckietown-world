from abc import ABC
from typing import Union

from .AbstractLayer import AbstractLayer


class LayerGroups(AbstractLayer, ABC):
    def __init__(self, data: dict, dm: 'DuckietownMap'):
        super().__init__()
        dm_items = dm.inner_items()
        self._items = data
        for group_name, group_data in self._items.items():
            member_objects = []
            for member in group_data["members"]:
                if member not in dm_items:
                    msg = "not found object " + member + " for group " + group_name
                    raise ValueError(msg)
                member_objects.append(dm_items[member])
            group_data["members"] = {k: v for k, v in list(zip(group_data["members"], member_objects))}

    @classmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'LayerGroups':
        return LayerGroups(data, dm)

    def serialize(self) -> dict:
        yaml_dict = {}
        for item_name, item_data in self._items.items():
            yaml_dict[item_name] = {"description": item_data["description"], "members": list(item_data["members"])}
        return {"groups": yaml_dict}
