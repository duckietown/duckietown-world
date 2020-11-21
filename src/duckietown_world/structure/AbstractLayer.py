from abc import ABCMeta, abstractmethod
from typing import Union

from . import AbstractEntity


class AbstractLayer(AbstractEntity, metaclass=ABCMeta):
    _items: dict

    def __init__(self):
        self._items = {}

    def __getattr__(self, item):
        return self._items[item]

    def __getitem__(self, item):
        return self._items[item]

    def __setitem__(self, key, value):
        self._items[key] = value

    def get(self, item, default=None):
        return self._items.get(item, default)

    def __iter__(self):
        return self._items.__iter__()

    def items(self):
        return self._items.items()

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict, dm: 'DuckietownMap') -> 'AbstractLayer':
        pass
