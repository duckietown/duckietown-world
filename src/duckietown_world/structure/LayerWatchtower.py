from typing import Union

from .AbstractLayer import AbstractLayer


class LayerWatchtower(AbstractLayer):

    def serialize(self) -> dict:
        pass

    def deserialize(self, data: Union[str, dict]) -> 'LayerWatchtower':
        pass

