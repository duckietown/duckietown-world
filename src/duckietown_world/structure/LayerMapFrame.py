from typing import Union

from .AbstractLayer import AbstractLayer


class LayerMapFrame(AbstractLayer):

    def serialize(self) -> dict:
        pass

    def deserialize(self, data: Union[str, dict]) -> 'LayerMapFrame':
        pass

