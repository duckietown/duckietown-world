from typing import Union

from .AbstractLayer import AbstractLayer


class LayerGeneral(AbstractLayer):

    def serialize(self) -> dict:
        pass

    def deserialize(self, data: Union[str, dict]) -> 'LayerGeneral':
        pass

