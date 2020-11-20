from typing import Union

from . import AbstractLayer


class LayerGeneral(AbstractLayer):

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: Union[str, dict]) -> 'LayerGeneral':
        pass

