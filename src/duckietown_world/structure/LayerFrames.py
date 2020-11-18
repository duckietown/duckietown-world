from typing import Union

from .AbstractLayer import AbstractLayer


class LayerFrames(AbstractLayer):

    def serialize(self) -> dict:
        pass

    def deserialize(self, data: Union[str, dict]) -> 'LayerFrames':
        pass
