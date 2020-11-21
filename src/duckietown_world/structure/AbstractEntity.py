from abc import ABC, abstractmethod
from typing import Union


class AbstractEntity(ABC):

    @abstractmethod
    def serialize(self) -> dict:
        pass

    # @classmethod
    # @abstractmethod
    # def deserialize(cls, data: dict, **kwargs) -> 'AbstractEntity':
    #     pass

    # @abc.abstractmethod
    # def draw_svg(self):
    #     pass