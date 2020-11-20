import abc
from typing import Union


class AbstractEntity(abc.ABC):

    @abc.abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: dict, **kwargs) -> 'AbstractEntity':
        pass

    # @abc.abstractmethod
    # def draw_svg(self):
    #     pass