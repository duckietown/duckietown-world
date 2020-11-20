from abc import ABCMeta
from typing import Union

from . import AbstractEntity


class AbstractLayer(AbstractEntity, metaclass=ABCMeta):

    def items(self):
        pass
