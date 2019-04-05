# coding=utf-8
from dataclasses import dataclass
from typing import *

from .tsequence import GenericSequence, X, Timestamp

__all__ = [
    'Constant',
]


@dataclass
class Constant(GenericSequence[X]):
    always: X

    def at(self, t: Timestamp) -> X:
        return self.always

    # noinspection PyUnusedLocal
    def at_or_previous(self, t: Timestamp) -> X:
        return self.always

    def get_sampling_points(self) -> str:
        return GenericSequence.CONTINUOUS

    def get_start(self) -> Optional[Timestamp]:
        return None

    def get_end(self) -> Optional[Timestamp]:
        return None
