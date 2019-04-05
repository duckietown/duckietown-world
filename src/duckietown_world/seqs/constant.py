# coding=utf-8
from dataclasses import dataclass
from typing import *

from duckietown_serialization_ds1.serialization1 import as_json_dict

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

    def as_json_dict(self):
        return {'always': as_json_dict(self.always)}
