# coding=utf-8
import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List, Optional, Type, TypeVar, Union

from zuper_typing import Generic

__all__ = [
    "Sequence",
    "GenericSequence",
    "UndefinedAtTime",
    "SampledSequence",
    "IterateDT",
    "iterate_with_dt",
]


class UndefinedAtTime(Exception):
    pass


X = TypeVar("X")
Y = TypeVar("Y")
Timestamp = float


class GenericSequence(Generic[X]):
    CONTINUOUS = "continuous-sampling"

    @abstractmethod
    def at(self, t: Timestamp) -> Generic:
        """ Raises UndefinedAtTime if not defined at t. """

    @abstractmethod
    def get_start(self) -> Optional[Timestamp]:
        """ Returns the timestamp for start, or None if -infinity. """

    @abstractmethod
    def get_end(self) -> Optional[Timestamp]:
        """ Returns the timestamp for start, or None if +infinity. """

    @abstractmethod
    def get_sampling_points(self) -> Union[str, typing.Sequence[Timestamp]]:
        """
            Returns the lists of interesting points.

            Returns the special value CONTINUOUS if
            dense sampling is possible
        """


Sequence = GenericSequence
 

@dataclass
class SampledSequence(GenericSequence):
    """ A sampled time sequence. Only defined at certain points. """

    timestamps: List[Timestamp]
    values: List[X]

    XT: ClassVar[Type[X]] = Any

    def __post_init__(self):
        values = list(self.values)
        timestamps = list(self.timestamps)

        if len(timestamps) != len(values):
            msg = "Length mismatch"
            raise ValueError(msg)

        for t in timestamps:
            if not isinstance(t, (float, int)):
                msg = "I expected a number, got %s" % type(t)
                raise ValueError(msg)
        for i in range(len(timestamps) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0:
                msg = "Invalid dt = %s at i = %s; ts= %s" % (dt, i, timestamps)
                raise ValueError(msg)
        timestamps = list(map(Timestamp, timestamps))
        self.timestamps = timestamps
        self.values = values

    def at(self, t: Timestamp) -> X:
        try:
            i = self.timestamps.index(t)
        except ValueError:
            msg = "Could not find timestamp %s in %s" % (t, self.timestamps)
            raise UndefinedAtTime(msg)
        else:
            return self.values[i]

    def at_or_previous(self, t: Timestamp) -> X:
        try:
            return self.at(t)
        except UndefinedAtTime:
            pass

        # last_t = self.timestamps[0]
        last_i = 0
        for i in range(len(self.timestamps)):
            if self.timestamps[i] < t:
                # last_t = self.timestamps[i]
                last_i = i
            else:
                break
        return self.values[last_i]

    def get_sampling_points(self) -> List[Timestamp]:
        return list(self.timestamps)

    def get_start(self) -> Timestamp:
        if not self.timestamps:
            msg = "Empty sequence"
            raise ValueError(msg)
        return self.timestamps[0]

    def get_end(self) -> Timestamp:
        if not self.timestamps:
            msg = "Empty sequence"
            raise ValueError(msg)
        return self.timestamps[-1]

    def __iter__(self):
        return zip(self.timestamps, self.values).__iter__()

    @classmethod
    def from_iterator(cls, i: typing.Iterator[X]) -> "SampledSequence[X]":
        timestamps = []
        values = []
        for t, v in i:
            timestamps.append(t)
            values.append(v)
        return SampledSequence[Any](timestamps, values)

    def __len__(self) -> int:
        return len(self.timestamps)

    def transform_values(
        self, f: Callable[[X], Y], Y: type = object
    ) -> "SampledSequence[Y]":
        values = []
        timestamps = []
        for t, _ in self:
            res = f(_)
            if res is not None:
                values.append(res)
                timestamps.append(t)

        return SampledSequence[Y](timestamps, values)

    def upsample(self, n: int) -> "SampledSequence[X]":
        timestamps = []
        values = []
        for i in range(len(self.timestamps) - 1):
            for k in range(n):
                t0 = self.timestamps[i]
                t1 = self.timestamps[i + 1]
                t = t0 + (k * 1.0 / n) * (t1 - t0)
                timestamps.append(t)
                values.append(self.values[i])
        timestamps.append(self.timestamps[-1])
        values.append(self.values[-1])
        return SampledSequence[self.XT](timestamps, values)


@dataclass
class IterateDT(Generic[Y]):
    t0: Timestamp
    t1: Timestamp
    dt: float
    v0: Y
    v1: Y


def iterate_with_dt(sequence: SampledSequence[X]) -> typing.Iterator[IterateDT[X]]:
    """ yields t0, t1, dt, v0, v1 """
    timestamps = sequence.timestamps
    values = sequence.values
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        v0 = values[i]
        v1 = values[i + 1]
        dt = t1 - t0
        yield IterateDT[type(sequence).XT](t0, t1, dt, v0, v1)  # XXX


@dataclass
class SampledSequenceBuilder(Generic[X]):
    timestamps: List[Timestamp] = None
    values: List[X] = None

    XT: typing.ClassVar[Type[X]] = Any

    def __post_init__(self):
        self.timestamps = self.timestamps or []
        self.values = self.values or []

    def add(self, t: Timestamp, v: X):
        # self.timestamps = self.timestamps or []
        # self.values = self.values or []

        # print(type(self), self.__dict__)
        self.timestamps.append(t)
        self.values.append(v)

    def __len__(self) -> int:
        return len(self.timestamps)

    def as_sequence(self) -> SampledSequence:
        return SampledSequence[self.XT](timestamps=self.timestamps, values=self.values)
