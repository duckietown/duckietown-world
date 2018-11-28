# coding=utf-8
from abc import abstractmethod
from collections import namedtuple

from contracts import describe_value

from duckietown_serialization_ds1 import Serializable

__all__ = [
    'Sequence',
    'UndefinedAtTime',
    'SampledSequence',
    'IterateDT',
    'iterate_with_dt',
]


class UndefinedAtTime(Exception):
    pass


class Sequence(Serializable):
    CONTINUOUS = 'continuous-sampling'

    # __metaclass__ = ABCMeta

    # def params_to_json_dict(self):
    #     return None

    # @classmethod
    # def params_from_json_dict(cls, d):
    #     return {}

    @abstractmethod
    def at(self, t):
        """ Raises UndefinedAtTime if not defined at t. """

    @abstractmethod
    def get_start(self):
        """ Returns the timestamp for start, or None if -infinity. """

    @abstractmethod
    def get_end(self):
        """ Returns the timestamp for start, or None if +infinity. """

    @abstractmethod
    def get_sampling_points(self):
        """
            Returns the lists of interesting points.

            Returns the special value CONTINUOUS if
            dense sampling is possible
        """


class SampledSequence(Sequence):
    """ A sampled time sequence. Only defined at certain points. """

    def __init__(self, timestamps, values):
        values = list(values)
        timestamps = list(timestamps)
        if not timestamps:
            msg = 'Empty sequence.'
            raise ValueError(msg)

        if len(timestamps) != len(values):
            msg = 'Length mismatch'
            raise ValueError(msg)

        for t in timestamps:
            if not isinstance(t, (float, int)):
                msg = 'I expected a number, got %s' % describe_value(t)
                raise ValueError(msg)
        for i in range(len(timestamps) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0:
                msg = 'Invalid dt = %s at i = %s; ts= %s' % (dt, i, timestamps)
                raise ValueError(msg)
        timestamps = list(map(float, timestamps))
        self.timestamps = timestamps
        self.values = values

    def at(self, t):
        try:
            i = self.timestamps.index(t)
        except ValueError:
            msg = 'Could not find timestamp %s in %s' % (t, self.timestamps)
            raise UndefinedAtTime(msg)
        else:
            return self.values[i]

    def get_sampling_points(self):
        return list(self.timestamps)

    def get_start(self):
        return self.timestamps[0]

    def get_end(self):
        return self.timestamps[-1]

    def __iter__(self):
        return zip(self.timestamps, self.values).__iter__()

    @classmethod
    def from_iterator(cls, i):
        timestamps = []
        values = []
        for t, v in i:
            timestamps.append(t)
            values.append(v)
        return SampledSequence(timestamps, values)

    def __len__(self):
        return len(self.timestamps)

    def transform_values(self, f):
        values = []
        timestamps = []
        for t, _ in self:
            res = f(_)
            if res is not None:
                values.append(res)
                timestamps.append(t)

        return SampledSequence(timestamps, values)

    def upsample(self, n):
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
        return SampledSequence(timestamps, values)


IterateDT = namedtuple('IterateDT', 't0 t1 dt v0 v1')


def iterate_with_dt(sequence):
    """ yields t0, t1, dt, v0, v1 """
    timestamps = sequence.timestamps
    values = sequence.values
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        v0 = values[i]
        v1 = values[i + 1]
        dt = t1 - t0
        yield IterateDT(t0, t1, dt, v0, v1)


class SampledSequenceBuilder(object):
    def __init__(self):
        self.timestamps = []
        self.values = []

    def add(self, t, v):
        self.timestamps.append(t)
        self.values.append(v)

    def __len__(self):
        return len(self.timestamps)

    def as_sequence(self):
        return SampledSequence(self.timestamps, self.values)
