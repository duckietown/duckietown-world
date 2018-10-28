# coding=utf-8
from abc import abstractmethod
from itertools import izip

from duckietown_serialization_ds1 import Serializable

__all__ = ['Sequence']


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
        """ Raises KeyError if not defined at t. """
        pass

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
                msg = 'I expected a number, got %s' % t
                raise ValueError(t)

        self.timestamps = timestamps
        self.values = values

    def at(self, t):
        try:
            i = self.timestamps.index(t)
        except ValueError:
            raise KeyError(t)
        else:
            return self.values[i]

    def get_sampling_points(self):
        return list(self.timestamps)

    def get_start(self):
        return self.timestamps[0]

    def get_end(self):
        return self.timestamps[-1]

    def __iter__(self):
        return izip(self.timestamps, self.values)

    def __len__(self):
        return len(self.timestamps)
