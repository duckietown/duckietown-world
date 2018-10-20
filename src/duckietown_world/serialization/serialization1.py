# coding=utf-8
import json
import traceback
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from copy import deepcopy

import numpy as np

__all__ = ['Serializable', 'from_json_dict2']


class CouldNotDeserialize(Exception):
    pass


class Serializable0(object):
    __metaclass__ = ABCMeta

    def __repr__(self):
        params = self.params_to_json_dict()
        s = ",".join('%s=%s'%(k,v) for k,v in params.items())
        return '%s(%s)' % (type(self).__name__, s)

    @abstractmethod
    def params_to_json_dict(self):
        pass

    def as_json_dict(self):
        mro = type(self).mro()
        res = {}
        for k in mro:
            if k is object or k is Serializable0 or k is Serializable:
                continue
            # noinspection PyUnresolvedReferences
            if hasattr(k, 'params_to_json_dict'):
                params = k.params_to_json_dict(self)
                if params is not None:
                    params = as_json_dict(params)

                    res[k.__name__] = params
        return res

    @classmethod
    def params_from_json_dict(cls, d):
        if not isinstance(d, dict):
            msg = 'Expected d to be a dict, got %s' % type(d).__name__
            raise ValueError(msg)
        params = {}
        mro = cls.mro()
        for k in mro:
            if k is object or k is Serializable0 or k is Serializable:
                continue
            f = d[k.__name__]
            print(cls, k, f)
            params.update(f)
        return params

    registered = OrderedDict()


from future.utils import with_metaclass


def register_class(cls):
    Serializable0.registered[cls.__name__] = cls
    print('registering %s' % cls.__name__)


class MetaSerializable(ABCMeta):
    def __new__(mcs, name, bases, class_dict):
        cls = type.__new__(mcs, name, bases, class_dict)
        register_class(cls)
        return cls


class Serializable(with_metaclass(MetaSerializable, Serializable0)):
    pass


def as_json_dict(x):
    if x is None:
        return None
    elif isinstance(x, (int, str, float)):
        return x
    elif isinstance(x, list):
        return [as_json_dict(_) for _ in x]
    elif isinstance(x, dict):
        return dict([(k, as_json_dict(v)) for k, v in x.items()])
    elif isinstance(x, Serializable):
        return x.as_json_dict()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        msg = 'Invalid class %s' % type(x).__name__
        raise ValueError(msg)



def from_json_dict2(d):
    if not isinstance(d, dict):
        msg = 'Expected dict for %s' % d
        raise CouldNotDeserialize(msg)
    for name, klass in reversed(Serializable.registered.items()):
        if name in d:
            d2 = deepcopy(d)
            try:
                res = klass.params_from_json_dict(d2)
            except BaseException as e:
                msg = 'Cannot interpret data using %s' % klass.__name__
                msg += '\n\n%s' % json.dumps(d, indent=4)[:100]
                msg += '\n\n%s' % traceback.format_exc()
                raise CouldNotDeserialize(msg)
            return klass(**res)

    msg = 'Cannot interpret any of %s' % list(d)
    raise CouldNotDeserialize(msg)
