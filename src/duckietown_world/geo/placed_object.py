# coding=utf-8
from __future__ import unicode_literals

import copy

import six
from contracts import contract, check_isinstance
from duckietown_serialization_ds1 import Serializable
from duckietown_world.seqs import UndefinedAtTime, Sequence

from .rectangular_area import RectangularArea
from .transforms import Transform

__all__ = [
    'PlacedObject',
    'SpatialRelation',
    'GroundTruth',
    'get_object_tree',
]


class SpatialRelation(Serializable):

    @contract(a='seq(string)', b='seq(string)')
    def __init__(self, a, transform, b):
        check_isinstance(transform, (Transform, Sequence))
        self.a = tuple(a)
        self.transform = transform
        self.b = tuple(b)

    def filter_all(self, f):
        t2 = f(self.transform)
        if t2 == self.transform:
            return self
        else:
            return SpatialRelation(self.a, t2, self.b)

    @classmethod
    def params_from_json_dict(cls, d):
        a = d.pop('a', [])
        b = d.pop('b')
        transform = d.pop('transform')
        transform = Serializable.from_json_dict(transform)
        return dict(a=a, b=b, transform=transform)

    def params_to_json_dict(self):
        res = {}
        if self.a:
            res['a'] = list(self.a)
        res['b'] = self.b
        res['transform'] = self.transform
        return res


class GroundTruth(SpatialRelation):

    def __repr__(self):
        return 'GroundTruth(%r -> %r  %s)' % (self.a, self.b, self.transform)

    @classmethod
    def params_from_json_dict(cls, d):
        return {}

    def params_to_json_dict(self):
        return {}


class PlacedObject(Serializable):
    def __init__(self, children=None, spatial_relations=None):
        children = children or {}
        spatial_relations = spatial_relations or {}

        self.children = children

        for k, v in list(spatial_relations.items()):
            from .transforms import Transform
            if isinstance(v, Transform):
                if k in self.children:
                    sr = GroundTruth(a=(), b=(k,), transform=v)
                    spatial_relations[k] = sr
                else:
                    msg = 'What is the "%s" referring to?' % k
                    raise ValueError(msg)

        self.spatial_relations = spatial_relations

        if not spatial_relations:
            for child in self.children:
                from duckietown_world import SE2Transform
                sr = GroundTruth(a=(), b=(child,), transform=SE2Transform.identity())
                self.spatial_relations[child] = sr

    def remove_object(self, k):
        self.children.pop(k)
        for sr_id, sr in list(self.spatial_relations.items()):
            if sr.b == (k,):
                self.spatial_relations.pop(sr_id)

    def _simplecopy(self, *args, **kwargs):
        children = dict((k, v) for k, v in self.children.items())
        spatial_relations = dict((k, v) for k, v in self.spatial_relations.items())
        kwargs.update(dict(children=children, spatial_relations=spatial_relations))
        return type(self)(*args, **kwargs)

    def _copy(self):
        if type(self) is PlacedObject:
            return self._simplecopy()
        else:
            # logger.debug('no _copy for %s' % type(self).__name__)
            return copy.copy(self)

    def filter_all(self, f):

        children = {}
        spatial_relations = {}

        no_child = []
        for child_name, child in list(self.children.items()):
            try:
                child2 = f(child.filter_all(f))
            except UndefinedAtTime:
                no_child.append(child_name)
            else:
                children[child_name] = child2

        for sr_name, sr in list(self.spatial_relations.items()):
            if sr.b and sr.b[0] in no_child:
                pass
            else:
                try:
                    sr2 = f(sr.filter_all(f))
                except UndefinedAtTime:
                    pass
                else:
                    spatial_relations[sr_name] = sr2

        if children == self.children and spatial_relations == self.spatial_relations:
            # logger.debug('could save a copy of %s' % type(self).__name__)
            return self
        else:
            # print('need a copy of %s' % self)
            x = self._copy()
            x.children = children
            x.spatial_relations = spatial_relations
            return x

    def __getitem__(self, item):
        """

        Either url-like:

            child1/sub
            .

        or tuple like:

            ('child1', 'sub')
            ()

        :param item:
        :return:
        """
        if isinstance(item, str):
            item = fqn_from_url(item)
        check_isinstance(item, tuple)
        return self.get_object_from_fqn(item)

    def get_object_from_fqn(self, fqn):
        if fqn == ():
            return self
        first, rest = fqn[0], fqn[1:]
        if first in self.children:
            return self.children[first].get_object_from_fqn(rest)
        else:
            msg = 'Cannot find child %s in %s' % (first, list(self.children))
            raise KeyError(msg)

    def params_to_json_dict(self):
        res = {}

        if self.children:
            res['children'] = self.children
        if self.spatial_relations:
            res['spatial_relations'] = self.spatial_relations

        return res

    def set_object(self, name, ob, **transforms):
        check_isinstance(name, six.string_types)
        assert self is not ob
        self.children[name] = ob
        type2klass = {
            'ground_truth': GroundTruth
        }
        for k, v in transforms.items():
            klass = type2klass[k]
            st = klass(a=(), b=(name,), transform=v)
            i = len(self.spatial_relations)
            self.spatial_relations[i] = st

    # @abstractmethod
    def draw_svg(self, drawing, g):
        from duckietown_world.svg_drawing import draw_axes
        draw_axes(drawing, g)

    def get_drawing_children(self):
        return sorted(self.children)

    def extent_points(self):
        return [(0.0, 0.1), (0.1, 0.0)]

    def get_footprint(self):
        return RectangularArea([-0.1, -0.1], [0.1, 0.1])


from contracts import indent
import yaml


def get_object_tree(po, levels=100, spatial_relations=False, attributes=False):
    ss = []
    ss.append('%s' % type(po).__name__)
    d = po.params_to_json_dict()
    d.pop('children', None)
    d.pop('spatial_relations', None)

    if attributes:
        if d:
            ds = yaml.safe_dump(d, encoding='utf-8', indent=4, allow_unicode=True, default_flow_style=False)
            ss.append('\n' + indent(ds, ' '))

    if po.children and levels >= 1:
        ss.append('')
        N = len(po.children)
        for i, (child_name, child) in enumerate(po.children.items()):

            if i != N - 1:
                prefix1 = u'├ %s ┐ ' % child_name
                prefix2 = u'│ %s │ ' % (' ' * len(child_name))
            else:
                prefix1 = u'└ %s ┐ ' % child_name
                prefix2 = u'  %s │ ' % (' ' * len(child_name))
            c = get_object_tree(child, attributes=attributes, spatial_relations=spatial_relations, levels=levels - 1)
            sc = indent(c, prefix2, prefix1)
            n = max(len(_) for _ in sc.split('\n'))
            sc += '\n' + prefix2[:-2] + u'└' + u'─' * (n - len(prefix2) + 3)
            ss.append(sc)

    if spatial_relations:
        if po.spatial_relations and levels >= 1:
            ss.append('')
            for r_name, rel in po.spatial_relations.items():
                ss.append('- from "%s" to "%s"  %s ' % (url_from_fqn(rel.a), url_from_fqn(rel.b), rel.transform))

    return "\n".join(ss)


def url_from_fqn(x):
    if not x:
        return '.'
    else:
        return '/'.join(x)


def fqn_from_url(u):
    if u == '.':
        return ()
    else:
        return tuple(u.split('/'))
