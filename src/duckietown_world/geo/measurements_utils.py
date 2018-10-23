# coding=utf-8
from __future__ import unicode_literals

from networkx import MultiDiGraph

from duckietown_world.geo.placed_object import SpatialRelation

__all__ = [
    'iterate_measurements_relations',
    'get_meausurements_graph',
]


def iterate_measurements_relations(po_name, po):
    assert isinstance(po_name, tuple)
    for sr_name, sr in po.spatial_relations.items():
        a = po_name + sr.a
        b = po_name + sr.b
        s = SpatialRelation(a=a, b=b, sr_type=sr.sr_type, transform=sr.transform)
        yield po_name + (sr_name,), s

    for child_name, child in po.children.items():
        cname = po_name + (child_name,)
        for _ in iterate_measurements_relations(cname, child):
            yield _


def get_meausurements_graph(po):
    G = MultiDiGraph()

    for name, sr in iterate_measurements_relations((), po):
        a = sr.a
        b = sr.b

        def v(x):
            if x:
                return "/".join("{}".format(_) for _ in x)
            else:
                return '/'

        print('rel %s from %s to %s' % (v(name), v(a), v(b)))

        attr_dict = dict(sr=sr)
        G.add_edge(a, b, attr_dict=attr_dict)
    # pass

    # print(G)
    return G
