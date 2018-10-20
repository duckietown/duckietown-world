# coding=utf-8
from networkx import MultiDiGraph

from duckietown_world.geo import ChildObject

__all__ = [
    'iterate_measurements_relations',
    'get_meausurements_graph',
]


def iterate_measurements_relations(po_name, po):
    for child_name, co in po.children.items():
        assert isinstance(co, ChildObject)
        cname = po_name + (child_name,)
        yield po_name, po, cname, co.child, co.transforms
        for _ in iterate_measurements_relations(cname, co.child):
            yield _


def get_meausurements_graph(po):
    G = MultiDiGraph()

    for parent_name, parent, child_name, child, transforms in iterate_measurements_relations((), po):
        # print(parent_name, child_name)
        for t, value in transforms.items():
            attr_dict = dict(name=t, transform=value)
            G.add_edge(parent_name, child_name, attr_dict=attr_dict)
        # pass

    # print(G)
    return G


