# coding=utf-8
from __future__ import unicode_literals

from collections import namedtuple

from contracts import contract
from networkx import MultiDiGraph

from duckietown_world.seqs import Sequence
from .placed_object import PlacedObject
from .rectangular_area import RectangularArea
from .transforms import VariableTransformSequence

__all__ = [
    'iterate_measurements_relations',
    'get_meausurements_graph',
    'get_extent_points',
    'get_static_and_dynamic',
    'iterate_by_class',
]


def iterate_measurements_relations(po_name, po):
    assert isinstance(po_name, tuple)
    for sr_name, sr in po.spatial_relations.items():
        a = po_name + sr.a
        b = po_name + sr.b
        klass = type(sr)
        s = klass(a=a, b=b, transform=sr.transform)
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
        attr_dict = dict(sr=sr)
        G.add_edge(a, b, attr_dict=attr_dict)
    return G


import networkx as nx

FlattenResult = namedtuple('FlattenResult', 'po old2new')


@contract(po=PlacedObject)
def get_static_and_dynamic(po):
    assert isinstance(po, PlacedObject)

    G = get_flattened_measurement_graph(po)

    static = []

    static.append(())
    dynamic = []
    root_name = ()
    for name in G.nodes():
        if name == root_name:
            continue
        edge_data = G.get_edge_data(root_name, name)

        transform = edge_data['transform_sequence']
        from duckietown_world.world_duckietown.transformations import is_static
        it_is = is_static(transform)

        if it_is:

            static.append(name)
        else:
            dynamic.append(name)

    return static, dynamic

#
# @contract(po=PlacedObject, returns=PlacedObject)
# def flatten_hierarchy(po):
#     assert isinstance(po, PlacedObject)
#     res = PlacedObject()
#     G = get_flattened_measurement_graph(po)
#
#     root_name = ()
#     for name in G.nodes():
#         if name == root_name:
#             continue
#         edge_data = G.get_edge_data(root_name, name)
#
#         transform = edge_data['transform_sequence']
#         ob = po.get_object_from_fqn(name)
#         name2 = "/".join(name)
#         res.set_object(name2, ob, ground_truth=transform)
#     return res


def get_flattened_measurement_graph(po, include_root_to_self=False):
    G = get_meausurements_graph(po)
    G2 = nx.DiGraph()
    root_name = ()
    for name in G.nodes():
        if name == root_name:
            continue
        path = nx.shortest_path(G, (), name)
        transforms = []
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            edges = G.get_edge_data(a, b)

            k = list(edges)[0]
            v = edges[k]
            sr = v['attr_dict']['sr'].transform

            transforms.append(sr)

        from duckietown_world import TransformSequence
        if any(isinstance(_, Sequence) for _ in transforms):
            res = VariableTransformSequence(transforms)
        else:
            res = TransformSequence(transforms)
        G2.add_edge(root_name, name, transform_sequence=res)

    if include_root_to_self:
        from duckietown_world import SE2Transform
        transform_sequence = SE2Transform.identity()
        G2.add_edge(root_name, root_name, transform_sequence=transform_sequence)

    return G2


IterateByTestResult = namedtuple('IterateByTestResult', 'fqn transform_sequence object')


def iterate_by_class(po, klass):
    for _ in iterate_by_test(po, lambda _: isinstance(_, klass)):
        yield _


def iterate_by_test(po, testf):
    """

    :param po: root object
    :param testf: boolean test on the object
    :return: Iterator of IterateByTestResult
    """
    G = get_flattened_measurement_graph(po, include_root_to_self=True)
    root_name = ()
    for name in G:
        ob = po.get_object_from_fqn(name)
        if testf(ob):
            transform_sequence = G.get_edge_data(root_name, name)['transform_sequence']
            yield IterateByTestResult(fqn=name, transform_sequence=transform_sequence, object=ob)


import numpy as np


def get_extent_points(root):
    assert isinstance(root, PlacedObject)

    G = get_flattened_measurement_graph(root, include_root_to_self=True)
    points = []

    root_name = ()
    for name in G.nodes():

        transform_sequence = G.get_edge_data(root_name, name)['transform_sequence']
        extent_points = root.get_object_from_fqn(name).extent_points()
        m2d = transform_sequence.asmatrix2d()
        for _ in extent_points:
            p = np.dot(m2d.m, [_[0], _[1], 1])[:2]
            points.append(p)

    if not points:
        msg = 'No points'
        raise ValueError(msg)

    pmin = points[0]
    pmax = points[0]

    for p in points:
        pmin = np.minimum(pmin, p)
        pmax = np.maximum(pmax, p)

    return RectangularArea(pmin, pmax)
