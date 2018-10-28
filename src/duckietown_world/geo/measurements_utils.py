# coding=utf-8
from __future__ import unicode_literals

from networkx import MultiDiGraph

from duckietown_world.geo.placed_object import SpatialRelation, PlacedObject
from duckietown_world.geo.rectangular_area import RectangularArea

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
        attr_dict = dict(sr=sr)
        G.add_edge(a, b, attr_dict=attr_dict)
    return G


import networkx as nx


def get_flattened_measurement_graph(po):
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
        res = TransformSequence(transforms)
        G2.add_edge(root_name, name, transform_sequence=res)
    return G2


import numpy as np


def get_extent_points(root):
    assert isinstance(root, PlacedObject)
    # iterate_in_frame(root)
    G = get_flattened_measurement_graph(root)
    points = []

    root_name = ()
    for name in G.nodes():
        if name == root_name:
            continue
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
