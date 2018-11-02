# coding=utf-8
import os

import numpy as np
from comptests import comptest, run_module_tests, get_comptests_output_dir

from duckietown_world.geo import PlacedObject, SE2Transform, get_meausurements_graph
from duckietown_world.seqs import Constant
from duckietown_world.utils.gvgen_ac import ACGvGen
from duckietown_world.world_duckietown.map_loading import load_map


@comptest
def m1():
    outdir = get_comptests_output_dir()
    gm = load_map('udem1')

    # dw = DuckietownWorld()
    # for map_name, tm in gym_maps.items():
    #     DW.root.set_object(map_name, tm)

    root = PlacedObject()

    world = PlacedObject()
    root.set_object('world', world)
    origin = SE2Transform([1, 10], np.deg2rad(10))
    world.set_object('tile_map', gm, ground_truth=Constant(origin))

    # d = dw.as_json_dict()
    # print(json.dumps(d, indent=4))
    # print(yaml.safe_dump(d, default_flow_style=False))
    #

    G = get_meausurements_graph(root)
    fn = os.path.join(outdir, 'out1.pdf')
    plot_measurement_graph(root, G, fn)
    #
    # world = ('world',)
    # for node in G.nodes():
    #     paths = list(all_simple_paths(G, world, node))
    #     paths = [transform_from_path(G, _) for _ in paths]
    #     if paths:
    #         mpath = paths[0]
    #         path = squash_path(mpath)
    #         print(path)
    #         # frozen = [_.at(t=0) for _ in path]
    #         # print(frozen)


class NoMeasurements(Exception):
    pass


def squash_one(p, preference):
    for m in p:
        if m.sr_type in preference:
            return m.transform

    msg = 'Missing measurement of type %r' % ", ".join(preference)
    msg += '\n\nFound:\n%r' % "\n".join(_.sr_type.__repr__() for _ in p)
    raise NoMeasurements(msg)


def squash_path(path, preference=('ground_truth',)):
    res = []
    for p in path:
        res.append(squash_one(p, preference))
    return res


def transform_from_path(G, path):
    transforms = []
    for u, v in pairwise(list(path)):
        t = []
        for k, edge_data in G.get_edge_data(u, v).items():
            attr_dict = edge_data['attr_dict']
            transform = attr_dict['sr']
            t.append(transform)
        transforms.append(t)
    return transforms


import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return \
        zip(a, b)


def plot_measurement_graph(root, G, out):
    d = os.path.dirname(out)
    if not os.path.exists(d):
        os.makedirs(d)
    gg = ACGvGen()

    node2item = {}
    for node in G.nodes():
        label = node[-1] if node else "root"
        ob = root.get_object_from_fqn(node)
        label += '\n%s' % ob
        item = gg.newItem(label)
        node2item[node] = item
    for u, v in G.edges():
        for k, data in G.get_edge_data(u, v).items():
            attr_dict = data['attr_dict']
            transform = attr_dict['sr']
            label = str(transform)
            src = node2item[u]
            dst = node2item[v]
            gg.newLink(src=src, dst=dst, label=label)

    fn = out + '.dot'
    with open(fn, 'w') as f:
        f.write(gg.dot2())

    cmd = 'dot -o%s -Tpdf %s' % (out, fn)
    os.system(cmd)
    print('Written to %s' % out)


if __name__ == '__main__':
    run_module_tests()
