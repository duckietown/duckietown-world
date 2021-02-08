import numpy as np
import networkx as nx
from dataclasses import dataclass
from functools import reduce
from duckietown_world.seqs import GenericSequence
from networkx import MultiDiGraph
from typing import Callable, Generic, Iterator, List, Tuple, Type, TypeVar

from .placed_object import FQN, PlacedObject, SpatialRelation
from .rectangular_area import RectangularArea
from .transforms import TransformSequence, Transform, Scale2D, SE2Transform
from duckietown_world.structure.bases import _Frame, IBaseMap

__all__ = [
    "get_extent_points",
    "get_static_and_dynamic",
    "get_transforms",
    "iterate_by_class",
    "IterateByTestResult",
]


def frame2transforms(frame: "_Frame") -> List["Transform"]:
    print([frame.pose.x, frame.pose.y])
    se2 = SE2Transform(p=[frame.pose.x, frame.pose.y], theta=frame.pose.yaw)
    print(se2)
    if frame.scale == 1.0:
        return [se2]
    else:
        return [se2, Scale2D(frame.scale)]


def iterate_measurements_relations(po_name: FQN, po: PlacedObject) -> Iterator[Tuple[FQN, SpatialRelation]]:
    assert isinstance(po_name, tuple)
    for sr_name, sr in po.spatial_relations.items():
        a = po_name + sr.a
        b = po_name + sr.b
        klass = type(sr)
        # noinspection PyArgumentList
        s = klass(a=a, b=b, transform=sr.transform)
        yield po_name + (sr_name,), s


def get_transform_sequence(dm: "IBaseMap", frame: "_Frame") -> "TransformSequence":
    frames_list = dm.get_relative_frames_list(frame)
    transforms_lists = list(map(frame2transforms, frames_list))
    return TransformSequence([transform for transforms_list in transforms_lists for transform in transforms_list])


def get_transforms(frame: "_Frame") -> "TransformSequence":
    return TransformSequence(frame2transforms(frame))


def get_static_and_dynamic(dm: "IBaseMap") -> Tuple[List[Tuple[str, type]], List[Tuple[str, type]]]:
    placed_objects = dm.get_placed_objects()

    static, dynamic = [], []
    for (nm, tp) in placed_objects:
        frame = dm.get_frame_by_name(nm)
        transform_sequence = get_transform_sequence(dm, frame)
        from duckietown_world.world_duckietown.transformations import is_static

        if is_static(transform_sequence):
            static.append((nm, tp))
        else:
            dynamic.append((nm, tp))

    return static, dynamic


def get_flattened_measurement_graph(po: PlacedObject, include_root_to_self: bool = False) -> nx.DiGraph:
    from duckietown_world import TransformSequence
    from duckietown_world import SE2Transform

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
            sr = v["attr_dict"]["sr"].transform

            transforms.append(sr)

        if any(isinstance(_, GenericSequence) for _ in transforms):
            res = VariableTransformSequence(transforms)
        else:
            res = TransformSequence(transforms)
        G2.add_edge(root_name, name, transform_sequence=res)

    if include_root_to_self:
        transform_sequence = SE2Transform.identity()
        G2.add_edge(root_name, root_name, transform_sequence=transform_sequence)

    return G2


X = TypeVar("X")

@dataclass
class IterateByTestResult(Generic[X]):
    fqn: Tuple[str, ...]
    transform_sequence: TransformSequence
    object: X
    parents: Tuple[PlacedObject, ...]


def iterate_by_class(po: PlacedObject, klass: Type[X]) -> Iterator[IterateByTestResult]:
    t = lambda _: isinstance(_, klass)
    yield from iterate_by_test(po, t)


def iterate_by_test(po: PlacedObject, testf: Callable[[PlacedObject], bool]) -> Iterator[IterateByTestResult]:
    G = get_flattened_measurement_graph(po, include_root_to_self=True)
    root_name = ()

    for name in G:
        ob = po.get_object_from_fqn(name)

        if testf(ob):
            transform_sequence = G.get_edge_data(root_name, name)["transform_sequence"]

            parents = get_parents(po, name)

            yield IterateByTestResult(
                fqn=name, transform_sequence=transform_sequence, object=ob, parents=parents,
            )


def get_parents(root: PlacedObject, child_fqn: FQN) -> Tuple[PlacedObject]:
    parents = []
    n = len(child_fqn)
    for i in range(0, n - 1):
        name_parent: FQN = child_fqn[:i]
        parent = root.get_object_from_fqn(name_parent)
        parents.append(parent)
    return tuple(parents)


def get_extent_points(dm: "IBaseMap") -> "RectangularArea":
    placed_objects = dm.get_placed_objects()

    points = []
    for (nm, _), ob in placed_objects.items():
        frame = dm.get_frame_by_name(nm)
        transform_sequence = get_transform_sequence(dm, frame)
        m2d = transform_sequence.asmatrix2d()
        extent_points = ob.extent_points()
        for ep in extent_points:
            p = np.dot(m2d.m, [ep[0], ep[1], 1])[:2]
            points.append(p)

    if not points:
        msg = "No points"
        raise ValueError(msg)

    p_min = reduce(np.minimum, points)
    p_max = reduce(np.maximum, points)

    return RectangularArea(p_min, p_max)
