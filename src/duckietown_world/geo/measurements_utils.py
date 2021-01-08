import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Iterator
from functools import reduce

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
    se2 = SE2Transform(p=[frame.pose.x, frame.pose.y], theta=frame.pose.yaw)
    if frame.scale == 1.0:
        return [se2]
    else:
        return [se2, Scale2D(frame.scale)]


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


@dataclass
class IterateByTestResult:
    fqn: Tuple[str, ...]
    transform_sequence: TransformSequence
    object: PlacedObject
    parents: Tuple[PlacedObject, ...]


def iterate_by_class(po: PlacedObject, klass: type) -> Iterator[IterateByTestResult]:
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
