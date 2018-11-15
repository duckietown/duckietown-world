# coding=utf-8
from duckietown_world import Sequence, TransformSequence

__all__ = [
    'ChooseTime',
    # 'RemoveVariable',
    # 'RemoveStatic',
    'get_sampling_points',
]


class ChooseTime(object):
    def __init__(self, t):
        self.t = t

    def __call__(self, ob):
        if isinstance(ob, Sequence):
            ob = ob.at(self.t)
            return ob
        else:
            return ob
#
#
# class RemoveVariable(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, ob):
#         if isinstance(ob, PlacedObject):
#             toremove = []
#             for sr_name, sr in ob.spatial_relations.items():
#                 if not sr.a and isinstance(sr.transform, Sequence):
#                     toremove.append(sr.b)
#             return remove_stuff_and_children(ob, toremove)
#
#         else:
#             return ob

#
# class RemoveStatic(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, ob):
#         if isinstance(ob, PlacedObject):
#             toremove = []
#             for sr_name, sr in ob.spatial_relations.items():
#                 # print(sr_name, sr)
#                 if not sr.a and isinstance(sr.transform, Sequence) and is_static(sr.transform):
#                     # print('removing')
#                     toremove.append(sr.b)
#             if toremove:
#                 print('removing %s' % toremove)
#             return remove_stuff_and_children(ob, toremove)
#         else:
#             return ob


def is_static(transform):
    if isinstance(transform, Sequence):
        return False
    if isinstance(transform, TransformSequence):
        return all(is_static(_) for _ in transform.transforms)
    return True


# def remove_stuff_and_children(ob, toremove):
#     assert isinstance(ob, PlacedObject)
#     children = {}
#     for child_name, child in ob.children.items():
#         if child_name in toremove:
#             pass
#         else:
#             children[child_name] = child
#     srs = {}
#     for sr_name, sr in ob.spatial_relations.items():
#         if sr.b[0] in toremove:
#             pass
#         else:
#             srs[sr_name] = sr
#     return PlacedObject(children=children, spatial_relations=srs)

#
# class Flatten(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, ob):
#         if isinstance(ob, Transform):
#             return ob.asmatrix2d()
#         else:
#             return ob


def get_sampling_points(ob0):
    points = set()

    def f(ob):
        # print(ob)
        if isinstance(ob, Sequence):
            sp = ob.get_sampling_points()
            if sp == Sequence.CONTINUOUS:
                pass
            else:
                points.update(sp)
        return ob

    ob0.filter_all(f)
    return sorted(points)
