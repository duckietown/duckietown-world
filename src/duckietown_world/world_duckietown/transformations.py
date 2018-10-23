from duckietown_world import Transform, Sequence


class ChooseTime(object):
    def __init__(self, t):
        self.t = t

    def __call__(self, ob):
        # if isinstance(ob, Sequence):
        if hasattr(ob, 'at'):
            # assert hasattr(ob, 'at'), ob
            ob = ob.at(self.t)
            return ob
        else:
            return ob


class Flatten(object):
    def __init__(self):
        pass

    def __call__(self, ob):
        if isinstance(ob, Transform):
            return ob.asmatrix2d()
        else:
            return ob


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
