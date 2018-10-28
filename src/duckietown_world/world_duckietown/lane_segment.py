import numpy as np
from contracts import contract, check_isinstance, new_contract

import geometry as geo
from duckietown_world import SE2Transform, PlacedObject
from duckietown_world.utils.poses import SE2_interpolate, SE2_apply_R2

__all__ = [
    'LaneSegment',
]


class LaneSegment(PlacedObject):

    @contract(width='>0', control_points='list[>=2](SE2Transform)')
    def __init__(self, width, control_points, *args, **kwargs):
        PlacedObject.__init__(self, *args, **kwargs)
        self.width = width
        self.control_points = control_points

        for p in control_points:
            check_isinstance(p, SE2Transform)

    def params_to_json_dict(self):
        return dict(width=self.width, control_points=self.control_points)

    def extent_points(self):
        return self.lane_profile()

    def draw_svg(self, drawing, g):
        points = self.lane_profile(points_per_segment=10)
        p = drawing.polygon(points=points,
                            fill='#709457',
                            fill_opacity=0.5)
        g.add(p)

        for x in self.control_points:
            q = x.asmatrix2d().m
            p1, _ = geo.translation_angle_from_SE2(q)
            delta_arrow = np.array([self.width / 4, 0])
            p2 = SE2_apply_R2(q, delta_arrow)
            l = drawing.line(start=p1.tolist(), end=p2.tolist(),
                             stroke='black',
                             stroke_width=self.width / 20.0)
            g.add(l)
            c = drawing.circle(center=p1.tolist(),
                               r=self.width / 8,
                               fill='white',
                               stroke='black',
                               stroke_width=self.width / 20.0)
            g.add(c)

    def lane_profile(self, points_per_segment=5):
        points_left = []
        points_right = []
        n = len(self.control_points)
        num = n * points_per_segment
        betas = np.linspace(0, n, num=num)
        for beta in betas:
            i = int(np.floor(beta))
            if i >= n - 1:
                q = self.control_points[-1].asmatrix2d().m
            else:
                alpha = beta - i
                q0 = self.control_points[i].asmatrix2d().m
                q1 = self.control_points[i + 1].asmatrix2d().m
                q = SE2_interpolate(q0, q1, alpha)
            delta_left = np.array([0, self.width/2])
            delta_right = np.array([0, -self.width/2])
            points_left.append(SE2_apply_R2(q, delta_left))
            points_right.append(SE2_apply_R2(q, delta_right))

        return points_right + list(reversed(points_left))


new_contract('LaneSegment', LaneSegment)
