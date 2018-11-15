# coding=utf-8
import geometry as geo
import numpy as np
import svgwrite
from contracts import contract, check_isinstance, new_contract
from duckietown_serialization_ds1 import Serializable
from duckietown_serialization_ds1.serialization1 import as_json_dict
from duckietown_world.geo import SE2Transform, PlacedObject
from duckietown_world.utils import memoized_reset, SE2_interpolate, SE2_apply_R2

from .tile import relative_pose

__all__ = [
    'LaneSegment',
    'LanePose',

]


class LanePose(Serializable):

    # coordinate frame

    # longitudinal => along_lane
    # lateral => lateral
    # relative heading

    #

    # lateral_right => lateral position of closest right lane boundary
    # lateral_left => lateral position of closest left lane boundary

    # center_point: anchor point on the center lane

    def __init__(self, inside, outside_left, outside_right,
                 along_lane, lateral, relative_heading,
                 distance_from_left, distance_from_right,
                 along_inside, along_before, along_after,
                 lateral_left, lateral_right, lateral_inside, distance_from_center,
                 center_point, correct_direction):
        self.inside = inside
        self.lateral_inside = lateral_inside
        self.outside_left = outside_left
        self.outside_right = outside_right
        self.lateral = lateral
        self.lateral_left = lateral_left
        self.lateral_right = lateral_right

        self.along_inside = along_inside
        self.along_before = along_before
        self.along_after = along_after
        self.along_lane = along_lane

        self.relative_heading = relative_heading

        self.distance_from_left = distance_from_left
        self.distance_from_right = distance_from_right
        self.distance_from_center = distance_from_center
        self.center_point = center_point
        self.correct_direction = correct_direction


def almost_equal(a, b):
    return (np.abs(a - b)) < 1e-7


class LaneSegment(PlacedObject):

    @contract(width='>0', control_points='list[>=2](SE2Transform)')
    def __init__(self, width, control_points, *args, **kwargs):
        PlacedObject.__init__(self, *args, **kwargs)
        self.width = float(width)
        self.control_points = control_points

        for p in control_points:
            check_isinstance(p, SE2Transform)

        for i in range(len(control_points)-1):
            a = control_points[i]
            b = control_points[i+1]
            ta, _ = geo.translation_angle_from_SE2(a.as_SE2())
            tb, _ = geo.translation_angle_from_SE2(b.as_SE2())
            d = np.linalg.norm(ta-tb)
            if d < 0.001:
                msg = 'Two points are two close: \n%s\n%s' % (a, b)
                raise ValueError(msg)

    def _copy(self):
        return self._simplecopy(width=self.width, control_points=self.control_points)

    @memoized_reset
    def get_lane_lengths(self):
        res = []
        for i in range(len(self.control_points) - 1):
            p0 = self.control_points[i]
            p1 = self.control_points[i + 1]
            sd = get_distance_two(p0.as_SE2(), p1.as_SE2())
            res.append(sd)
        return res

    def get_lane_length(self):
        return sum(self.get_lane_lengths())

    def lane_pose_random(self):
        along_lane = np.random.uniform(0, self.get_lane_length())
        W = self.width
        lateral = np.random.uniform(-W / 2, +W / 2)
        angle = np.random.uniform(-np.pi / 2, +np.pi / 2)
        return self.lane_pose(along_lane=along_lane, lateral=lateral, relative_heading=angle)

    @contract(along_lane='float', lateral='float', relative_heading='float')
    def lane_pose(self, along_lane, lateral, relative_heading):
        beta = self.beta_from_along_lane(along_lane)
        center_point = self.center_point(beta)

        W2 = self.width / 2
        lateral_inside = -W2 <= lateral <= W2
        outside_right = lateral < -W2
        outside_left = W2 < lateral
        distance_from_left = np.abs(+ W2 - lateral)
        distance_from_right = np.abs(- W2 - lateral)
        distance_from_center = np.abs(lateral)

        L = self.get_lane_length()
        along_inside = 0 <= along_lane < L
        along_before = along_lane < 0
        along_after = along_lane > L
        inside = lateral_inside and along_inside

        correct_direction = np.abs(relative_heading) <= np.pi / 2
        return LanePose(inside=inside,
                        lateral_inside=lateral_inside,
                        outside_left=outside_left,
                        outside_right=outside_right,
                        distance_from_left=distance_from_left,
                        distance_from_right=distance_from_right,
                        relative_heading=relative_heading,
                        along_inside=along_inside,
                        along_before=along_before,
                        along_after=along_after,
                        along_lane=along_lane,
                        lateral=lateral,
                        lateral_left=W2,
                        lateral_right=-W2,
                        distance_from_center=distance_from_center,
                        center_point=SE2Transform.from_SE2(center_point),
                        correct_direction=correct_direction)

    def params_to_json_dict(self):
        return dict(width=self.width, control_points=as_json_dict(self.control_points))

    def extent_points(self):
        return self.lane_profile()

    @contract(qt=SE2Transform, returns=LanePose)
    def lane_pose_from_SE2Transform(self, qt, tol=0.001):
        return self.lane_pose_from_SE2(qt.as_SE2(), tol=tol)

    @contract(  # q='SE2',
            returns=LanePose)
    def lane_pose_from_SE2(self, q, tol=0.001):
        return (self.is_straight()
                and self.lane_pose_from_SE2_straight(q)
                or self.lane_pose_from_SE2_generic(q, tol=tol))

    # return condition ? if_true : if_false
    # return condition and if_true or if_false

    def is_straight(self):
        if not len(self.control_points) == 2:
            return False

        cp1 = self.control_points[0]
        cp2 = self.control_points[1]

        r = relative_pose(cp1.as_SE2(), cp2.as_SE2())
        p, theta = geo.translation_angle_from_SE2(r)
        return almost_equal(theta, 0) and almost_equal(p[1], 0)

    @contract(q='euclidean2')
    def lane_pose_from_SE2_straight(self, q):
        cp1 = self.control_points[0]
        rel = relative_pose(cp1.as_SE2(), q)
        tas = geo.translation_angle_scale_from_E2(rel)
        lateral = tas.translation[1]
        along_lane = tas.translation[0]
        relative_heading = tas.angle
        return self.lane_pose(relative_heading=relative_heading,
                              lateral=lateral,
                              along_lane=along_lane)

    @contract(q='euclidean2')
    def lane_pose_from_SE2_generic(self, q, tol=0.001):
        p, _, _ = geo.translation_angle_scale_from_E2(q)

        beta, q0 = self.find_along_lane_closest_point(p, tol=tol)
        along_lane = self.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)

        r, relative_heading, _ = geo.translation_angle_scale_from_E2(rel)
        lateral = r[1]
        # extra = r[0]
        # along_lane -= extra
        # logger.info('ms: %s' % r[0])
        return self.lane_pose(relative_heading=relative_heading,
                              lateral=lateral,
                              along_lane=along_lane)

    def find_along_lane_closest_point(self, p, tol=0.001):

        def get_delta(beta):
            q0 = self.center_point(beta)
            t0, _ = geo.translation_angle_from_SE2(q0)
            d = np.linalg.norm(p - t0)

            d1 = np.array([0, -d])
            p1 = SE2_apply_R2(q0, d1)

            d2 = np.array([0, +d])
            p2 = SE2_apply_R2(q0, d2)

            D2 = np.linalg.norm(p2 - p)
            D1 = np.linalg.norm(p1 - p)
            res = np.minimum(D1, D2)
            # print('%10f: q %s %f' % (beta, geo.SE2.friendly(q0), res))
            return res

        import scipy.optimize
        bracket = (-1.0, len(self.control_points))
        res = scipy.optimize.minimize_scalar(get_delta, bracket=bracket, tol=tol)
        # print(res)
        beta = res.x
        q0 = self.center_point(beta)
        return beta, q0

    @contract(lane_pose=LanePose, returns=SE2Transform)
    def SE2Transform_from_lane_pose(self, lane_pose):
        beta = self.beta_from_along_lane(lane_pose.along_lane)
        # logger.info('beta: %s' % beta)
        center_point = self.center_point(beta)
        delta = np.array([0, lane_pose.lateral])

        rel = geo.SE2_from_translation_angle(delta, lane_pose.relative_heading)
        # logger.info('rel: %s' % geo.SE2.friendly(rel))
        res = geo.SE2.multiply(center_point, rel)

        return SE2Transform.from_SE2(res)

    def along_lane_from_beta(self, beta):
        lengths = self.get_lane_lengths()
        if beta < 0:
            return beta
        elif beta >= len(self.control_points) - 1:
            rest = beta - (len(self.control_points) - 1)
            return sum(lengths) + rest
        else:
            i = int(np.floor(beta))

            rest = beta - i

            res = sum(lengths[:i]) + lengths[i] * rest

            return res

    def beta_from_along_lane(self, along_lane):
        lengths = self.get_lane_lengths()
        x0 = along_lane
        n = len(self.control_points)
        S = sum(lengths)

        if x0 < 0:
            beta = x0
            return beta

        if x0 > S:
            beta = (n - 1.0) + (x0 - S)
            return beta

        if almost_equal(x0, S):
            beta = (n - 1.0)
            return beta

        assert 0 <= x0 < S, (x0, S)

        for i in range(n - 1):
            start_x = sum(lengths[:i])
            end_x = sum(lengths[:i + 1])
            if start_x <= x0 < end_x:
                beta = i + (x0 - start_x) / lengths[i]
                return beta

        assert False

    def draw_svg(self, drawing, g):
        assert isinstance(drawing, svgwrite.Drawing)
        glane = drawing.g()
        glane.attribs['class'] = 'lane'

        cp1 = self.control_points[0]
        cp2 = self.control_points[-1]
        rel = relative_pose(cp1.as_SE2(), cp2.as_SE2())
        _, theta = geo.translation_angle_from_SE2(rel)

        delta = int(np.sign(theta))
        fill = {
            -1: '#577094',
            0: '#709457',
            1: '#947057',

        }[delta]

        points = self.lane_profile(points_per_segment=10)
        p = drawing.polygon(points=points,
                            fill=fill,
                            fill_opacity=0.5)

        glane.add(p)

        center_points = self.center_line_points(points_per_segment=10)
        center_points = [geo.translation_angle_from_SE2(_)[0] for _ in center_points]
        p = drawing.polyline(points=center_points,
                             stroke=fill,
                             fill='none',
                             stroke_dasharray="0.02",
                             stroke_width=0.01)
        glane.add(p)

        g.add(glane)

        for x in self.control_points:
            q = x.asmatrix2d().m
            p1, _ = geo.translation_angle_from_SE2(q)
            delta_arrow = np.array([self.width / 4, 0])
            p2 = SE2_apply_R2(q, delta_arrow)
            gp = drawing.g()
            gp.attribs['class'] = 'control-point'
            l = drawing.line(start=p1.tolist(), end=p2.tolist(),
                             stroke='black',
                             stroke_width=self.width / 20.0)
            gp.add(l)
            c = drawing.circle(center=p1.tolist(),
                               r=self.width / 8,
                               fill='white',
                               stroke='black',
                               stroke_width=self.width / 20.0)
            gp.add(c)
            g.add(gp)

    @contract(returns='SE2')
    def center_point(self, beta):
        n = len(self.control_points)
        i = int(np.floor(beta))

        if i < 0:
            q0 = self.control_points[0].asmatrix2d().m
            q1 = geo.SE2.multiply(q0, geo.SE2_from_translation_angle([0.1, 0], 0))
            alpha = beta

        elif i >= n - 1:
            # q0 = self.control_points[-2].asmatrix2d().m
            q0 = self.control_points[-1].asmatrix2d().m
            q1 = geo.SE2.multiply(q0, geo.SE2_from_translation_angle([0.1, 0], 0))
            alpha = beta - (n - 1)
        else:
            alpha = beta - i
            q0 = self.control_points[i].asmatrix2d().m
            q1 = self.control_points[i + 1].asmatrix2d().m
        q = SE2_interpolate(q0, q1, alpha)
        return q

    @memoized_reset
    def center_line_points(self, points_per_segment=5):
        n = len(self.control_points) - 1
        num = n * points_per_segment
        betas = np.linspace(0, n, num=num)
        res = []
        for beta in betas:
            q = self.center_point(beta)
            res.append(q)
        return res

    @memoized_reset
    def lane_profile(self, points_per_segment=5):
        points_left = []
        points_right = []
        n = len(self.control_points) - 1
        num = n * points_per_segment
        betas = np.linspace(0, n, num=num)
        for beta in betas:
            q = self.center_point(beta)
            delta_left = np.array([0, self.width / 2])
            delta_right = np.array([0, -self.width / 2])
            points_left.append(SE2_apply_R2(q, delta_left))
            points_right.append(SE2_apply_R2(q, delta_right))

        return points_right + list(reversed(points_left))


new_contract('LaneSegment', LaneSegment)


def get_distance_two(q0, q1):
    g = geo.SE2.multiply(geo.SE2.inverse(q0), q1)
    v = geo.SE2.algebra_from_group(g)
    linear, angular = geo.linear_angular_from_se2(v)
    return np.linalg.norm(linear)
