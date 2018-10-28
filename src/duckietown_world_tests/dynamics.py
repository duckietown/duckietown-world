# coding=utf-8
import numpy as np
from comptests import comptest, run_module_tests
from numpy.testing import assert_almost_equal

import geometry as geo
from duckietown_world.world_duckietown import Integrator2D, GenericKinematicsSE2
from duckietown_world.world_duckietown.differential_drive_dynamics import DifferentialDriveDynamicsParameters, \
    WheelVelocityCommands


@comptest
def integrator2D_test1():
    q0 = geo.SE2_from_translation_angle([0, 0], 0)
    v0 = geo.se2.zero()
    c0 = q0, v0
    s0 = Integrator2D.initialize(c0)

    commands = [1, 2]
    dt = 2
    s1 = s0.integrate(dt, commands)

    q1, v1 = s1.TSE2_from_state()


@comptest
def kinematics2d_test():
    q0 = geo.SE2_from_translation_angle([0, 0], 0)
    v0 = geo.se2.zero()
    c0 = q0, v0
    s0 = GenericKinematicsSE2.initialize(c0)

    k = 0.4
    radius = 1.0 / k
    print('radius: %s' % radius)
    v = 3.1
    perimeter = 2 * np.pi * radius
    dt_loop = perimeter / v

    w = 2 * np.pi / dt_loop

    dt = dt_loop * .25
    commands = geo.se2_from_linear_angular([v, 0], w)

    s1 = s0.integrate(dt, commands)
    s2 = s1.integrate(dt, commands)
    s3 = s2.integrate(dt, commands)
    s4 = s3.integrate(dt, commands)
    seq = [s0, s1, s2, s3, s4]
    for _ in seq:
        q1, v1 = _.TSE2_from_state()
        print('%s' % geo.SE2.friendly(q1))

    assert_almost_equal(geo.translation_from_SE2(s1.TSE2_from_state()[0]), [radius, radius])
    assert_almost_equal(geo.translation_from_SE2(s2.TSE2_from_state()[0]), [0, radius * 2])
    assert_almost_equal(geo.translation_from_SE2(s3.TSE2_from_state()[0]), [-radius, radius])
    assert_almost_equal(geo.translation_from_SE2(s4.TSE2_from_state()[0]), [0, 0])


@comptest
def dd_test():
    radius = 0.1
    radius_left = radius
    radius_right = radius
    wheel_distance = 0.5
    dddp = DifferentialDriveDynamicsParameters(radius_left=radius_left, radius_right=radius_right,
                                               wheel_distance=wheel_distance)
    q0 = geo.SE2_from_translation_angle([0, 0], 0)
    v0 = geo.se2.zero()
    c0 = q0, v0
    s0 = dddp.initialize(c0)

    omega = 0.3
    commands = WheelVelocityCommands(omega, omega)
    dt = 4.0
    s1 = s0.integrate(dt, commands)

    q1, v1 = s1.TSE2_from_state()
    print(geo.se2.friendly(v1))
    print(geo.SE2.friendly(q1))

    p1, theta = geo.translation_angle_from_SE2(q1)
    assert_almost_equal(p1, [dt * radius * omega, 0])

    omega_left = 0.3
    omega_right = 0
    commands = WheelVelocityCommands(omega_left, omega_right)
    dt = 4.0
    s1 = s0.integrate(dt, commands)

    q1, v1 = s1.TSE2_from_state()
    p1, theta = geo.translation_angle_from_SE2(q1)
    print(geo.se2.friendly(v1))
    print(geo.SE2.friendly(q1))

    # TODO: finish these tests
    # assert_almost_equal(p1[0], [dt * radius * omega_left / 2, 0])


if __name__ == '__main__':
    run_module_tests()
