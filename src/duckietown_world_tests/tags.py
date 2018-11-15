# coding=utf-8
import numpy as np
from comptests import comptest, run_module_tests
from numpy.testing import assert_almost_equal

import geometry as geo
from duckietown_world.world_duckietown import Integrator2D, GenericKinematicsSE2
from duckietown_world.world_duckietown.differential_drive_dynamics import DifferentialDriveDynamicsParameters, \
    WheelVelocityCommands


@comptest
def tag_positions():

    q0 = geo.SE2_from_translation_angle([0, 0], 0)
    v0 = geo.se2.zero()

if __name__ == '__main__':
    run_module_tests()
