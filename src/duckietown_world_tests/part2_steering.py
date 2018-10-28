# coding=utf-8
import numpy as np
from comptests import comptest, run_module_tests

import geometry as geo
from duckietown_world import SE2Transform, PlacedObject
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static


if __name__ == '__main__':
    run_module_tests()
