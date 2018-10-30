# coding=utf-8
from .dynamics import *
from .measurements import *
from .svg import *
from .world_building import *
from .lane_pose import *

def jobs_comptests(context):
    # instantiation
    # from comptests import jobs_registrar
    from comptests.registrar import jobs_registrar_simple
    jobs_registrar_simple(context)
