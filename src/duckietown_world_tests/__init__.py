from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .dynamics import *
from .lane_pose import *
from .measurements import *
from .segment import *
from .svg import *
from .tags import *
from .world_building import *
from .sampling_poses import *
from .pwm_dynamics import *
from .lane_at_point import *
from .typing_tests import *


def jobs_comptests(context):
    # instantiation
    # from comptests import jobs_registrar
    from comptests.registrar import jobs_registrar_simple

    jobs_registrar_simple(context)
