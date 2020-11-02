# coding=utf-8
__version__ = "6.0.21"

import logging

logging.basicConfig()
logger = logging.getLogger("dt-world")
logger.setLevel(logging.DEBUG)

logger.info("duckietown-world %s" % __version__)

# remove noisy logging
from duckietown_serialization_ds1 import logger as dslogger

dslogger.setLevel(logging.CRITICAL)

from contracts import disable_all

disable_all()

from .geo import *
from .seqs import *
from .svg_drawing import *
from .world_duckietown import *
