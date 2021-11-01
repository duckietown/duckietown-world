# coding=utf-8
__version__ = "6.2.38"

import logging

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)
import os

path = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"duckietown-world version {__version__} path {path}")

# remove noisy logging
from duckietown_serialization_ds1 import logger as dslogger

dslogger.setLevel(logging.CRITICAL)

from contracts import disable_all

disable_all()

from .geo import *
from .seqs import *
from .resources import *
from .svg_drawing import *
from .world_duckietown import *
