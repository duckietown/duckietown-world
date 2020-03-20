# coding=utf-8
__version__ = "5.1.2"

# import zuper_json
import logging


logging.basicConfig()
logger = logging.getLogger("dt-world")
logger.setLevel(logging.DEBUG)

logger.info("duckietown-world %s" % __version__)
# for templating to work
# from zuper_ipce import logger as l
# _ = l

# remove noisy logging
from duckietown_serialization_ds1 import logger as dslogger

dslogger.setLevel(logging.CRITICAL)

from contracts import disable_all, __version__ as contracts_version

# logger.info('contracts %s ' % contracts_version)
disable_all()


from .geo import *
from .seqs import *
from .svg_drawing import *
from .world_duckietown import *
