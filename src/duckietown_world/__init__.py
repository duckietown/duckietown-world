# coding=utf-8
__version__ = '1.0.3'

import logging
logger = logging.getLogger('dt-world')
logger.setLevel(logging.DEBUG)


from .geo import *
from .seqs import *
from .world_duckietown import *
