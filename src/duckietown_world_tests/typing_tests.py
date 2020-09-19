from duckietown_world import GenericSequence, SampledSequence
from zuper_commons.logs import ZLogger
from zuper_typing import can_be_used_as2

logger = ZLogger(__name__)


def test_typing():
    s = SampledSequence.from_iterator([(0, "a"), (1, "b"), (2, "c")], str)
    T = GenericSequence[object]

    r = can_be_used_as2(type(s), T)
    logger.info(r=r)
    assert isinstance(s, T)
