# coding=utf-8

from comptests import comptest, run_module_tests

# from duckietown_world import SampledSequence, PWMCommands
from duckietown_world import PWMCommands
from duckietown_world.seqs.tsequence import SampledSequenceBuilder


@comptest
def test1():
    # print(SampledSequence)
    # print(SampledSequence[int])
    # print(SampledSequence[float])
    # print(SampledSequence[int])
    # print(SampledSequenceBuilder[PWMCommands])
    # print(SampledSequenceBuilder[PWMCommands]())
    B = SampledSequenceBuilder
    print('B:')
    print(B)
    print(B.__annotations__)

    print('C:')
    C = SampledSequenceBuilder[PWMCommands]
    print(C)
    print(C.__annotations__)

    print('b:')
    b = C()

    print('x:')
    x = b.as_sequence()
    print(x)

    print(x.__dict__)
    print(type(x))
    print(type(x).__annotations__)

    assert 'PWMCommands' in str(type(x).__annotations__)


if __name__ == '__main__':
    run_module_tests()
