# coding=utf-8
import argparse
import json
import os
import sys
from collections import namedtuple, OrderedDict, defaultdict

from duckietown_serialization_ds1 import Serializable
from duckietown_world import logger
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import make_timeseries
from duckietown_world.seqs import SampledSequence
from duckietown_world.seqs.tsequence import SampledSequenceBuilder
from .misc import draw_static, TimeseriesPlot

__all__ = [
    'draw_logs_main',
    'draw_logs_main_',
]


def draw_logs_main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output dir")

    parser.add_argument("--filename", required=True)
    parsed = parser.parse_args(args=args)

    filename = parsed.filename
    output = parsed.output
    draw_logs_main_(output, filename)


def draw_logs_main_(output, filename):
    if output is None:
        output = filename + '.out'
    if not os.path.exists(output):
        os.makedirs(output)
    logger.info('processing file %s' % filename)

    log = read_simulator_log(filename)
    duckietown_env = log.duckietown
    if log.observations:
        images = {'observations': log.observations}
    else:
        images = None

    interval = SampledSequence.from_iterator(enumerate(log.trajectory.timestamps))
    evaluated = evaluate_rules(poses_sequence=log.trajectory,
                               interval=interval, world=duckietown_env, ego_name='ego')
    timeseries = make_timeseries(evaluated)
    timeseries.update(timeseries_actions(log))
    print(evaluated)
    draw_static(duckietown_env, output, images=images, timeseries=timeseries)
    return evaluated


def timeseries_actions(log):
    timeseries = OrderedDict()
    sequences = OrderedDict()
    sequences['action0'] = log.actions.transform_values(lambda _: _[0])
    sequences['action1'] = log.actions.transform_values(lambda _: _[1])

    timeseries['actions'] = TimeseriesPlot('Actions', 'actions', sequences)
    return timeseries


def read_log(filename):
    with open(filename) as i:
        for k, line in enumerate(i.readlines()):
            try:
                j = json.loads(line)
            except BaseException:
                msg = 'Cannot interpret json in line %s: "%s"' % (k, line)
                raise Exception(msg)
            try:
                ob = Serializable.from_json_dict(j)
            except BaseException:
                msg = 'Cannot de-serialize in line %s:\n%s' % (k, j)
                raise Exception(msg)

            yield ob


SimulatorLog = namedtuple('SimulatorLog', 'observations duckietown trajectory render_time actions')



def read_simulator_log(filename):
    from duckietown_world.world_duckietown import DB18, construct_map

    duckietown_map = None

    B_CURPOS = 'curpos'
    TOPIC_OBSERVATIONS = 'observations'
    TOPIC_ACTIONS = 'actions'
    TOPIC_RENDER_TIME = 'render_time'
    other_topics = [TOPIC_RENDER_TIME, TOPIC_ACTIONS, TOPIC_OBSERVATIONS]
    builders = defaultdict(SampledSequenceBuilder)

    for ob in read_log(filename):
        if ob.topic == 'map_info':
            map_data = ob.data['map_data']
            tile_size = ob.data['tile_size']
            duckietown_map = construct_map(map_data, tile_size)

        for _ in other_topics:
            if ob.topic == _:
                builders[_].add(ob.timestamp, ob.data)

        if ob.topic == 'Simulator':
            sim = ob.data
            cur_pos = sim['cur_pos']
            cur_angle = sim['cur_angle']

            builders[B_CURPOS].add(ob.timestamp, (cur_pos, cur_angle))

    if len(builders[TOPIC_OBSERVATIONS]):

        observations = builders[TOPIC_OBSERVATIONS].as_sequence()
        logger.info('Found %d observations' % len(observations))
    else:
        observations = None

    if not duckietown_map:
        msg = 'Could not find duckietown_map.'
        raise Exception(msg)

    curpos_sequence = builders[B_CURPOS].as_sequence()

    def curpos2transform(_):
        cur_pos, cur_angle = _
        return duckietown_map.se2_from_curpos(cur_pos, cur_angle)

    trajectory = curpos_sequence.transform_values(curpos2transform)

    if not len(trajectory):
        msg = 'Could not find any position.'
        raise Exception(msg)

    robot = DB18()
    duckietown_map.set_object('ego', robot, ground_truth=trajectory)

    render_time = builders[TOPIC_RENDER_TIME].as_sequence()
    actions = builders[TOPIC_ACTIONS].as_sequence()
    return SimulatorLog(duckietown=duckietown_map, observations=observations,
                        trajectory=trajectory, render_time=render_time,
                        actions=actions)


if __name__ == '__main__':
    draw_logs_main()
