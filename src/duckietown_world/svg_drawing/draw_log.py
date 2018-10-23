import argparse
import json
import os
import sys
from collections import namedtuple

import numpy as np
import oyaml as yaml
from bs4 import Tag

from duckietown_serialization_ds1 import Serializable
from duckietown_world import logger
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing.misc import get_basic_upright, draw_recursive
from duckietown_world.world_duckietown.duckiebot import Duckiebot
from duckietown_world.world_duckietown.map_loading import load_gym_map
from duckietown_world.world_duckietown.tile import data_encoded_for_src
from duckietown_world.world_duckietown.transformations import get_sampling_points, ChooseTime, Flatten


def draw_logs_main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output dir")

    parser.add_argument("--filename", required=True)
    parsed = parser.parse_args(args=args)

    filename = parsed.filename
    output = parsed.output
    if output is None:
        output = filename + '.out'
    if not os.path.exists(output):
        os.makedirs(output)

    log = read_simulator_log(filename)
    duckietown_env = log.duckietown

    fn_svg = os.path.join(output, 'drawing.svg')
    fn_html = os.path.join(output, 'drawing-interactive.html')

    tilemap = duckietown_env.children['tilemap']
    gh, gw = tilemap.H * duckietown_env.tile_size, tilemap.W * duckietown_env.tile_size
    # gh = int(math.ceil(gh))
    # gw = int(math.ceil(gw))
    B = 640
    pixel_size = (B, B * gh / gw)
    space = (gh, gw)
    drawing, base = get_basic_upright(fn_svg, space, pixel_size)

    # print(yaml.dump(duckietown_env.as_json_dict(), default_flow_style=False, allow_unicode=True))

    timestamps = get_sampling_points(duckietown_env)
    # print('timestamps: %s' % timestamps)

    t = timestamps[0]
    gm2 = duckietown_env.filter_all(ChooseTime(t))
    gm2 = gm2.filter_all(Flatten())

    gmg = drawing.g()
    base.add(gmg)
    draw_recursive(drawing, gm2, gmg)

    dt = 0.3
    last = -np.inf
    keyframe = 0

    div = Tag(name='div')

    for t in timestamps:
        if t - last < dt:
            continue
        last = t
        duckietown_env2 = duckietown_env.filter_all(ChooseTime(t))
        duckietown_env2 = duckietown_env2.filter_all(Flatten())

        duckietown_env2.children = {'duckiebot': duckietown_env2.children.get('duckiebot')}

        gmg = drawing.g()
        gmg.attribs['class'] = 'keyframe keyframe%d' % keyframe
        draw_recursive(drawing, duckietown_env2, gmg)
        base.add(gmg)

        if log.observations:
            try:
                obs = log.observations.at(t)
            except KeyError as e:
                print(str(e))
            else:
                img = Tag(name='img')
                # print(obs)
                img.attrs['src'] = data_encoded_for_src(obs.bytes_contents, obs.content_type)
                img.attrs['class'] = 'keyframe keyframe%d' % keyframe
                img.attrs['visualize'] = 'hide'
                div.append(img)

        keyframe += 1

    # print(yaml.dump(gm2.as_json_dict(), default_flow_style=False, allow_unicode=True))

    drawing.filename = fn_svg
    drawing.save(pretty=True)

    fn = fn_html

    other = Tag(name='div')

    summary = Tag(name='summary')
    summary.append('Log data')
    details = Tag(name='details')

    details.append(summary)
    pre = Tag(name='pre')
    code = Tag(name='code')
    pre.append(code)
    y = yaml.safe_dump(duckietown_env.as_json_dict(), default_flow_style=False)
    code.append(y)
    details.append(pre)

    other.append(details)

    other = str(other)
    html = make_html_slider(drawing, nkeyframes=keyframe, obs_div=str(div), other=other)
    with open(fn, 'w') as f:
        f.write(html)

    print(fn)


def make_html_slider(drawing, nkeyframes, obs_div, other):
    # language=html
    controls = """\
<p id="valBox"></p>
<div class="slidecontainer">
    <input type="range" min="0" max="%s" value="1" class="slider" id="myRange" onchange="showVal(this.value)"/>
</div>
<style type='text/css'>
    .keyframe[visualize="hide"] {
        display: none;
    }
    .keyframe[visualize="show"] {
        display: inherit;
    }
    td#obs {
        padding: 1em;
        vertical-align: top;
    }
    td#obs img { width: 90%%;} 
</style>
<script type='text/javascript'>
    function showVal(newVal) {
        let elements = document.querySelectorAll('.keyframe');
        elements.forEach(_ => _.setAttribute('visualize', 'hide'));
        let elements_show = document.querySelectorAll('.keyframe' + newVal );  
        elements_show.forEach(_ => _.setAttribute('visualize', 'show'));
    }
    document.addEventListener("DOMContentLoaded", function(event) {
        showVal(0);
    });
</script>
""" % (nkeyframes - 1)

    drawing_svg = drawing.tostring()
    doc = """\
<html>
<head></head>
<body>
{controls}
<table>
<tr>
<td style="width: 640; height: 80vh; vertical-align:top;">
{drawing}
</td>
<td id="obs" >
<p>Robot observations</p>
<div id="observation_sequence">
{obs_div}
</div>
</td>
</tr>
</table>
{other}
</body>
</html>
    """.format(controls=controls, drawing=drawing_svg, obs_div=obs_div, other=other)
    return doc


def read_log(filename):
    with open(filename) as i:
        for line in i.readlines():
            j = json.loads(line)
            ob = Serializable.from_json_dict(j)
            yield ob


SimulatorLog = namedtuple('SimulatorLog', 'observations duckietown')


def read_simulator_log(filename):
    map_name = None
    curpos_timestamps = []
    curpos_values = []

    timestamps_observations = []
    observations = []
    for ob in read_log(filename):
        if ob.topic == 'env_parameters':
            map_name = ob.data['map_name']
        if ob.topic == 'observations':
            timestamps_observations.append(ob.timestamp)
            observations.append(ob.data)
        if ob.topic == 'misc':
            sim = ob.data['Simulator']
            cur_pos = sim['cur_pos']
            # p = [gx, gz]
            # print(sim['cur_pos'])
            cur_angle = sim['cur_angle']

            curpos_values.append((cur_pos, cur_angle))
            curpos_timestamps.append(ob.timestamp)

    if timestamps_observations:
        logger.info('Found %d observations' % len(timestamps_observations))
        observations = SampledSequence(timestamps_observations, observations)
    else:
        observations = None

    if not map_name:
        msg = 'Could not find env_parameters.'
        raise Exception(msg)

    duckietown_map = load_gym_map(map_name)

    transforms = []
    for cur_pos, cur_angle in curpos_values:
        transform = duckietown_map.se2_from_curpos(cur_pos, cur_angle)
        transforms.append(transform)

    trajectory = SampledSequence(curpos_timestamps, transforms)

    # root = PlacedObject()
    # transform = SE2Transform([0.5, 0.5], np.deg2rad(10))
    # root.set_object('duckietown', duckietown_map, ground_truth=transform)

    from gym_duckietown.simulator import ROBOT_WIDTH, ROBOT_LENGTH, ROBOT_HEIGHT

    robot = Duckiebot(length=ROBOT_LENGTH, height=ROBOT_HEIGHT, width=ROBOT_WIDTH - 0.02)
    duckietown_map.set_object('duckiebot', robot, ground_truth=trajectory)
    return SimulatorLog(duckietown=duckietown_map, observations=observations)


if __name__ == '__main__':
    draw_logs_main()
