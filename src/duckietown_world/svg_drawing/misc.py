# coding=utf-8
import base64
import itertools
import logging
import math
import os
from six import BytesIO

import svgwrite
import yaml
from bs4 import Tag
from contracts import contract
from past.builtins import reduce

from duckietown_world import logger
from duckietown_world.geo import RectangularArea
from duckietown_world.geo.measurements_utils import get_extent_points, get_static_and_dynamic
from duckietown_world.seqs.tsequence import SampledSequence, UndefinedAtTime
from duckietown_world.utils.memoizing import memoized_reset
from duckietown_world.world_duckietown import get_sampling_points, ChooseTime

__all__ = [
    'draw_recursive',
    'get_basic_upright2',
    'draw_static',
    'draw_axes',
    'draw_children',
    'data_encoded_for_src',
]


@contract(area=RectangularArea)
def get_basic_upright2(filename, area, size=(1024, 768)):
    drawing = svgwrite.Drawing(filename, size=size, debug=False)

    origin = area.pmin
    other = area.pmax
    space = other - origin

    sw = size[0] * 1.0 / space[0]
    sh = size[1] * 1.0 / space[1]

    s = min(sh, sw)
    # print(space, size, sw, sh, s)
    tx = 0 - origin[0] * s
    ty = (space[1] + origin[1]) * s

    base = drawing.g(transform='translate(%s, %s) scale(%s) scale(+1,-1)' % (tx, ty, s))

    i0 = int(math.floor(area.pmin[0]))
    j0 = int(math.floor(area.pmin[1]))
    i1 = int(math.ceil(area.pmax[0]))
    j1 = int(math.ceil(area.pmax[1]))

    grid = drawing.g(id='grid')
    for i, j in itertools.product(range(i0, i1), range(j0, j1)):
        where = (i, j)
        rect = drawing.rect(insert=where,
                            fill="none",
                            size=(1, 1,),
                            stroke_width="0.01",
                            stroke="#eeeeee", )
        grid.add(rect)

    base.add(grid)
    draw_axes(drawing, base, L=0.5, stroke_width=0.03)

    inside = drawing.g()
    tofill = drawing.g()
    inside.add(tofill)

    view = drawing.rect(insert=area.pmin.tolist(),
                        size=(area.pmax - area.pmin).tolist(),
                        stroke_width=0.01,
                        fill='none',
                        stroke='black')
    inside.add(view)
    base.add(inside)

    drawing.add(base)
    return drawing, tofill


def draw_recursive(drawing, po, g, draw_list=()):
    if () in draw_list:
        po.draw_svg(drawing, g)
    draw_children(drawing, po, g, draw_list=draw_list)


def draw_children(drawing, po, g, draw_list=()):
    for child_name in po.get_drawing_children():
        child = po.children[child_name]
        transforms = [_ for _ in po.spatial_relations.values() if _.a == () and _.b == (child_name,)]
        if transforms:

            rlist = recurive_draw_list(draw_list, child_name)

            if rlist:
                M = transforms[0].transform.asmatrix2d().m
                svg_transform = 'matrix(%s,%s,%s,%s,%s,%s)' % (M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])

                g2 = drawing.g(id=child_name, transform=svg_transform)
                classes = get_typenames_for_class(child)
                if classes:
                    g2.attribs['class'] = " ".join(classes)
                draw_recursive(drawing, child, g2, draw_list=rlist)

                g.add(g2)


def get_typenames_for_class(ob):
    mro = type(ob).mro()
    names = [_.__name__ for _ in mro]
    for n in ['Serializable', 'Serializable0', 'PlacedObject', 'object']:
        names.remove(n)
    return names


def recurive_draw_list(draw_list, prefix):
    res = []
    for _ in draw_list:
        if _ and _[0] == prefix:
            res.append(_[1:])
    return res


def draw_static(root, output_dir, pixel_size=(640, 640), area=None, images=None):
    images = images or {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn_svg = os.path.join(output_dir, 'drawing.svg')
    fn_html = os.path.join(output_dir, 'drawing.html')

    timestamps = get_sampling_points(root)
    if len(timestamps) == 0:
        keyframes = SampledSequence([0], [0])
    else:
        keyframes = SampledSequence(range(len(timestamps)), timestamps)
    nkeyframes = len(keyframes)

    # t0 = keyframes.at(0)
    # root_t0 = root.filter_all(ChooseTime(t0))

    if area is None:
        areas = []
        for i, t in keyframes:
            root_t = root.filter_all(ChooseTime(t))
            rarea = get_extent_points(root_t)
            areas.append(rarea)
        area = reduce(RectangularArea.join, areas)
        print('using area: %s' % area)

    drawing, base = get_basic_upright2(fn_svg, area, pixel_size)

    gmg = drawing.g()
    base.add(gmg)

    static, dynamic = get_static_and_dynamic(root)

    t0 = keyframes.values[0]
    root_t0 = root.filter_all(ChooseTime(t0))
    g_static = drawing.g()
    g_static.attribs['class'] = 'static'

    draw_recursive(drawing, root_t0, g_static, draw_list=static)
    base.add(g_static)

    obs_div = Tag(name='div')
    imagename2div = {}
    for name in images:
        imagename2div[name] = Tag(name='div')
        obs_div.append(imagename2div[name])

    for i, t in keyframes:
        g_t = drawing.g()
        g_t.attribs['class'] = 'keyframe keyframe%d' % i

        root_t = root.filter_all(ChooseTime(t))

        draw_recursive(drawing, root_t, g_t, draw_list=dynamic)
        base.add(g_t)

        for name, sequence in images.items():
            try:
                obs = sequence.at(t)
            except UndefinedAtTime as e:
                print(str(e))
            else:
                img = Tag(name='img')
                img.attrs['src'] = data_encoded_for_src(obs.bytes_contents, obs.content_type)
                img.attrs['class'] = 'keyframe keyframe%d' % i
                img.attrs['visualize'] = 'hide'
                imagename2div[name].append(img)

    other = Tag(name='div')

    summary = Tag(name='summary')
    summary.append('Log data')
    details = Tag(name='details')

    details.append(summary)
    pre = Tag(name='pre')
    code = Tag(name='code')
    pre.append(code)
    y = yaml.safe_dump(root.as_json_dict(), default_flow_style=False)
    code.append(y)
    details.append(pre)

    other.append(details)
    other1 = str(other)

    # language=html
    other2 = """\
            <style>
            *[visualize_parts=false] {
                display: none;
            }
            </style>

            <input id='checkbox-static' type="checkbox"  onclick="hideshow(this);" checked>static data</input>
            <input id='checkbox-textures' type="checkbox"  onclick="hideshow(this);" checked>textures</input>
            <input id='checkbox-axes' type="checkbox"  onclick="hideshow(this);">axes</input>
            <input id='checkbox-lane_segments' type="checkbox"  onclick="hideshow(this);">map lane segments</input>
            <input id='checkbox-lane_segments-control_points' type="checkbox"  onclick="hideshow(this);">map lane segments control points</input>
            <input id='checkbox-current_lane' type="checkbox"  onclick="hideshow(this);">current lane</input>
            <input id='checkbox-vehicles' type="checkbox"  onclick="hideshow(this);" checked>other vehicles</input>
            <input id='checkbox-duckies' type="checkbox"  onclick="hideshow(this);" checked>duckies</input>
            <input id='checkbox-decorations' type="checkbox"  onclick="hideshow(this);" checked>decorations</input>
            <input id='checkbox-signs' type="checkbox"  onclick="hideshow(this);" checked>signs</input>
            <script>
                var checkboxValues = JSON.parse(localStorage.getItem('checkboxValues')) || {};
                console.log(checkboxValues);
                name2selector = {
                    "checkbox-static": "g.static",
                    "checkbox-textures": "g.static .tile-textures",
                    "checkbox-axes": "g.axes",
                    "checkbox-lane_segments": "g.static .LaneSegment",
                    "checkbox-lane_segments-control_points": ".LaneSegment .control-point",
                    "checkbox-current_lane": "g.keyframe .LaneSegment",
                    "checkbox-duckies": ".Duckie",
                    "checkbox-signs": ".Sign",
                    "checkbox-vehicles": ".Vehicle",
                    "checkbox-decorations": ".Decoration",
                };
                function hideshow(element) {
                    console.log(element);
                    element_name = element.id;
                    console.log(element_name);
                    selector = name2selector[element_name];
                    checked = element.checked;
                    console.log(selector);
                    console.log(checked);
                    elements = document.querySelectorAll(selector);
                    elements.forEach(_ => _.setAttribute('visualize_parts', checked));
                    checkboxValues[element_name] = checked;
                    localStorage.setItem("checkboxValues", JSON.stringify(checkboxValues));
                }
                document.addEventListener("DOMContentLoaded", function(event) {
                    for(var name in name2selector) {
                        console.log(name);
                        element = document.getElementById(name);
                        if(name in checkboxValues) {
                            element.checked = checkboxValues[name];
                        }
                        
                        
                        hideshow(element);
                    } 
                     
                });
            </script>
        """
    other = other2 + other1

    obs_div = str(obs_div)
    html = make_html_slider(drawing, nkeyframes=nkeyframes, obs_div=obs_div, other=other)
    with open(fn_html, 'w') as f:
        f.write(html)

    drawing.save(pretty=True)
    logger.info('Written SVG to %s' % fn_svg)
    logger.info('Written HTML to %s' % fn_html)

    return [fn_svg, fn_html]


def make_html_slider(drawing, nkeyframes, obs_div, other):
    # language=html
    controls = """\
<p id="valBox"></p>
<div class="slidecontainer">
    <input autofocus type="range" min="0" max="%s" value="0" class="slider" id="myRange" onchange="showVal(this.value)"/>
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
        elements = document.querySelectorAll('.keyframe');
        elements.forEach(_ => _.setAttribute('visualize', 'hide'));
        elements_show = document.querySelectorAll('.keyframe' + newVal );  
        elements_show.forEach(_ => _.setAttribute('visualize', 'show'));
    }
    document.addEventListener("DOMContentLoaded", function(event) {
        showVal(0);
    });
</script>
""" % (nkeyframes - 1)

    drawing_svg = drawing.tostring()
    # language=html
    doc = """\
<html>
<head></head>
<body>
{controls}
<table>
<tr>
<td style="width: 640px; vertical-align:top;">
{drawing}
</td>
<td id="obs" >
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


def data_encoded_for_src(data, mime):
    """ data =
        ext = png, jpg, ...

        returns "data: ... " sttring
    """
    encoded = base64.b64encode(data).decode()
    link = 'data:%s;base64,%s' % (mime, encoded)
    return link


def draw_axes(drawing, g, L=0.1, stroke_width=0.01):
    g2 = drawing.g()
    g2.attribs['class'] = 'axes'
    line = drawing.line(start=(0, 0),
                        end=(L, 0),
                        stroke_width=stroke_width,
                        stroke="red")
    g2.add(line)

    line = drawing.line(start=(0, 0),
                        end=(0, L),
                        stroke_width=stroke_width,
                        stroke="green")
    g2.add(line)

    g.add(g2)


@memoized_reset
def get_jpeg_bytes(fn):
    from PIL import Image
    pl = logging.getLogger('PIL')
    pl.setLevel(logging.ERROR)

    image = Image.open(fn).convert('RGB')

    out = BytesIO()
    image.save(out, format='jpeg')
    return out.getvalue()
