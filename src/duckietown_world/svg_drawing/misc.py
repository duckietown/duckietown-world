import itertools
import math
import os

import svgwrite
import yaml
from bs4 import Tag
from contracts import contract
from past.builtins import reduce

from duckietown_world import logger
from duckietown_world.geo import RectangularArea
from duckietown_world.geo.measurements_utils import get_extent_points
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.world_duckietown.transformations import get_sampling_points, ChooseTime

__all__ = [
    'draw_recursive',
    'get_basic_upright2',
    'draw_static',
]


@contract(area=RectangularArea)
def get_basic_upright2(filename, area, size=(1024, 768)):
    drawing = svgwrite.Drawing(filename, size=size)

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

    from duckietown_world.world_duckietown.duckiebot import draw_axes


    # l = drawing.line(start=(-10, 0), end=(10, 0), stroke_width=0.01, stroke='red')
    # base.add(l)
    # l = drawing.line(start=(0, -10), end=(0, +10), stroke_width=0.01, stroke='green')
    # base.add(l)

    grid = drawing.g(id='grid')
    for i, j in itertools.product(range(i0, i1), range(j0, j1)):
        where = (i, j)
        rect = drawing.rect(insert=where,
                            fill="none",
                            size=(1, 1,),
                            stroke_width="0.01",
                            stroke="#eeeeee", )
        grid.add(rect)

    if False:
        for i, j in itertools.product(range(H), range(W)):
            where = (i, j)
            t = drawing.text('x = %s, y = %s' % (i, j), insert=where,
                             # ,
                             style="font-size: 0.1",
                             # transform='scale(1,-1)'
                             )
            grid.add(t)

    base.add(grid)
    draw_axes(drawing, base, L=0.5, stroke_width=0.03)

    inside = drawing.g()
    tofill =drawing.g()
    inside.add(tofill)

    view = drawing.rect(insert=area.pmin.tolist(),
                     size=(area.pmax - area.pmin).tolist(),
                     stroke_width=0.1,
                     fill='none',
                     stroke='gray')
    inside.add(view)
    base.add(inside)

    drawing.add(base)
    return drawing, tofill


def draw_recursive(drawing, po, g):
    po.draw_svg(drawing, g)

    for child_name in po.get_drawing_children():
        child = po.children[child_name]
        # find transformations
        transforms = [_ for _ in po.spatial_relations.values() if _.a == () and _.b == (child_name,)]
        # print('draw_recursive %s %s' % (child, transforms))
        if transforms:
            # print('ok')
            M = transforms[0].transform.asmatrix2d().m

            # print(M)
            svg_transform = 'matrix(%s,%s,%s,%s,%s,%s)' % (M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])
            # print(svg_transform)
            g2 = drawing.g(id=child_name, transform=svg_transform)

            # print(g2.transform)
            t = drawing.text(child_name, style='font-size: 0.03em; transform: scaleY(-1)')
            t.attribs['class'] = 'labels'
            # t.attribs['style'] = 'z-index: 1000'
            # g2.add(t)
            draw_recursive(drawing, child, g2)

            # print(g2.tostring())
            g.add(g2)


def draw_static(root, output_dir, pixel_size=(640, 640), area=None):
    # space = (10, 10)
    # origin = (-1, -1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn_svg = os.path.join(output_dir, 'drawing.svg')
    fn_html = os.path.join(output_dir, 'drawing.html')

    timestamps = get_sampling_points(root)
    keyframes = SampledSequence(range(len(timestamps)), timestamps)
    nkeyframes = len(keyframes)

    t0 = timestamps[0]
    root_t0 = root.filter_all(ChooseTime(t0))

    if area is None:
        areas = []
        for i, t in keyframes:
            root_t = root.filter_all(ChooseTime(t))
            rarea = get_extent_points(root_t)
            areas.append(rarea)
        area = reduce(RectangularArea.join, areas)

    drawing, base = get_basic_upright2(fn_svg, area, pixel_size)

    gmg = drawing.g()
    base.add(gmg)

    for i, t in keyframes:
        g_t = drawing.g()
        g_t.attribs['class'] = 'keyframe keyframe%d' % i

        root_t = root.filter_all(ChooseTime(t))

        draw_recursive(drawing, root_t, g_t)
        base.add(g_t)

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
    other = str(other)

    from duckietown_world.svg_drawing.draw_log import make_html_slider
    html = make_html_slider(drawing, nkeyframes=nkeyframes, obs_div='', other=other)
    with open(fn_html, 'w') as f:
        f.write(html)

    drawing.save(pretty=True)
    logger.info('Written SVG to %s' % fn_svg)
    logger.info('Written HTML to %s' % fn_html)

    return [fn_svg, fn_html]
