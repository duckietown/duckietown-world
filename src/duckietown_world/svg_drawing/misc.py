# coding=utf-8
import base64
import itertools
import logging
import math
import os
from collections import OrderedDict

import svgwrite
from bs4 import Tag, BeautifulSoup
from contracts import contract, check_isinstance
from duckietown_world import logger
from duckietown_world.geo import RectangularArea, get_extent_points, get_static_and_dynamic
from duckietown_world.seqs import SampledSequence, UndefinedAtTime
from duckietown_world.utils import memoized_reset
from past.builtins import reduce
from six import BytesIO

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

    tx = 0 - origin[0] * s
    ty = (space[1] + origin[1]) * s

    base = drawing.g(transform='translate(%s, %s) scale(%s) scale(+1,-1)' % (tx, ty, s))
    base.attribs['id'] = 'base'
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
                            stroke_width=0.01,
                            stroke="#eeeeee", )
        grid.add(rect)

    base.add(grid)
    draw_axes(drawing, base, L=0.5, stroke_width=0.03)

    inside = drawing.g()
    inside.attribs['id'] = 'inside'
    tofill = drawing.g()
    tofill.attribs['id'] = 'tofill'
    inside.add(tofill)

    view = drawing.rect(insert=area.pmin.tolist(),
                        size=(area.pmax - area.pmin).tolist(),
                        stroke_width=0.01,
                        fill='none',
                        stroke='black')
    view.attribs['id'] = 'view'
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


def draw_static(root, output_dir, pixel_size=(480, 480), area=None, images=None,
                timeseries=None):
    from duckietown_world.world_duckietown import get_sampling_points, ChooseTime
    images = images or {}
    timeseries = timeseries or {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn_svg = os.path.join(output_dir, 'drawing.svg')
    fn_html = os.path.join(output_dir, 'drawing.html')

    timestamps = get_sampling_points(root)
    if len(timestamps) == 0:
        keyframes = SampledSequence([0], [0])
    else:
        keyframes = SampledSequence(range(len(timestamps)), timestamps)
    # nkeyframes = len(keyframes)

    if area is None:
        areas = []
        for i, t in keyframes:
            root_t = root.filter_all(ChooseTime(t))
            rarea = get_extent_points(root_t)
            areas.append(rarea)
        area = reduce(RectangularArea.join, areas)

    logger.info('area: %s' % area)
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

    # logger.debug('dynamic: %s' % dynamic)
    for i, t in keyframes:
        g_t = drawing.g()
        g_t.attribs['class'] = 'keyframe keyframe%d' % i

        root_t = root.filter_all(ChooseTime(t))

        draw_recursive(drawing, root_t, g_t, draw_list=dynamic)
        base.add(g_t)

        for name, sequence in images.items():
            try:
                obs = sequence.at(t)
            except UndefinedAtTime:
                pass
            else:
                img = Tag(name='img')
                resized = get_resized_image(obs.bytes_contents, 200)
                img.attrs['src'] = data_encoded_for_src(resized, 'image/jpeg')
                # print('image %s %s: %.4fMB ' % (i, t, len(resized) / (1024 * 1024.0)))
                img.attrs['class'] = 'keyframe keyframe%d' % i
                img.attrs['visualize'] = 'hide'
                imagename2div[name].append(img)

    # other = Tag(name='div')
    #
    # summary = Tag(name='summary')
    # summary.append('Log data')
    # details = Tag(name='details')
    #
    # details.append(summary)
    # pre = Tag(name='pre')
    # code = Tag(name='code')
    # pre.append(code)
    # y = yaml.safe_dump(root.as_json_dict(), default_flow_style=False)
    # code.append(y)
    # details.append(pre)
    #
    # other.append(details)
    # other1 = str(other)
    other = ""

    # language=html
    visualize_controls = """\
            <style>
            *[visualize_parts=false] {
                display: none;
            }
            </style>
        
            <p>
            <input id='checkbox-static' type="checkbox"  onclick="hideshow(this);" checked>static data</input>
            <input id='checkbox-textures' type="checkbox"  onclick="hideshow(this);" checked>textures</input>
            <input id='checkbox-axes' type="checkbox"  onclick="hideshow(this);">axes</input>
            <br/>
            <input id='checkbox-lane_segments' type="checkbox"  onclick="hideshow(this);">map lane segments</input>
            (<input id='checkbox-lane_segments-control_points' type="checkbox"  onclick="hideshow(this);">control points</input>)</p>
            </p>
           
            
            <p>
            <input id='checkbox-vehicles' type="checkbox"  onclick="hideshow(this);" checked>vehicles</input>
            <input id='checkbox-duckies' type="checkbox"  onclick="hideshow(this);" checked>duckies</input>
            <input id='checkbox-signs' type="checkbox"  onclick="hideshow(this);" checked>signs</input>
            <input id='checkbox-sign-papers' type="checkbox"  onclick="hideshow(this);" checked>signs textures</input>
            <input id='checkbox-decorations' type="checkbox"  onclick="hideshow(this);" checked>decorations</input>
          
            </p>
             <p>
            <input id='checkbox-current_lane' type="checkbox"  onclick="hideshow(this);">current lane</input>
            <input id='checkbox-anchors' type="checkbox"  onclick="hideshow(this);">anchor point</input>
            </p>
            <script>
                var checkboxValues = JSON.parse(localStorage.getItem('checkboxValues')) || {};
                console.log(checkboxValues);
                name2selector = {
                    "checkbox-static": "g.static",
                    "checkbox-textures": "g.static .tile-textures",
                    "checkbox-axes": "g.axes",
                    "checkbox-lane_segments": "g.static .LaneSegment",
                    "checkbox-lane_segments-control_points": " .control-point",
                    "checkbox-current_lane": "g.keyframe .LaneSegment",
                    "checkbox-duckies": ".Duckie",
                    "checkbox-signs": ".Sign",
                    "checkbox-sign-papers": ".Sign .sign-paper",
                    "checkbox-vehicles": ".Vehicle",
                    "checkbox-decorations": ".Decoration",
                    'checkbox-anchors': '.Anchor',
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

    div_timeseries = str(make_tabs(timeseries))

    obs_div = str(obs_div)
    html = make_html_slider(drawing, keyframes, obs_div=obs_div, other=other,
                            div_timeseries=div_timeseries,
                            visualize_controls=visualize_controls)
    with open(fn_html, 'w') as f:
        f.write(html)

    # language=css
    style = """
        .sign-paper {
            display: none;
        }
        g.axes, .LaneSegment {
            display: none;
        }
         
    """
    drawing.defs.add(drawing.style(style))

    drawing.save(pretty=True)
    logger.info('Written SVG to %s' % fn_svg)
    logger.info('Written HTML to %s' % fn_html)

    return [fn_svg, fn_html]


def get_resized_image(bytes_content, width):
    from PIL import Image
    pl = logging.getLogger('PIL')
    pl.setLevel(logging.ERROR)
    idata = BytesIO(bytes_content)
    image = Image.open(idata).convert('RGB')
    size = image.size
    height = int(size[1] * 1.0 / size[0] * width)
    image = image.resize((width, height))
    out = BytesIO()
    image.save(out, format='jpeg')
    return out.getvalue()


class TimeseriesPlot(object):

    def __init__(self, title, long_description, sequences):
        check_isinstance(title, six.string_types)
        self.title = title
        self.long_description = long_description
        self.sequences = sequences

    def get_title(self):
        return self.title

    def get_long_description(self):
        return self.long_description


def make_tabs(timeseries):
    tabs = OrderedDict()
    import plotly.offline as offline
    i = 0
    for name, tsp in timeseries.items():
        assert isinstance(tsp, TimeseriesPlot)

        div = Tag(name='div')
        table = Tag(name='table')
        table.attrs['style'] = 'width: 100%'
        tr = Tag(name='tr')

        td = Tag(name='td')
        td.attrs['style'] = 'width: 15em; min-height: 20em; vertical-align: top;'
        td.append(get_markdown(tsp.long_description))
        tr.append(td)

        td = Tag(name='td')
        td.attrs['style'] = 'width: calc(100%-16em); min-height: 20em; vertical-align: top;'

        import plotly.graph_objs as go
        import plotly.tools as tools
        scatters = []
        for name_sequence, sequence in tsp.sequences.items():
            assert isinstance(sequence, SampledSequence)

            trace = go.Scatter(
                    x=sequence.timestamps,
                    y=sequence.values,
                    mode='lines+markers',
                    name=name_sequence,
            )
            scatters.append(trace)

        layout = {'font': dict(size=10), 'margin': dict(t=0)}

        n = len(scatters)
        fig = tools.make_subplots(rows=1, cols=n)
        fig.layout.update(layout)
        for j, scatter in enumerate(scatters):
            fig.append_trace(scatter, 1, j + 1)

        # include_plotlyjs = True if i == 0 else False
        include_plotlyjs = True
        res = offline.plot(fig,
                           output_type='div',
                           show_link=False,
                           include_plotlyjs=include_plotlyjs)
        td.append(bs(res))
        i += 1

        tr.append(td)
        table.append(tr)

        div.append(table)

        tabs[name] = Tab(title=tsp.title, content=div)

    return render_tabs(tabs)


import six


class Tab(object):
    def __init__(self, title, content):
        check_isinstance(title, six.string_types)
        self.title = title
        self.content = content


def render_tabs(tabs):
    div_buttons = Tag(name='div')
    div_buttons.attrs['class'] = 'tab'
    div_content = Tag(name='div')

    for i, (name, tab) in enumerate(tabs.items()):
        assert isinstance(tab, Tab), tab

        tid = 'tab%s' % i
        button = Tag(name='button')
        button.attrs['class'] = 'tablinks'
        button.attrs['onclick'] = "open_tab(event,'%s')" % tid
        button.append(tab.title)
        div_buttons.append(button)

        div_c = Tag(name='div')
        div_c.attrs['id'] = tid
        div_c.attrs['style'] = ''  # ''display: none; width:100%; height:100vh'

        div_c.attrs['class'] = 'tabcontent'

        div_c.append(tab.content)

        div_content.append(div_c)

    script = Tag(name='script')
    # language=javascript
    js = """
function open_tab(evt, cityName) {
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(cityName).style.display = "block";
    document.getElementById(cityName).style.opacity = 1.0;
    evt.currentTarget.className += " active";
} 
    
    """
    script.append(js)

    style = Tag(name='style')
    # language=css
    style.append('''\
/* Style the tab */
.tab {
    overflow: hidden;
    border: 1px solid #ccc;
    background-color: #f1f1f1;
}

/* Style the buttons that are used to open the tab content */
.tab button {
    
    font-size: 80%;
    background-color: inherit;
    float: left;
    border: solid 0.5px gray;
    outline: none;
    cursor: pointer;
    /* padding: 14px 16px;*/
    transition: 0.3s;
}

.tab button + button {
    margin-left: 10px;
}

/* Change background color of buttons on hover */
.tab button:hover {
    background-color: #ddd;
}

/* Create an active/current tablink class */
.tab button.active {
    background-color: #ccc;
}

/* Style the tab content */
.tabcontent {
    /*display: none;*/
    opacity: 0;
    padding: 6px 12px;
    border: 1px solid #ccc;
    border-top: none;
    width: 100%;
}
    
    ''')
    main = Tag(name='div')
    main.attrs['id'] = 'tabs'
    main.append(style)
    main.append(script)
    main.append(div_buttons)
    main.append(div_content)
    return main


def make_html_slider(drawing, keyframes, obs_div, other, div_timeseries, visualize_controls):
    nkeyframes = len(keyframes.timestamps)

    # language=html
    controls_html = """\

<div id="slidecontainer">
<div id='fixedui'>
    Select time: <input autofocus type="range" min="0" max="%s" value="0" class="slider" id="time-range" onchange="showVal(this.value)" oninput="showVal(this.value)"/>
    <span id="time-display"></span>
    </div>
</div>
<style type='text/css'>
    #slidecontainer {
    height: 3em;
    }
    #time-range {
    width: 50%%;
    }
    #fixedui { 
    position: fixed; 
    width: 100%%;
    height: 3em;
    background-color: white;
    }
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
    
    #observation_sequence {
        width: 220px;
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

    if nkeyframes <= 1:
        controls_html += ('''
        <style>
        .slidecontainer {
        display: none;
        }
        </style>
        ''')

    controls = bs(controls_html)

    valbox = controls.find('span', id='time-display')
    assert valbox is not None
    for i, timestamp in keyframes:
        t = Tag(name='span')
        t.attrs['class'] = 'keyframe keyframe%d' % i
        t.attrs['visualize'] = 'hide'
        t.append('t = %.2f' % timestamp)

        valbox.append(t)

    from six import StringIO
    f = StringIO()
    drawing.write(f, pretty=True)
    drawing_svg = f.getvalue()
    f.close()
    # drawing_svg = drawing.tostring(pretty=True)
    # language=html
    doc = """\
<html>
<head></head>
<body>
<style>
/*svg {{ background-color: #eee;}}*/
body {{
    font-family: system-ui, sans-serif;
}}
</style>
{controls}
<table>
<tr>
<td style="width: 640px; vertical-align:top;">
{drawing}
</td>
<td id="obs" >
{visualize_controls}
<div id="observation_sequence">
{obs_div}
</div>
</td>
</tr>
</table>
{div_timeseries}
{other}
</body>
</html>
    """.format(controls=str(controls), drawing=drawing_svg, obs_div=obs_div, other=other,
               div_timeseries=div_timeseries, visualize_controls=visualize_controls)
    return doc


def mime_from_fn(fn):
    if fn.endswith('png'):
        return 'image/png'
    elif fn.endswith('jpg'):
        return 'image/jpeg'
    else:
        raise ValueError(fn)


def data_encoded_for_src(data, mime):
    """ data =
        ext = png, jpg, ...

        returns "data: ... " sttring
    """
    encoded = base64.b64encode(data).decode()
    link = 'data:%s;base64,%s' % (mime, encoded)
    return link


def draw_axes(drawing, g, L=0.1, stroke_width=0.01, klass='axes'):
    g2 = drawing.g()
    g2.attribs['class'] = klass
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


def bs(fragment):
    """ Returns the contents wrapped in an element called "fragment".
        Expects fragment as a str in utf-8 """

    check_isinstance(fragment, six.string_types)

    if six.PY2:
        if isinstance(fragment, unicode):
            fragment = fragment.encode('utf8')
    s = u'<fragment>%s</fragment>' % fragment

    wire = s.encode('utf-8')
    parsed = BeautifulSoup(wire, 'lxml', from_encoding='utf-8')
    res = parsed.html.body.fragment
    assert res.name == 'fragment'
    return res


def get_markdown(md):
    import markdown

    extensions = ['extra', 'smarty']
    html = markdown.markdown(md, extensions=extensions, output_format='html5')

    res = bs(html)
    return res
