import os

from duckietown_world import draw_static

d = 'out'


def ipython_draw_html(po, outdir=None, area=None):
    from IPython.display import IFrame, display
    if outdir is None:
        outdir = os.path.join(d, 'ipython_draw_html', '%s' % id(po))
    draw_static(po, outdir, area=area)

    iframe = IFrame(src=outdir + '/drawing.html', width='100%', height=600)
    # noinspection PyTypeChecker
    display(iframe)
    return iframe


def ipython_draw_svg(m, outdir=None):
    from IPython.display import SVG, display


    if outdir is None:
        outdir = os.path.join(d, 'ipython_draw_svg', '%s' % id(m))
    draw_static(m, outdir)

    svg = SVG(os.path.join(outdir, 'drawing.svg'))
    # noinspection PyTypeChecker
    display(svg)
    return svg
