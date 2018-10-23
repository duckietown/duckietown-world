import itertools
import math

import svgwrite


def get_basic_upright(filename, space, size=(1024, 768)):
    drawing = svgwrite.Drawing(filename, size=size)

    sw = size[0]*1.0 / space[0]
    sh = size[1]*1.0 / space[1]

    s = min(sh, sw)
    print(space, size, sw, sh, s)
    tx = 0
    ty = space[1] * s
    base = drawing.g(transform='translate(%s, %s) scale(%s) scale(+1,-1)' % (tx, ty, s))

    grid = drawing.g(id='grid')
    H0 = int(math.ceil(space[0]))
    W0 = int(math.ceil(space[1]))
    for i, j in itertools.product(range(H0), range(W0)):
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
    drawing.add(base)
    return drawing, base


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
