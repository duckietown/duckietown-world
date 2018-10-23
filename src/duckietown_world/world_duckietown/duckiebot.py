from duckietown_world import PlacedObject


class Duckiebot(PlacedObject):

    def __init__(self, width, length, height, *args, **kwargs):
        PlacedObject.__init__(self, *args, **kwargs)

        self.width = width
        self.height = height
        self.length = length

    def draw_svg(self, drawing, g):
        L, W = self.length, self.width
        rect = drawing.rect(insert=(-L * 0.5, -W * 0.5),
                            size=(L, W),
                            fill="red",
                            style='opacity:0.4',
                            stroke_width="0.01",
                            stroke="black", )
        rect.width = "0.1em"
        g.add(rect)

        draw_axes(drawing, g)
        # print(g.tostring())


def draw_axes(drawing, g, L=0.1):
    g2 = drawing.g()
    g2.attribs['class'] = 'axes'
    line = drawing.line(start=(0, 0),
                        end=(L, 0),
                        stroke_width="0.01",
                        stroke="red")
    g2.add(line)

    line = drawing.line(start=(0, 0),
                        end=(0, L),
                        stroke_width="0.01",
                        stroke="green")
    g2.add(line)

    g.add(g2)
