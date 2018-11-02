# coding=utf-8
from duckietown_world.geo import PlacedObject
from duckietown_world.svg_drawing import draw_axes
from .other_objects import Vehicle

__all__ = [
    'Duckiebot',
    'DB18',
]


class Duckiebot(Vehicle):

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
                            # style='opacity:0.4',
                            stroke_width="0.01",
                            stroke="black", )
        rect.width = "0.1em"
        g.add(rect)

        draw_axes(drawing, g)
        # print(g.tostring())


class DB18(Duckiebot):
    def __init__(self, *args, **kwargs):
        width = 0.13 + 0.02
        length = 0.18
        height = 0.12
        Duckiebot.__init__(self, width=width, length=length, height=height, *args, **kwargs)
