# coding=utf-8
from svgwrite.container import Use

from duckietown_world.geo import PlacedObject
from duckietown_world.svg_drawing import draw_axes
from .other_objects import Vehicle

__all__ = ["Duckiebot", "DB18"]


class Duckiebot(Vehicle):
    def __init__(self, width, length, height, *args, color: str = None, **kwargs):
        # noinspection PyArgumentList
        PlacedObject.__init__(self, *args, **kwargs)

        self.width = width
        self.height = height
        self.length = length
        self.color = color or "red"

    def draw_svg(self, drawing, g):
        ID = str(id(self))
        for element in drawing.defs.elements:
            if element.attribs.get("id", None) == ID:
                break
        else:
            template = drawing.g(id=ID)

            L, W = self.length, self.width
            W = 0.11
            Lb = 0.11
            Lf = 0.08
            d = 0.10
            d = 0.12
            wheel_width = 0.027
            wheel_radius = 0.032
            rx_robot = 0.02
            rx_wheel = 0.005
            fancy = True

            if fancy:

                rect = drawing.rect(
                    insert=(-wheel_radius, -d * 0.5 - wheel_width / 2),
                    size=(+wheel_radius * 2, wheel_width),
                    fill="black",
                    stroke_width="0.01",
                    stroke="black",
                    rx=rx_wheel,
                )
                rect.width = "0.1em"
                template.add(rect)
                rect = drawing.rect(
                    insert=(-wheel_radius, +d * 0.5 - wheel_width / 2),
                    size=(+wheel_radius * 2, wheel_width),
                    fill="black",
                    stroke_width="0.01",
                    stroke="black",
                    rx=rx_wheel,
                )
                rect.width = "0.1em"
                template.add(rect)

                rect = drawing.rect(
                    insert=(-Lb, -W * 0.5),
                    size=(Lb + Lf, W),
                    fill=self.color,
                    stroke_width="0.01",
                    stroke="black",
                    rx=rx_robot,
                )
                rect.width = "0.1em"
                template.add(rect)
            else:

                rect = drawing.rect(
                    insert=(-Lb, -W * 0.5),
                    size=(Lb + Lf, W),
                    fill=self.color,
                    # style='opacity:0.4',
                    stroke_width="0.01",
                    stroke="black",
                    # rx=rx_robot,
                )
                template.add(rect)

            drawing.defs.add(template)

        use = Use(f"#{ID}")
        g.add(use)

        draw_axes(drawing, g)
        # print(g.tostring())


class DB18(Duckiebot):
    def __init__(self, *args, **kwargs):
        width = 0.13  # + 0.02
        length = 0.18
        height = 0.12
        Duckiebot.__init__(self, width=width, length=length, height=height, *args, **kwargs)
