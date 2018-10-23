# coding=utf-8
import base64

from duckietown_world import logger
from duckietown_world.geo import PlacedObject

__all__ = [
    'Tile',
]
draw_directions_lanes = False


class Tile(PlacedObject):
    def __init__(self, kind, drivable, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.kind = kind
        self.drivable = drivable

        try:
            from duckietown_world.world_duckietown.map_loading import get_texture_file
            self.fn = get_texture_file(kind)

            self.texture = open(self.fn, 'rb').read()
        except KeyError:
            msg = 'Cannot find texture for %s' % kind
            logger.warning(msg)
            self.texture = None
            self.fn = None

    def params_to_json_dict(self):
        return dict(kind=self.kind, drivable=self.drivable)

    def draw_svg(self, drawing, g):
        # rect = drawing.rect(width=1,height=1,)
        # g.add(rect)
        rect = drawing.rect(insert=(-0.5, -0.5),
                            width=1,
                            height=1,
                            fill="#eee",
                            style='opacity:0.4',
                            stroke_width="0.01",
                            stroke="none", )
        # g.add(rect)

        if self.fn:
            href = data_encoded_for_src(self.texture, 'image/png')
            img = drawing.image(href=href,
                                size=(1, 1),
                                insert=(-0.5, -0.5),
                                style='transform: rotate(90deg) scaleX(-1)')

            g.add(img)

        if draw_directions_lanes:
            if self.kind != 'floor':
                start = (-0.5, -0.25)
                end = (+0, -0.25)
                line = drawing.line(start=start, end=end, stroke='blue', stroke_width='0.01')
                g.add(line)

        from duckietown_world.world_duckietown.duckiebot import draw_axes
        draw_axes(drawing, g)


def data_encoded_for_src(data, mime):
    """ data =
        ext = png, jpg, ...

        returns "data: ... " sttring
    """
    encoded = base64.b64encode(data)
    link = 'data:%s;base64,%s' % (mime, encoded)
    return link
