# coding=utf-8
import base64
import logging

from six import BytesIO

from duckietown_world import logger
from duckietown_world.geo import PlacedObject

from duckietown_world.utils.memoizing import memoized_reset

__all__ = [
    'Tile',
]
draw_directions_lanes = False


@memoized_reset
def get_jpeg_bytes(fn):
    from PIL import Image
    pl = logging.getLogger('PIL')
    pl.setLevel(logging.ERROR)

    image = Image.open(fn).convert('RGB')

    out = BytesIO()
    image.save(out, format='jpeg')
    return out.getvalue()


class Tile(PlacedObject):
    def __init__(self, kind, drivable, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.kind = kind
        self.drivable = drivable

        from duckietown_world.world_duckietown.map_loading import get_texture_file

        try:
            self.fn = get_texture_file(kind)
        except KeyError:
            msg = 'Cannot find texture for %s' % kind
            logger.warning(msg)

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
            texture = get_jpeg_bytes(self.fn)
            href = data_encoded_for_src(texture, 'image/jpeg')
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

        from duckietown_world.svg_drawing.misc import draw_children
        draw_children(drawing, self, g)


def data_encoded_for_src(data, mime):
    """ data =
        ext = png, jpg, ...

        returns "data: ... " sttring
    """
    encoded = base64.b64encode(data).decode()
    link = 'data:%s;base64,%s' % (mime, encoded)
    return link
