# coding=utf-8

from duckietown_world import logger
from duckietown_world.svg_drawing.misc import mime_from_fn, draw_axes

from ..geo import PlacedObject

__all__ = [
    'GenericObject',
    'Duckie',
    'Decoration',
    'Sign',
    'Sign4WayIntersect',
    'SignLeftTIntersect',
    'SignRightTIntersect',
    'SignStop',
    'SignTIntersect',
    'SignTLightAhead',
    'SignOneWayRight',
    'SignOneWayLeft',
    'SignDuckCrossing',
    'SignYield',
    'SignNoLeftTurn',
    'SignNoRightTurn',
    'SignDoNotEnter',
    'SignParking',
    'SignPedestrian',
    'Bus',
    'House',
    'Tree',
    'Truck',
    'Cone',
    'Vehicle',
    'Barrier',
    'Building',
    'SIGNS',
    'SIGNS_ALIASES',
    'get_canonical_sign_name',
]


class GenericObject(PlacedObject):
    def __init__(self, kind, **kwargs):
        self.kind = kind
        PlacedObject.__init__(self, **kwargs)

    def params_to_json_dict(self):
        return dict(kind=self.kind)

    def draw_svg(self, drawing, g):
        c = drawing.circle(center=(0, 0), r=0.1)
        g.add(c)


class Duckie(PlacedObject):
    def draw_svg(self, drawing, g):
        c = drawing.circle(center=(0, 0), r=0.065,
                           fill='yellow', stroke='black', stroke_width=0.01)
        g.add(c)


class Decoration(PlacedObject):
    pass


class Tree(Decoration):
    def draw_svg(self, drawing, g):
        c = drawing.circle(center=(0, 0), r=0.25,
                           fill='green', stroke='black', stroke_width=0.01)
        g.add(c)


class Vehicle(PlacedObject):
    pass


class Cone(PlacedObject):
    def draw_svg(self, drawing, g):
        c = drawing.circle(center=(0, 0), r=0.03,
                           fill='orange', stroke='black', stroke_width=0.01)
        g.add(c)


class Bus(Vehicle):
    def draw_svg(self, drawing, g):
        L, W = 0.3, 0.2
        c = drawing.rect(insert=(-L / 2, -W / 2),
                         fill="grey",
                         size=(L, W,),
                         stroke_width="0.01",
                         stroke="#eeeeee", )
        g.add(c)


class Truck(Vehicle):
    def draw_svg(self, drawing, g):
        L, W = 0.4, 0.2
        c = drawing.rect(insert=(-L / 2, -W / 2),
                         fill="blue",
                         size=(L, W,),
                         stroke_width="0.01",
                         stroke="#eeeeee", )
        g.add(c)


class House(Decoration):
    def draw_svg(self, drawing, g):
        L, W = 0.5, 0.25
        c = drawing.rect(insert=(-L / 2, -W / 2),
                         fill="red",
                         size=(L, W,),
                         stroke_width="0.01",
                         stroke="#eeeeee", )
        g.add(c)


class Barrier(Decoration):
    def draw_svg(self, drawing, g):
        L, W = 0.5, 0.25
        c = drawing.rect(insert=(-L / 2, -W / 2),
                         fill="pink",
                         size=(L, W,),
                         stroke_width="0.01",
                         stroke="#eeeeee", )
        g.add(c)


class Building(Decoration):
    # TODO
    def draw_svg(self, drawing, g):
        L, W = 0.5, 0.25
        c = drawing.rect(insert=(-L / 2, -W / 2),
                         fill="red",
                         size=(L, W,),
                         stroke_width="0.01",
                         stroke="#eeeeee", )
        g.add(c)


class Sign(PlacedObject):

    def __init__(self, tag=None, **kwargs):
        PlacedObject.__init__(self, **kwargs)
        self.tag = tag

    def draw_svg(self, drawing, g):
        texture = self.get_name_texture()
        # x = -0.2
        CM = 0.01
        PAPER_WIDTH, PAPER_HEIGHT = 8.5 * CM, 15.5 * CM
        PAPER_THICK = 0.01

        BASE_SIGN = 5 * CM
        WIDTH_SIGN = 1.1 * CM

        y = -1.5 * PAPER_HEIGHT  # XXX not sure why this is negative
        y = BASE_SIGN
        x = -PAPER_WIDTH/2
        try:
            from .map_loading import get_texture_file
            fn = get_texture_file(texture)
        except KeyError:
            msg = 'Cannot find texture for %s' % texture
            logger.warning(msg)

            c = drawing.rect(insert=(x, y),
                             fill="white",
                             size=(PAPER_WIDTH, PAPER_HEIGHT,),
                             stroke_width="0.01",
                             stroke="black", )
            g.add(c)

        else:
            texture = open(fn, 'rb').read()
            from duckietown_world.world_duckietown.tile import data_encoded_for_src

            href = data_encoded_for_src(texture, mime_from_fn(fn))

            img = drawing.image(href=href,
                                size=(PAPER_WIDTH, PAPER_HEIGHT),
                                insert=(x, y),

                                stroke_width=0.001,
                                stroke='black',
                                style='transform: rotate(90deg) rotate(90deg) scaleX(-1)  rotate(-90deg);'
                                      'border: solid 1px black '
                                )
            img.attribs['class'] = 'sign-paper'
            g.add(img)

        c = drawing.rect(insert=(-BASE_SIGN / 2, -BASE_SIGN / 2),
                         fill="lightgreen",

                         stroke_width=0.001,
                         stroke='black',
                         size=(BASE_SIGN, BASE_SIGN,))
        g.add(c)
        draw_axes(drawing, g)
        alpha = 1.2
        c = drawing.rect(insert=(-WIDTH_SIGN / 2, -BASE_SIGN / 2 * alpha),
                         fill="#FFCB9C",
                         stroke_width=0.001,
                         stroke='black',
                         size=(WIDTH_SIGN, BASE_SIGN * alpha,))
        g.add(c)

        c = drawing.rect(insert=(+0.01 + -PAPER_THICK / 2, -PAPER_WIDTH / 2),
                         fill="white",
                         stroke_width=0.001,
                         stroke='black',
                         size=(PAPER_THICK, PAPER_WIDTH,))
        g.add(c)

    def get_name_texture(self):
        for k, v in SIGNS.items():
            if v.__name__ == type(self).__name__:
                return k

        raise KeyError(type(self).__name__)


class SignStop(Sign):
    pass


class SignLeftTIntersect(Sign):
    pass


class SignRightTIntersect(Sign):
    pass


class SignTIntersect(Sign):
    pass


class Sign4WayIntersect(Sign):
    pass


class SignTLightAhead(Sign):
    pass


class SignOneWayRight(Sign): pass


class SignOneWayLeft(Sign): pass


class SignDuckCrossing(Sign): pass


class SignYield(Sign): pass


class SignNoLeftTurn(Sign): pass


class SignNoRightTurn(Sign): pass


class SignDoNotEnter(Sign): pass


class SignParking(Sign): pass


class SignPedestrian(Sign): pass


SIGNS = {
    'sign_left_T_intersect': SignLeftTIntersect,
    'sign_right_T_intersect': SignRightTIntersect,
    'sign_T_intersect': SignTIntersect,
    'sign_4_way_intersect': Sign4WayIntersect,
    'sign_t_light_ahead': SignTLightAhead,
    'sign_stop': SignStop,
    'sign_1_way_right': SignOneWayRight,
    'sign_1_way_left': SignOneWayLeft,
    'sign_duck_crossing': SignDuckCrossing,
    'sign_yield': SignYield,
    'sign_parking': SignParking,
    'sign_pedestrian': SignPedestrian,
    'sign_do_not_enter': SignDoNotEnter,
    'sign_no_left_turn': SignNoLeftTurn,
    'sign_no_right_turn': SignNoRightTurn,

}

SIGNS_ALIASES = {
    'T-intersection': 'sign_T_intersect',
    'oneway-right': 'sign_1_way_right',
    'oneway-left': 'sign_1_way_left',
    'duck-crossing': 'sign_duck_crossing',
    'stop': 'sign_stop',
    'yield': 'sign_yield',
    'no-left-turn': 'sign_no_left_turn',
    't-light-ahead': 'sign_t_light_ahead',
    'pedestrian': 'sign_pedestrian',
    'no-right-turn': 'sign_no_right_turn',
    'parking': 'sign_parking',
    'right-T-intersect': 'sign_right_T_intersect',
    'left-T-intersect': 'sign_left_T_intersect',
    '4-way-intersect': 'sign_4_way_intersect',
    'do-not-enter': 'sign_do_not_enter',
}


def get_canonical_sign_name(sign_name):
    if sign_name in SIGNS:
        return sign_name
    if sign_name in SIGNS_ALIASES:
        return SIGNS_ALIASES[sign_name]
    raise KeyError(sign_name)
