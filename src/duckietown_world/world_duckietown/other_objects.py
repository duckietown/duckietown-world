# coding=utf-8
from abc import abstractmethod

from duckietown_world import logger
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
    'SingTLightAhead',
    'Bus',
    'House',
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

    def draw_svg(self, drawing, g):
        texture = self.get_name_texture()
        L, W = 0.2, 0.5
        try:
            from .map_loading import get_texture_file
            fn = get_texture_file(texture)
        except KeyError:
            msg = 'Cannot find texture for %s' % texture
            logger.warning(msg)

            c = drawing.rect(insert=(-L / 2, -W / 2),
                             fill="white",
                             size=(L, W,),
                             stroke_width="0.01",
                             stroke="black", )
            g.add(c)

        else:
            texture = open(fn, 'rb').read()
            from duckietown_world.world_duckietown.tile import data_encoded_for_src
            href = data_encoded_for_src(texture, 'image/jpeg')

            img = drawing.image(href=href,
                                size=(L, W),
                                insert=(-L / 2, -W / 2),
                                style='transform: rotate(90deg) scaleX(-1)  rotate(-90deg) '
                                )
            g.add(img)

    @abstractmethod
    def get_name_texture(self):
        pass


class SignStop(Sign):
    def get_name_texture(self):
        return 'sign_stop'


class SignLeftTIntersect(Sign):
    def get_name_texture(self):
        return 'sign_left_T_intersect'


class SignRightTIntersect(Sign):
    def get_name_texture(self):
        return 'sign_right_T_intersect'


class SignTIntersect(Sign):
    def get_name_texture(self):
        return 'sign_T_intersect'


class Sign4WayIntersect(Sign):
    def get_name_texture(self):
        return 'sign_4_way_intersect'


class SingTLightAhead(Sign):

    def get_name_texture(self):
        return 'sign_t_light_ahead'
