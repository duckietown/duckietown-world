# coding=utf-8
from ..geo import PlacedObject

__all__ = ['GenericObject']


class GenericObject(PlacedObject):
    def __init__(self, kind, **kwargs):
        self.kind = kind
        PlacedObject.__init__(self, **kwargs)

    def params_to_json_dict(self):
        return dict(kind=self.kind)

    def draw_svg(self, drawing, g):
        c = drawing.circle(center=(0, 0), r=0.1)
        g.add(c)
