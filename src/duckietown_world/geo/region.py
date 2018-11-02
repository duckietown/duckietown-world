# coding=utf-8
__all__ = [
    'Region',
    'EmptyRegion',
]


class Region(object):

    def contains(self, p):
        """ Return true if it contains the 2D point p. """


class EmptyRegion(Region):

    def contains(self, p):
        return False
