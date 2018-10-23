from duckietown_world import PlacedObject, SE2Transform


class DuckietownMap(PlacedObject):
    def __init__(self, tile_size, *args, **kwargs):
        self.tile_size = tile_size
        PlacedObject.__init__(self, *args, **kwargs)

    def se2_from_curpos(self, cur_pos, cur_angle):
        H = self.children['tilemap'].H
        gx,gy,gz = cur_pos
        p = [gx, (H - 1) * self.tile_size - gz]
        transform = SE2Transform(p, cur_angle)
        return transform
