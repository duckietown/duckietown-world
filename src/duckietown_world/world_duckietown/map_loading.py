# coding=utf-8
import itertools
import numpy as np
import yaml

from duckietown_world.seqs import Constant
from duckietown_world.world_duckietown import TileMap, Tile, TileRelativeTransform, TrafficLight, GenericObject
from gym_duckietown import get_subdir_path
from gym_duckietown.simulator import ROAD_TILE_SIZE

__all__ = ['create_map', 'load_gym_maps']

import os


def create_map(H=3, W=3):
    tile_map = TileMap(H=H, W=W, tile_size=0.4)

    for i, j in itertools.product(range(H), range(W)):
        tile = Tile(kind='mykind', drivable=True)
        tile_map.add_tile(i, j, 'N', tile)

    return tile_map


def load_gym_maps():
    maps_dir = get_subdir_path('maps')
    res = {}
    for map_file in os.listdir(maps_dir):
        map_name = map_file.split('.')[0]
        data = open(os.path.join(maps_dir, map_file)).read()
        res[map_name] = interpret_gym_map(yaml.load(data))

    return res


def interpret_gym_map(data):
    tiles = data['tiles']
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    H = len(tiles)
    W = len(tiles[0])
    tm = TileMap(H=H, W=W, tile_size=ROAD_TILE_SIZE)

    # For each row in the grid
    for j, row in enumerate(tiles):
        if len(row) != W:
            msg = "each row of tiles must have the same length"
            raise ValueError(msg)

        # For each tile in this row
        for i, tile in enumerate(row):
            tile = tile.strip()

            if tile == 'empty':
                continue

            if '/' in tile:
                kind, orient = tile.split('/')
                kind = kind.strip()
                orient = orient.strip()
                # angle = ['S', 'E', 'N', 'W'].index(orient)
                drivable = True
            elif '4' in tile:
                kind = '4way'
                # angle = 2
                orient = 'N'
                drivable = True
            else:
                kind = tile
                # angle = 0
                orient = 'S'
                drivable = False

            # tile = {
            #     'coords': (i, j),
            #     'kind': kind,
            #     # 'angle': angle,
            #     'drivable': drivable
            # }

            tile = Tile(kind=kind, drivable=drivable)
            tm.add_tile(i, j, orient, tile)

            # if drivable:
            #     tile['curves'] = self._get_curve(i, j)
            # self.drivable_tiles.append(tile)
    #
    # self._load_objects(self.map_data)
    #

    # # Get the starting tile from the map, if specified
    # self.start_tile = None
    # if 'start_tile' in self.map_data:
    #     coords = self.map_data['start_tile']
    #     self.start_tile = self._get_tile(*coords)

    # # Arrays for checking collisions with N static objects
    # # (Dynamic objects done separately)
    # # (N x 2): Object position used in calculating reward
    # self.collidable_centers = []
    #
    # # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
    # self.collidable_corners = []
    #
    # # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
    # self.collidable_norms = []
    #
    # # (N): Safety radius for object used in calculating reward
    # self.collidable_safety_radii = []

    # For each object
    for obj_idx, desc in enumerate(data.get('objects', [])):
        kind = desc['kind']

        pos = desc['pos']
        x, z = pos[0:2]

        i = int(np.floor(x))
        j = int(np.floor(z))
        dx = x - i
        dy = z - j

        z = pos[2] if len(pos) == 3 else 0.0

        rotate = desc['rotate']
        optional = desc.get('optional', False)
        height = desc.get('height', None)

        # pos = ROAD_TILE_SIZE * np.array((x, y, z))

        # Load the mesh
        # mesh = ObjMesh.get(kind)

        # TODO
        # if 'height' in desc:
        #     scale = desc['height'] / mesh.max_coords[1]
        # else:
        #     scale = desc['scale']
        # assert not ('height' in desc and 'scale' in desc), "cannot specify both height and scale"

        # static = desc.get('static', True)

        # obj_desc = {
        #     'kind': kind,
        #     # 'mesh': mesh,
        #     'pos': pos,
        #     'scale': scale,
        #     'y_rot': rotate,
        #     'optional': optional,
        #     'static': static,
        # }

        if kind == "trafficlight":
            status = Constant("off")
            obj = TrafficLight(status)
        else:
            print('do not know kind %s' % kind)
            obj = GenericObject(kind=kind)

        tile = tm[(i, j)]
        transform = Constant(TileRelativeTransform([dx, dy], z, rotate))
        tile.set_object('obj%s' % obj_idx, obj, ground_truth=transform)
        # obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)

        # obj = None
        # if static:
        #     if kind == "trafficlight":
        #         obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
        #     else:
        #         obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
        # else:
        #     obj = DuckieObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, ROAD_TILE_SIZE)
        #
        # self.objects.append(obj)

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # Find drivable tiles object could intersect with

    return tm
