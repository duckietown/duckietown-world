# coding=utf-8
import itertools

import numpy as np

from duckietown_world import Scale2D, logger, SE2Transform
from duckietown_world.seqs import Constant
from duckietown_world.world_duckietown import TileMap, Tile, TrafficLight, GenericObject
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap
from duckietown_world.world_duckietown.tile_template import load_tile_types

import oyaml as yaml
import os


__all__ = ['create_map', 'list_gym_maps']


def create_map(H=3, W=3):
    tile_map = TileMap(H=H, W=W)

    for i, j in itertools.product(range(H), range(W)):
        tile = Tile(kind='mykind', drivable=True)
        tile_map.add_tile(i, j, 'N', tile)

    return tile_map


def list_gym_maps():
    maps_dir = get_maps_dir()
    for map_file in os.listdir(maps_dir):
        map_name = map_file.split('.')[0]
        yield map_name



def get_maps_dir():
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    d = os.path.join(module_dir, '../data/gd1/maps')
    assert os.path.exists(d), d
    return d

def get_texture_dirs():
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    d = os.path.join(module_dir, '../data/gd1/textures')
    assert os.path.exists(d), d
    return d


def load_gym_map(map_name):
    logger.info('loading map %s' % map_name)
    maps_dir = get_maps_dir()
    fn = os.path.join(maps_dir, map_name + '.yaml')
    if not os.path.exists(fn):
        msg = 'Could not find file %s' % fn
        raise ValueError(msg)
    data = open(fn).read()
    yaml_data = yaml.load(data)
    # from gym_duckietown.simulator import ROAD_TILE_SIZE
    tile_size = 0.61 # XXX
    return construct_map(yaml_data, tile_size)


def construct_map(yaml_data, tile_size):
    tiles = interpret_gym_map(yaml_data)
    tilemap0 = DuckietownMap(tile_size)
    tilemap0.set_object('tilemap', tiles, ground_truth=Scale2D(tile_size))
    return tilemap0


def interpret_gym_map(data):
    tiles = data['tiles']
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    A = len(tiles)
    B = len(tiles[0])
    tm = TileMap(H=B, W=A)

    # print(tiles)
    # For each row in the grid

    templates = load_tile_types()
    for a, row in enumerate(tiles):
        if len(row) != B:
            msg = "each row of tiles must have the same length"
            raise ValueError(msg)

        # For each tile in this row
        for b, tile in enumerate(row):
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
            if kind in templates:
                tile.set_object('template', templates[kind],
                                ground_truth=SE2Transform.identity())
            else:
                pass
                # msg = 'Could not find %r in %s' % (kind, templates)
                # logger.debug(msg)

            tm.add_tile(b, (A - 1) - a, orient, tile)

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
            logger.debug('Do not know special kind %s' % kind)
            obj = GenericObject(kind=kind)

        # tile = tm[(i, j)]
        # transform = TileRelativeTransform([dx, dy], z, rotate)
        transform = SE2Transform([float(pos[0]), float(tm.W - pos[1])], rotate)
        tm.set_object('obj%s' % obj_idx, obj, ground_truth=transform)
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


def get_texture_file(tex_name):
    d = get_texture_dirs()
    suffixes = ['', '_1', '_2', '_3', '_4']
    for s in suffixes:
        path = os.path.join(d, tex_name + s + '.png')
        if os.path.exists(path):
            return path
    msg = 'Could not find any texture for %s' % tex_name
    raise KeyError(msg)
