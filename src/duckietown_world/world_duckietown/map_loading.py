# coding=utf-8
import itertools
import os

import oyaml as yaml

from duckietown_world import Scale2D, logger, SE2Transform
from duckietown_world.seqs import Constant
from .duckiebot import DB18
from .duckietown_map import DuckietownMap
from .other_objects import Duckie, SignLeftTIntersect, SignRightTIntersect, \
    SignTIntersect, SignStop, Tree, House, Bus, Truck, Cone, Barrier, Building, Sign4WayIntersect, SingTLightAhead, \
    GenericObject
from .tile import Tile
from .tile_map import TileMap
from .tile_template import load_tile_types
from .traffic_light import TrafficLight

__all__ = [
    'create_map',
    'list_maps',
    'construct_map',
    'load_map',
]


def create_map(H=3, W=3):
    tile_map = TileMap(H=H, W=W)

    for i, j in itertools.product(range(H), range(W)):
        tile = Tile(kind='mykind', drivable=True)
        tile_map.add_tile(i, j, 'N', tile)

    return tile_map


def list_maps():
    maps_dir = get_maps_dir()

    def f():
        for map_file in os.listdir(maps_dir):
            map_name = map_file.split('.')[0]
            yield map_name

    return list(f())


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
    d2 = os.path.join(module_dir, '../data/gd1/meshes')
    assert os.path.exists(d2), d2
    return [d, d2]


def load_map(map_name):
    logger.info('loading map %s' % map_name)
    maps_dir = get_maps_dir()
    fn = os.path.join(maps_dir, map_name + '.yaml')
    if not os.path.exists(fn):
        msg = 'Could not find file %s' % fn
        raise ValueError(msg)
    data = open(fn).read()
    yaml_data = yaml.load(data)
    # from gym_duckietown.simulator import ROAD_TILE_SIZE
    tile_size = 0.61  # XXX
    return construct_map(yaml_data, tile_size)


def construct_map(yaml_data, tile_size):
    tiles = interpret_map_data(yaml_data)
    tilemap0 = DuckietownMap(tile_size)
    tilemap0.set_object('tilemap', tiles, ground_truth=Scale2D(tile_size))
    return tilemap0


def interpret_map_data(data):
    tiles = data['tiles']
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    A = len(tiles)
    B = len(tiles[0])
    tm = TileMap(H=B, W=A)

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

            tile = Tile(kind=kind, drivable=drivable)
            if kind in templates:
                tile.set_object(kind, templates[kind],
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

        rotate = desc['rotate']
        transform = SE2Transform([float(pos[0]), float(tm.W - pos[1])], rotate)

        # x, z = pos[0:2]
        #
        # i = int(np.floor(x))
        # j = int(np.floor(z))
        # dx = x - i
        # dy = z - j
        #
        # z = pos[2] if len(pos) == 3 else 0.0

        # optional = desc.get('optional', False)
        # height = desc.get('height', None)

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
            kind2klass = {
                'duckie': Duckie,
                'cone': Cone,
                'barrier': Barrier,
                'building': Building,
                'duckiebot': DB18,
                'sign_left_T_intersect': SignLeftTIntersect,
                'sign_right_T_intersect': SignRightTIntersect,
                'sign_T_intersect': SignTIntersect,
                'sign_4_way_intersect': Sign4WayIntersect,
                'sign_t_light_ahead': SingTLightAhead,
                'sign_stop': SignStop,
                'tree': Tree,
                'house': House,
                'bus': Bus,
                'truck': Truck,
            }
            if kind in kind2klass:
                klass = kind2klass[kind]
                obj = klass()
            else:
                logger.debug('Do not know special kind %s' % kind)
                obj = GenericObject(kind=kind)
        obj_name = 'ob%02d-%s' % (obj_idx, kind)
        # tile = tm[(i, j)]
        # transform = TileRelativeTransform([dx, dy], z, rotate)

        tm.set_object(obj_name, obj, ground_truth=transform)

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
    res = []
    for d in get_texture_dirs():
        suffixes = ['', '_1', '_2', '_3', '_4']
        for s in suffixes:
            for ext in ['jpg', 'png']:
                path = os.path.join(d, tex_name + s + '.' + ext)
                if os.path.exists(path):
                    res.append(path)

    if not res:
        msg = 'Could not find any texture for %s' % tex_name
        raise KeyError(msg)
    return res[0]
