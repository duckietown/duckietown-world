import csv

from duckietown_world import SE2Transform
from duckietown_world.world_duckietown.tags_db import get_sign_type_from_tag_id

in_file = open("autolab_tiles_map.csv", "rb")
reader = csv.reader(in_file)

data = list(reader)[1:]

map_data = {}
map_data['version'] = 2

H = W = 0
for entry in data:
    xs, ys, kind, rotation = entry
    x = int(xs)
    y = int(ys)
    H = max(H, y + 1)
    W = max(W, x + 1)

tiles = [[None for _ in range(W)] for __ in range(H)]

for entry in data:
    xs, ys, kind, rotation = entry
    x = int(xs)
    y = int(ys)
    rotation = int(rotation)
    direction = {
        90: 'N',
        180: 'W',
        270: 'S',
        0: 'E',
    }
    translate = {
        'empty': 'asphalt',
        'turn': 'curve_left',
        '3way': '3way_left',
    }
    kind = translate.get(kind, kind)
    k = kind + '/' + direction[rotation]
    tiles[H - 1 - y][x] = k
map_data['tiles'] = tiles
#
# tiles2 = []
# for row in tiles:
#     row_string = '  '.join('%10s' % _ for _ in row)
#     tiles2.append(row_string)
#
#
#
# map_data['tiles-string'] = "\n".join(tiles2)

import yaml

in_file = open("autolab_tags_map.csv", "rb")
reader = csv.reader(in_file)
data = list(reader)[1:]

objects = map_data['objects'] = {}
import numpy as np

APRIL_TAG_SIZE = 0.08

for i, entry in enumerate(data):
    tag_ID, x, y, slot, rotation = entry
    tag_ID = int(tag_ID)

    kind = get_sign_type_from_tag_id(tag_ID)
    x = int(x)
    y = int(y)
    slot = int(slot)

    rotation = rotation or 0
    rotation = int(rotation)
    # bug in the map file
    # rotation = rotation + 90
    sign = dict(kind=kind,
                tag={'~TagInstance': dict(tag_id=int(tag_ID), family='36h11', size=APRIL_TAG_SIZE)},
                attach=dict(tile=[x, y], slot=slot)
                )

    sign['pose'] = SE2Transform([0, 0], np.deg2rad(rotation)).as_json_dict()

    objects['tag%02d' % i] = sign

in_file = open("AprilTag position duckietown.csv", "rb")
reader = csv.reader(in_file)
data = list(reader)[1:]
CM = 0.01
origin = -5 * CM, -15.5 * CM
# Tag ID,Quadrant,x location,Y-location,Roation

# FAMILY = 'default'
# SIZE = 0.035
for entry in data:
    tag_ID, quadrant, x, y, rotation = entry
    x = float(x) + origin[0]
    y = float(y) + origin[1]

    rotation = rotation or 0
    rotation = int(rotation)

    sign = dict(
            kind='floor_tag',
            tag={'~TagInstance': dict(tag_id=int(tag_ID), family='36h11', size=APRIL_TAG_SIZE)},
    )
    sign['pose'] = SE2Transform([x, y], np.deg2rad(rotation)).as_json_dict()

    objects[quadrant] = sign

s = yaml.safe_dump(map_data, encoding='utf-8', indent=4, allow_unicode=True,
                   default_flow_style=False)
print(s)
