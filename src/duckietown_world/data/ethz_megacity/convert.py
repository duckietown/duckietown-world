import csv

in_file = open("autolab_tiles_map.csv", "rb")
reader = csv.reader(in_file)

data = list(reader)[1:]

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


map_data = dict(tiles=tiles)
import yaml


in_file = open("autolab_tags_map.csv", "rb")
reader = csv.reader(in_file)
data = list(reader)[1:]

signs = map_data['intersection_signs'] = []

for entry in data:
    tag_ID, x, y, position, rotation = entry
    tag_ID = int(tag_ID)
    x=int(x)
    y=int(y)
    position=int(position)
    rotation = int(rotation)
    sign = dict(tag_id=tag_ID,
                cell=[x,y],
                predefined_position=position,
                rotation_deg=rotation
                )
    signs.append(sign)


s = yaml.dump(map_data, default_flow_style=False)
print(s)
