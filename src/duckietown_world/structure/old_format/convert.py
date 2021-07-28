import os
from typing import Optional

import numpy as np
import yaml

from duckietown_world.resources import get_data_dir
# from duckietown_world.structure.utils import get_canonical_sign_name
from duckietown_world.world_duckietown.other_objects import get_canonical_sign_name

CITIZENS = 'citizens'
DECORATIONS = 'decorations'
FRAMES = 'frames'
TAGS = 'groundtags'
TILE_MAP = 'tile_maps'
TILES = 'tiles'
VEHICLES = 'vehicles'
SIGNS = 'trafficsigns'

ROOT_OF_MAP = 'map_1'

LAYERS_NAME = [CITIZENS, DECORATIONS, FRAMES, TILES, TILE_MAP, TILES, TAGS, SIGNS]


def get_id_by_type(type_of_obj: str) -> Optional[int]:
    with open(f"{get_data_dir()}/apriltagsDB.yaml") as file:
        content = yaml.safe_load(file)
        for tag in content:
            if tag['traffic_sign_type'] == type_of_obj:
                return int(tag['tag_id'])
    return 0


def load() -> dict:
    layers: dict = {}
    for layer_name in LAYERS_NAME:
        with open(f"{get_data_dir()}/maps/empty/{layer_name}.yaml") as file:
            layers[layer_name] = yaml.safe_load(file)
            print(layers[layer_name])
            if layers[layer_name][layer_name] is None:
                layers[layer_name][layer_name] = {}
            print(layers[layer_name])
    return layers


def dump(new_format: dict):
    for layer_name in LAYERS_NAME:
        # new_format[layer_name].pop(layer_name)
        with open(f"{os.getcwd()}/output/{layer_name}.yaml", "w") as file:
            file.write(yaml.dump(new_format[layer_name], Dumper=yaml.Dumper))


def convert_new_format(map_data: str):
    old_format_data = yaml.load(map_data, Loader=yaml.Loader)
    nf = load()
    frames = nf[FRAMES][FRAMES]

    # tile maps layer
    tile_size = float(old_format_data['tile_size'])
    nf[TILE_MAP][TILE_MAP][ROOT_OF_MAP]["tile_size"]["x"] = tile_size
    nf[TILE_MAP][TILE_MAP][ROOT_OF_MAP]["tile_size"]["y"] = tile_size

    # tiles layer
    tiles = old_format_data["tiles"]
    new_tiles = nf[TILES][TILES]

    for i in range(len(tiles)):
        for j in range(len(tiles[0])):
            name_tile = tiles[i][j]
            new_name_of_tile = f"{ROOT_OF_MAP}/tile-{j}-{i}"
            if "/" in name_tile:
                type_tile, orientation = name_tile.split("/")
            else:
                type_tile, orientation = name_tile, "E"
            new_tiles[new_name_of_tile] = {
                "i": i,
                "j": j,
                "orientation": orientation,
                "type": type_tile
            }
            frames[new_name_of_tile] = {
                "relative_to": ROOT_OF_MAP,
                "pose": {
                    "x": i * tile_size,
                    "y": j * tile_size,
                    "z": 0,
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0
                }
            }

    # objects
    objects = old_format_data["objects"]
    for key in objects:
        obj: dict = objects[key]
        kind: str = obj["kind"]
        x, y = [tile_size * float(i) for i in obj["pos"]]
        rotate: float = float(obj["rotate"])
        new_name_obj = f"{ROOT_OF_MAP}/{key}"
        if rotate < 0:
            rotate += 360
        rotate = rotate % 360
        rotate = np.deg2rad(rotate)

        frames[new_name_obj] = {
            "relative_to": ROOT_OF_MAP,
            "pose": {
                "x": x,
                "y": y,
                "z": 0,
                "roll": 0,
                "pitch": 0,
                "yaw": float(rotate)
            }
        }

        if kind.startswith("sign"):
            type_sign = kind.split("_")[1]
            id = get_id_by_type(type_sign)
            nf[SIGNS][SIGNS][new_name_obj] = {
                "type": get_canonical_sign_name(kind),  # get_canonical_sign_name(type_sign),
                "id": id
            }
        elif kind == "duckie":
            nf[CITIZENS][CITIZENS][new_name_obj] = {
                "color": "yellow"
            }
        elif "duckiebot" in kind:
            nf[VEHICLES][VEHICLES][new_name_obj] = {
                "configuration": "DB19",
                "id": "",
                "color": "red"
            }
        else:  # decorations
            nf[DECORATIONS][DECORATIONS][new_name_obj] = {
                "type": kind,
                "colors": {
                    "trunk": "brown",
                    "foliage": "green"
                }
            }

    return nf


if __name__ == '__main__':
    with open("udem.yaml") as file:
        new_format_map = convert_new_format(file.read())
        dump(new_format_map)
