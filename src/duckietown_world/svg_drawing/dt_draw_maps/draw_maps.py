# coding=utf-8
import argparse
import json
import os
import sys
from typing import cast, TYPE_CHECKING

import coloredlogs
from zuper_commons.fs import AbsDirPath, FilePath

from duckietown_world.utils import save_rgb_to_jpg

if TYPE_CHECKING:
    from duckietown_world.world_duckietown import DuckietownMap

from duckietown_world import logger
from ..misc import draw_static

__all__ = ["draw_maps_main"]


def draw_maps_main(args=None):
    coloredlogs.install(level="DEBUG")
    from duckietown_world.world_duckietown import load_map
    from duckietown_world.resources import list_maps

    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output dir", default="out-draw_maps")
    parser.add_argument("map_names", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(args)

    output = parsed.output

    if parsed.map_names:
        map_names = parsed.map_names
    else:
        map_names = list_maps()
    logger.info("Drawing the maps %s." % ", ".join(map_names))

    from duckietown_world.gltf import export_gltf

    for map_name in map_names:
        duckietown_map = load_map(map_name)
        out = os.path.join(output, map_name)

        draw_map(out, duckietown_map)
        style = "synthetic-F"
        export_gltf(duckietown_map, out)
        draw_map_gymd(map_name, out, style)

        y = duckietown_map.as_json_dict()
        fn = os.path.join(out, "map.json")
        with open(fn, "w") as f:
            f.write(json.dumps(y, indent=4))
        # print('written to %s' % fn)


def draw_map(output: str, duckietown_map: "DuckietownMap") -> None:
    from duckietown_world.world_duckietown import DuckietownMap

    if not os.path.exists(output):
        os.makedirs(output)
    assert isinstance(duckietown_map, DuckietownMap)

    fns = draw_static(duckietown_map, output_dir=output, pixel_size=(640, 640), area=None)
    for fn in fns:
        logger.info(f"Written to {fn}")


def draw_map_gymd(map_name: str, output: AbsDirPath, style: str):
    try:
        from gym_duckietown.simulator import Simulator
    except ImportError:
        return

    sim = Simulator(
        map_name,
        enable_leds=True,
        domain_rand=False,
        num_tris_distractors=0,
        camera_width=640,
        camera_height=480,
        # distortion=True,
        color_ground=[0, 0.3, 0],  # green
        style=style,
    )

    sim.reset()

    logger.info("rendering obs")
    img = sim.render_obs()

    out = os.path.join(output, "cam.jpg")
    save_rgb_to_jpg(img, out)

    sim.cur_pos = [-100.0, -100.0, -100.0]
    from gym_duckietown.simulator import FrameBufferMemory

    td = FrameBufferMemory(width=1024, height=1024)
    # noinspection PyProtectedMember
    horiz = sim._render_img(
        width=td.width,
        height=td.height,
        multi_fbo=td.multi_fbo,
        final_fbo=td.final_fbo,
        img_array=td.img_array,
        top_down=True,
    )
    # img = sim.render("top_down")
    out = cast(FilePath, os.path.join(output, "top_down.jpg"))
    save_rgb_to_jpg(horiz, out)


if __name__ == "__main__":
    draw_maps_main()
