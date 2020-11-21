import argparse
import os

from zuper_commons.logs import setup_logging, ZLogger

from .tile import get_fancy_textures

logger = ZLogger(__name__)


def compile_textures_main(args=None):
    setup_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", help="Destination directory", default="out-textures")
    parser.add_argument(
        "--styles",
        default="all",
        help="Draw preview in various styles, comma separated. (needs gym duckietown)",
    )

    parsed = parser.parse_args(args=args)
    if parsed.styles == "all":
        styles = ["synthetic", "synthetic-F", "photos", "smooth"]
    else:
        styles = parsed.styles.split(",")

    tile_types = [
        "straight",
        "floor",
        "asphalt",
        "4way",
        "3way_left",
        "3way_right",
        "curve_left",
        "curve_right",
        "grass",
    ]
    for style in styles:
        for kind in tile_types:
            ft = get_fancy_textures(style, kind)
            out = os.path.join(parsed.output, style, kind)
            ft.write(out)
