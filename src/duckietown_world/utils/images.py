import numpy as np
from PIL import Image
from zuper_commons.fs import FilePath, make_sure_dir_exists

from . import logger

__all__ = ["save_rgb_to_png", "save_rgb_to_jpg"]


def save_rgb_to_png(img: np.ndarray, out: FilePath):
    make_sure_dir_exists(out)
    image = Image.fromarray(img)
    image.save(out, format="png")
    # logger.info(f"written {out}")


def save_rgb_to_jpg(img: np.ndarray, out: FilePath):
    make_sure_dir_exists(out)
    image = Image.fromarray(img)
    image = image.convert("RGB")
    image.save(out, format="jpeg")
    logger.info(f"written {out}")
