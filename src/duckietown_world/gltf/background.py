import numpy as np
from geometry import SE3_roty, SE3_rotz, SE3_trans

from gltflib import Node

__all__ = ["add_background"]

from gltflib import GLTF

from . import logger
from ..world_duckietown.map_loading import get_resource_path


def add_background(gltf: GLTF) -> int:
    model = gltf.model
    resources = gltf.resources
    root = "pannello %02d.pdf.jpg"
    found = []
    for i in range(1, 27):
        basename = root % i
        try:
            fn = get_resource_path(basename)
        except KeyError:
            logger.warn(f"not found {basename!r}")
        else:
            found.append(fn)

    found = found[:9]
    n = len(found)
    if n == 0:
        raise ValueError(root)
    from .export import make_material
    from .export import get_square
    from .export import add_polygon
    from .export import add_node
    from .export import gm

    dist = 30
    fov_y = np.deg2rad(45)
    nodes_panels = []
    for i, fn in enumerate(found):
        # if i > 5:
        #     break
        material_index = make_material(
            gltf,
            doubleSided=True,
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            fn_emissive=fn
            # fn=fn, fn_normals=None
        )
        print(fn, material_index)

        mi = get_square()

        mesh_index = add_polygon(
            gltf,
            f"bg-{i}",
            vertices=mi.vertices,
            texture=mi.textures,
            colors=mi.color,
            normals=mi.normals,
            material=material_index,
        )
        v = (np.pi * 2) / n
        theta = v * i

        a = 2 * np.tan((2 * np.pi / n) / 2) * dist
        matrix = (
            SE3_rotz(theta)
            @ SE3_trans(np.array((dist, 0, 0)))
            @ SE3_roty(np.pi / 2)
            @ SE3_rotz(np.pi / 2)
            @ np.diag(np.array((a, a, 1, 1)))
            @ SE3_trans(np.array([0, 0.35, 0]))
        )

        node1_index = add_node(gltf, Node(name=f"panel-{i}", mesh=mesh_index, matrix=gm(matrix)))
        nodes_panels.append(node1_index)
    node = Node(children=nodes_panels)
    return add_node(gltf, node)
