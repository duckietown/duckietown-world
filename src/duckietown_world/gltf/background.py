import os
from typing import List

import numpy as np
from geometry import rotx, roty, rotz, SE3_from_SO3, SE3value
from geometry.poses import pose_from_rotation_translation

from gltflib import GLTFModel, Node


def add_background(model: GLTFModel, resources: List) -> int:
    root = "resources/banners/pannello %02d.pdf.jpg"
    found = []
    for i in range(1, 27):
        fn = root % i
        if os.path.exists(fn):
            found.append(fn)
        else:
            print(f"not found {fn}")

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
            model,
            resources,
            doubleSided=True,
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            fn_emissive=fn
            # fn=fn, fn_normals=None
        )
        print(fn, material_index)

        mi = get_square()

        mesh_index = add_polygon(
            model,
            resources,
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

        node1_index = add_node(model, Node(name=f"panel-{i}", mesh=mesh_index, matrix=gm(matrix)))
        nodes_panels.append(node1_index)
    node = Node(children=nodes_panels)
    return add_node(model, node)


def SE3_rotz(alpha: float) -> SE3value:
    return SE3_from_SO3(rotz(alpha))


def SE3_roty(alpha: float) -> SE3value:
    return SE3_from_SO3(roty(alpha))


def SE3_rotx(alpha: float) -> SE3value:
    return SE3_from_SO3(rotx(alpha))


def SE3_trans(t: np.ndarray) -> SE3value:
    return pose_from_rotation_translation(np.eye(3), np.array(t))
