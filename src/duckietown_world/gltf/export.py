import argparse
import json
import os
from dataclasses import dataclass, replace
from typing import cast, List, Optional, Tuple

__all__ = ["gltf_export_main", "export_gltf"]

import trimesh
from networkx import DiGraph, find_cycle, NetworkXNoCycle
from zuper_commons.types import ZValueError

from duckietown_world.gltf.background import add_background
from duckietown_world.world_duckietown.tile_map import ij_from_tilename
from zuper_commons.logs import ZLogger
from zuper_commons.text import get_md5

import struct
import operator

from gltflib import (
    Accessor,
    AccessorType,
    Camera,
    ComponentType,
    Image,
    Material,
    NormalTextureInfo,
    PBRMetallicRoughness,
    PerspectiveCameraInfo,
    Sampler,
    Texture,
    TextureInfo,
)


from duckietown_world.world_duckietown.map_loading import get_resource_path, get_texture_file

import numpy as np
from geometry import rotx, roty, rotz, SE3_from_SE2, SE3value
from geometry.poses import SE3_from_rotation_translation
from zuper_commons.fs import make_sure_dir_exists

from duckietown_world import (
    DB18,
    DuckietownMap,
    iterate_by_class,
    IterateByTestResult,
    load_map,
    PlacedObject,
    Sign,
    Tile,
)

from gltflib import (
    GLTF as GLTF0,
    GLTFModel,
    Asset,
    Scene,
    Node,
    Mesh,
    Primitive,
    Attributes,
    Buffer,
    BufferView,
    BufferTarget,
    FileResource,
)

logger = ZLogger(__name__)


class GLTF(GLTF0):
    def __init__(self):
        _resources = []
        _model = GLTFModel(
            asset=Asset(version="2.0"),
            scenes=[],
            nodes=[],
            buffers=[],
            bufferViews=[],
            cameras=[],
            images=[],
            materials=[],
            meshes=[],
            samplers=[Sampler()],
            skins=[],
            textures=[],
            extensionsRequired=[],
            extensionsUsed=[],
            accessors=[],
        )

        GLTF0.__init__(self, _model, _resources)
        self.md52PV = {}
        self.fn2node = {}

        self.fn2texture_index = {}

        self.cacheMaterial = {}


def gltf_export_main(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map", type=str,
    )

    parser.add_argument("--output", type=str)

    parsed = parser.parse_args(args=args)
    # ---

    dm: DuckietownMap = load_map(parsed.map)

    export_gltf(dm, parsed.output)


@dataclass
class PV:
    buffer_index: int
    bytelen: int
    buffer_view_index: int
    accessor_index: int


def pack_values(gltf: GLTF, uri: str, x: List[Tuple[float, ...]], btype) -> PV:
    md52PV = gltf.md52PV
    vertex_bytearray = bytearray()
    for vertex in x:
        for value in vertex:
            vertex_bytearray.extend(struct.pack("f", value))
    width = len(x[0])
    nelements = len(x)
    bytelen = len(vertex_bytearray)
    mins = [min([operator.itemgetter(i)(vertex) for vertex in x]) for i in range(width)]
    maxs = [max([operator.itemgetter(i)(vertex) for vertex in x]) for i in range(width)]

    md5 = get_md5(bytes(vertex_bytearray))
    if md5 not in md52PV:
        # noinspection PyTypeChecker

        fr = FileResource(uri, data=vertex_bytearray)
        gltf.resources.append(fr)
        buffer = Buffer(byteLength=bytelen, uri=uri)

        bi = add_buffer(gltf, buffer)
        bv = BufferView(buffer=bi, byteOffset=0, byteLength=bytelen, target=BufferTarget.ARRAY_BUFFER.value)
        buffer_view_index = add_buffer_view(gltf, bv)
        # noinspection PyTypeChecker
        accessor = Accessor(
            bufferView=buffer_view_index,
            byteOffset=0,
            componentType=ComponentType.FLOAT.value,
            count=nelements,
            type=btype,
            min=mins,
            max=maxs,
        )

        accessor_index = add_accessor(gltf, accessor)
        md52PV[md5] = PV(
            buffer_index=bi,
            accessor_index=accessor_index,
            buffer_view_index=buffer_view_index,
            bytelen=bytelen,
        )
    return md52PV[md5]


def add_polygon(
    gltf: GLTF,
    name: str,
    vertices: List[Tuple[float, float, float]],
    normals: List[Tuple[float, float, float]],
    texture: List[Tuple[float, float]],
    colors: List[Tuple[float, float, float]],
    material: int,
) -> int:
    pos = pack_values(gltf, f"{name}.position.bin", vertices, btype=AccessorType.VEC3.value)
    tex = pack_values(gltf, f"{name}.texture.bin", texture, btype=AccessorType.VEC2.value)
    col = pack_values(gltf, f"{name}.colors.bin", colors, btype=AccessorType.VEC3.value)
    normal = pack_values(gltf, f"{name}.normals.bin", normals, btype=AccessorType.VEC3.value)

    attributes = Attributes(
        POSITION=pos.accessor_index,
        COLOR_0=col.accessor_index,
        TEXCOORD_0=tex.accessor_index,
        NORMAL=normal.accessor_index,
    )
    primitives = []
    primitives.append(Primitive(attributes=attributes, material=material))
    mesh = Mesh(primitives=primitives)

    return add_mesh(gltf, mesh)


@dataclass
class MeshInfo:
    textures: List
    color: List
    normals: List
    vertices: List


def get_square() -> MeshInfo:
    #   - +            ++
    #   0 1  b          1 1  c
    #
    #   0 0            1 0
    #   - -  a          +-  d

    points = [
        (-0.5, -0.5, 0),
        (-0.5, 0.5, 0),
        (0.5, 0.5, 0),
        (0.5, -0.5, 0),
    ]
    tex = [
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    triangles = [0, 2, 1, 3, 2, 0]
    vertices = [points[_] for _ in triangles]
    textures = [tex[_] for _ in triangles]

    normals = [(0, 0, 1)] * 6
    color = [
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]

    return MeshInfo(textures=textures, color=color, normals=normals, vertices=vertices)


def export_gltf(dm: DuckietownMap, outdir: str, background: bool = True):
    gltf = GLTF()
    # setattr(gltf, 'md52PV', {})

    scene_index = add_scene(gltf, Scene(nodes=[]))

    map_nodes = []
    it: IterateByTestResult

    tiles = list(iterate_by_class(dm, Tile))
    if not tiles:
        raise ZValueError("no tiles?")
    for it in tiles:
        tile = cast(Tile, it.object)
        name = it.fqn[-1]
        fn = tile.fn
        fn_normal = get_resource_path("normalmap.png")
        # fn_normal = None
        material_index = make_material(
            gltf, doubleSided=False, baseColorFactor=[1, 1, 1, 1.0], fn=fn, fn_normals=fn_normal
        )
        mi = get_square()
        mesh_index = add_polygon(
            gltf,
            name + "-mesh",
            vertices=mi.vertices,
            texture=mi.textures,
            colors=mi.color,
            normals=mi.normals,
            material=material_index,
        )
        node1_index = add_node(gltf, Node(mesh=mesh_index))

        i, j = ij_from_tilename(name)
        c = (i + j) % 2
        color = [1, 0, 0, 1.0] if c else [0, 1, 0, 1.0]
        add_back = False
        if add_back:

            material_back = make_material(gltf, doubleSided=False, baseColorFactor=color)
            back_mesh_index = add_polygon(
                gltf,
                name + "-mesh",
                vertices=mi.vertices,
                texture=mi.textures,
                colors=mi.color,
                normals=mi.normals,
                material=material_back,
            )

            flip = np.diag([1.0, 1.0, -1.0, 1.0])
            flip[2, 3] = -0.01
            back_index = add_node(gltf, Node(mesh=back_mesh_index, matrix=gm(flip)))
        else:
            back_index = None

        tile_transform = it.transform_sequence
        tile_matrix2d = tile_transform.asmatrix2d().m
        s = dm.tile_size
        scale = np.diag([s, s, s, 1])
        tile_matrix = SE3_from_SE2(tile_matrix2d)
        tile_matrix = tile_matrix @ scale

        tile_matrix_float = list(tile_matrix.T.flatten())
        if back_index is not None:
            children = [node1_index, back_index]
        else:
            children = [node1_index]
        tile_node = Node(name=name, matrix=tile_matrix_float, children=children)
        tile_node_index = add_node(gltf, tile_node)

        map_nodes.append(tile_node_index)

    if background:
        bg_index = add_background(gltf)
        add_node_to_scene(gltf, scene_index, bg_index)

    exports = {
        "Sign": export_sign,
        # XXX: the tree model is crewed up
        # 'Tree': export_tree,
        "Tree": None,
        "Duckie": export_duckie,
        "DB18": export_DB18,
        # 'Duckie': None,
        "TileMap": None,
        "LaneSegment": None,
        "PlacedObject": None,
        "DuckietownMap": None,
        "Tile": None,
    }

    for it in iterate_by_class(dm, PlacedObject):
        ob = it.object

        K = type(ob).__name__
        if isinstance(ob, Sign):
            K = "Sign"

        if K not in exports:
            logger.warn(f"cannot convert {type(ob)}")
            continue

        f = exports[K]
        if f is None:
            continue
        thing_index = f(gltf, it.fqn[-1], ob)

        tile_transform = it.transform_sequence
        tile_matrix2d = tile_transform.asmatrix2d().m
        tile_matrix = SE3_from_SE2(tile_matrix2d)
        sign_node_index = add_node(gltf, Node(children=[thing_index]))
        tile_matrix_float = list(tile_matrix.T.flatten())
        tile_node = Node(name=it.fqn[-1], matrix=tile_matrix_float, children=[sign_node_index])
        tile_node_index = add_node(gltf, tile_node)
        map_nodes.append(tile_node_index)

    mapnode = Node(name="tiles", children=map_nodes)
    map_index = add_node(gltf, mapnode)
    add_node_to_scene(gltf, scene_index, map_index)

    # add_node_to_scene(model, scene_index, node1_index)
    yfov = np.deg2rad(60)
    camera = Camera(
        name="perpcamera",
        type="perspective",
        perspective=PerspectiveCameraInfo(aspectRatio=4 / 3, yfov=yfov, znear=0.01, zfar=1000),
    )
    gltf.model.cameras.append(camera)

    t = np.array([2, 2, 0.15])
    matrix = look_at(pos=t, target=np.array([0, 2, 0]))
    cam_index = add_node(gltf, Node(name="cameranode", camera=0, matrix=list(matrix.T.flatten())))
    add_node_to_scene(gltf, scene_index, cam_index)

    cleanup_model(gltf)

    fn = os.path.join(outdir, "main.gltf")
    make_sure_dir_exists(fn)
    logger.info(f"writing to {fn}")
    gltf.export(fn)
    if True:
        with open(fn) as f:
            data = f.read()
        j = json.loads(data)
        with open(fn, "w") as f:
            f.write(json.dumps(j, indent=2))
    fnb = os.path.join(outdir, "main.glb")
    logger.info(f"writing to {fnb}")
    gltf.export(fnb)

    if False:
        res = trimesh.load(fn)
        # camera_pose, _ = res.graph['cameranode']
        # logger.info(res=res)
        import pyrender

        scene = pyrender.Scene.from_trimesh_scene(res)
    # r = pyrender.OffscreenRenderer(640, 480)
    # cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    # scene.add(cam, pose=camera_pose)
    # color, depth = r.render(scene)


def export_tree(gltf: GLTF, name, ob):
    _ = get_resource_path("tree/main.gltf")
    # fn = "src/duckietown_world/data/gd1/meshes/tree/main.gltf"
    tree_node_index = embed_external(gltf, _)
    return tree_node_index


def export_duckie(gltf: GLTF, name, ob):
    _ = get_resource_path("duckie2/main.gltf")
    return embed_external(gltf, _)


def export_DB18(gltf: GLTF, name, ob: DB18) -> int:
    _ = get_resource_path("duckiebot3/main.gltf")

    g2 = GLTF.load(_)
    color = [0, 1, 0, 1]
    set_duckiebot_color(g2, "gkmodel0_chassis_geom0_mat_001-material.001", color)
    set_duckiebot_color(g2, "gkmodel0_chassis_geom0_mat_001-material", color)
    index = embed(gltf, g2)
    return index


def set_duckiebot_color(gltf: GLTF, mname: str, color: List[float]):
    for m in gltf.model.materials:
        if m.name == mname:
            m.pbrMetallicRoughness.baseColorFactor = color
            logger.info("found material", m=m)
            break
    else:
        logger.error("could not find material")


def embed_external(gltf: GLTF, fn: str) -> int:
    if fn not in gltf.fn2node:
        g2 = GLTF.load(fn)

        index = gltf.fn2node[fn] = embed(gltf, g2)
        return index
    index = gltf.fn2node[fn]
    return make_node_copy(gltf, index)


def make_node_copy(gltf: GLTF, index: int) -> int:
    node = gltf.model.nodes[index]
    if node.children:
        children = [make_node_copy(gltf, i) for i in node.children]
    else:
        children = None
    n2 = replace(node, children=children)
    return add_node(gltf, n2)


def embed(gltf: GLTF, g2: GLTF) -> int:
    lenn = lambda x: 0 if x is None else len(x)
    model = gltf.model
    off_accessors = lenn(model.accessors)
    off_animations = lenn(model.animations)

    off_buffers = lenn(model.buffers)
    off_bufferViews = lenn(model.bufferViews)
    off_cameras = lenn(model.cameras)
    off_images = lenn(model.images)
    off_materials = lenn(model.materials)
    off_meshes = lenn(model.meshes)
    off_nodes = lenn(model.nodes)
    off_samplers = lenn(model.samplers)
    off_scenes = lenn(model.scenes)
    off_skins = lenn(model.skins)
    off_textures = len(model.textures)
    model2 = g2.model

    def add_if_not_none(a, b):
        if a is None:
            return None
        else:
            return a + b

    # b: Buffer
    if model2.buffers:
        for b in model2.buffers:
            model.buffers.append(b)
    if model2.bufferViews:
        for bv in model2.bufferViews:
            buffer = add_if_not_none(bv.buffer, off_buffers)
            bv2 = replace(bv, buffer=buffer)
            model.bufferViews.append(bv2)
    if model2.samplers:
        for s in model2.samplers:
            model.samplers.append(s)
    if model2.cameras:
        for c in model2.cameras:
            model.cameras.append(c)
    if model2.images:
        for i in model2.images:
            i2 = replace(i, bufferView=add_if_not_none(i.bufferView, off_bufferViews))
            model.images.append(i2)

    def convert_texture_info(t: Optional[TextureInfo]) -> Optional[TextureInfo]:
        if t is None:
            return None
        index = add_if_not_none(t.index, off_textures)
        return replace(t, index=index)

    def convert_metallic_r(t: Optional[PBRMetallicRoughness]) -> Optional[PBRMetallicRoughness]:
        if t is None:
            return None
        baseColorTexture = convert_texture_info(t.baseColorTexture)

        metallicRoughnessTexture = convert_texture_info(t.metallicRoughnessTexture)

        return replace(
            t, baseColorTexture=baseColorTexture, metallicRoughnessTexture=metallicRoughnessTexture
        )

    if model2.materials:
        m: Material
        for m in model2.materials:
            pbrMetallicRoughness = convert_metallic_r(m.pbrMetallicRoughness)
            normalTexture = convert_texture_info(m.normalTexture)
            occlusionTexture = convert_texture_info(m.occlusionTexture)
            emissiveTexture = convert_texture_info(m.emissiveTexture)

            m2 = replace(
                m,
                normalTexture=normalTexture,
                occlusionTexture=occlusionTexture,
                emissiveTexture=emissiveTexture,
                pbrMetallicRoughness=pbrMetallicRoughness,
            )
            model.materials.append(m2)

    def convert_node(n1: Node) -> Node:
        camera2 = add_if_not_none(n1.camera, off_cameras)
        mesh2 = add_if_not_none(n1.mesh, off_meshes)
        if n1.children is None:
            children2 = None
        else:
            children2 = [_ + off_nodes for _ in n1.children]
        return replace(n1, camera=camera2, mesh=mesh2, children=children2)

    if model2.nodes:
        n_: Node
        for n_ in model2.nodes:
            n2 = convert_node(n_)
            model.nodes.append(n2)

    def replace_attributes(a: Attributes) -> Attributes:
        # class Attributes:
        #     """
        #     Helper type for describing the attributes of a primitive. Each property corresponds to mesh
        #     attribute semantic and
        #     each value is the index of the accessor containing the attribute's data.
        #     """
        return Attributes(
            POSITION=add_if_not_none(a.POSITION, off_accessors),
            NORMAL=add_if_not_none(a.NORMAL, off_accessors),
            TANGENT=add_if_not_none(a.TANGENT, off_accessors),
            TEXCOORD_0=add_if_not_none(a.TEXCOORD_0, off_accessors),
            TEXCOORD_1=add_if_not_none(a.TEXCOORD_1, off_accessors),
            COLOR_0=add_if_not_none(a.COLOR_0, off_accessors),
            JOINTS_0=add_if_not_none(a.JOINTS_0, off_accessors),
            WEIGHTS_0=add_if_not_none(a.WEIGHTS_0, off_accessors),
        )

    def replace_primitives(p: Primitive) -> Primitive:
        # TODO: targets
        attributes = replace_attributes(p.attributes)
        indices = add_if_not_none(p.indices, off_accessors)
        material = add_if_not_none(p.material, off_materials)
        return replace(p, attributes=attributes, indices=indices, material=material)

    if model2.meshes:
        mesh: Mesh
        for mesh in model2.meshes:
            primitives = [replace_primitives(_) for _ in mesh.primitives]
            mesh = replace(mesh, primitives=primitives)
            model.meshes.append(mesh)

    if model2.textures:
        tex: Texture
        for tex in model2.textures:
            sampler = add_if_not_none(tex.sampler, off_samplers)
            source = add_if_not_none(tex.source, off_images)
            tex2 = replace(tex, sampler=sampler, source=source)
            model.textures.append(tex2)

    if model2.accessors:
        a: Accessor
        for a in model2.accessors:
            bufferView = add_if_not_none(a.bufferView, off_bufferViews)
            a2 = replace(a, bufferView=bufferView)
            model.accessors.append(a2)

    assert len(model2.scenes) == 1, model2
    # scene: Scene
    # for scene in model2.scenes:
    #     nodes = [add_if_not_none(_, off_nodes) for _ in scene.nodes]
    #     scene2 = replace(scene, nodes=nodes)
    #     model.scenes.append(scene2)
    scene0 = model2.scenes[0]
    scene_nodes = [add_if_not_none(_, off_nodes) for _ in scene0.nodes]

    node = Node(children=scene_nodes)

    node_index = add_node(gltf, node)
    logger.info(f"the main scene node for imported is {node_index}")
    # model.scenes[0].nodes.extend(nodes)

    gltf.resources.extend(g2.resources)
    check_loops(gltf)
    return node_index


def check_loops(gltf: GLTF):
    n = DiGraph()
    all_nodes = list(range(len(gltf.model.nodes)))
    for i in all_nodes:
        n.add_node(i)

    for node_index, node in enumerate(gltf.model.nodes):
        if node.children:
            for c in node.children:
                n.add_edge(node_index, c)

    try:
        c = find_cycle(n)
    except NetworkXNoCycle:
        pass
    else:
        logger.info(c=c)
        raise ZValueError("cycle found", c=c)


def make_texture(gltf: GLTF, fn: str):
    if fn not in gltf.fn2texture_index:
        uri = os.path.basename(fn)
        gltf.resources.append(resource_from_file(fn))
        image = Image(uri=uri)
        image_index = add_image(gltf, image)
        texture = Texture(sampler=0, source=image_index)
        texture_index = add_texture(gltf, texture)
        gltf.fn2texture_index[fn] = texture_index

    return gltf.fn2texture_index[fn]


def make_material(
    gltf: GLTF,
    *,
    doubleSided: bool,
    baseColorFactor: List = None,
    fn: str = None,
    fn_normals: str = None,
    fn_emissive: str = None,
):
    key = (tuple(baseColorFactor), fn, fn_normals, fn_emissive, doubleSided)
    if key not in gltf.cacheMaterial:
        res = make_material_(
            gltf,
            doubleSided=doubleSided,
            baseColorFactor=baseColorFactor,
            fn=fn,
            fn_normals=fn_normals,
            fn_emissive=fn_emissive,
        )
        gltf.cacheMaterial[key] = res
    return gltf.cacheMaterial[key]


def make_material_(
    gltf: GLTF,
    *,
    doubleSided: bool,
    baseColorFactor: List = None,
    fn: str = None,
    fn_normals: str = None,
    fn_emissive: str = None,
) -> int:
    # uri = os.path.basename(fn)
    # doubleSided = True
    if baseColorFactor is None:
        baseColorFactor = [1, 1, 1, 1.0]

    if fn is not None:
        texture_index = make_texture(gltf, fn)
        baseColorTexture = TextureInfo(index=texture_index)
    else:
        baseColorTexture = None

    if fn_normals is not None:
        normal_map_index = make_texture(gltf, fn_normals)
        normalTexture = NormalTextureInfo(index=normal_map_index)
    else:
        normalTexture = None

    if fn_emissive is not None:
        emissive_map_index = make_texture(gltf, fn_emissive)
        emissiveTexture = NormalTextureInfo(index=emissive_map_index)
        s = 1.0
        emissiveFactor = [s, s, s]
    else:
        emissiveTexture = None
        emissiveFactor = None

    m = Material(
        doubleSided=doubleSided,
        pbrMetallicRoughness=PBRMetallicRoughness(
            baseColorFactor=baseColorFactor, baseColorTexture=baseColorTexture,
        ),
        alphaMode="BLEND",
        emissiveTexture=emissiveTexture,
        emissiveFactor=emissiveFactor,
        normalTexture=normalTexture,
    )

    return add_material(gltf, m)


def look_at(pos: np.ndarray, target: np.ndarray, gltf: bool = True) -> SE3value:
    diff = target - pos
    dist = np.linalg.norm(diff)
    x = diff / dist
    up = np.array([0, 0, 1])
    y = -np.cross(up, x)
    z = np.cross(x, y)
    R = np.eye(3)

    # +X axis is to the right, the lens looks towards the local -Z axis, and the top of the camera is
    # aligned with the local +Y axis.
    R0 = np.eye(3)
    R0[:, 0] = x
    R0[:, 1] = y
    R0[:, 2] = z
    R[:, 0] = y
    R[:, 1] = -z
    R[:, 2] = -x

    M = SE3_from_rotation_translation(R=R, t=pos)

    l = (M @ [dist, 0, 0, 1])[:3]

    eps = np.linalg.norm(target - l)

    # logger.info(diff=diff, dist=dist, up=up, x=x, y=y, z=z, R=R, M=M, l=l, eps=eps)
    return M


def cleanup_model(gltf: GLTF):
    model = gltf.model
    attrs = [
        "scenes",
        "nodes",
        "buffers",
        "bufferViews",
        "cameras",
        "images",
        "materials",
        "meshes",
        "samplers",
        "skins",
        "textures",
        "extensionsRequired",
        "extensionsUsed",
        "accessors",
    ]
    for a in attrs:
        v = getattr(model, a)
        if not v:
            # logger.debug(f'cleaning up attribute {a!r} because empty', v=v)
            setattr(model, a, None)


def resource_from_file(fn: str) -> FileResource:
    with open(fn, "rb") as f:
        data = f.read()

    bn = os.path.basename(fn)
    return FileResource(bn, data=data)


def add_mesh(gltf: GLTF, mesh: Mesh) -> int:
    model = gltf.model
    n = len(model.meshes)
    model.meshes.append(mesh)
    return n


def add_node(gltf: GLTF, node: Node) -> int:
    model = gltf.model
    n = len(model.nodes)
    model.nodes.append(node)
    return n


def add_node_to_scene(gltf: GLTF, scene, node):
    model = gltf.model
    model.scenes[scene].nodes.append(node)


def add_scene(gltf: GLTF, scene: Scene) -> int:
    n = len(gltf.model.scenes)
    gltf.model.scenes.append(scene)
    return n


def add_image(gltf: GLTF, image: Image) -> int:
    model = gltf.model
    n = len(model.images)
    model.images.append(image)
    return n


def add_texture(gltf: GLTF, texture: Texture) -> int:
    model = gltf.model
    n = len(model.textures)
    model.textures.append(texture)
    return n


def add_material(gltf: GLTF, material: Material) -> int:
    model = gltf.model
    n = len(model.materials)
    model.materials.append(material)
    return n


def add_accessor(gltf: GLTF, accessor: Accessor) -> int:
    model = gltf.model
    n = len(model.accessors)
    model.accessors.append(accessor)
    return n


def add_buffer_view(gltf: GLTF, bf: BufferView) -> int:
    model = gltf.model
    n = len(model.bufferViews)
    model.bufferViews.append(bf)
    return n


def add_buffer(gltf: GLTF, bf: Buffer) -> int:
    model = gltf.model

    n = len(model.buffers)
    model.buffers.append(bf)

    return n


def export_sign(gltf: GLTF, name: str, sign: Sign) -> int:
    texture = sign.get_name_texture()
    # x = -0.2
    CM = 0.01
    PAPER_WIDTH, PAPER_HEIGHT = 8.5 * CM, 15.5 * CM
    PAPER_THICK = 0.01

    BASE_SIGN = 5 * CM
    WIDTH_SIGN = 1.1 * CM

    y = -1.5 * PAPER_HEIGHT  # XXX not sure why this is negative
    y = BASE_SIGN
    x = -PAPER_WIDTH / 2

    fn = get_texture_file(texture)

    material_index = make_material(gltf, doubleSided=False, baseColorFactor=[1, 1, 1, 1.0], fn=fn)
    mi = get_square()
    mesh_index = add_polygon(
        gltf,
        name + "-mesh",
        vertices=mi.vertices,
        texture=mi.textures,
        colors=mi.color,
        normals=mi.normals,
        material=material_index,
    )
    sign_node = Node(mesh=mesh_index)
    sign_node_index = add_node(gltf, sign_node)

    fn = get_texture_file("wood")
    material_index_white = make_material(gltf, doubleSided=False, baseColorFactor=[0.5, 0.5, 0.5, 1], fn=fn)
    back_mesh_index = add_polygon(
        gltf,
        name + "-mesh",
        vertices=mi.vertices,
        texture=mi.textures,
        colors=mi.color,
        normals=mi.normals,
        material=material_index_white,
    )
    back_rot = SE3_from_rotation_translation(roty(np.pi), np.array([0, 0, 0]))

    back_node = Node(matrix=gm(back_rot), mesh=back_mesh_index)
    back_node_index = add_node(gltf, back_node)

    scale = np.diag([PAPER_WIDTH, PAPER_HEIGHT, PAPER_WIDTH, 1])
    rot = SE3_from_rotation_translation(
        rotz(np.pi / 2) @ rotx(np.pi / 2) @ rotz(np.pi), np.array([0, 0, 0.8 * PAPER_HEIGHT])
    )
    M = rot @ scale

    node = Node(matrix=gm(M), children=[sign_node_index, back_node_index])
    return add_node(gltf, node)


def gm(m):
    return list(m.T.flatten())
