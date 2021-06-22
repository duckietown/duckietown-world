import argparse
import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import cast, Dict, List, Sequence, Tuple, Union

import geometry as g
import numpy as np
import yaml
from geometry import SE2_from_translation_angle, SE2value
from zuper_commons.fs import (
    FilePath,
    read_ustring_from_utf8_file,
    write_ustring_to_utf8_file,
)
from zuper_commons.logs import setup_logging, ZLogger
from zuper_commons.types import ZException, ZValueError
from zuper_ipce import IEDO, IESO, ipce_from_object, object_from_ipce

import duckietown_world as dw
from aido_schemas import (
    FriendlyVelocity,
    ProtocolDesc,
    RobotConfiguration,
    RobotName,
    Scenario,
    ScenarioDuckieSpec,
    ScenarioRobotSpec,
)
from .map_loading import _get_map_yaml, construct_map
from .sampling_poses import sample_good_starting_pose
from .utils import friendly_from_pose, pose_from_friendly
from ..geo.measurements_utils import iterate_by_class, iterate_by_test
from ..gltf.export import export_gltf
from ..utils.images import save_rgb_to_jpg, save_rgb_to_png

logger = ZLogger(__name__)


@dataclass
class ScenarioGenerationParam:
    map_name: str
    # sampling robots
    robots_npcs: List[str]
    robots_pcs: List[str]
    robots_parked: List[str]
    # where should they be?
    theta_tol_deg: Union[float, int]
    dist_tol_m: float
    min_dist: float
    """ min distance among robots """
    delta_y_m: float
    """ with respect to center of lane """
    only_straight: bool
    """ only sample in straight """

    # duckie parameters
    nduckies: int
    duckie_min_dist_from_other_duckie: float
    duckie_min_dist_from_robot: float
    duckie_y_bounds: List[float]

    pc_robot_protocol: ProtocolDesc
    npc_robot_protocol: ProtocolDesc

    tree_density: float = 0.5
    tree_min_dist: float = 0.4


@dataclass
class MultiConfigEntry:
    repeat: int
    config: ScenarioGenerationParam


@dataclass
class MultiConfig:
    sample: Dict[str, MultiConfigEntry]


iedo = IEDO(use_remembered_classes=True, remember_deserialized_classes=True)
ieso = IESO(with_schema=False)


def make_scenario_main(args=None):
    setup_logging()
    parser = argparse.ArgumentParser()

    # parser.add_argument("--config", help="Configuration", required=True)
    parser.add_argument("-o", "--output", help="Destination directory", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-n", "--num", type=int, help="Number of scenarios to generate", required=True)
    parser.add_argument(
        "--styles",
        default="synthetic-F",
        help="Draw preview in various styles, comma separated. (needs gym duckietown)",
    )

    parsed, rest = parser.parse_known_args(args=args)
    if parsed.styles == "all":
        styles = ["synthetic", "synthetic-F", "photos", "smooth"]
    else:
        styles = parsed.styles.split(",")

    if not rest:
        msg = "Need to provide at least one config file"
        raise Exception(msg)
    for config in rest:
        # basename = os.path.basename(config).split(".")[0]
        data = read_ustring_from_utf8_file(config)
        interpreted = yaml.load(data, Loader=yaml.Loader)
        # n: int = parsed.num
        output: str = parsed.output
        multi: MultiConfig = object_from_ipce(interpreted, MultiConfig, iedo=iedo)

        tosample = multi.sample
        for k, v in tosample.items():
            n = parsed.num * v.repeat
            params = v.config
            for i in range(n):
                scenario_name = f"{k}-{i:03d}"
                yaml_str = _get_map_yaml(params.map_name)
                scenario = make_scenario(
                    seed=parsed.seed + i * 10000,
                    yaml_str=yaml_str,
                    scenario_name=scenario_name,
                    only_straight=params.only_straight,
                    min_dist=params.min_dist,
                    delta_y_m=params.delta_y_m,
                    robots_npcs=params.robots_npcs,
                    robots_parked=params.robots_parked,
                    robots_pcs=params.robots_pcs,
                    nduckies=params.nduckies,
                    duckie_min_dist_from_other_duckie=params.duckie_min_dist_from_other_duckie,
                    duckie_min_dist_from_robot=params.duckie_min_dist_from_robot,
                    duckie_y_bounds=params.duckie_y_bounds,
                    delta_theta_rad=np.deg2rad(params.theta_tol_deg),
                    pc_robot_protocol=params.pc_robot_protocol,
                    npc_robot_protocol=params.npc_robot_protocol,
                    tree_density=params.tree_density,
                    tree_min_dist=params.tree_min_dist,
                )

                # styles = ['smooth']
                for style in styles:
                    try:
                        from gym_duckietown.simulator import Simulator
                    except ImportError:
                        logger.warning(traceback.format_exc())
                        # noinspection PyUnusedLocal
                        Simulator = None
                    else:
                        sim = Simulator(
                            "4way",
                            enable_leds=True,
                            domain_rand=False,
                            num_tris_distractors=0,
                            camera_width=640,
                            camera_height=480,
                            # distortion=True,
                            color_ground=[0, 0.3, 0],  # green
                            style=style,
                        )
                        logger.info("resetting")
                        sim.reset()
                        m = cast(dw.MapFormat1, yaml.load(scenario.environment, Loader=yaml.Loader))

                        if "objects" not in m:
                            m["objects"] = {}
                        obs: Dict[str, object] = m["objects"]

                        for robot_name, srobot in scenario.robots.items():
                            p = pose_from_friendly(srobot.configuration.pose)
                            st = dw.SE2Transform.from_SE2(p)

                            obs[robot_name] = dict(
                                kind="duckiebot", pose=st.as_json_dict(), height=0.12, color=srobot.color
                            )

                        for duckie_name, duckie in scenario.duckies.items():
                            p = pose_from_friendly(duckie.pose)
                            st = dw.SE2Transform.from_SE2(p)
                            obs[duckie_name] = dict(
                                kind="duckie", pose=st.as_json_dict(), height=0.08, color=duckie.color
                            )

                        sim._interpret_map(m)
                        sim.start_pose = None
                        sim.reset()
                        logger.info("rendering obs")
                        img = sim.render_obs()
                        out = os.path.join(output, scenario_name, style, "cam.png")
                        save_rgb_to_png(img, out)
                        out = os.path.join(output, scenario_name, style, "cam.jpg")
                        save_rgb_to_jpg(img, out)

                        sim.cur_pos = [-100.0, -100.0, -100.0]
                        from gym_duckietown.simulator import FrameBufferMemory

                        td = FrameBufferMemory(width=1024, height=1024)
                        horiz = sim._render_img(
                            width=td.width,
                            height=td.height,
                            multi_fbo=td.multi_fbo,
                            final_fbo=td.final_fbo,
                            img_array=td.img_array,
                            top_down=True,
                        )
                        # img = sim.render("top_down")
                        out = cast(FilePath, os.path.join(output, scenario_name, style, "top_down.jpg"))
                        save_rgb_to_jpg(horiz, out)
                        out = cast(FilePath, os.path.join(output, scenario_name, style, "top_down.png"))
                        save_rgb_to_png(horiz, out)

                        dw.Tile.style = style
                        dm = interpret_scenario(scenario)
                        output_dir = os.path.join(output, scenario_name, style)
                        dw.draw_static(dm, output_dir=output_dir)
                        export_gltf(dm, output_dir, background=False)

                scenario_struct = ipce_from_object(scenario, Scenario, ieso=ieso)
                scenario_yaml = yaml.dump(scenario_struct)
                filename = os.path.join(output, scenario_name, f"scenario.yaml")
                write_ustring_to_utf8_file(scenario_yaml, filename)


def interpret_scenario(s: Scenario) -> dw.DuckietownMap:
    """ """
    y = yaml.load(s.environment, Loader=yaml.SafeLoader)

    dm = construct_map(y)
    if True:
        for robot_name, robot_spec in s.robots.items():
            pose = pose_from_friendly(robot_spec.configuration.pose)
            # gt = dw.Constant[dw.SE2Transform](dw.SE2Transform.from_SE2(pose))
            gt = dw.SE2Transform.from_SE2(pose)
            ob = dw.DB18(color=robot_spec.color)
            # noinspection PyTypeChecker
            dm.set_object(robot_name, ob, ground_truth=gt)

    if True:
        for duckie_name, duckie_spec in s.duckies.items():
            pose = pose_from_friendly(duckie_spec.pose)

            # gt = dw.Constant[dw.SE2Transform](dw.SE2Transform.from_SE2(pose))

            gt = dw.SE2Transform.from_SE2(pose)

            ob = dw.Duckie(color=duckie_spec.color)

            # noinspection PyTypeChecker
            dm.set_object(duckie_name, ob, ground_truth=gt)
    return dm


def make_scenario(
    yaml_str: str,
    scenario_name: str,
    only_straight: bool,
    min_dist: float,
    delta_y_m: float,
    delta_theta_rad: float,
    robots_pcs: List[RobotName],
    robots_npcs: List[RobotName],
    robots_parked: List[RobotName],
    nduckies: int,
    duckie_min_dist_from_other_duckie: float,
    duckie_min_dist_from_robot: float,
    tree_density: float,
    tree_min_dist: float,
    duckie_y_bounds: Sequence[float],
    pc_robot_protocol: ProtocolDesc,
    npc_robot_protocol: ProtocolDesc,
    seed: int,
) -> Scenario:
    np.random.seed(seed)
    yaml_data = yaml.load(yaml_str, Loader=yaml.SafeLoader)
    if "objects" not in yaml_data:
        yaml_data["objects"] = {}

    objects = yaml_data["objects"]
    if isinstance(objects, list):
        raise ZValueError(yaml_data=yaml_data)
    po = dw.construct_map(yaml_data)
    num_pcs = len(robots_pcs)
    num_npcs = len(robots_npcs)
    num_parked = len(robots_parked)
    nrobots = num_npcs + num_pcs + num_parked

    all_robot_poses = sample_many_good_starting_poses(
        po,
        nrobots,
        only_straight=only_straight,
        min_dist=min_dist,
        delta_theta_rad=delta_theta_rad,
        delta_y_m=delta_y_m,
    )
    remaining_robot_poses = list(all_robot_poses)

    poses_pcs = remaining_robot_poses[:num_pcs]
    remaining_robot_poses = remaining_robot_poses[num_pcs:]
    #
    poses_npcs = remaining_robot_poses[:num_npcs]
    remaining_robot_poses = remaining_robot_poses[num_npcs:]
    #
    poses_parked = remaining_robot_poses[:num_parked]
    remaining_robot_poses = remaining_robot_poses[num_parked:]
    assert len(remaining_robot_poses) == 0

    COLORS_PLAYABLE = ["red", "green", "blue"]
    COLOR_NPC = "blue"
    COLOR_PARKED = "grey"
    robots = {}
    for i, robot_name in enumerate(robots_pcs):
        pose = poses_pcs[i]
        fpose = friendly_from_pose(pose)
        velocity = FriendlyVelocity(0.0, 0.0, 0.0)
        configuration = RobotConfiguration(fpose, velocity)

        color = COLORS_PLAYABLE[i % len(COLORS_PLAYABLE)]
        robots[robot_name] = ScenarioRobotSpec(
            description=f"Playable robot {robot_name}",
            controllable=True,
            configuration=configuration,
            # motion=None,
            color=color,
            protocol=pc_robot_protocol,
        )

    for i, robot_name in enumerate(robots_npcs):
        pose = poses_npcs[i]

        fpose = friendly_from_pose(pose)
        velocity = FriendlyVelocity(0.0, 0.0, 0.0)
        configuration = RobotConfiguration(fpose, velocity)

        robots[robot_name] = ScenarioRobotSpec(
            description=f"NPC robot {robot_name}",
            controllable=True,
            configuration=configuration,
            color=COLOR_NPC,
            protocol=npc_robot_protocol,
        )

    for i, robot_name in enumerate(robots_parked):
        pose = poses_parked[i]

        fpose = friendly_from_pose(pose)
        velocity = FriendlyVelocity(0.0, 0.0, 0.0)
        configuration = RobotConfiguration(fpose, velocity)

        robots[robot_name] = ScenarioRobotSpec(
            description=f"Parked robot {robot_name}",
            controllable=False,
            configuration=configuration,
            # motion=MOTION_PARKED,
            color=COLOR_PARKED,
            protocol=None,
        )
    # logger.info(duckie_y_bounds=duckie_y_bounds)
    names = [f"duckie{i:02d}" for i in range(nduckies)]
    poses = sample_duckies_poses(
        po,
        nduckies,
        robot_positions=all_robot_poses,
        min_dist_from_other_duckie=duckie_min_dist_from_other_duckie,
        min_dist_from_robot=duckie_min_dist_from_robot,
        from_side_bounds=(duckie_y_bounds[0], duckie_y_bounds[1]),
        delta_theta_rad=np.pi,
    )
    d = [ScenarioDuckieSpec("yellow", friendly_from_pose(_)) for _ in poses]
    duckies = dict(zip(names, d))

    trees = sample_trees(po, tree_density, tree_min_dist)
    for i, pose_tree in enumerate(trees):
        obname = f"tree_random_{i}"
        st = dw.SE2Transform.from_SE2(pose_tree)
        objects[obname] = {"kind": "tree", "pose": st.as_json_dict(), "height": random.uniform(0.15, 0.4)}

    add_signs(po, objects)

    yaml_str = yaml.dump(yaml_data)
    # logger.info(trees=trees)
    payload = {"info": "fill here the yaml"}
    ms = Scenario(
        payload_yaml=yaml.dump(payload),
        scenario_name=scenario_name,
        environment=yaml_str,
        robots=robots,
        duckies=duckies,
        player_robots=list(robots_pcs),
    )
    return ms


from numpy.random import choice


def add_signs(po: dw.PlacedObject, objects):
    def choose_tile(x: dw.PlacedObject) -> bool:
        return isinstance(x, dw.Tile) and x.kind == "3way_left"

    tiles = list(iterate_by_test(po, choose_tile))
    i = 0

    P = 0.54
    D = 0.12
    for tile in tiles:
        # logger.info('found', tile=tile)

        def add_sign(k: str, rel, theta_deg):
            nonlocal i
            obname = f"{i}-{k}"
            i += 1

            q = SE2_from_translation_angle(rel, np.deg2rad(theta_deg))
            # logger.info(tile.transform_sequence)
            m1 = tile.transform_sequence.asmatrix2d().m
            pose = m1 @ q
            st = dw.SE2Transform.from_SE2(pose)
            objects[obname] = {"kind": k, "pose": st.as_json_dict(), "height": 0.18}

        # add_sign('sign_stop', [0.1, 0], 20)

        # for me
        add_sign("sign_stop", [P, P], 180)
        add_sign("sign_left_T_intersect", [P, -P], 180)

        # coming from left
        add_sign("sign_stop", [P + D, -P], 90)
        add_sign("sign_T_intersect", [-P, -P], 90)

        # coming from top
        add_sign("sign_stop", [-P + D, -P], 0)
        add_sign("sign_right_T_intersect", [-P, +P], 0)

    def choose_tile(x: dw.PlacedObject) -> bool:
        return isinstance(x, dw.Tile) and x.kind == "4way"

    tiles = list(iterate_by_test(po, choose_tile))
    i = 0
    for tile in tiles:
        # logger.info('found', tile=tile)

        def add_sign(k: str, rel, theta_deg):
            nonlocal i
            obname = f"{i}-{k}"
            i += 1

            q = SE2_from_translation_angle(rel, np.deg2rad(theta_deg))
            # logger.info(tile.transform_sequence)
            m1 = tile.transform_sequence.asmatrix2d().m
            pose = m1 @ q
            st = dw.SE2Transform.from_SE2(pose)
            objects[obname] = {"kind": k, "pose": st.as_json_dict(), "height": 0.18}

        # add_sign('sign_stop', [0.1, 0], 20)

        # for me
        add_sign("sign_t_light_ahead", [P, P + D], 180)
        add_sign("sign_4_way_intersect", [P, -P], 180)

        # coming from left
        add_sign("sign_t_light_ahead", [P + D, -P], 90)
        add_sign("sign_4_way_intersect", [-P, -P], 90)

        # coming from top
        add_sign("sign_t_light_ahead", [-P, -P - D], 0)
        add_sign("sign_4_way_intersect", [-P, +P], 0)

        # coming from right
        add_sign("sign_t_light_ahead", [-P - D, P], 270)
        add_sign("sign_4_way_intersect", [P, P], 270)


def sample_trees(po: dw.PlacedObject, tree_density: float, tree_min_dist: float) -> List[SE2value]:
    if tree_density == 0:
        return []

    good_tile_types = ["floor", "grass"]

    def choose_grass(x: dw.PlacedObject) -> bool:
        if isinstance(x, dw.Tile) and x.kind in good_tile_types:
            return True
        else:
            return False

    def the_others(x: dw.PlacedObject) -> bool:
        if isinstance(x, dw.Tile) and x.kind not in good_tile_types:
            return True
        else:
            return False

    choices = list(iterate_by_test(po, choose_grass))
    non_grass = list(iterate_by_test(po, the_others))

    # logger.info(choices=choices)
    if not choices:
        msg = "Cannot put trees in grass"
        tiles = list(_.object.kind for _ in iterate_by_class(po, dw.Tile))
        raise ZValueError(msg, tiles=tiles)

    ntrees = int(len(choices) * tree_density)
    results: List[SE2value] = []

    def far_enough(pose_):
        for p in results:
            if distance_poses(p, pose_) < tree_min_dist:
                return False
        return True

    options = list(range(len(choices)))
    attitude = [None] * len(choices)
    a: int
    for a, c in enumerate(choices):
        p1 = c.transform_sequence.asmatrix2d().m
        min_distance = 100.0
        sum_distances = 0.0
        for k, s in enumerate(non_grass):
            p2 = s.transform_sequence.asmatrix2d().m
            d = distance_poses(p1, p2)
            sum_distances += d
            # if d < 0.4:
            #     logger.info(a=a, k=k, d=d)
            #     raise ZValueError()
            min_distance = min(d, min_distance)
        # noinspection PyTypeChecker
        attitude[a] = 1.0 / sum_distances

    # logger.info(attitude=attitude)
    nplaced = [0] * len(choices)
    for i in range(ntrees):

        while True:
            weights = 1.0 / (np.array(nplaced, dtype=float) * 3 + 1)
            weights2 = weights * np.array(attitude)
            weights3 = weights2 / np.sum(weights2)
            chosen = choice(options, p=weights3)

            itr = choices[chosen]

            m1 = itr.transform_sequence.asmatrix2d().m

            t = [random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)]
            pos_in_tile = SE2_from_translation_angle(t, 0.0)

            pose = m1 @ pos_in_tile
            if far_enough(pose):
                results.append(pose)
                nplaced[chosen] += 1
                break

    return results


def sample_many_good_starting_poses(
    po: dw.PlacedObject,
    nrobots: int,
    only_straight: bool,
    min_dist: float,
    delta_theta_rad: float,
    delta_y_m: float,
    timeout: float = 10,
) -> List[np.ndarray]:
    poses = []

    def far_enough(pose_):
        for p in poses:
            if distance_poses(p, pose_) < min_dist:
                return False
        return True

    t0 = time.time()
    while len(poses) < nrobots:
        pose = sample_good_starting_pose(po, only_straight=only_straight, along_lane=0.2)
        if far_enough(pose):
            theta = np.random.uniform(-delta_theta_rad, +delta_theta_rad)
            y = np.random.uniform(-delta_y_m, +delta_y_m)
            t = [0, y]
            q = g.SE2_from_translation_angle(t, theta)
            pose = g.SE2.multiply(pose, q)
            poses.append(pose)

        dt = time.time() - t0
        if dt > timeout:
            msg = "Cannot sample the poses"
            raise ZException(msg)
    return poses


def sample_duckies_poses(
    po: dw.PlacedObject,
    nduckies: int,
    robot_positions: List[SE2value],
    min_dist_from_robot: float,
    min_dist_from_other_duckie: float,
    from_side_bounds: Tuple[float, float],
    delta_theta_rad: float,
    timeout: float = 10,
) -> List[np.ndarray]:
    poses: List[SE2value] = []

    def far_enough(pose_: SE2value) -> bool:
        for p in poses:
            if distance_poses(p, pose_) < min_dist_from_other_duckie:
                return False
        for p in robot_positions:
            if distance_poses(p, pose_) < min_dist_from_robot:
                return False
        return True

    t0 = time.time()
    while len(poses) < nduckies:
        along_lane = np.random.uniform(0, 1)
        pose = sample_good_starting_pose(po, only_straight=False, along_lane=along_lane)
        if not far_enough(pose):
            continue

        theta = np.random.uniform(-delta_theta_rad, +delta_theta_rad)
        y = np.random.uniform(from_side_bounds[0], from_side_bounds[1])
        t = [0, y]
        q = g.SE2_from_translation_angle(t, theta)
        pose = g.SE2.multiply(pose, q)
        poses.append(pose)

        dt = time.time() - t0
        if dt > timeout:
            msg = "Cannot sample in time."
            raise ZException(msg)
    return poses


def distance_poses(q1: SE2value, q2: SE2value) -> float:
    SE2 = g.SE2
    d = SE2.multiply(SE2.inverse(q1), q2)
    t, _a = g.translation_angle_from_SE2(d)
    return np.linalg.norm(t)
