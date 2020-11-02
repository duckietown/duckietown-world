import argparse
import os
from dataclasses import dataclass
from typing import cast, List, Sequence, Tuple, Union

import duckietown_world as dw
import geometry as g
import numpy as np
import yaml
from aido_schemas import (
    PROTOCOL_FULL,
    PROTOCOL_NORMAL,
    RobotConfiguration,
    RobotName,
    Scenario,
    ScenarioDuckieSpec,
    ScenarioRobotSpec,
)
from geometry import SE2value
from zuper_commons.fs import read_ustring_from_utf8_file, write_ustring_to_utf8_file
from zuper_commons.logs import setup_logging, ZLogger
from zuper_ipce import IEDO, IESO, ipce_from_object, object_from_ipce

from .map_loading import _get_map_yaml, construct_map
from .sampling_poses import sample_good_starting_pose
from ..gltf.export import export_gltf

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


iedo = IEDO(use_remembered_classes=True, remember_deserialized_classes=True)
ieso = IESO(with_schema=False)


def make_scenario_main(args=None):
    setup_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Configuration", required=True)
    parser.add_argument("-o", "--output", help="Destination directory", required=True)
    parser.add_argument("-n", "--num", type=int, help="Number of scenarios to generate", required=True)

    parsed = parser.parse_args(args=args)
    config: str = parsed.config
    basename = os.path.basename(config).split(".")[0]
    data = read_ustring_from_utf8_file(config)
    interpreted = yaml.load(data, Loader=yaml.Loader)
    n: int = parsed.num
    output: str = parsed.output
    params = object_from_ipce(interpreted, ScenarioGenerationParam, iedo=iedo)
    for i in range(n):
        scenario_name = f"{basename}-{i:03d}"
        yaml_str = _get_map_yaml(params.map_name)
        scenario = make_scenario(
            yaml_str=yaml_str,
            scenario_name=scenario_name,
            only_straight=params.only_straight,
            min_dist=params.min_dist,
            delta_y_m=params.delta_y_m,
            robots_npcs=params.robots_npcs,
            robots_parked=params.robots_parked,
            robots_pcs=params.robots_parked,
            nduckies=params.nduckies,
            duckie_min_dist_from_other_duckie=params.duckie_min_dist_from_other_duckie,
            duckie_min_dist_from_robot=params.duckie_min_dist_from_robot,
            duckie_y_bounds=params.duckie_y_bounds,
            delta_theta_rad=np.deg2rad(params.theta_tol_deg),
        )
        scenario_struct = ipce_from_object(scenario, Scenario, ieso=ieso)
        scenario_yaml = yaml.dump(scenario_struct)
        filename = os.path.join(output, f"{scenario_name}.scenario.yaml")
        write_ustring_to_utf8_file(scenario_yaml, filename)
        dm = interpret_scenario(scenario)
        output_dir = os.path.join(output, scenario_name)
        dw.draw_static(dm, output_dir=output_dir)
        export_gltf(dm, output_dir, background=False)


def interpret_scenario(s: Scenario) -> dw.DuckietownMap:
    """  """
    y = yaml.load(s.environment, Loader=yaml.SafeLoader)
    dm = construct_map(y)
    if True:
        for robot_name, robot_spec in s.robots.items():
            pose = cast(g.SE2value, robot_spec.configuration.pose)
            gt = dw.Constant[dw.SE2Transform](dw.SE2Transform.from_SE2(pose))
            gt = dw.SE2Transform.from_SE2(pose)
            ob = dw.DB18(color=robot_spec.color)
            # noinspection PyTypeChecker
            dm.set_object(robot_name, ob, ground_truth=gt)

    if True:
        for duckie_name, duckie_spec in s.duckies.items():
            pose = cast(g.SE2value, duckie_spec.pose)

            gt = dw.Constant[dw.SE2Transform](dw.SE2Transform.from_SE2(pose))

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
    duckie_y_bounds: Sequence[float],
) -> Scenario:
    yaml_data = yaml.load(yaml_str, Loader=yaml.SafeLoader)
    po = dw.construct_map(yaml_data)
    num_pcs = len(robots_pcs)
    num_npcs = len(robots_npcs)
    num_parked = len(robots_parked)
    nrobots = num_npcs + num_pcs + num_parked

    poses = sample_many_good_starting_poses(
        po,
        nrobots,
        only_straight=only_straight,
        min_dist=min_dist,
        delta_theta_rad=delta_theta_rad,
        delta_y_m=delta_y_m,
    )

    poses_pcs = poses[:num_pcs]
    poses = poses[num_pcs:]
    #
    poses_npcs = poses[:num_npcs]
    poses = poses[num_npcs:]
    #
    poses_parked = poses[:num_parked]
    poses = poses[num_parked:]
    assert len(poses) == 0

    COLOR_PLAYABLE = "red"
    COLOR_NPC = "blue"
    COLOR_PARKED = "grey"
    robots = {}
    for i, robot_name in enumerate(robots_pcs):
        pose = poses_pcs[i]
        vel = g.se2_from_linear_angular([0, 0], 0)

        configuration = RobotConfiguration(pose=pose, velocity=vel)

        robots[robot_name] = ScenarioRobotSpec(
            description=f"Playable robot {robot_name}",
            controllable=True,
            configuration=configuration,
            # motion=None,
            color=COLOR_PLAYABLE,
            protocol=PROTOCOL_NORMAL,
        )

    for i, robot_name in enumerate(robots_npcs):
        pose = poses_npcs[i]
        vel = g.se2_from_linear_angular([0, 0], 0)

        configuration = RobotConfiguration(pose=pose, velocity=vel)

        robots[robot_name] = ScenarioRobotSpec(
            description=f"NPC robot {robot_name}",
            controllable=True,
            configuration=configuration,
            # motion=MOTION_MOVING,
            color=COLOR_NPC,
            protocol=PROTOCOL_FULL,
        )

    for i, robot_name in enumerate(robots_parked):
        pose = poses_parked[i]
        vel = g.se2_from_linear_angular([0, 0], 0)

        configuration = RobotConfiguration(pose=pose, velocity=vel)

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
        robot_positions=poses,
        min_dist_from_other_duckie=duckie_min_dist_from_other_duckie,
        min_dist_from_robot=duckie_min_dist_from_robot,
        from_side_bounds=(duckie_y_bounds[0], duckie_y_bounds[1]),
        delta_theta_rad=np.pi,
    )
    d = [ScenarioDuckieSpec("yellow", _) for _ in poses]
    duckies = dict(zip(names, d))
    ms = Scenario(
        scenario_name=scenario_name,
        environment=yaml_str,
        robots=robots,
        duckies=duckies,
        player_robots=list(robots_pcs),
    )
    return ms


def sample_many_good_starting_poses(
    po: dw.PlacedObject,
    nrobots: int,
    only_straight: bool,
    min_dist: float,
    delta_theta_rad: float,
    delta_y_m: float,
) -> List[np.ndarray]:
    poses = []

    def far_enough(pose_):
        for p in poses:
            if distance_poses(p, pose_) < min_dist:
                return False
        return True

    while len(poses) < nrobots:
        pose = sample_good_starting_pose(po, only_straight=only_straight, along_lane=0.2)
        if far_enough(pose):
            theta = np.random.uniform(-delta_theta_rad, +delta_theta_rad)
            y = np.random.uniform(-delta_y_m, +delta_y_m)
            t = [0, y]
            q = g.SE2_from_translation_angle(t, theta)
            pose = g.SE2.multiply(pose, q)
            poses.append(pose)
    return poses


def sample_duckies_poses(
    po: dw.PlacedObject,
    nduckies: int,
    robot_positions: List[SE2value],
    min_dist_from_robot: float,
    min_dist_from_other_duckie: float,
    from_side_bounds: Tuple[float, float],
    delta_theta_rad: float,
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

    while len(poses) < nduckies:
        pose = sample_good_starting_pose(po, only_straight=False, along_lane=0.2)
        if not far_enough(pose):
            continue

        theta = np.random.uniform(-delta_theta_rad, +delta_theta_rad)
        y = np.random.uniform(from_side_bounds[0], from_side_bounds[1])
        t = [0, y]
        q = g.SE2_from_translation_angle(t, theta)
        pose = g.SE2.multiply(pose, q)
        poses.append(pose)
    return poses


def distance_poses(q1: SE2value, q2: SE2value) -> float:
    SE2 = g.SE2
    d = SE2.multiply(SE2.inverse(q1), q2)
    t, _a = g.translation_angle_from_SE2(d)
    return np.linalg.norm(t)
