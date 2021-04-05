import argparse
import time
from typing import List

import geometry as g
import numpy as np
import yaml
from geometry import SE2_from_xytheta, SE2value
from zuper_commons.fs import write_ustring_to_utf8_file
from zuper_commons.types import ZException
from zuper_ipce import ipce_from_object

import duckietown_world as dw
from aido_schemas import PROTOCOL_NORMAL, RobotConfiguration, Scenario, ScenarioDuckieSpec, ScenarioRobotSpec
from duckietown_world.geo.rectangular_area import RectangularArea, sample_in_rect
from duckietown_world.world_duckietown.sampling import distance_poses, ieso


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--nduckies", type=int, required=True)
    parser.add_argument("--ntiles", type=int, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)
    parsed = parser.parse_args()

    scenario_name = parsed.scenario_name

    scenario = get_base_scenario(scenario_name=scenario_name, nduckies=parsed.nduckies, ntiles=parsed.ntiles)
    scenario_struct = ipce_from_object(scenario, Scenario, ieso=ieso)
    scenario_yaml = yaml.dump(scenario_struct)
    filename = parsed.output
    write_ustring_to_utf8_file(scenario_yaml, filename)


def sample_duckies(
    nduckies: int,
    robot_positions: List[SE2value],
    min_dist_from_robot: float,
    min_dist_from_other_duckie: float,
    bounds: dw.RectangularArea,
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

        theta = np.random.uniform(-np.pi, +np.pi)
        t = sample_in_rect(bounds)
        pose = g.SE2_from_translation_angle(t, theta)

        if far_enough(pose):
            poses.append(pose)

        dt = time.time() - t0
        if dt > timeout:
            msg = "Cannot sample in time."
            raise ZException(msg)
    return poses


def get_base_scenario(scenario_name: str, nduckies: int, ntiles: int) -> Scenario:

    tile_size = 0.585
    themap = {"tiles": [], "tile_size": tile_size}
    themap["tiles"] = [["asphalt"] * ntiles] * ntiles
    area = RectangularArea([0, 0], [tile_size * ntiles, tile_size * ntiles])

    robots = {}

    x = y = tile_size * (ntiles / 2)

    pose = SE2_from_xytheta([x, y, 0])
    vel = np.zeros((3, 3))
    robots["ego0"] = ScenarioRobotSpec(
        color="blue",
        configuration=RobotConfiguration(pose, vel),
        controllable=True,
        protocol=PROTOCOL_NORMAL,
        description="",
    )
    yaml_str = yaml.dump(themap)
    nduckies = 10
    duckie_poses = sample_duckies(
        nduckies,
        [robots["ego0"].configuration.pose],
        min_dist_from_robot=1.0,
        min_dist_from_other_duckie=0.2,
        bounds=area,
    )

    duckies = {}
    for i, pose in enumerate(duckie_poses):
        duckies[f"duckie{i}"] = ScenarioDuckieSpec("yellow", pose)
    ms = Scenario(
        scenario_name=scenario_name,
        environment=yaml_str,
        robots=robots,
        duckies=duckies,
        player_robots=list(robots),
    )
    return ms


if __name__ == "__main__":
    main()
