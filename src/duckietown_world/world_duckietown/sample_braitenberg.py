import argparse

import numpy as np
import yaml
from geometry import SE2_from_xytheta
from zuper_commons.fs import write_ustring_to_utf8_file
from zuper_ipce import ipce_from_object

from aido_schemas import PROTOCOL_NORMAL, RobotConfiguration, Scenario, ScenarioRobotSpec
from duckietown_world.world_duckietown.sampling import ieso


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", "-o", required=True)
    parsed = parser.parse_args()

    scenario = get_base_scenario(10)
    scenario_struct = ipce_from_object(scenario, Scenario, ieso=ieso)
    scenario_yaml = yaml.dump(scenario_struct)
    filename = parsed.output
    write_ustring_to_utf8_file(scenario_yaml, filename)


def get_base_scenario(ntiles: int) -> Scenario:
    scenario_name = "map1"
    tile_size = 0.585
    themap = {"tiles": [], "tile_size": tile_size}
    themap["tiles"] = [["asphalt"] * ntiles] * ntiles

    duckies = {}
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
