import numpy as np
import yaml
from comptests import comptest, get_comptests_output_dir, run_module_tests
from . import logger
import duckietown_world as dw
from aido_schemas import Scenario
from duckietown_world import draw_static
from duckietown_world.world_duckietown.map_loading import _get_map_yaml
from duckietown_world.world_duckietown.sampling import make_scenario
from duckietown_world.world_duckietown.sampling_poses import sample_good_starting_pose


@comptest
def test_pose_sampling_1():
    m = dw.load_map("4way")

    along_lane = 0.2
    only_straight = True
    for i in range(45):
        q = sample_good_starting_pose(m, only_straight=only_straight, along_lane=along_lane)
        ground_truth = dw.SE2Transform.from_SE2(q)
        m.set_object(f"sampled-{i}", dw.DB18(), ground_truth=ground_truth)

    outdir = get_comptests_output_dir()
    draw_static(m, outdir)


@comptest
def test_scenario_making1():
    map_name = "udem1"
    s: str = _get_map_yaml(map_name)

    yaml_data = yaml.load(s, Loader=yaml.SafeLoader)
    # update_map(yaml_data)
    yaml_str = yaml.dump(yaml_data)

    s: Scenario = make_scenario(
        yaml_str,
        scenario_name=map_name + "01",
        only_straight=True,
        min_dist=0.5,
        delta_y_m=0.1,
        delta_theta_rad=np.deg2rad(15),
        robots_pcs=["ego"],
        robots_npcs=["v1", "v2"],
        robots_parked=[],
        nduckies=10,
        duckie_min_dist_from_other_duckie=0.1,
        duckie_min_dist_from_robot=0.4,
        duckie_y_bounds=[-0.1, -0.3],
    )
    logger.info(s=s)


if __name__ == "__main__":
    run_module_tests()
