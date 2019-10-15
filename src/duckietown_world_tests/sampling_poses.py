# coding=utf-8
from comptests import comptest, get_comptests_output_dir, run_module_tests
from duckietown_world import draw_static
from duckietown_world.world_duckietown.sampling_poses import sample_good_starting_pose


@comptest
def test_pose_sampling_1():
    import duckietown_world as dw

    m = dw.load_map("4way")

    along_lane = 0.2
    only_straight = True
    for i in range(45):
        q = sample_good_starting_pose(
            m, only_straight=only_straight, along_lane=along_lane
        )
        ground_truth = dw.SE2Transform.from_SE2(q)
        m.set_object(f"sampled-{i}", dw.DB18(), ground_truth=ground_truth)

    outdir = get_comptests_output_dir()
    draw_static(m, outdir)


if __name__ == "__main__":
    run_module_tests()
