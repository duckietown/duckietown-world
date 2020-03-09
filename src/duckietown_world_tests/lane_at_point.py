from networkx import connected_components

import duckietown_world as dw
import geometry as geo
from comptests import comptest
from duckietown_world.world_duckietown.segmentify import MeetingPoint
from duckietown_world.world_duckietown.tile_utils import get_tile_at_point
from zuper_typing import debug_print


@comptest
def get_lane_at_point_test():
    m: dw.DuckietownMap = dw.load_map("robotarium2")

    p = geo.SE2_from_translation_angle([1.3, 0.3], 0.2)
    r = get_tile_at_point(m, p)
    assert r.i, r.j == (2, 0)


@comptest
def get_lane_at_point_test2():
    m: dw.DuckietownMap = dw.load_map("robotarium2")

    sk2 = dw.get_skeleton_graph(m)
    G0 = sk2.G0
    c = list(connected_components(G0.to_undirected()))
    assert len(c) == 6

    p = list(G0.nodes)[0]
    mp: MeetingPoint = G0.nodes[p]["meeting_point"]
    print(debug_print(mp))


if __name__ == "__main__":
    get_lane_at_point_test2()
