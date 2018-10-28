import yaml

from duckietown_serialization_ds1 import Serializable

# __all__ = [
#     'TileTemplate',
# ]
#
#
# class TileTemplate(PlacedObject):
#     '''
#
#         Describes a modular element of the city.
#
#
#     '''
#
#     # def __init__(self):
#     #     pass
#
#     @contract(returns='list(f)')
#     def get_lane_segments(self):
#         """
#             Returns a list of LaneSegments that describe
#             the connectivity of this tile.
#
#         """
#         pass


# language=yaml
data = """


go_right: &go_right
    ~LaneSegment:
      width: &width 0.47 
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.25]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.30, -0.30]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.25, -0.50]
            theta_deg: -90

go_straight: &go_straight
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.25]
            theta_deg: 0
        - ~SE2Transform:
            p: [+0.5, -0.25]
            theta_deg: 0

go_left: &go_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.25]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.0, -0.25]
            theta_deg: 0

        - ~SE2Transform:
            p: [+0.25, 0.0]
            theta_deg: 90
        - ~SE2Transform:
            p: [+0.25, +0.50]
            theta_deg: 90 

            
straight:
    ~PlacedObject:
        children:
            lane1: *go_straight
            lane2: *go_straight
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform: {theta_deg: 180}}

curve_right: &curve_right
    ~PlacedObject:
        children:
            lane1: *go_right
            lane2: *go_left
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform: {theta_deg: 90}}

curve_left: &curve_left
    ~PlacedObject:
        children:
            template: *curve_right
        spatial_relations: 
            template: {~SE2Transform: {theta_deg: 270}}
             

1way: &1way
     ~PlacedObject:
        children:
            go_right: *go_right
            go_left: *go_left
            go_straight: *go_straight
            
4way:
    ~PlacedObject:
        children:
            a: *1way
            b: *1way
            c: *1way
            d: *1way
        spatial_relations:
            a: {~SE2Transform: {theta_deg: 0}}
            b: {~SE2Transform: {theta_deg: 90}}
            c: {~SE2Transform: {theta_deg: 180}}
            d: {~SE2Transform: {theta_deg: 270}}
            
#    |    |        
# ---      ----
#  
# -----------

3way_left: &3way_left
    ~PlacedObject:
        children:
            l1: *go_straight
            l2: *go_left
            top1: *go_left
            top2: *go_right
            r1: *go_right
            r2: *go_straight
        spatial_relations:
            l1: {~SE2Transform: {theta_deg: 0}}
            l2: {~SE2Transform: {theta_deg: 0}}
            top1: {~SE2Transform: {theta_deg: -90}}
            top2: {~SE2Transform: {theta_deg: -90}}
            r1: {~SE2Transform: {theta_deg: 180}}
            r2: {~SE2Transform: {theta_deg: 180}}

3way_right:
    ~PlacedObject:  
        children:
            template: *3way_left           
        spatial_relations:
            template: {~SE2Transform: {theta_deg: 180}}   
            
 

"""


def load_tile_types():
    s = yaml.load(data)
    templates = Serializable.from_json_dict(s)
    return templates
    # 'straight':
    # if kind.startswith('straight'):
    #     pts = np.array([
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, -0.25],
    #             [-0.20, 0, 0.25],
    #             [-0.20, 0, 0.50],
    #         ],
    #         [
    #             [0.20, 0, 0.50],
    #             [0.20, 0, 0.25],
    #             [0.20, 0, -0.25],
    #             [0.20, 0, -0.50],
    #         ]
    #     ]) * ROAD_TILE_SIZE
    #
    # elif kind == 'curve_left':
    #     pts = np.array([
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, 0.00],
    #             [0.00, 0, 0.20],
    #             [0.50, 0, 0.20],
    #         ],
    #         [
    #             [0.20, 0, -0.50],
    #             [0.20, 0, -0.30],
    #             [0.30, 0, -0.20],
    #             [0.50, 0, -0.20],
    #         ]
    #     ]) * ROAD_TILE_SIZE
    #
    # elif kind == 'curve_right':
    #     pts = np.array([
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, -0.20],
    #             [-0.30, 0, -0.20],
    #             [-0.50, 0, -0.20],
    #         ],
    #
    #         [
    #             [-0.50, 0, 0.20],
    #             [-0.30, 0, 0.20],
    #             [0.30, 0, 0.00],
    #             [0.20, 0, -0.50],
    #         ]
    #     ]) * ROAD_TILE_SIZE
    #
    # # Hardcoded all curves for 3way intersection
    # elif kind.startswith('3way'):
    #     pts = np.array([
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, -0.25],
    #             [-0.20, 0, 0.25],
    #             [-0.20, 0, 0.50],
    #         ],
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, 0.00],
    #             [0.00, 0, 0.20],
    #             [0.50, 0, 0.20],
    #         ],
    #         [
    #             [0.20, 0, 0.50],
    #             [0.20, 0, 0.25],
    #             [0.20, 0, -0.25],
    #             [0.20, 0, -0.50],
    #         ],
    #         [
    #             [0.50, 0, -0.20],
    #             [0.30, 0, -0.20],
    #             [0.20, 0, -0.20],
    #             [0.20, 0, -0.50],
    #         ],
    #         [
    #             [0.20, 0, 0.50],
    #             [0.20, 0, 0.20],
    #             [0.30, 0, 0.20],
    #             [0.50, 0, 0.20],
    #         ],
    #         [
    #             [0.50, 0, -0.20],
    #             [0.30, 0, -0.20],
    #             [-0.20, 0, 0.00],
    #             [-0.20, 0, 0.50],
    #         ],
    #     ]) * ROAD_TILE_SIZE
    #
    # # Template for each side of 4way intersection
    # elif kind.startswith('4way'):
    #     pts = np.array([
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, 0.00],
    #             [0.00, 0, 0.20],
    #             [0.50, 0, 0.20],
    #         ],
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, -0.25],
    #             [-0.20, 0, 0.25],
    #             [-0.20, 0, 0.50],
    #         ],
    #         [
    #             [-0.20, 0, -0.50],
    #             [-0.20, 0, -0.20],
    #             [-0.30, 0, -0.20],
    #             [-0.50, 0, -0.20],
    #         ]
    #     ]) * ROAD_TILE_SIZE
    # else:
    #     assert False, kind
