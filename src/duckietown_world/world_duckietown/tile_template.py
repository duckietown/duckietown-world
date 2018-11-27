# coding=utf-8
import yaml

from duckietown_serialization_ds1 import Serializable


from duckietown_world.utils.memoizing import memoized_reset

# language=yaml
data = """

# lane = 22cm
# tile = 58.5
# lane_rel = 22/58.5
go_right: &go_right
    ~LaneSegment:
      width: &width 0.376
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.22]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.30, -0.30]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.22, -0.50]
            theta_deg: -90

go_straight: &go_straight
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.22]
            theta_deg: 0
        - ~SE2Transform:
            p: [+0.5, -0.22]
            theta_deg: 0

go_left: &go_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.22]
            theta_deg: 0 

        - ~SE2Transform:
            p: [0.0, 0.0]
            theta_deg: 45
 
        - ~SE2Transform:
            p: [+0.22, +0.50]
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
            curve: *curve_right
        spatial_relations: 
            curve: {~SE2Transform: {theta_deg: 270}}
             

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
            west_go_straight: *go_straight
            west_go_left: *go_left
            north_go_left: *go_left
            north_go_right: *go_right
            east_go_right: *go_right
            east_go_straight: *go_straight
        spatial_relations:
            west_go_straight: {~SE2Transform: {theta_deg: 0}}
            west_go_left: {~SE2Transform: {theta_deg: 0}}
            north_go_left: {~SE2Transform: {theta_deg: -90}}
            north_go_right: {~SE2Transform: {theta_deg: -90}}
            east_go_right: {~SE2Transform: {theta_deg: 180}}
            east_go_straight: {~SE2Transform: {theta_deg: 180}}

3way_right:
    ~PlacedObject:  
        children:
            template: *3way_left           
        spatial_relations:
            template: {~SE2Transform: {theta_deg: 180}}   
            
 

"""


@memoized_reset
def load_tile_types():
    s = yaml.load(data)
    templates = Serializable.from_json_dict(s)
    return templates
