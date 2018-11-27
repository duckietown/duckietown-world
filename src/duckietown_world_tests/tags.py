# coding=utf-8
import yaml
from comptests import comptest, run_module_tests, get_comptests_output_dir

# language=yaml
from duckietown_world import construct_map, draw_static, get_object_tree

map_yaml = """
objects:
- kind: sign_right_T_intersect
  attach:
    tile: [0, 0]
    slot: 2
    
    
tiles:
- [ asphalt, straight/S, asphalt]
- [ straight/E, 4way, straight/W]
- [ asphalt, straight/N, asphal]  
- [ asphalt, curve_right/W, straight/W]
"""




@comptest
def tag_positions():
    map_yaml_data = yaml.load(map_yaml)
    m = construct_map(map_yaml_data, tile_size=0.61)
    print(get_object_tree(m, attributes=True))
    outdir = get_comptests_output_dir()
    draw_static(m, outdir)


if __name__ == '__main__':
    run_module_tests()
