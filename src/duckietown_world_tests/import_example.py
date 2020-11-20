import yaml
from duckietown_world.yaml_include import YamlIncludeConstructor

def make_map(map_dir_name):
    import logging
    logging.basicConfig()
    logger = logging.getLogger("dt-world")

    logger.info("loading map from %s" % map_dir_name)

    import os
    abs_path_module = os.path.realpath(__file__)
    logger.info("abs_path_module: " + str(abs_path_module))
    module_dir = os.path.dirname(abs_path_module)
    logger.info("module_dir: " + str(module_dir))
    map_dir = os.path.join(module_dir, "../duckietown_world/data/gd2", map_dir_name)
    logger.info("map_dir: " + str(map_dir))
    assert os.path.exists(map_dir), map_dir

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=map_dir)

    yaml_data_layers = {}
    for layer in os.listdir(map_dir):
        fn = os.path.join(map_dir, layer)
        with open(fn) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        yaml_data_layers[layer] = yaml_data

    yaml_data = yaml_data_layers["main.yaml"]
    from pprint import pprint
    pprint(yaml_data)


if __name__ == "__main__":
    make_map("import_example")
