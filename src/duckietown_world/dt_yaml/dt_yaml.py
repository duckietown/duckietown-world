from typing import Any, Dict, TextIO
import yaml

from .constructor import DTYamlConstructor
from .representer import DTYamlRepresenter


class DTYaml:
    YAML_LOADER = yaml.FullLoader
    constructor = DTYamlConstructor.add_to_loader_class(loader_class=YAML_LOADER)
    DTYamlRepresenter.add_representers()

    @classmethod
    def load(cls, map_path: str, main_layer_file_stream: TextIO) -> Dict[str, Dict[str, Dict[str, Any]]]:
        cls.constructor.base_dir = map_path
        return yaml.load(main_layer_file_stream, Loader=cls.YAML_LOADER)

    @classmethod
    def dump(cls, layer_dict: Dict[str, Any]) -> str:
        return yaml.dump(layer_dict)
