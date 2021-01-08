import yaml


__all__ = ['DTYamlRepresenter', 'DTYamlLayer']


def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '~')


def quoted_presenter(dumper, data):
    if len(data.split()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


class DTYamlLayer:
    name: str

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def to_yaml(dumper, data):
        return dumper.represent_scalar("!include", data.name, style=None)


class DTYamlRepresenter:

    @classmethod
    def add_representers(cls):
        yaml.add_representer(type(None), represent_none)
        yaml.add_representer(str, quoted_presenter)
        yaml.add_representer(DTYamlLayer, DTYamlLayer.to_yaml)
