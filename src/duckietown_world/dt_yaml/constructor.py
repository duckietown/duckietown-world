import yaml
from ast import literal_eval
import re

from yamlinclude import YamlIncludeConstructor


__all__ = ['DTYamlConstructor']


def yml_tuple_constructor(loader, node):
    value = loader.construct_scalar(node)
    return literal_eval(value)


def get_tuple_regex():
    FLOAT_REGEX = r"([-+]?\d*\.\d+|\d+)"
    INNER_TUPLE_REGEX = r"\(\s*%s+(\s*,\s*%s+)+\s*\)" % (FLOAT_REGEX, FLOAT_REGEX)
    TUPLE_REGEX = r"\(\s*(%s|%s+)(\s*,\s*%s+)+\s*\)" % (INNER_TUPLE_REGEX, FLOAT_REGEX, FLOAT_REGEX)
    return re.compile(TUPLE_REGEX)


def construct_yaml_map(self, node):
    mapping = {}
    yield mapping
    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        val = self.construct_object(value_node, deep=True)
        if key in mapping:
            if isinstance(val, dict) and key in val:
                mapping[key].update(val[key])
            else:
                mapping[key].update(val)
        else:
            if isinstance(val, dict) and key in val:
                mapping[key] = val[key]
            else:
                mapping[key] = val
    return mapping


def _get_from_url(file_dict, url):
    names = url.split('/')
    val = file_dict
    for n in names:
        try:
            tup = literal_eval(n)
            val = val[tup]
        except ValueError:
            val = val[n]
    return _make_dict(names, val)


def _make_dict(names, val):
    try:
        n = literal_eval(names[0])
    except ValueError:
        n = names[0]
    if len(names) == 1:
        return {n: val}
    return {n: _make_dict(names[1:], val)}


def _del_from_url(file_dict, url):
    names = url.split('/')
    root = list(file_dict)[0]
    val = file_dict[root]
    for n in names[:-1]:
        try:
            tup = literal_eval(n)
            val = val[tup]
        except ValueError:
            val = val[n]
    del val[names[-1]]


class DTYamlConstructor(YamlIncludeConstructor):

    def __call__(self, loader, node):
        args = []
        kwargs = {}
        if isinstance(node, yaml.nodes.ScalarNode):
            args = [loader.construct_scalar(node)]
        elif isinstance(node, yaml.nodes.SequenceNode):
            args = loader.construct_sequence(node)
        elif isinstance(node, yaml.nodes.MappingNode):
            kwargs = loader.construct_mapping(node)
        else:
            raise TypeError('Un-supported YAML node {!r}'.format(node))
        if args:
            splt = re.split('only | except', args[0])
            splt = [s.strip() for s in splt]
            args = []
            kwargs['pathname'] = splt[0]
            if len(splt) > 1:
                kwargs['only'] = literal_eval(splt[1])
            if len(splt) > 2:
                kwargs['except'] = literal_eval(splt[2])
        return self.load(loader, *args, **kwargs)

    def load(self, loader, pathname, recursive=False, encoding=None, reader=None, **kwargs):
        file_data = super().load(loader, pathname, recursive, encoding, reader)
        if 'only' in kwargs:
            only_file_dict = {}
            root = list(file_data)[0]
            for o in kwargs['only']:
                only_file_dict.update(_get_from_url(file_data[root], o))
            file_data = {root: only_file_dict}
        if 'except' in kwargs:
            for e in kwargs['except']:
                _del_from_url(file_data, e)
        return file_data

    @classmethod
    def _add_map_parser(cls):
        yaml.add_constructor(u'tag:yaml.org,2002:map', construct_yaml_map)

    @classmethod
    def _add_tuple_parser(cls):
        yaml.add_constructor(u'!tuple', yml_tuple_constructor)
        yaml.add_implicit_resolver(u'!tuple', get_tuple_regex())

    @classmethod
    def add_to_loader_class(cls, loader_class=None, tag=None, **kwargs):
        instance = super().add_to_loader_class(loader_class, tag, **kwargs)
        cls._add_tuple_parser()
        cls._add_map_parser()
        return instance
