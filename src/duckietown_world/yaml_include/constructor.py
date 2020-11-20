# -*- coding: utf-8 -*-

"""
Include YAML files within YAML
"""

import os.path
import re
from glob import iglob
from sys import version_info

import yaml
from ast import literal_eval
import re

from .readers import get_reader_class_by_name, get_reader_class_by_path

__all__ = ['YamlIncludeConstructor']

PYTHON_MAYOR_MINOR = '{0[0]}.{0[1]}'.format(version_info)

WILDCARDS_REGEX = re.compile(r'^.*(\*|\?|\[!?.+\]).*$')


class YamlIncludeConstructor:
    """The `include constructor` for PyYAML Loaders

    Call :meth:`add_to_loader_class` or :meth:`yaml.Loader.add_constructor` to add it into loader.

    In YAML files, use ``!include`` to load other YAML files as below::

        !include [dir/**/*.yml, true]

    or::

        !include {pathname: dir/abc.yml, encoding: utf-8}

    """

    DEFAULT_ENCODING = 'utf-8'
    DEFAULT_TAG_NAME = '!include'

    def __init__(self, base_dir=None, encoding=None, reader_map=None, **_):
        # type:(str, str)->YamlIncludeConstructor
        """
        :param str base_dir: Base directory where search including YAML files

            :default: ``None``:  include YAML files from current working directory.

        :param str encoding: Encoding of the YAML files

            :default: ``None``:  Not specified

        :param dict reader_map: A dictionary of `{path-pattern : reader-class}`

            :default: ``None``: set :data:`readers.READER_TABLE` as default readers map
        """
        self._base_dir = base_dir
        self._encoding = encoding
        self._reader_map = reader_map

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

    @property
    def base_dir(self):  # type: ()->str
        """Base directory where search including YAML files

        :rtype: str
        """
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value):  # type: (str)->None
        self._base_dir = value

    @property
    def encoding(self):  # type: ()->str
        """Encoding of the YAML files

        :rtype: str
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):  # type: (str)->None
        self._encoding = value

    def load(self, loader, pathname, recursive=False, encoding=None, reader=None, **kwargs):
        """Once add the constructor to PyYAML loader class,
        Loader will use this function to include other YAML fils
        on parsing ``"!include"`` tag

        :param loader: Instance of PyYAML's loader class
        :param str pathname: pathname can be either absolute (like `/usr/src/Python-1.5/*.yml`) or relative (like `../../Tools/*/*.yml`), and can contain shell-style wildcards

        :param bool recursive: If recursive is true, the pattern ``"**"`` will match any files and zero or more directories and subdirectories. If the pattern is followed by an os.sep, only directories and subdirectories match.

            .. note:: Using the ``"**"`` pattern in large directory trees may consume an inordinate amount of time.

        :param str encoding: YAML file encoding

            :default: ``None``: Attribute :attr:`encoding` or constant :attr:`DEFAULT_ENCODING` will be used to open it

        :param str reader: name of the reader for loading files

            it's typically one of:

            - `ini`
            - `json`
            - `yaml`
            - `toml`
            - `txt`

            if not specified, reader would be decided by `reader_map` parameter passed in constructor

        :return: included YAML file, in Python data type

        .. warning:: It's called by :mod:`yaml`. Do NOT call it yourself.
        """
        if not encoding:
            encoding = self._encoding or self.DEFAULT_ENCODING
        if self._base_dir:
            pathname = os.path.join(self._base_dir, pathname)
        reader_clz = None
        if reader:
            reader_clz = get_reader_class_by_name(reader)
        if re.match(WILDCARDS_REGEX, pathname):
            result = []
            if PYTHON_MAYOR_MINOR >= '3.5':
                iterable = iglob(pathname, recursive=recursive)
            else:
                iterable = iglob(pathname)
            for path in filter(os.path.isfile, iterable):
                if reader_clz:
                    result.append(reader_clz(path, encoding=encoding, loader_class=type(loader))())
                else:
                    result.append(self._read_file(path, loader, encoding))
            return result
        if reader_clz:
            return reader_clz(pathname, encoding=encoding, loader_class=type(loader))()
        file_data = self._read_file(pathname, loader, encoding)
        if 'only' in kwargs:
            only_file_dict = {}
            root = list(file_data)[0]
            for o in kwargs['only']:
                only_file_dict.update(self._get_from_url(file_data[root], o))
            file_data = {root: only_file_dict}
        if 'except' in kwargs:
            for e in kwargs['except']:
                self._del_from_url(file_data, e)
        return file_data

    def _get_from_url(self, file_dict, url):
        names = url.split('/')
        val = file_dict
        for n in names:
            try:
                tup = literal_eval(n)
                val = val[tup]
            except ValueError:
                val = val[n]
        return self._make_dict(names, val)

    def _make_dict(self, names, val):
        try:
            n = literal_eval(names[0])
        except ValueError:
            n = names[0]
        if len(names) == 1:
            return {n: val}
        return {n: self._make_dict(names[1:], val)}

    def _del_from_url(self, file_dict, url):
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

    def _read_file(self, path, loader, encoding):
        reader_clz = get_reader_class_by_path(path, self._reader_map)
        reader_obj = reader_clz(path, encoding=encoding, loader_class=type(loader))
        return reader_obj()

    @classmethod
    def _add_map_parser(cls):
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

        yaml.add_constructor(u'tag:yaml.org,2002:map', construct_yaml_map)

    @classmethod
    def _add_tuple_parser(cls):
        import re

        def yml_tuple_constructor(loader, node):
            value = loader.construct_scalar(node)
            return literal_eval(value)

        FLOAT_REGEX = r"([-+]?\d*\.\d+|\d+)"
        INNER_TUPLE_REGEX = r"\(\s*%s+(\s*,\s*%s+)+\s*\)" % (FLOAT_REGEX, FLOAT_REGEX)
        TUPLE_REGEX = r"\(\s*(%s|%s+)(\s*,\s*%s+)+\s*\)" % (INNER_TUPLE_REGEX, FLOAT_REGEX, FLOAT_REGEX)

        yaml.add_constructor(u'!tuple', yml_tuple_constructor)
        yaml.add_implicit_resolver(u'!tuple', re.compile(TUPLE_REGEX))

    @classmethod
    def add_to_loader_class(cls, loader_class=None, tag=None, **kwargs):
        # type: (type(yaml.Loader), str, **str)-> YamlIncludeConstructor
        """
        Create an instance of the constructor, and add it to the YAML `Loader` class

        :param loader_class: The `Loader` class add constructor to.

            .. attention:: This parameter **SHOULD** be a **class type**, **NOT** an object.

            It's one of followings:

                - :class:`yaml.BaseLoader`
                - :class:`yaml.UnSafeLoader`
                - :class:`yaml.SafeLoader`
                - :class:`yaml.Loader`
                - :class:`yaml.FullLoader`
                - :class:`yaml.CBaseLoader`
                - :class:`yaml.CUnSafeLoader`
                - :class:`yaml.CSafeLoader`
                - :class:`yaml.CLoader`
                - :class:`yaml.CFullLoader`

            :default: ``None``:

                - When :mod:`pyyaml` `3.*`: :class:`yaml.Loader`
                - When :mod:`pyyaml` `5.*`: :class:`yaml.FullLoader`

        :type loader_class: type

        :param str tag: Tag's name of the include constructor.

          :default: ``""``: Use :attr:`DEFAULT_TAG_NAME` as tag name.

        :param kwargs: Arguments passed to construct function

        :return: New created object
        :rtype: YamlIncludeConstructor
        """
        if tag is None:
            tag = ''
        tag = tag.strip()
        if not tag:
            tag = cls.DEFAULT_TAG_NAME
        if not tag.startswith('!'):
            raise ValueError('`tag` argument should start with character "!"')
        if loader_class is None:
            if yaml.__version__ >= '5.0':
                loader_class = yaml.FullLoader
            else:
                loader_class = yaml.Loader
        instance = cls(**kwargs)
        yaml.add_constructor(tag, instance, loader_class)

        cls._add_tuple_parser()
        cls._add_map_parser()

        return instance
