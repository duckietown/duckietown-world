# coding=utf-8
from setuptools import find_packages, setup


def get_version(filename):
    import ast
    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError('No version found in %r.' % filename)
    if version is None:
        raise ValueError(filename)
    return version


shell_version = get_version(filename='src/duckietown_world/__init__.py')

setup(name='duckietown-world',

      version=shell_version,
      download_url='http://github.com/duckietown/duckietown-world/tarball/%s' % shell_version,
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=[
          'numpy',
          'PyYAML',
          'networkx>=2.2,<3',
          'duckietown-serialization-ds1>=1,<2',
          'svgwrite',
          'PyGeometry>=1.5.6',
          'beautifulsoup4>=4.6.3',
          'lxml',
          'pillow',
          'future',
          'scipy',
          'plotly',
          'oyaml',
          'markdown',
      ],

      tests_require=[
          'comptests',
      ],

      # This avoids creating the egg file, which is a zip file, which makes our data
      # inaccessible by dir_from_package_name()
      zip_safe=False,

      # without this, the stuff is included but not installed
      include_package_data=True,

      entry_points={
          'console_scripts': [
              'dt-world-draw-log = duckietown_world.svg_drawing:draw_logs_main',
              'dt-world-draw-maps = duckietown_world.svg_drawing:draw_maps_main',
          ]
      }
      )
