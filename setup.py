# coding=utf-8
import sys

from setuptools import find_packages, setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename="src/duckietown_world/__init__.py")

install_requires = [
    "numpy",
    "PyYAML",
    "networkx>=2.2,<3",
    "duckietown-serialization-ds1<2",
    "svgwrite",
    "PyGeometry-z6>=2.0.4",
    "beautifulsoup4>=4.6.3,<=4.7.1",
    "lxml",
    "pillow",
    "future",
    "scipy",
    "plotly",
    "oyaml",
    "markdown",
    "zuper-typing-z6>=6.0.66",
    "trimesh",
    "gltflib",
    "pyrender",
]
tests_require = ["comptests-z6"]
system_version = tuple(sys.version_info)[:3]

if system_version < (3, 7):
    install_requires.append("dataclasses")

line = "daffy"

setup(
    name=f"duckietown-world-{line}",
    version=version,
    download_url="http://github.com/duckietown/duckietown-world/tarball/%s" % version,
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    tests_require=tests_require,
    # This avoids creating the egg file, which is a zip file, which makes our data
    # inaccessible by dir_from_package_name()
    zip_safe=False,
    # without this, the stuff is included but not installed
    include_package_data=True,
    entry_points={
        "console_scripts": [
            # 'dt-world-draw-log = duckietown_world.svg_drawing:draw_logs_main',
            "dt-world-draw-maps = duckietown_world.svg_drawing:draw_maps_main",
            "dt-world-export-gltf = duckietown_world.gltf:gltf_export_main",
            "dt-make-scenarios =duckietown_world.world_duckietown.sampling:make_scenario_main",
        ]
    },
)
