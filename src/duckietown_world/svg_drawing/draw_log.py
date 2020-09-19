# coding=utf-8
from dataclasses import dataclass
from typing import Any, Dict, Optional

from duckietown_world import SampledSequence, SE2Transform
from ..world_duckietown import DuckietownMap

__all__ = [
    "RobotTrajectories",
    "SimulatorLog",
]


@dataclass
class RobotTrajectories:
    pose: SampledSequence[SE2Transform]
    # wheels_velocities: SampledSequence
    # actions: SampledSequence
    velocity: SampledSequence
    observations: SampledSequence
    commands: SampledSequence


@dataclass
class SimulatorLog:
    duckietown: DuckietownMap

    render_time: Optional[Any]
    robots: Dict[str, RobotTrajectories]
