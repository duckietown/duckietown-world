# coding=utf-8
from duckietown_serialization_ds1.serialization1 import as_json_dict
from duckietown_world.geo import PlacedObject

__all__ = ["TrafficLight"]


# TrafficLightStatus = str


class TrafficLight(PlacedObject):
    # status: GenericSequence[TrafficLightStatus]

    def __init__(self, status=None, **kwargs):
        # if status is None:
        #     status = Constant[TrafficLightStatus]("off")
        PlacedObject.__init__(self, **kwargs)
        self.status = status

    def params_to_json_dict(self):
        return dict(status=as_json_dict(self.status))
