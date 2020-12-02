from abc import ABCMeta, abstractmethod


class IBaseMap(metaclass=ABCMeta):

    @abstractmethod
    def get_object_name(self, obj):
        pass

    @abstractmethod
    def get_object_frame(self, obj):
        pass

    @abstractmethod
    def get_object(self, name, obj_type):
        pass

    @abstractmethod
    def get_layer_objects(self, obj_type):
        pass
