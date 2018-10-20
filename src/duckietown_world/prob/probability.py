# # coding=utf-8
#
# class Distribution(object):
#     __metaclass__ = ABCMeta
#
#     @abstractmethod
#     def mle(self):
#         pass
#
#     @abstractmethod
#     def samples(self, n):
#         pass
#
#
# class Singleton(Distribution):
#     def __init__(self, only):
#         self.only = only
#
#     def mle(self):
#         return self.only
#
#     def samples(self, n):
#         return [self.only]
#
#     def as_json_dict(self):
#         x = {'only': self.only.as_json_dict()}
#         return {'Singleton': x}
#
#     @classmethod
#     def params_from_json_dict(cls, d):
#         x = d['Singleton']
#         return dict(only=from_json_dict(x['only']))
