"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import json
from utils_ import isJsonable

class Config(object):
	def __init__(self, d=None):
		if d is not None:
			self.assign(d)


	def __str__(self):
		return '\n'.join('%s: %s' % item for item in self.__dict__.items())


	def save(self, path, deleteUnjson=False):
		if deleteUnjson:
			json.dump(self.jsonableFilter(self.__dict__), open(path, 'w'), indent=2)
		else:
			json.dump(self.__dict__, open(path, 'w'), indent=2)


	def load(self, path):
		self.assign(json.load(open(path)))


	def assign(self, valueDict):
		for key in valueDict:
			setattr(self, key, valueDict[key])


	def jsonableFilter(self, d):
		return {k:v for k, v in d.items() if isJsonable(v)}


if __name__ == '__main__':
	print(Config({'a': 1, 'b': 2}))

