'''
A module for hash unordered elements
'''

from typing import Union
from collections import OrderedDict
import hashlib
import json
import warnings


class UnorderedSha256:
	'''
		Using SHA256 on unordered elements
	'''

	def __init__(self):
		self.result = [0] * 32

	def update_data(self, data: Union[bytes, bytearray, memoryview]):
		'''update digest by data. type(data)=bytes'''
		digest = hashlib.sha256(data).digest()
		self.update_hash(digest)

	def update_hash(self, hashvalue):
		'''update digest by hash. type(hashvalue)=bytes'''
		for i, bit in enumerate(list(hashvalue)):
			self.result[i] = (self.result[i] + bit) & 0xFF

	def digest(self) -> bytes:
		'''return unordered hashvalue'''
		return bytes(self.result)

	def hexdigest(self) -> str:
		'''return unordered hashvalue'''
		return bytes(self.result).hex()


def dumps_json(obj) -> bytes:
	'''Generate bytes to identify the object by json serialization'''
	if isinstance(obj, (str, int, float, bool)):
		return str(obj).encode('utf-8')
	return json.dumps(obj, sort_keys=True).encode('utf-8')


def dumps(obj) -> bytes:
	'''Generate bytes to identify the object by repr'''
	return simple_dumps(convert_obj(obj))


def simple_dumps(obj) -> bytes:
	return repr(obj).encode('utf-8')


def convert_obj(obj):
	if isinstance(obj, OrderedDict):
		return convert_ordered_dict(obj)
	for cls, func in special_type_processing_functions.items():
		if isinstance(obj, cls):
			return func(obj)
	if not isinstance(obj, common_types):
		warnings.warn("It's unsupported to dumps a %s object. The result may not be expected." % type(obj).__name__)
	return obj


def convert_dict(obj):
	return type(obj), [(convert_obj(k), convert_obj(v)) for k, v in sorted(obj.items())]


def convert_ordered_dict(obj):
	return type(obj), [(convert_obj(k), convert_obj(v)) for k, v in obj.items()]


def convert_ordered_iterable(obj):
	return type(obj), [convert_obj(item) for item in obj]


def convert_unordered_iterable(obj):
	# Elements in a set or a frozenset is unordered. Sort them before dumps.
	return type(obj), [convert_obj(item) for item in sorted(obj)]


special_type_processing_functions = {
	tuple: convert_ordered_iterable,
	list: convert_ordered_iterable,
	set: convert_unordered_iterable,
	frozenset: convert_unordered_iterable,
	dict: convert_dict,
	OrderedDict: convert_ordered_dict
}
common_types = (str, int, float, bytes, bytearray, bool, type, type(None))
