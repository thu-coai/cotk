'''
A module for hash unordered elements
'''

from typing import Union
import hashlib
import json

import numpy as np


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
	return repr(obj).encode('utf-8')
