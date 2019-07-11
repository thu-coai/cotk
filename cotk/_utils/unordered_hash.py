'''
A module for hash unordered elements
'''

import hashlib

import numpy as np


class UnorderedSha256:
	'''
		Using SHA256 on unordered elements
	'''
	def __init__(self):
		self.result = np.array([0] * 32, dtype=np.uint8)

	def update_data(self, data):
		'''update digest by data. type(data)=bytes'''
		digest = hashlib.sha256(data).digest()
		self.update_hash(digest)

	def update_hash(self, hashvalue):
		'''update digest by hash. type(hashvalue)=bytes'''
		self.result += np.array(list(hashvalue), dtype=np.uint8)

	def digest(self):
		'''return unordered hashvalue'''
		return bytes(self.result.tolist()).hex()
