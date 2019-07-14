r"""
``cotk._utils.imports`` provides classes that fake a uninstalled module.
"""
class DummyObject(dict):
	r'''Dummy Object for uninstalled modules.
	Examples:
		>>> try:
		...   import torch
		... except ImportError as err:
		...   torch = DummyObject(err)
		...
		>>> torch.Tensor
		ModuleNotFoundError: No module named 'torch'
		>>> torch.Tensor = DummyObject(torch.err)
		>>> torch.Tensor
		>>> torch.Tensor()
		ModuleNotFoundError: No module named 'torch'
	'''
	def __init__(self, err):
		self.err = err
		super().__init__()

	def __getattr__(self, key):
		if key in self:
			return self[key]
		elif key == "__bases__":
			return tuple()
		else:
			raise self["err"]

	def __setattr__(self, key, value):
		self[key] = value

	def __delattr__(self, key):
		del self[key]

	def __call__(self, *args, **kwargs):
		raise self['err']
