'''
A module for dataloader
'''
class Dataloader():
	'''Base class of Dataloader.
	'''
	def __init__(self):
		pass

	@classmethod
	def get_all_subclasses(cls):
		'''Return a generator of all subclasses.
		'''
		for subclass in cls.__subclasses__():
			yield from subclass.get_all_subclasses()
			yield subclass

	@classmethod
	def load_class(cls, class_name):
		'''Return a subclass of `class_name`.

		Arguments:
			class_name (str): target class name.
		'''
		for subclass in cls.get_all_subclasses():
			if subclass.__name__ == class_name:
				return subclass
		return None
