class WordVector:
	r'''Base of all word vector loader
	'''
	def __init__(self):
		pass

	def load(self, n_dims, vocab_list):
		'''Load word vector and return a numpy array. The ith row is the feature
		of the ith word in `vocab_list`.

		Parameters:
			n_dims (int): specify the dimension size of word vector. If `n_dims`
				is bigger than pretrained word vector, the rest feature will be
				randomly initialized by normal distriution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				be randomly initialized by normal distribution
		'''
		raise NotImplementedError("WordVector.load is a virtual function.")


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
