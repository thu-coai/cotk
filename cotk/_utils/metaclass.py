"""A lib for decorator and metaclass"""
import re
import inspect

class DocStringInheritor(type):
	"""
	A meta class. It make the class:

	* Docstring can inherit the parent class.
	* {STRING} in docs will be replaced by self.STRING
	* {BaseClassName.STRING} in docs will be replaced by BaseClassName.STRING

	A variation on
	http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
	by Paul McGuire
	from https://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
	"""
	def __new__(cls, name, bases, clsdict):

		def find_base(base_name):
			for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
				if mro_cls.__name__ == base_name:
					return mro_cls
			raise ValueError("No bases named %s" % base_name)

		def find_attr(attr_name):
			if "." in attr_name:
				base_name, attr_name = attr_name.split(".")
				base = find_base(base_name)
				return getattr(base, attr_name)
			else:
				return clsdict[attr_name]

		def replace_for_clsdict(matched):
			attr_name = matched.group(1)
			try:
				return find_attr(attr_name)
			except ValueError as err:
				if err.args[0].startswith("No bases"):
					raise ValueError("Can't find %s when interpreting docstring of class %s, becausethe class doesn't have a baseclass named %s." \
						% (attr_name, name, attr_name.split(".")[0]))
				else:
					raise
			except (AttributeError, KeyError):
				raise ValueError("Can't find %s when interpreting docstring of class %s, please check whether the CONSTANT exists." \
					% (attr_name, name))

		def replace_for(attr):
			def replace(matched):
				attr_name = matched.group(1)
				try:
					return find_attr(attr_name)
				except ValueError as err:
					if err.args[0].startswith("No bases"):
						raise ValueError("Can't find %s when interpreting docstring of %s.%s, because the class doesn't have a baseclass named %s." \
							% (attr_name, name, attr, attr_name.split(".")[0]))
					else:
						raise
				except (AttributeError, KeyError):
					raise ValueError("Can't find %s when interpreting docstring of %s.%s, please check whether the CONSTANT exists." \
						% (attr_name, name, attr))
			return replace

		# modify class docstring
		if not('__doc__' in clsdict and clsdict['__doc__']):
			# first inherit docstring from bases
			for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
				# iterate from bases in MRO
				doc = mro_cls.__doc__
				if doc:
					clsdict['__doc__'] = doc
					break
		else:
			# else do substitution for CONSTANT
			while True:
				doc = re.sub(r'\{\b((\w*\.)?[A-Z_]+?)\}', replace_for_clsdict, clsdict['__doc__'])
				if doc == clsdict['__doc__']:
					break
				clsdict['__doc__'] = doc

		# modify attribute docstring
		for attr, attribute in clsdict.items():
			if not attribute.__doc__:
				# inherit docstring from bases
				for mro_cls in (mro_cls for base in bases for mro_cls in base.mro() \
								if hasattr(mro_cls, attr)):
					doc = getattr(getattr(mro_cls, attr), '__doc__')
					if doc:
						if isinstance(attribute, property):
							clsdict[attr] = property(attribute.fget, attribute.fset, \
													 attribute.fdel, doc)
						else:
							attribute.__doc__ = doc
						break
			else:
				while True:
					# else do substitution for CONSTANT
					doc = re.sub(r'\{\b(((\w*\.)?)[A-Z_]+?)\}', replace_for(attr), attribute.__doc__)
					if doc == attribute.__doc__:
						break

					if isinstance(attribute, property):
						clsdict[attr] = property(attribute.fget, attribute.fset, \
													attribute.fdel, doc)
					else:
						attribute.__doc__ = doc
		return type.__new__(cls, name, bases, clsdict)

class LoadClassInterface:
	r"""The support of dynamic class load."""
	@classmethod
	def get_all_subclasses(cls):
		'''Return a generator of all subclasses.

		Returns:
			(generator) A iterator over all subclasses.
		'''
		for subclass in cls.__subclasses__():
			yield from subclass.get_all_subclasses()
			yield subclass

	@classmethod
	def load_class(cls, class_name):
		'''Return a subclass of ``class_name``.

		Arguments:
			class_name (str): target class name.

		Returns:
			(class) The subclass specified by ``class_name``
		'''
		result = None
		for subclass in cls.get_all_subclasses():
			if subclass.__name__ == class_name:
				if result is None:
					result = subclass
				else:
					raise RuntimeError('There are two classes with the name "{}" located at "{}" and "{}". \
						You have to remove one of them to make "load_class" work normally.'.format(\
						class_name, inspect.getfile(result), inspect.getfile(subclass)))
		return result
