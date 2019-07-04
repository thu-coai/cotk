"""A lib for decorator and metaclass"""
import re

class DocStringInheritor(type):
	"""
	A variation on
	http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
	by Paul McGuire
	from https://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
	"""
	def __new__(cls, name, bases, clsdict):

		def replace_for_clsdict(matched):
			return clsdict[matched.group(1)]
		def replace_for(obj):
			def replace(matched):
				#return obj.__getattr__(matched.group(1))
				return getattr(obj, matched.group(1))
			return replace

		if not('__doc__' in clsdict and clsdict['__doc__']):
			for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
				doc = mro_cls.__doc__
				if doc:
					clsdict['__doc__'] = doc
					break
		else:
			doc = re.sub(r'\{([A-Z]+?)\}', replace_for_clsdict, clsdict['__doc__'])
			if doc != clsdict['__doc__']:
				clsdict['__doc__'] = doc

		for attr, attribute in clsdict.items():
			if not attribute.__doc__:
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
				doc = re.sub(r'\{([A-Z]+?)\}', replace_for(attribute), attribute.__doc__)
				if doc == attribute.__doc__:
					continue
				if isinstance(attribute, property):
					clsdict[attr] = property(attribute.fget, attribute.fset, \
											 attribute.fdel, doc)
				else:
					#print(attr, attribute, doc)
					attribute.__doc__ = doc
		return type.__new__(cls, name, bases, clsdict)

class LoadClassInterface:
	r"""Implement two classmethods"""
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
