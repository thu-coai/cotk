
from typing import List, Any, Dict, Union, Optional

from .._utils.metaclass import DocStringInheritor

# For type checking
if False: #pylint: disable=using-constant-test
	#pylint: disable=unused-import
	from .vocab import Vocab
	from .tokenizer import Tokenizer
	from collections import OrderedDict

class _UNDEFINED():
	pass

class Context(metaclass=DocStringInheritor):
	'''An abstract base class for context manager.

	This class is used for setting default parameters
	for :class:`Field` or :class:`Vocab`, without directly
	passing parameters to ``__init__`` of the object.

	See :ref:`examples<dataloader_context_ref>` for how to use context manager.

	Arguments:
		parameter_dict (Dict[str, Any]): Key-value dict for changed parameters.
		weak (bool, optional): When ``False``, overwrite existing parameters. Default: ``False``.
		none_as_ignored (bool, optional): When ``True``, ``None`` values in ``parameter_dict`` are ignored.
			Otherwise, the corresponding key will be set to ``None``.
			Default: ``True``.
	'''

	context_dict: Dict[str, Any] = {}
	corrupted = False

	UNDEFINED = _UNDEFINED()

	def __init__(self, parameter_dict: Dict[str, Any], weak=False, none_as_ignored=True):
		if self.__class__.corrupted:
			raise RuntimeError("A context object do not close before becoming invalid. Use ``with`` statement, " \
				"or make sure of calling close.")

		self._old_parameters = self._set_parameters(parameter_dict, weak=weak, none_as_ignored=none_as_ignored)
		self._closed = False

	@classmethod
	def _set_parameters(cls, parameter_dict: Dict[str, Any], weak=False, none_as_ignored=True) -> Dict[str, Any]:
		old_parameters: Dict[str, Any] = {}
		for key, value in parameter_dict.items():
			old_parameters[key] = cls.set(key, value, weak=weak, none_as_ignored=none_as_ignored)
		return old_parameters

	@classmethod
	def get(cls, key: str, default: Any = None, no_default=False) -> Any:
		'''Get the value of parameter named ``key`` stored in this class.

		Arguments:
			key (str): name of the parameter
			default (Any, optional): Default value if ``key`` is not set. Defaults: ``None``.
			no_default (bool, optional): When ``True``, Raise ``KeyError`` if ``key`` is not set. Defaults: ``False``.
		'''
		if key in cls.context_dict:
			return cls.context_dict[key]
		else:
			if no_default:
				raise KeyError("Must specify %s in Context." % key)
			else:
				return default

	WEAK_ARGS = r'''
			weak (bool, optional): When ``False``, overwrite existing parameters. Defaults: ``False``.
	'''
	NONE_AS_IGNORED_ARGS = r'''
			none_as_ignored (bool, optional): When ``True``, ``None`` values in ``parameter_dict`` are ignored.
				Otherwise, the corresponding value will be set to ``None``.
				Default: ``True``.
	'''

	@classmethod
	def set(cls, key: str, value: Any, weak=False, none_as_ignored=True) -> Any:
		'''Set the parameter named ``key`` to ``value``, stored in this class.
		If weak is ``True``, do not overwrite if ``key`` is already set.
		Return the old value.

		Arguments:
			key (str): The name of the changed parameter.
			value (Any): The new value of changed parameter.
				If want to delete the key, use ``Context.UNDEFINED``.
			{WEAK_ARGS}
			{NONE_AS_IGNORED_ARGS}
		'''
		if key not in cls.context_dict:
			old = Context.UNDEFINED
			if value or not none_as_ignored:
				cls.context_dict[key] = value
			return old

		old = cls.context_dict[key]
		if not weak:
			if value is Context.UNDEFINED:
				del cls.context_dict[key]
			elif value is not None or not none_as_ignored:
				cls.context_dict[key] = value
		return old

	def __enter__(self):
		'''Enter a context'''
		return self

	@classmethod
	def _restore(cls, old_parameters):
		for name, param in old_parameters.items():
			if name not in cls.context_dict:
				continue
			if param is Context.UNDEFINED:
				del cls.context_dict[name]
			else:
				cls.context_dict[name] = param

	def __exit__(self, exc_type, exc_val, exc_tb):
		'''Exit the context and restore the old parameter.'''
		self.close()

	def close(self):
		'''Restore the old parameter.'''
		self._restore(self._old_parameters)
		self._closed = True

	def __del__(self):
		if hasattr(self, "_closed") and not self._closed:
			self.__class__.corrupted = True
			raise RuntimeError("A context object do not close before becoming invalid. Use ``with`` statement, " \
				"or make sure of calling close.")

class FieldContext(Context):
	'''Bases: :class:`.dataloader.Context`

	A context class for setting default parameters for :class:`.Field`.
	'''

	context_dict: Dict[str, Any] = {}
	corrupted = False
	UNDEFINED = Context.UNDEFINED

	NONE_AS_IGNORED_ARGS = Context.NONE_AS_IGNORED_ARGS.replace("``parameter_dict``", "``kwargs``")

	# pylint: disable=unused-argument
	@classmethod
	def set_parameters(cls, *, weak=False, none_as_ignored=True, **kwargs) -> "FieldContext":
		'''Set a context for initialization of :class:`Field`.
		See :ref:`examples<dataloader_context_ref>` for how to use context manager.

		Arguments:
			{WEAK_ARGS}
			{NONE_AS_IGNORED_ARGS}
			\*\*kwargs: Any parameters to be set. Set ``key`` to ``FieldContext.UNDEFINED`` to delete a parameter.
		'''
		return FieldContext(kwargs, weak=weak, none_as_ignored=none_as_ignored)

class VocabContext(Context):
	'''Bases: :class:`.dataloader.Context`

	A context class for setting default parameters for :class:`.Vocab`.
	'''

	context_dict: Dict[str, Any] = {}
	corrupted = False
	UNDEFINED = Context.UNDEFINED

	NONE_AS_IGNORED_ARGS = Context.NONE_AS_IGNORED_ARGS.replace("``parameter_dict``", "``kwargs``")

	# pylint: disable=unused-argument
	@classmethod
	def set_parameters(cls, *, weak=False, none_as_ignored=True, **kwargs) -> "VocabContext":
		'''Set a context for initialization of :class:`Vocab`.
		See :ref:`examples<dataloader_context_ref>` for how to use context manager.

		Arguments:
			{WEAK_ARGS}
			{NONE_AS_IGNORED_ARGS}
			\*\*kwargs: Any parameters to be set. Set ``key`` to ``VocabContext.UNDEFINED`` to delete a parameter.
		'''
		return VocabContext(kwargs, weak=weak, none_as_ignored=none_as_ignored)
