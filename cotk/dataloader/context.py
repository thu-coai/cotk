
from typing import List, Any, Dict, Union, Optional

from .._utils.metaclass import DocStringInheritor
from .._utils.typehint import OrderedDictType

# For type checking
if False: #pylint: disable=using-constant-test
	#pylint: disable=unused-import
	from .vocab import Vocab
	from .tokenizer import Tokenizer

class Context(metaclass=DocStringInheritor):
	'''A base class for Context. This class is used for setting parameters
	for :class:`Field` or :class:`Vocab`, without directly
	passing parameters to ``__init__`` of the object.
	When initialized, the object write the list of parameters stored in the class.
	The old parameters are restored when :meth:`.close` or :meth:`.__exit__` is called.
	TODO: add an example for context manager, use ``with``.
	Arguments:
		parameter_dict (Dict[str, Any]): Key-value dict for changed parameters.
		weak (bool, optional): Overwrite existing parameters. Default: False.
	'''

	context_dict: Dict[str, Any] = {}
	corrupted = False

	def __init__(self, parameter_dict: Dict[str, Any], weak=False):
		if self.__class__.corrupted:
			raise RuntimeError("A context object do not close before becoming invalid. Use ``with`` statement, " \
				"or make sure of calling close.")

		self._parameter_keys = list(parameter_dict)
		self._old_parameters = self._set_parameters(parameter_dict, weak=weak)
		self._closed = False

	@classmethod
	def _set_parameters(cls, parameter_dict: Dict[str, Any], weak=False) -> List[Any]:
		old_parameters: List[Any] = []
		for key, value in parameter_dict.items():
			old_parameters.append(cls.set(key, value, weak=weak))
		return old_parameters

	@staticmethod
	def _prune_parameter(parameter_keys: List[str], parameter_dict: Dict[str, Any]) -> Dict[str, Any]:
		return {key : parameter_dict[key] for key in parameter_keys}

	@classmethod
	def get(cls, key: str, default: Any = None, no_default=False) -> Any:
		'''Get the value of parameter named ``key`` stored in this class.
		If ``key`` is not set, return ``default``.
		When ``no_default`` is ``True``, raise a KeyError if ``key`` is not set.
		Arguments:
			key (str): name of the parameter
			default (Any, optional): Default value if ``key`` is not set. Defaults: None.
			no_default (bool, optional): When ``True``, Raise KeyError if ``key`` is not set. Defaults: False.
		'''
		if cls.context_dict[key]:
			return cls.context_dict[key]
		else:
			if no_default:
				raise KeyError("Must specify %s in Context." % key)
			else:
				return default

	@classmethod
	def set(cls, key: str, value: Any, weak=False) -> Any:
		'''Set the parameter named ``key`` to ``value``, stored in this class.
		If weak is ``True``, do not overwrite if ``key`` is already set.
		Return the old value.
		Arguments:
			key (str): The name of the changed parameter.
			value (Any): The new value of changed parameter. If None, do nothing.
				If want to set the value to None, use ``"force_none"``.
			weak (bool, optional): Whether overwrite it if the parameter existing. Defaults: False.
		'''
		old = cls.context_dict[key]
		if weak:
			if old is None:
				cls.context_dict[key] = value
		else:
			if value == "force_none":
				cls.context_dict[key] = None
			elif value is not None:
				cls.context_dict[key] = value
		return old

	def __enter__(self):
		'''Enter a context'''
		return self

	@classmethod
	def _restore(cls, parameter_keys, old_parameters):
		for name, param in zip(parameter_keys, old_parameters):
			cls.context_dict[name] = param

	def __exit__(self, exc_type, exc_val, exc_tb):
		'''Exit the context and restore the old parameter.'''
		self.close()

	def close(self):
		'''Restore the old parameter.'''
		self._restore(self._parameter_keys, self._old_parameters)
		self._closed = True

	def __del__(self):
		if hasattr(self, "_closed") and not self._closed:
			self.__class__.corrupted = True
			raise RuntimeError("A context object do not close before becoming invalid. Use ``with`` statement, " \
				"or make sure of calling close.")

class FieldContext(Context):
	'''Bases: :class:`.dataloader.Context`
	A context class for setting parameters for :class:`.Field`.
	'''

	PARAMETER_LIST = ["tokenizer", "vocab", "vocab_from", "max_sent_length", "max_turn_length", "convert_to_lower_letter"]
	context_dict = {key: None for key in PARAMETER_LIST}
	corrupted = False

	# pylint: disable=unused-argument
	@classmethod
	def set_parameters(cls, *,
			tokenizer: Union[None, "Tokenizer", str] = None,
			vocab: Optional["Vocab"] = None,
			vocab_from: Optional[Dict[str, str]] = None,
			max_sent_length: Optional[int] = None,
			max_turn_length: Optional[int] = None,
			convert_to_lower_letter: Optional[bool] = None,
			weak=False) -> "FieldContext":
		'''Set a Context for initialization of :class:`Field`.
		See the example at TODO: write an example for how to use field context.
		Arguments:
			TODO: fill the parameters from Field classes.
		'''
		parameter_dict = cls._prune_parameter(cls.PARAMETER_LIST, locals())
		return FieldContext(parameter_dict, weak=weak)

class VocabContext(Context):
	'''Bases: :class:`.dataloader.Context`
	A context class for setting parameters for :class:`.Vocab`.
	'''

	PARAMETER_LIST = ["min_frequent_vocab_times", "min_rare_vocab_times", "special_tokens_mapping", "special_appeared_in_data"]
	context_dict = {key: None for key in PARAMETER_LIST}
	corrupted = False

	# pylint: disable=unused-argument
	@classmethod
	def set_parameters(cls, *, \
			min_frequent_vocab_times: Optional[int] = None, \
			min_rare_vocab_times: Optional[int] = None, \
			special_tokens_mapping: Optional[OrderedDictType[str, str]] = None, \
			special_appeared_in_data: Optional[bool] = None, \
			weak=False) -> "VocabContext":
		'''Set a Context for initialization of :class:`Vocab`.
		See the example at TODO: write an example for how to use field context.
		Arguments:
			TODO: fill the parameters from Vocab classes.
		'''
		parameter_dict = cls._prune_parameter(cls.PARAMETER_LIST, locals())
		return VocabContext(parameter_dict, weak=weak)