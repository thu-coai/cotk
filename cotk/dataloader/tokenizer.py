"""A module for Tokenizer"""
import typing
from nltk.tokenize import WordPunctTokenizer
from .._utils.metaclass import DocStringInheritor
from .._utils.imports import LazyObject

PreTrainedTokenizer = LazyObject('transformers.PreTrainedTokenizer')


class BaseTokenizer(metaclass=DocStringInheritor):
	"""Base class of Tokenizer"""

	def tokenize(self, sentence: str, **kwargs) -> typing.List[str]:
		raise NotImplementedError

	@classmethod
	def _check_callable_tokenizer(cls, tokenizer: typing.Callable):
		if not callable(tokenizer):
			raise TypeError('Expected a callable object, but get %s' % type(tokenizer))

		sentence = 'Foo'
		rv = tokenizer(sentence)
		if not isinstance(rv, list):
			raise ValueError('`tokenizer(sentence)` should return a list of str.')


class TokenizerAdapter(BaseTokenizer):
	"""Convert a callable object to a tokenizer"""
	def __init__(self, tokenizer: typing.Callable):
		try:
			if callable(tokenizer):
				self._check_callable_tokenizer(tokenizer)
				self._callable_tokenizer = tokenizer
			else:
				raise TypeError
		except:
			msg = '`tokenizer` should be a callable object, which accepts a string and returns a list of strings.'
			raise TypeError(msg) from None

	def tokenize(self, sentence: str, **kwargs) -> typing.List[str]:
		return self._callable_tokenizer(sentence, **kwargs)


class Tokenizer(BaseTokenizer):
	"""A general Tokenizer.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = """
			tokenizer(object): It's assigned to the attribute `self.tokenizer` and it'll be used to tokenize a sentence.
				If it's a string, `Tokenizer.from_string` is invoked and it's converted to a `BaseTokenizer` instance.
				Otherwise, it must be callable or have a method `tokenize`. The callable object(`tokenizer` itself) or the method
			 	`tokenize` accepts a string and returns a list of strings.
	"""

	ATTRIBUTES = """
			tokenizer(object): It has a method `tokenize`, which converts a sentence(string) to a list of tokens(a list of strings).
	"""
	def __init__(self, tokenizer: typing.Union[BaseTokenizer, typing.Callable, str, typing.Any]):
		"""
		Args:{ARGUMENTS}
		"""
		try:
			if isinstance(tokenizer, BaseTokenizer) or hasattr(tokenizer, 'tokenize'):
				self._check_callable_tokenizer(getattr(tokenizer, 'tokenize'))
				self.tokenizer = tokenizer
			elif callable(tokenizer):
				self.tokenizer = TokenizerAdapter(tokenizer)
			elif isinstance(tokenizer, str):
				self.tokenizer = self.from_string(tokenizer)
			else:
				raise TypeError
		except:
			msg = r"""If `tokenizer` is a string, it must be a valid value :meth:`Tokenizer.from_string` specifies.
			 Otherwise, it must be callable or have method `tokenize`. The callable object(`tokenizer` itself) or the method
			 `tokenize` accepts a string and returns a list of strings.
			"""
			raise TypeError(msg) from None

		# shorten the length of ref-chain
		if isinstance(self.tokenizer, __class__):
			self.tokenizer = self.tokenizer.tokenizer

	def tokenize(self, sentence: str, **kwargs) -> typing.List[str]:
		return self.tokenizer.tokenize(sentence, **kwargs)

	def is_tokenizer_pretrained(self):
		try:
			return PreTrainedTokenizer.__instancecheck__(self.tokenizer)
		except:
			return False

	attrs = {
		'_check_callable_tokenizer',
		'tokenizer',
		'tokenize',
		'from_string',
		'is_tokenizer_pretrained',
		'__getattribute__'
	}
	def __getattribute__(self, item):
		if item in __class__.attrs:
			return super().__getattribute__(item)
		try:
			return getattr(super().__getattribute__('tokenizer'), item)
		except AttributeError:
			return super().__getattribute__(item)


	@staticmethod
	def from_string(tokenizer: str) -> BaseTokenizer:
		"""

		Args:
			tokenizer: How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified.

		Returns:
			An instance of `BaseTokenizer`
		"""
		if not isinstance(tokenizer, str):
			raise TypeError('`tokenizer` must be a str, but get a %s' % type(tokenizer))
		if tokenizer == 'nltk':
			return __class__(WordPunctTokenizer())
		elif tokenizer == 'space':
			return TokenizerAdapter(str.split)
		else:
			raise ValueError('Invalid value {}. `tokenizer` should be "nltk" or "space" '.format(tokenizer))
