'''
A module for dataloader
'''
import random
from .._utils import trim_before_target
from .._utils.metaclass import DocStringInheritor, LoadClassInterface

class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''
	def __init__(self):
		pass

class BasicLanguageGeneration(Dataloader):
	r"""Base class for all language generation datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = r"""
			ext_vocab (list): special tokens. default: `["<pad>", "<unk>", "<go>", "<eos>"]`
			key_name (list): name of subsets of the data. default: `["train", "dev", "test"]`
	"""
	ATTRIBUTES = r"""
			ext_vocab (list): special tokens, be placed at beginning of `vocab_list`.
					For example: `["<pad>", "<unk>", "<go>", "<eos>"]`
			pad_id (int): token for padding, always equal to `0`
			unk_id (int): token for unknown words, always equal to `1`
			go_id (int): token at the beginning of sentences, always equal to `2`
			eos_id (int): token at the end of sentences, always equal to `3`
			key_name (list): name of subsets of the data. For example: `["train", "dev", "test"]`
			all_vocab_list (list): vocabulary list of the datasets,
					including valid vocabs and invalid vocabs.
			word2id (dict): a dict mapping all vocab to index.
					Maybe you want to use :meth:`sen_to_index` instead.
	"""

	def __init__(self, \
				 ext_vocab=None, \
				 key_name=None):
		super().__init__()

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = self.ext_vocab.index("<pad>")
		self.unk_id = self.ext_vocab.index("<unk>")
		self.go_id = self.ext_vocab.index("<go>")
		self.eos_id = self.ext_vocab.index("<eos>")
		self.key_name = key_name or ["train", "dev", "test"]

		# initialize by subclass
		self.all_vocab_list, self.valid_vocab_len, self.data, self.data_size = self._load_data()
		self.word2id = {w: i for i, w in enumerate(self.all_vocab_list)}

		# postprocess initialization
		self.index = {}
		self.batch_id = {}
		self.batch_size = {}
		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None
			self.index[key] = list(range(self.data_size[key]))

	def _valid_word2id(self, word):
		'''This function return the id for a valid word (``id < valid_vocab_len``),
		otherwise return `unk_id`.
		'''
		idx = self.word2id.get(word, self.unk_id)
		if idx >= self.vocab_size:
			idx = self.unk_id
		return idx

	def _load_data(self):
		r'''This function is called during the initialization.

		Returns:
			(tuple): tuple containing:

			* all_vocab_list (list): vocabulary list of the datasets,
			  including valid and invalid vocabs
			* valid_vocab_len (int): the number of valid vocab.
			  ``vocab_list[:valid_vocab_len]`` will be regarded as valid vocabs,
			  while ``vocab_list[valid_vocab_len:]`` regarded as invalid vocabs
			* data (dict): a dict contains data.
			* data_size (dict): a dict contains size of each item in data.
		'''
		raise NotImplementedError( \
			"This function should be implemented by subclasses.")

	@property
	def vocab_size(self):
		'''int: equals to ``valid_vocab_len``. Read only.
		'''
		return self.valid_vocab_len

	@property
	def all_vocab_size(self):
		'''int: equals to ``len(self.all_vocab_list)``. Read only.
		'''
		return len(self.all_vocab_list)

	@property
	def vocab_list(self):
		r'''list: valid vocab list, equals to ``all_vocab_list[：valid_vocab_len]``.
		Read only.
		'''
		# using utf-8 ：instead of : for avoiding bug in sphinx
		return self.all_vocab_list[:self.valid_vocab_len]

	def restart(self, key, batch_size=None, shuffle=True):
		'''Initialize mini-batches. Must call this function before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
				key (str): must be contained in `key_name`.
				batch_size (None or int): default (None): use last batch_size.
				shuffle (bool): whether to shuffle the data. default: `True`
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if batch_size is None and self.batch_size[key] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			random.shuffle(self.index[key])

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" % (key, \
				len(self.index[key]) // self.batch_size[key], \
				len(self.index[key]) % self.batch_size[key]))

	def get_batch(self, key, index, needhash=False):
		'''Get a batch of specified `index`.

		Arguments:
				key (str): must be contained in `key_name`
				index (list): a list of specified index

		Returns:
				A dict. See examples in subclasses.
		'''
		raise NotImplementedError( \
			"This function should be implemented by subclasses.")

	def get_next_batch(self, key, ignore_left_samples=False, needhash=False):
		'''Get next batch.

		Arguments:
				key (str): must be contained in `key_name`
				ignore_left_samples (bool): Ignore the last batch, whose sample num
						is not equal to `batch_size`. Default: `False`

		Returns:
				A dict like :func:`get_batch`, or None if the epoch is end.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if self.batch_size[key] is None:
			raise RuntimeError( \
				"Please run restart before calling this function.")
		batch_id = self.batch_id[key]
		start, end = batch_id * \
			self.batch_size[key], (batch_id + 1) * self.batch_size[key]
		if start >= len(self.index[key]):
			return None
		if ignore_left_samples and end > len(self.index[key]):
			return None
		index = self.index[key][start:end]
		res = self.get_batch(key, index, needhash=needhash)
		self.batch_id[key] += 1
		return res

	def sen_to_index(self, sen, invalid_vocab=False):
		'''Convert a sentence from string to index representation.

		Arguments:
			sen (list): a list of str, representing each token of the sentences.
			invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be replaced by `unk_id`.
					If ``True``, invalid vocabs will using their own id.
					Default: `False`

		Examples:
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "Sichuan"]
			>>> dataloader.sen_to_index(
			...	["<go>", "I", "have", "been", "to", "Sichuan", "<eos>"])
			>>> [2, 4, 5, 6, 7 ,8 ,3]

		TODO:
			* add invalid vocab example
		'''
		if invalid_vocab:
			return list(map(lambda word: self.word2id.get(word, self.unk_id), sen))
		else:
			return list(map(self._valid_word2id, sen))

	def trim_index(self, index):
		'''Trim index. There will be two steps:

			* If there is an end token (`<eos>`) in sentences,
			  find first end token and abondon words after it (included the end token).
			* ignore `<pad>` s at the end of the sentence.

		Arguments:
			index (list): a list of int

		Examples:

			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "Sichuan"]
			>>> dataloader.trim_index(
			...	[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0])
			... # <go> I have been to Sichuan <pad> <pad> <eos> I <eos> <pad>
			>>> [2, 4, 5, 6, 7, 8] # <go> I have been to Sichuan
		'''

		index = trim_before_target(list(index), self.eos_id)
		idx = len(index)
		while idx > 0 and index[idx-1] == self.pad_id:
			idx -= 1
		index = index[:idx]
		return index

	def index_to_sen(self, index, trim=True):
		'''Convert a sentence from index to string representation.

		Arguments:
				index (list): a list of int.
				trim (bool): if True, call :func:`trim_index` before convertion.

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "Sichuan"]
			>>> dataloader.index_to_sen(
			...		[2, 4, 5, 6, 7, 8, 3, 0, 0], trim = True)
			>>> ["<go>", "I", "have", "been", "to", "Sichuan"]
			>>> dataloader.index_to_sen(
			...		[2, 4, 5, 6, 7, 8, 3, 0, 0], trim = False)
			>>> ["<go>", "I", "have", "been", "to", "Sichuan", "<eos>", "<pad>", "<pad>"]

		'''
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.all_vocab_list[word], index))
