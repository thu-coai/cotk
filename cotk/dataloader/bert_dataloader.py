'''
A module for BERT dataloader
'''
from .dataloader import LanguageProcessingBase
from .._utils import trim_before_target
from .._utils.imports import LazyObject

BertTokenizer = LazyObject("transformers.BertTokenizer")

#pylint: disable=abstract-method
class BERTLanguageProcessingBase(LanguageProcessingBase):
	r"""Base class for all BERT-based language processing with BERT tokenizer.
	This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	BERT_VOCAB_NAME = r"""
			bert_vocab_name (str): A string indicates which bert model is used, it will be a
					parameter passed to `transformers.BertTokenizer.from_pretrained
					<https://github.com/huggingface/transformers#berttokenizer>`_.
					It can be 'bert-[base|large]-[uncased|cased]' or a local path."""

	ARGUMENTS = LanguageProcessingBase.ARGUMENTS + BERT_VOCAB_NAME

	ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES + r"""
			bert_id2word (list): Vocabulary list mapping bert ids to tokens,
					including valid vocabs and invalid vocabs.
			word2bert_id (dict): A dict mapping all tokens to its bert id. You don't need to use it 
					at most times, see :meth:`convert_tokens_to_bert_ids` instead.
	"""

	_version = 1

	def __init__(self, ext_vocab=None, \
					key_name=None, \
					bert_vocab_name='bert-base-uncased'):

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

		self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_name)
		self._build_bert_vocab()

		super().__init__(self.ext_vocab, key_name)

	def _build_bert_vocab(self):
		self.word2bert_id = dict(self.tokenizer.vocab)
		self.bert_id2word = [None] * len(self.word2bert_id)
		for key, value in self.word2bert_id.items():
			self.bert_id2word[value] = key

		self.bert_pad_id = self.word2bert_id["[PAD]"]
		self.bert_unk_id = self.word2bert_id["[UNK]"]
		self.bert_go_id = self.word2bert_id["[CLS]"]
		self.bert_eos_id = self.word2bert_id["[SEP]"]

	def _valid_bert_id_to_id(self, bert_id):
		'''This function return the id for a valid bert id, otherwise return ``unk_id``.

		Arguments:
			bert_id (str): a bert id.

		Returns:
			int
		'''
		idx = self.word2id.get(bert_id, self.unk_id)
		if idx >= self.vocab_size:
			idx = self.unk_id
		return idx

	def tokenize(self, sentence):
		'''Convert sentence(str) to list of tokens(str)

		Arguments:
				sentence (str)

		Returns:
				sent (list): list of tokens(str)
		'''
		return self.tokenizer.tokenize(sentence)

	def convert_tokens_to_bert_ids(self, sent):
		'''Convert list of token(str) to list of bert id(int)

		Arguments:
				sent (list): list of token(str)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return self.tokenizer.convert_tokens_to_ids(sent)

	def convert_bert_ids_to_tokens(self, bert_ids, trim=True):
		'''Convert list of bert id(int) to list of token(str)

		Arguments:
				bert_ids (list): list of bert id(int)

		Returns:
				(list): list of token(str)
		'''
		if trim:
			bert_ids = trim_before_target(list(bert_ids), self.bert_eos_id)
			idx = len(bert_ids)
			while idx > 0 and bert_ids[idx-1] == self.bert_pad_id:
				idx -= 1
			bert_ids = bert_ids[:idx]
		return list(map(lambda word: self.bert_id2word[word], bert_ids))

	def convert_bert_ids_to_ids(self, bert_ids, invalid_vocab=False):
		'''Convert list of bert id(int) to list of id(int)

		Arguments:
				bert_ids (list): list of bert id(int)
				invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be replaced by ``unk_id``.
					If ``True``, invalid vocabs will using their own id.
					Default: ``False``

		Returns:
				(list): list of id(int)
		'''
		return self.convert_tokens_to_ids(\
			self.convert_bert_ids_to_tokens(bert_ids, False), invalid_vocab)

	def convert_ids_to_bert_ids(self, ids):
		'''Convert list of id(int) to list of bert id(int)

		Arguments:
				ids (list): list of id(int)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return self.convert_tokens_to_bert_ids(\
			self.convert_ids_to_tokens(ids, False))
