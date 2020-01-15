'''
A module for BERT dataloader
'''
from .dataloader import LanguageProcessingBase
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
		tokenizer = BertTokenizer.from_pretrained(bert_vocab_name)
		super().__init__(ext_vocab=self.ext_vocab,
						 key_name=key_name,
						 tokenizer=tokenizer)


	def _build_pretrained_vocab(self):
		super()._build_pretrained_vocab()
		self.word2bert_id = self.word2pretrained_id
		self.bert_id2word = self.pretrained_id2word
		self.bert_pad_id, self.bert_unk_id, self.go_id, self.bert_eos_id = \
			self.pad_id, self.unk_id, self.go_id, self.eos_id


	_valid_bert_id_to_id = LanguageProcessingBase._valid_pretrained_id_to_id
	convert_tokens_to_bert_ids = LanguageProcessingBase.convert_tokens_to_pretrained_ids
	convert_bert_ids_to_tokens = LanguageProcessingBase.convert_pretrained_ids_to_ids
	convert_bert_ids_to_ids = LanguageProcessingBase.convert_pretrained_ids_to_ids
	convert_ids_to_bert_ids = LanguageProcessingBase.convert_ids_to_pretrained_ids
