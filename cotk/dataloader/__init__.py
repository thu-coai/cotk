r"""
``cotk.dataloader`` provides classes and functions downloading and
loading benchmark data automatically. It reduces your cost preprocessing
data and provide a fair dataset for every model. It also helps you adapt
your model from one dataset to other datasets.
"""

from .tokenizer import BaseTokenizer, SimpleTokenizer, PretrainedTokenizer
from .vocab import BaseVocab, Vocab
from .field import Field, Sentence, Session
from .context import FieldContext, VocabContext
from .dataloader import Dataloader, LanguageProcessingBase
from .language_generation import LanguageGeneration, MSCOCO
from .single_turn_dialog import SingleTurnDialog, OpenSubtitles

__all__ = [ \
	'BaseTokenizer', 'SimpleTokenizer', 'PretrainedTokenizer', \
	'BaseVocab', 'Vocab', \
	'Field', 'Sentence', 'Session', \
	'FieldContext', 'VocabContext', \
	'Dataloader', 'LanguageProcessingBase', \
	'LanguageGeneration', 'MSCOCO', \
	'SingleTurnDialog', 'OpenSubtitles', \
]
