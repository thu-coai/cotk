r"""
``cotk.dataloader`` provides classes and functions downloading and
loading benchmark data automatically. It reduces your cost preprocessing
data and provide a fair dataset for every model. It also helps you adapt
your model from one dataset to other datasets.
"""

from .dataloader import Dataloader, GenerationBase
from .single_turn_dialog import SingleTurnDialog, OpenSubtitles
from .multi_turn_dialog import MultiTurnDialog, UbuntuCorpus, SwitchboardCorpus
from .language_generation import LanguageGeneration, MSCOCO

__all__ = ['Dataloader', 'SingleTurnDialog', 'OpenSubtitles', 'MultiTurnDialog', 'UbuntuCorpus', \
           'SwitchboardCorpus', 'LanguageGeneration', 'MSCOCO', 'GenerationBase']
