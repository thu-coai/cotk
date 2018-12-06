r"""
``contk.dataloader`` provides classes and functions downloading and
loading benchmark data automatically. It reduce your cost preprocessing
data and provide a fair dataset for every model. It also help you adapt
your model from one dataset to other datasets.
"""

from .dataloader import Dataloader
from .single_turn_dialog import SingleTurnDialog, OpenSubtitles

__all__ = ['Dataloader', 'SingleTurnDialog', 'OpenSubtitles']
