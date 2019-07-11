r"""
cotk is a python package providing utilities for natural language
generation. It contains benchmark data loader, word vector loader,
pretrained baseline models and other useful utilities for evaluating
new models fairly with baselines.
"""

from ._utils import start_recorder, close_recorder

from . import dataloader, metric, wordvector

__all__ = ["dataloader", "metric", "wordvector", "start_recorder", "close_recorder"]
