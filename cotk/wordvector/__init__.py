r"""
``cotk.wordvector`` provides classes and functions downloading and
loading wordvector automatically.
"""

from .wordvector import WordVector
from .gloves import Glove

__all__ = ["WordVector", "Glove"]
