r"""
``cotk`` is an open-source lightweight framework for model building and evaluation.
We provides standard dataset and evaluation suites in the domain of general language generation.
It easy to use and make you focus on designing your models!

Features include:

 * Light-weight, easy to start. Don't bother your way to construct models.
 * Predefined standard datasets, in the domain of language modeling, dialog generation and more.
 * Predefined evaluation suites, test your model with multiple metrics in several lines.
 * A dashboard to show the performance, compare your and others' models fairly.
 * Long-term maintenance and consistent development.
"""

from ._utils.imports import LazyModule

dataloader = LazyModule("cotk.dataloader", globals())
metric = LazyModule("cotk.metric", globals())
wordvector = LazyModule("cotk.wordvector", globals())
models = LazyModule("cotk.models", globals())
scripts = LazyModule("cotk.scripts", globals())
_utils = LazyModule("cotk._utils", globals())

__all__ = ['dataloader', 'metric', 'wordvector', 'models', 'scripts', '_utils']
