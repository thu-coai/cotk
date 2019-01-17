r"""
`contk.metrics` provides functions evaluating results of models. It provides
a fair metric for every model.
"""

from .metric import MetricBase, PerlplexityMetric, BleuCorpusMetric, \
                    SingleTurnDialogRecorder, LanguageGenerationRecorder, MetricChain, \
                    MultiTurnDialogRecorder, MultiTurnPerplexityMetric, MultiTurnBleuCorpusMetric

__all__ = ["MetricBase", "PerlplexityMetric", "BleuCorpusMetric", \
        "SingleTurnDialogRecorder", "LanguageGenerationRecorder", "MetricChain", \
        "MultiTurnDialogRecorder", "MultiTurnPerplexityMetric", "MultiTurnBleuCorpusMetric"]
