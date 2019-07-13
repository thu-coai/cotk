r"""
``cotk.metric`` provides commonly used metrics for ``cotk.dataloader``.
All metric objects receive a batch of data per one call of ``forward``.
The batch of data is represented by a dict which contains models' outputs and
answers. The answers are highly relevant to the corresponding
dataloader, where the type, shape are usually identical with the return value of
``get_batch`` in dataloader, as long as the correct key name is set.
``forward`` function can be called several times and at last
``close`` can be called for results.

Here is an example:

    >>> dm = OpenSubtitles()
    >>> metric = BleuCorpusMetric(gen_key="gen",\
    ...     reference_allvocabs_key="resp_allvocabs_key")
    ... # "resp_allvocabs_key" is a key name in get_batch()
    >>> for data in dm.get_batches("test", batch_size=32):
    ...     data["gen"] = predict(data["post"])
    ...     assert "resp_allvocabs_key" in data
    ...     metric.forward(data)
    >>> print(metric.close())
    {"bleu": 0.135, "bleu hashvalue": b"XXXX"}


We also provide default metrics in dataloader, you can use "get_metric"-like
functions (example: :meth:`.SingleTurnDialog.get_inference_metric`) to get
default metrics and avoid the mess with complex key name.

Here is an exmample:

    >>> dm = OpenSubtitles()
    >>> metric = dm.get_inference_metric(gen_key="gen")
    >>> for data in dm.get_batches("test", batch_size=32):
    ...     data["gen"] = predict(data["post"])
    ...     metric.forward(data)
    >>> print(metric.close())
    {"bleu": 0.135, "bleu hashvalue": b"XXXX", ...}
"""

from .metric import MetricBase, MetricChain
from .precision_recall import BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric
from .bleu import BleuCorpusMetric, SelfBleuCorpusMetric, FwBwBleuCorpusMetric, \
                    MultiTurnBleuCorpusMetric
from .perplexity import PerplexityMetric, MultiTurnPerplexityMetric
from .accuracy import AccuracyMetric
from .recorder import SingleTurnDialogRecorder, LanguageGenerationRecorder, MultiTurnDialogRecorder
from .ngram_perplexity import NgramFwBwPerplexityMetric

__all__ = ["MetricBase", "PerplexityMetric", "BleuCorpusMetric", "SelfBleuCorpusMetric", \
        "FwBwBleuCorpusMetric", "SingleTurnDialogRecorder", "LanguageGenerationRecorder", \
        "MetricChain", "MultiTurnDialogRecorder", "MultiTurnPerplexityMetric", \
        "MultiTurnBleuCorpusMetric", "BleuPrecisionRecallMetric", \
        "EmbSimilarityPrecisionRecallMetric", "AccuracyMetric", \
		"NgramFwBwPerplexityMetric"]
