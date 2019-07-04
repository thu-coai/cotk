Metric
#########
.. automodule:: cotk.metric

.. autoclass:: MetricBase
    :members:

Metric class
---------------------------------
.. autoclass:: BleuPrecisionRecallMetric
    :members: forward,score,close

.. autoclass:: EmbSimilarityPrecisionRecallMetric
    :members: forward,score,close

.. autoclass:: PerplexityMetric
    :members:

.. autoclass:: MultiTurnPerplexityMetric
    :members:

.. autoclass:: BleuCorpusMetric
    :members:

.. autoclass:: SelfBleuCorpusMetric
    :members:

.. autoclass:: FwBwBleuCorpusMetric
    :members:

.. autoclass:: MultiTurnBleuCorpusMetric
    :members:

Metric-like class
----------------------

.. autoclass:: SingleTurnDialogRecorder
    :members:

.. autoclass:: MultiTurnDialogRecorder
    :members:

.. autoclass:: LanguageGenerationRecorder
    :members:

.. autoclass:: HashValueRecorder
    :members:

.. autoclass:: MetricChain
    :members: