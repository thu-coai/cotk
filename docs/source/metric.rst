Metric
#########
.. automodule:: cotk.metric

.. autoclass:: MetricBase
    :members:
    
    .. automethod:: _hash_relevant_data
    .. automethod:: _hashvalue

Metric class
---------------------------------
.. autoclass:: BleuPrecisionRecallMetric
    :members: forward,_score,close

.. autoclass:: EmbSimilarityPrecisionRecallMetric
    :members: forward,_score,close

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

.. autoclass:: MetricChain
    :members:
