Metric
#########
.. automodule:: cotk.metric


Hash Value
--------------------------------
:meth:`.MetricBase.close` will return a dict containing hash value,
which can validate whether two models used the same test data and the
same setting. Only two models using the same metric with the same hash 
value returned, can compare with each other.

Basic Classes
--------------------------------
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
