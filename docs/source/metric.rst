Metric
#########
.. automodule:: cotk.metric


.. _hash_ref:

Hash Value
--------------------------------
:meth:`.MetricBase.close` will return a dict containing hash value,
which can validate whether two models used the same test data and the
same setting. Only two models using the same metric with the same hash 
value returned, can compare with each other.

.. _metric_ref:

Basic Classes
--------------------------------
.. autoclass:: MetricBase
    :members:
    
    .. automethod:: _hash_relevant_data
    .. automethod:: _hashvalue

Metric class
---------------------------------

PerplexityMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PerplexityMetric
    :members:

MultiTurnPerplexityMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiTurnPerplexityMetric
    :members:

BleuCorpusMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BleuCorpusMetric
    :members:

SelfBleuCorpusMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SelfBleuCorpusMetric
    :members:

FwBwBleuCorpusMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FwBwBleuCorpusMetric
    :members:

MultiTurnBleuCorpusMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiTurnBleuCorpusMetric
    :members:

BleuPrecisionRecallMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BleuPrecisionRecallMetric
    :members: forward,_score,close

EmbSimilarityPrecisionRecallMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: EmbSimilarityPrecisionRecallMetric
    :members: forward,_score,close

NgramFwBwPerplexityMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: NgramFwBwPerplexityMetric
    :members:

Metric-like class
----------------------

SingleTurnDialogRecorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SingleTurnDialogRecorder
    :members:

MultiTurnDialogRecorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiTurnDialogRecorder
    :members:

LanguageGenerationRecorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LanguageGenerationRecorder
    :members:

MetricChain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MetricChain
    :members:
