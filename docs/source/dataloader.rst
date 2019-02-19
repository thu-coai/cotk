Data Loader
===================================
.. automodule:: contk.dataloader
.. autoclass:: Dataloader

    .. automethod:: get_all_subclasses
    .. automethod:: load_class

LanguageGeneration
-----------------------------------
.. autoclass:: LanguageGeneration

    .. autoattribute:: vocab_list
    .. autoattribute:: vocab_size
    .. autoattribute:: all_vocab_size
    .. automethod:: _load_data
    .. automethod:: _valid_word2id
    .. automethod:: restart
    .. automethod:: get_batch
    .. automethod:: get_next_batch
    .. automethod:: trim_index
    .. automethod:: sen_to_index
    .. automethod:: index_to_sen
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

MSCOCO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MSCOCO
    :members:
    :private-members:

SingleTurnDialog
-----------------------------------
.. autoclass:: SingleTurnDialog

    .. autoattribute:: vocab_list
    .. autoattribute:: vocab_size
    .. autoattribute:: all_vocab_size
    .. automethod:: _load_data
    .. automethod:: _valid_word2id
    .. automethod:: restart
    .. automethod:: get_batch
    .. automethod:: get_next_batch
    .. automethod:: trim_index
    .. automethod:: sen_to_index
    .. automethod:: index_to_sen
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

OpenSubtitles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OpenSubtitles
    :members:
    :private-members:

MultiTurnDialog
-----------------------------------
.. autoclass:: MultiTurnDialog

    .. autoattribute:: vocab_list
    .. autoattribute:: vocab_size
    .. autoattribute:: all_vocab_size
    .. automethod:: _load_data
    .. automethod:: _valid_word2id
    .. automethod:: restart
    .. automethod:: get_batch
    .. automethod:: get_next_batch
    .. automethod:: trim_index
    .. automethod:: multi_turn_trim_index
    .. automethod:: sen_to_index
    .. automethod:: multi_turn_sen_to_index
    .. automethod:: index_to_sen
    .. automethod:: multi_turn_index_to_sen
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

UbuntuCorpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UbuntuCorpus
    :members:
    :private-members:

