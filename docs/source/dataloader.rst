Data Loader
===================================
.. automodule:: cotk.dataloader
.. autoclass:: Dataloader

    .. automethod:: get_all_subclasses
    .. automethod:: load_class

.. autoclass:: GenerationBase

    .. autoattribute:: vocab_list
    .. autoattribute:: vocab_size
    .. autoattribute:: all_vocab_size
    .. automethod:: _load_data
    .. automethod:: _valid_word2id
    .. automethod:: restart
    .. automethod:: get_batch
    .. automethod:: get_next_batch
    .. automethod:: get_batches
    .. automethod:: trim_index
    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_ids_to_tokens

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
    .. automethod:: get_batches
    .. automethod:: trim_index
    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_ids_to_tokens
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
    .. automethod:: get_batches
    .. automethod:: trim_index
    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_ids_to_tokens
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
    .. automethod:: get_batches
    .. automethod:: trim_index
    .. automethod:: multi_turn_trim_index
    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_multi_turn_tokens_to_ids
    .. automethod:: convert_ids_to_tokens
    .. automethod:: convert_multi_turn_ids_to_tokens
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

UbuntuCorpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UbuntuCorpus
    :members:
    :private-members:

SwitchBoardCorpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SwitchboardCorpus
    :members:

    .. automethod:: _load_data

