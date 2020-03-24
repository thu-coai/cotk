Data Loader
===================================
.. automodule:: cotk.dataloader


Building a Dataloader
----------------------------------

Dataloaders are essential components in ``CoTK`` to build models or do fair evaluation.
Here we introduce methods of building a dataloader.

Predefined Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CoTK`` provides several predefined tasks and benchmarks, including

:class:`LanguageGeneration`
   * :class:`MSCOCO`
:class:`SingleTurnDialog`
   * :class:`OpenSubtitles`
:class:`MultiTurnDialog`
   * :class:`UbuntuCorpus`
   * :class:`SwitchBoard`
:class:`SentenceClassification`
   * :class:`SST`

Choose an adequate class for your task, and it would be the simplest and best way to build a dataloader.

.. _customized_tasks_ref:

Customized Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the predefined classes do not satisfy your need, you can construct an instance of :class:`LanguageProcessing`.

To specify the data format of the customized tasks, :func:`LanguageProcessing.__init__` receives an argument named ``fields``.
The full description of ``fields`` should be like the example below.

>>> postField = SentenceDefault(...)
>>> respField = SentenceDefault(...)
>>> labelField = DenseLabel(...)
>>> fields = {
>>>    "train": [("post", postField), ("resp", respField)],
>>>    "test": [("post", postField), ('resp', respField), ('label', labelField)]
>>> }
>>> dataloader = LangaugeProcessing("/path/to/dataset", fields)

* ``"train"`` and ``"test"`` is the name of the split sets in the dataset. There should be two text file named ``train.txt`` and ``test.txt``
  under ``/path/to/dataset/``, corresponding to the two sets, ``"train"`` and ``"test"`` respectively.
* ``fields["train"]`` describes the data format of ``train.txt``. Every sample in ``train`` set has two :ref:`data fields<field_ref>`,
  which is represented by :class:`Field` objects.
  As :class:`SentenceDefault` (a subclass of :class:`Field`) only read one line per each sample, a sample in ``train.txt`` occupy two lines.
  The first line are named by ``"post"``, the second line are named ``"resp"``.
* Similarily, ``fields["test"]`` describes the data format of ``test.txt``. Every sample in ``test`` set occupies three lines,
  where the first line is ``"post"``, the second line is ``"resp"``, and the third line is an integer indicating ``"label"``.

An valid input example:

* ``/path/to/dataset/train.txt``

   .. code-block:: none

      How are you?
      I am fine.
      What's up?
      Everything is good.

* ``/path/to/dataset/test.txt``

   .. code-block:: none

      What is your name?
      Jack.
      1
      How about the food?
      Terrible.
      0

The data fields objects defines how dataloaders read the file, process the data, and provide the data to networks.
See :ref:`data fields<field_ref>` for further details.

**Omit Set Names**

If you have three sets named ``"train"``, ``"dev"``, ``"test"``, and the data format is the same, you can
specify the ``fields`` argument in :func:`LanguageProcessing.__init__` by the following code:

>>> fields = [("post", postField), ("resp", respField)]

equals to

>>> fields = {
>>>    "train": [("post", postField), ("resp", respField)],
>>>    "dev": [("post", postField), ("resp", respField)],
>>>    "test": [("post", postField), ("resp", respField)]
>>> }

**Use Simple Create**

You can use :func:`LanguageProcessing.simple_create` to initialize a dataloder, using the class name of :class:`Field`
instead of instances. The method receives arguments for initializing the common :class:`Field`.

>>> fields = {
>>>    "train": [("post", "SentenceDefault"), ("resp", "SentenceDefault")],
>>>    "dev": [("post", "SentenceDefault"), ("resp", "SentenceDefault")],
>>>    "test": [("post", "SentenceDefault"), ("resp", "SentenceDefault")],
>>> }
>>> #or fields = [("post", "SentenceDefault"), ("resp", "SentenceDefault")]
>>> dataloader = LanguageProcessing.simple_create("/path/to/dataset", fields, max_sent_length=10, min_frequent_vocab_times=10)

In this example, ``max_sent_length=10`` and ``min_frequent_vocab_times=10`` will be used to initialize the :class:`SentenceDefault` objects.

**Use Context Manager**

There is another way to use the class name of :class:`Field` instead of instances. Initialize the :class:`LanguageProcessing`
in the context of :class:`FieldContext` and :class:`VocabContext`.

>>> fields = [("post", "SentenceDefault"), ("resp", "SentenceDefault")]
>>> with FieldContext(max_sent_length=10):
>>>     with VocabContext(min_frequent_vocab_times=10):
>>>         dataloader = LanguageProcessing("/path/to/dataset", fields)

equals to

>>> fields = [("post", "SentenceDefault"), ("resp", "SentenceDefault")]
>>> dataloader = LanguageProcessing.simple_create("/path/to/dataset", fields, max_sent_length=10, min_frequent_vocab_times=10)

Context is used to provide default values for :class:`Field` and :class:`Vocab` instances.
See :ref:`Context<context_ref>` for further details.

.. _field_ref:

Field
----------------------------------

:class:`Field` indicates data fields, which defines how dataloaders read the file, process the data, and provide the data to networks.

``Cotk`` provides several fields, including

* :class:`Sentence`
   * :class:`SentenceDefault`
   * :class:`SentenceGPT2`
* :class:`Session`
   * :class:`SessionDefault`
   * :class:`SessionGPT2`
* :class:`DenseLabel`
* :class:`SparseLabel`

Read the File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`Field` defines the way to read the file. For example,

* :class:`Sentence` reads one line per sample, which is a string of sentence.
* :class:`Session` reads multiple lines per sample, stopped when a empty line is read.
* :class:`DenseLabel` reads one line per sample, which is an integer.

See the documentation in each class for details.

Process the Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each subclass of :class:`Field` defines the methods to process the input.

For example, :class:`Sentence` processes the sentence into different formats:

* (str) The whole sentences.
* (List[str]) The tokenized sentences.
* (List[id]) The index of tokens in the :ref:`vocabulary<vocabulary_ref>`.

:class:`Sentence` also provides methods to convert a sentence from one format to another:

* :meth:`Sentence.tokenize`
* :meth:`Sentence.convert_ids_to_sentence`
* :meth:`Sentence.convert_sentence_to_ids`
* :meth:`Sentence.convert_tokens_to_ids`
* :meth:`Sentence.convert_ids_to_tokens`

The dataloader has similar methods, which invoke the corresponding methods of the default field.
See :meth:`LanguageProcessing.set_default_field` for details.

Provide the Data to Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each subclass of :class:`Field` defines :meth:`Field.get_batch`, which is invoked by :meth:`LanguageProcessing.get_batch`.
This method defines the data format when training the networks.

For example, if an instance of :class:`SentenceDefault` is named with ``"sent"``, it returns a dict when :meth:`SentenceDefault.get_batch` called:

* sent (np.ndarray[batch_size, max_sent_length]): padded sentences in id formats with :ref:`frequent words<vocabulary_ref>`.
* sent_allvocabs(np.ndarray[batch_size, max_sent_length]): padded sentences in id formats with :ref:`frequent and rare words<vocabulary_ref>`.
* sent_length (np.ndarray[batch_size]): length of sentences
* sent_str (List[str]): the raw sentence

Relatively, a dataloader with two :class:`SentenceDefault` fields named ``"post"``, ``"resp"`` will returns a dict when :meth:`LanguageProcessing.get_batch` called:

* ``post``
* ``post_allvocabs``
* ``post_length``
* ``post_str``
* ``reps``
* ``resp_allvocabs``
* ``resp_length``
* ``resp_str``

This is the merge of two returned dicts by :meth:`SentenceDefault.get_batch`.

.. _vocabulary_ref:

Vocabulary
----------------------------------

:class:`Vocab` defines the , which is used by :class:`Field` and :class:`LanguageProcessing`.

``CoTK`` provides several vocabularies, including

* :class:`GeneralVocab`: A vocabulary for general use in ``CoTK``
* :class:`PretrainedVocab`: A pretrained vocabulary from the ``transformers`` package. For example, vocabulary for ``GPT2``.

Type of Tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All tokens appeared in dataset (including the ones only appear in test set) are split into
2 sets.

    Frequent Vocabularies(``frequent_vocabs``)
        * Tokens that the model **should** read, predict and generate.
        * These tokens are important in evaluation. They include
          common words and usually cover over most of tokens
          from dataset.
        * They are extracted from only training set, because models should be
          blind for test set. Hence they are defined as the tokens appear
          more than a specified number of times (``min_frequent_vocab_times``) in
          **training set**.

    Rare Vocabularies(``rare_vocabs``)
        * Tokens that the model can **optionally** read,
          but will **not** predict and generate at most times
          (**except** some models can generate rare words using
          copy mechanism or external knowledge).
        * These tokens are less important but **DO** affect the evaluation.
        * They are extracted from both training set and test set, because they
          are defined considering evaluation. Hence, they are defined as the tokens
          (excluded ``frequent_vocabs``) appear more than a specified number
          (``min_rare_vocab_times``) of times in **the whole dataset**.

There is also some other terms for vocabularies.

    All Vocabularies(``allvocabs``)
        * The union of `Frequent vocabularies` and `rare vocabularies` is called `all vocabularies`.

    Special Tokens(``special_tokens``)
        * Most used special tokens are ``<pad>``, ``<unk>``, ``<go>``, ``<eos>``.
        * Special tokens are counted as valid vocabularies.

    Unknown tokens (``<unk>``)
        * ``<unk>`` means "Out of Vocabularies", but we the meaning of ``<unk>`` may varies from situations.
        * If it appears at a list named with ``allvocabs`` (eg: ``sent_allvocabs``),
          ``<unk>`` indicates a token out of all vocabularies.
        * If it appears at a list named without ``allvocabs`` (eg: ``sent``),
          ``<unk>`` indicates a token out of frequent vocabularies, which means it may a ``rare vocabulary``.

Why CoTK Uses Rare Words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In traditional implementation, vocabulary only contains frequent vocabulary.
``CoTK`` use frequent vocabulary and rare vocabulary for supporting fair comparisons across different configurations.

For examples, we test two models under the same dataset, but with different vocabularies.

* Model A:  Frequent vocabulary ``F_A``; Rare vocabulary ``R_A``.
* Model B:  Frequent vocabulary ``F_B``; Rare vocabulary ``R_B``.

The fairness of comparisons can be gauranteed under the conditions:

* :class:`.metric.PerplexityMetric`: ``F_A + R_A == F_B + R_B``.
* :class:`.metric.BleuCorpusMetric`: ``F_A + R_A == F_B + R_B`` if tokenizer is ``None``; ``F_A + R_A == F_B + R_B`` if tokenizer is set.

See each metrics for when the fairness can be gauranteed. :ref:`Hash value of metrics<metric_hashvalue_ref>`
can help user determine whether the comparisons is fair.

.. _context_ref:

How to Use Context
----------------------------------


How to Use Dataloader
----------------------------------

.. _dataloader_hash_ref:

Hash Code
==================================


TODO: fill the documentation









Context
------------------------------------

.. autoclass:: Context

    .. automethod:: get
    .. automethod:: set
    .. automethod:: __enter__
    .. automethod:: __exit__
    .. automethod:: close

FieldContext
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: FieldContext

    .. automethod:: set_parameters

VocabContext
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: VocabContext

    .. automethod:: set_parameters

Tokenizer
-------------------------------------

.. autoclass:: Tokenizer

    .. automethod:: tokenize
    .. automethod:: tokenize_sentences
    .. automethod:: tokenize_sessions
    .. automethod:: convert_tokens_to_sentence

    .. automethod:: get_setting_hash

SimpleTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SimpleTokenizer


Pretrainedtokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PretrainedTokenizer

    .. automethod:: get_tokenizer_class

Vocab
-------------------------------------

.. autoclass:: Vocab

    .. automethod:: get_all_subclasses
    .. automethod:: load_class
    .. automethod:: add_tokens
    .. automethod:: build_vocab

    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_ids_to_tokens
    .. autoattribute:: frequent_vocab_size
    .. autoattribute:: all_vocab_size
    .. autoattribute:: frequent_vocab_list
    .. autoattribute:: all_vocab_list
    .. automethod:: get_special_tokens_mapping
    .. automethod:: get_special_tokens_id
    .. autoattribute:: pad_id
    .. autoattribute:: unk_id
    .. autoattribute:: go_id
    .. autoattribute:: eos_id

    .. automethod:: get_setting_hash
    .. automethod:: get_vocab_hash

GeneralVocab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GeneralVocab

    .. automethod:: from_predefined
    .. automethod:: from_predefined_vocab
    .. automethod:: from_frequent_word
    .. automethod:: from_frequent_word_of_vocab
    .. autoattribute:: frequent_vocab_list
    .. autoattribute:: all_vocab_list

PretrainedVocab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PretrainedVocab

    .. autoattribute:: frequent_vocab_list
    .. autoattribute:: all_vocab_list

Field
-------------------------------------

.. autoclass:: Field

    .. automethod:: get_all_subclasses
    .. automethod:: load_class
    .. automethod:: get_vocab
    .. automethod:: get_tokenizer
    .. automethod:: get_batch
    .. autoattribute:: DEFAULT_VOCAB_FROM

Sentence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Sentence

    .. automethod:: tokenize
    .. automethod:: tokenize_sentences
    .. automethod:: convert_tokens_to_ids
    .. automethod:: convert_ids_to_tokens
    .. automethod:: convert_ids_to_sentence
    .. automethod:: convert_sentence_to_ids
    .. automethod:: add_special_to_ids
    .. automethod:: remove_special_in_ids
    .. automethod:: process_sentences
    .. automethod:: trim_in_ids

    .. autoattribute:: frequent_vocab_size
    .. autoattribute:: all_vocab_size
    .. autoattribute:: frequent_vocab_list
    .. autoattribute:: all_vocab_list
    .. automethod:: get_special_tokens_mapping
    .. automethod:: get_special_tokens_id
    .. autoattribute:: pad_id
    .. autoattribute:: unk_id
    .. autoattribute:: go_id
    .. autoattribute:: eos_id

SentenceDefault
#####################################
.. autoclass:: SentenceDefault

    .. automethod:: get_batch

SentenceGPT2
#####################################
.. autoclass:: SentenceGPT2

    .. automethod:: get_batch

Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Session

SessionDefault
#####################################
.. autoclass:: SessionDefault

    .. automethod:: get_batch

DenseLabel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DenseLabel

    .. automethod:: get_batch

SparseLabel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SparseLabel

    .. automethod:: get_batch

Dataloader
------------------------------------
.. autoclass:: Dataloader

    .. automethod:: get_all_subclasses
    .. automethod:: load_class

LanguageProcessing
------------------------------------
.. autoclass:: LanguageProcessing

.. automethod:: LanguageProcessing.simple_create

Tokenizer, Vocabulary, and Field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: LanguageProcessing.get_default_tokenizer
.. automethod:: LanguageProcessing.get_default_vocab
.. automethod:: LanguageProcessing.get_default_field
.. automethod:: LanguageProcessing.set_default_field
.. automethod:: LanguageProcessing.get_field

Batched Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. automethod:: LanguageProcessing.get_batch
  .. automethod:: LanguageProcessing.restart
  .. automethod:: LanguageProcessing.get_next_batch
  .. automethod:: LanguageProcessing.get_batches
  .. automethod:: LanguageProcessing.get_all_batch

Sentences and Ids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: LanguageProcessing.tokenize
.. automethod:: LanguageProcessing.tokenize_sentences
.. automethod:: LanguageProcessing.convert_tokens_to_ids
.. automethod:: LanguageProcessing.convert_ids_to_tokens
.. automethod:: LanguageProcessing.convert_ids_to_sentence
.. automethod:: LanguageProcessing.convert_sentence_to_ids
.. automethod:: LanguageProcessing.add_special_to_ids
.. automethod:: LanguageProcessing.remove_special_in_ids
.. automethod:: LanguageProcessing.process_sentences
.. automethod:: LanguageProcessing.trim_in_ids

Vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: LanguageProcessing.frequent_vocab_size
.. autoattribute:: LanguageProcessing.all_vocab_size
.. autoattribute:: LanguageProcessing.frequent_vocab_list
.. autoattribute:: LanguageProcessing.all_vocab_list
.. automethod:: LanguageProcessing.get_special_tokens_mapping
.. automethod:: LanguageProcessing.get_special_tokens_id
.. autoattribute:: LanguageProcessing.pad_id
.. autoattribute:: LanguageProcessing.unk_id
.. autoattribute:: LanguageProcessing.go_id
.. autoattribute:: LanguageProcessing.eos_id

Hash
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: LanguageProcessing.get_general_hash
.. automethod:: LanguageProcessing.get_raw_data_hash
.. automethod:: LanguageProcessing.get_data_hash
.. automethod:: LanguageProcessing.get_vocab_hash
.. automethod:: LanguageProcessing.get_setting_hash

LanguageGeneration
---------------------------------------
.. autoclass:: LanguageGeneration

    .. automethod:: get_batch
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

MSCOCO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MSCOCO

SingleTurnDialog
---------------------------------------
.. autoclass:: SingleTurnDialog

    .. automethod:: get_batch
    .. automethod:: get_teacher_forcing_metric
    .. automethod:: get_inference_metric

OpenSubtitles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OpenSubtitles
