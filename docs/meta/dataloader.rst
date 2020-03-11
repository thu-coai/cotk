Data Loader
===================================
.. automodule:: cotk.dataloader


How to Build Dataloader
----------------------------------

Dataloaders are essential components in ``CoTK`` to build models or do fair evaluation.
Here we introduce methods of building a dataloader.

Predefined Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CoTK`` contains several predefined tasks, including

* :class:`LanguageGeneration`,
* :class:`SingleTurnDialog`
* :class:`MultiTurnDialog`
* :class:`SentenceClassification`.

Choose a adequate class for your task, and it would be the simplest and best way to build a dataloader.

Fast Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the predefined classes do not satisfy your task, you can construct an instance of :class:`LanguageProcessing`
using the fast initialization.

>> dl = LanuageProcessing(file_id, OrderedDict([("post", "SentenceDefault"), ('resp', 'SentenceDefault')]))

It means you will construct a dataloader with three set



Initialization for Complex Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. _vocab_ref:

Vocabulary
----------------------------------

Vocabulary for LanguageProcessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Language generation and evaluation highly depend on vocabulary,
thus we introduce a common setting in ``cotk``. All tokens appeared
in dataset (including the ones only appear in test set) are split into
3 set.

    Valid Vocabularies(``valid_vocabs``)
        * Tokens that the model **should** read, predict and generate.
        * These tokens are important in evaluation. They include
          common words and usually cover over 90% of tokens
          from dataset.
        * They are extracted from only training set, because models should
          blind for test set. Usually, they are defined as the tokens appear
          more than a specified number of times (``min_vocab_times``) in
          **training set**.

    Invalid Vocabularies(``invalid_vocabs``)
        * Tokens that the model can **optionally** read,
          but will **not** predict and generate at most times
          (**except** some models can generate rare words using
          copy mechanism or external knowledge).
        * These tokens are less important but **DO** affect the evaluation.
          They include rare words.
        * They are extracted from both training set and test set, because they
          are defined considering evaluation. Usually, they are defined as the tokens
          (excluded ``valid_vocabs``) appear more than a specified number 
          (``invalid_vocab_times``) of times in **the whole dataset**.

    Unknown Vocabularies
        * Tokens that model can **NOT** read, predict or generate. They are always
          replaced by a token like ``<unk>``, whenever training or evaluation.
          (While ``<unk>`` is not always `unknown vocabularies`, see below for
          explanation.)
        * These tokens will **not** affect the test. Whatever your model generated,
          the unknown words are always considered wrong or ignored.
        * Usually, they are defined as the tokens appear less than a specified
          number of times (``invalid_vocab_times``) in all data sets, i.e.,
          the ones are neither ``valid_vocabs`` nor ``invalid_vocabs``.
          By default, ``invalid_vocab_times`` is set to ``0``, thus no
          `unknown words` appear in the whole dataset. However, there can
          be words in our real language but out of the dataset, which
          are assumed as `unknown vocabularies`.

There is also some other terms for vocabularies.

    All Vocabularies(``allvocabs``)
        * `Valid vocabularies` and `invalid vocabularies` are also called `all vocabularies`.

    Special Tokens(``ext_vocab``)
        * Most used special tokens are ``<pad>``, ``<unk>``, ``<go>``, ``<eos>``.
        * ``<pad>``'s index is ``0``, representing the empty token, used for padding.
        * ``<unk>``'s index is ``1``, see the next section for details.
        * ``<go>``'s index is ``2``, representing the start of a sentence.
        * ``<eos>``'s index is ``3``, representing the end of a sentence.
        * Sometimes, special tokens are counted as valid vocabularies,
          but exactly, they should be specially handled.

There is a special token ``<unk>``, which is **different** from unknown vocabularies.

    Token <unk>
        * ``<unk>`` means "Out of Vocabularies", but we have different
          vocabularies, so the meaning of ``<unk>`` may varies from situations.
        * If it appears at a list named with ``allvocabs`` (eg: ``sent_allvocabs``),
          ``<unk>`` indicates `unknown vocabularies`.
        * If it appears at a list named without ``allvocabs`` (eg: ``sent``),
          ``<unk>`` indicates `invalid vocabularies` or `unknown vocabularies`.

    Convert from `all vocabularies` to `valid vocabularies`
        * If we have a sentence given in `all vocabularies`, just replace all
          `invalid vocabularies` to ``<unk>``.

    Convert from `valid vocabularies` to `all vocabularies`
        * There will sometime that ``cotk`` need to convert a sentence with 
          `valid vocabularies` to a sentence with `all vocabularies`. But 
          it is **never** returned to user, so you can just skip the section if you
          don't want to bother with too much of details.
          The conversion is performed depend on the condition.
        * For example, :class:`.metric.PerplexityMetric` can accept sentences with
          only `valid vocabularies`. A smoothing method will be applied that
          the probability on ``<unk>`` will be distributed evenly on
          `invalid vocabularies`.
        * For most time, ``<unk>`` in valid vocabularies will be replaced
          by a special token, which don't match any other tokens, including itself.
          The special token is only used inside ``cotk``.

We offer some tips for you to further understand how vocabularies work.

    Use ``allvocabs`` or not ?
        * First of all, you never need to consider `unknown vocabularies`, because a model
          can never generate a totally unknown word.
        * `Valid vocabularies` are often used as input, supervision of generated sentences.
          It is because `invalid vocabularies` are less appeared and hard to learn 
          the representation or generate, thus we can treat them as ``<unk>`` .
        * `All vocabularies` are always used as references of metrics. They can also be used
          as input or supervision, if your model can handle the rare word (like copyNet).

    Index and ``vocab_list``
        * Token's index is always equal to the position that the token in
          :attr:`.LanguageProcessingBase.vocab_list`.
        * Special tokens like ``<pad>``, ``<unk>``, ``<go>``, ``<eos>`` are always at
          the front of :attr:`.LanguageProcessingBase.vocab_list`.
        * Valid vocabularies are following the special tokens. 
          Thus ``vocab_list[:valid_vocab_size]`` includes valid words and special tokens.
        * Invalid vocabularies are the left. ``vocab_list[valid_vocab_size:]`` are
          invalid_vocabs.
        * Unknown vocabs are not in ``vocab_list``, therefore
          they don't have index.

Vocabulary for BERTLanguageProcessingBase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some pretrained model have its own vocabulary so it is hard directly adapted
to our framework. Before stepping in out workaround, make sure you have read
the previous section `Vocabulary for LanguageProcessingBase`.

To make sure the pretrained model works correctly, we use the pretrained
model's tokenizer. It is defined in
:meth:`.BERTLanguageProcessingBase.tokenize`.
For example, BERTLanguageProcessingBase use BERTtokenizer for tokenizing.

Then, we processing the tokenized sentences and define a mapping from
tokenized words to all vocabularies. This process is the same with the one
in the section `Vocabulary for LanguageProcessingBase`.

Also, we can obtained a conversion between tokenized sentences and pretrained
id from pretrained models. Now, we have 3 representation methods of sentences:

* Tokenized sentences.
* Pretrained model's word ids.
* Our word ids. (Both for ``all_vocabs`` and ``valid_vocabs``,
  because they share the id and the only difference is
  ``valid_vocabs`` is less than ``all_vocabs``.)

Here is the path of conversion.

1.  :meth:`.BERTLanguageProcessingBase.convert_bert_ids_to_tokens`.
    May cause ``<unk>`` due to model's id can't cover all tokens.
    (But this won't happen in BERT, because BERT has a tokenizer
    that sentences will always split to what it knows.)

2.  :meth:`.BERTLanguageProcessingBase.convert_ids_to_tokens`.
    May cause ``<unk>`` when there is unknown vocabs or invalid vocabs.

3.  :meth:`.LanguageProcessingBase.convert_ids_to_tokens`.
    May cause ``<unk>`` when there is unknown vocabs.

4.  The same except ``id > valid_vocab_size``.

5.  :meth:`.BERTLanguageProcessingBase.convert_tokens_to_bert_ids`.
    No more ``<unk>``.

6.  :meth:`.LanguageProcessingBase.convert_tokens_to_ids`
    No more ``<unk>``.

7.  The same. No more ``<unk>``.

There is also other conversions:

* :meth:`.BERTLanguageProcessingBase.convert_ids_to_bert_ids` : Just
  equal to (2 or 3) + 5.

* :meth:`.BERTLanguageProcessingBase.convert_bert_ids_to_ids` : Just
  equal to 1 + 6.

In most time, dataloader will provide sentences both in bert id and
our id. And unnecessary conversion should be avoided because it may
cause information loss.


.. _dataloader_hash:

Hash Code for Dataloader
----------------------------------


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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SentenceDefault

    .. automethod:: get_batch

SentenceGPT2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SentenceGPT2

    .. automethod:: get_batch

Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Session

SessionDefault
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

Vocabulary List
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
