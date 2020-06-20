Extending Cotk: More Data, More Metrics!
============================================

We provide easy APIs for extend ``cotk`` to custom datasets and tasks.

Add A New Dataset
----------------------------------------------

For Local Use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have a new dataset and want to load it with ``cotk`` and
the task is similar with the existed tasks, like :class:`.dataloader.LanguageGeneration`,
:class:`.dataloader.SingleTurnDialog` or :class:`.dataloader.MultiTurnDialog`.
The simplest way is to just adapt the format of your dataset.

For example, if you want to adapt your data to :class:`.dataloader.LanguageGeneration`,
you just have to generate 3 text files named ``train.txt``,
``dev.txt``, ``test.txt``.
Each file contains several lines and each line is a sentence.
The structure of your directory should be like:

.. code-block:: none

    mydata
    ├── train.txt
    ├── dev.txt
    └── test.txt

Then you can load your data using :class:`.dataloader.LanguageGeneration` with a local path.

.. code-block:: python

    dataloader = LanguageGeneration("./path/to/mydata", min_frequent_vocab_times=min_frequent_vocab_times,
                max_sent_length=max_sent_length, min_rare_vocab_times=min_rare_vocab_times,
                tokenizer=tokenizer, convert_to_lower_letter=convert_to_lower_letter)

.. note ::

    If you want to write your own ``dataloader`` with a complex data format,
    see `Add A New Task`_.

Download Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to publish your dataset and make it possible to download them automatically.
You can zip your data and upload to an server. The url path should be accessible for every one.

Using :class:`.dataloader.LanguageGeneration` with a url path is adequate for
the requirements.

.. code-block:: python

    dataloader = MSCOCO("http://url/to/new_data.zip")

.. note ::

    The zip file is downloaded then processed by
    :class:`.file_utils.resource_processor.DefaultResourceProcessor`.
    For more about ``ResourceProcessor``, refer to :ref:`this <resources_reference>`.

Add A Resource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _resources_desc:

This time, you have to change some codes in ``cotk``.
First you have to understand the usage of resources at
:ref:`here <resources_reference>`. Then see the json file under
``/cotk/resource_config/``, you will find each predefined resource
corresponds to a json file, like

.. code-block:: javascript

    {
        "name": "MSCOCO_small",
        "type": "MSCOCO",
        "hashtag": "68f6b8d764bff6f5440a63f87aeea97049a1d2e89942a7e524b7dabd475ffd79",
        "link": {
            "default":"https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip",
            "amazon": "https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip"
        }
    }

There are some places you have to pay attention to:

    * ``type`` is the prefix of its ``ResourceProcessor``.
    * ``link.default`` is necessary when no source is specified.
    * ``hashtag`` is required for checksum.

We use the following codes to hash the zip file.

.. code-block :: python

    def _get_file_sha256(file_path):
        '''Get sha256 of given file'''
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as fin:
            for chunk in iter(lambda: fin.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

After accomplishment the config file, you can use the following code to load the data.

.. code-block:: python

    dataloader = MSCOCO("resources://new_name")

We highly recommend that the developers make a **Pull Request** and add more datasets for ``cotk``.

Add A New Task
----------------------------------------------

Sometimes you want to deal with a totally different task
from the predefined ones.
In that case, you have to implement a subclass of :class:`.LanguageProcessing`,
and pass the parameters ``file_id`` and ``fields`` when invoking :meth:`.LanguageProcessing.__init__`.
For more details about ``file_id``, refer :ref:`this <resources_reference>`.
For more details about ``fields``, refer :ref:`this <dataloader_reference>`.

.. note ::

    In the method ``__init__`` of your own dataloader class, :meth:`.LanguageProcessing.set_default_field` must be
    called. If ``self.default_field_set_name`` and ``self.default_field_name`` are not set, some methods and properties
    (such as :meth:`.LanguageProcessing.tokenize`, :attr:`.LanguageProcessing.all_vocab_size`, etc.) aren't available.

For example, you can implement a new dataloader for sentence classification.

.. code-block:: python

    from collections import OrderedDict
    from cotk.dataloader import LanguageProcessing
    from cotk.dataloader.context import FieldContext, VocabContext
    class SentenceClassification(LanguageProcessing):
        def __init__(self, file_id: str,
                    tokenizer=None,
                    max_sent_length=None,
                    convert_to_lower_letter=None,
                    min_frequent_vocab_times=None,
                    min_rare_vocab_times=None):
            fields = OrderedDict([('sent', 'SentenceDefault'), ('label', 'DenseLabel')])
            with FieldContext.set_parameters(tokenizer=tokenizer,
                                                max_sent_length=max_sent_length,
                                                convert_to_lower_letter=convert_to_lower_letter):
                with VocabContext.set_parameters(min_rare_vocab_times=min_rare_vocab_times,
                                                    min_frequent_vocab_times=min_frequent_vocab_times):
                    super().__init__(file_id, fields)
            self.set_default_field('train', 'sent')

Assume that there is a directory named ``mydata``, which contains 3 text files (``train.txt``, ``dev.txt`` and ``test.txt``) in the same format.
For example, the content of ``test.txt`` is as follows. Each sentence is followed by an integer (the label), just as ``fields`` specifies.

.. code-block:: none

    effective but too-tepid biopic.
    2
    if you sometimes like to go to the movies to have fun, wasabi is a good place to start.
    3
    emerges as something rare, an issue movie that's so honest and keenly observed that it doesn't feel like one.
    4
    the film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game.
    2
    offers that rare combination of entertainment and education.
    4


Then, you can use ``SentenceClassification`` to build the dataloader.

.. code-block:: python

    dl = SentenceClassification("mydata", tokenizer="nltk", convert_to_lower_letter=True)
    dl.restart('test', batch_size=2, shuffle=False)
    dl.get_next_batch('test')

The returned value of ``dl.get_next_batch`` is as follows.

.. code-block:: javascript

    {'sent_length': array([ 9, 23]),
    'sent': array([[  2,   1,  31,   1,  11,   1,   1,   5,   3,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  2,   1, 112,   1,   1,  13,   1,  13,   4,   1,  13,  62,   1,
        9,   1,  12,   8,   1,   1,  13,   1,   5,   3]]),
    'sent_allvocabs': array([[  2, 138,  31, 191,  11, 189, 129,   5,   3,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  2, 114, 112, 185, 118,  13, 149,  13,   4, 165,  13,  62, 146,
        9, 198,  12,   8, 151, 174,  13, 186,   5,   3]]),
    'sent_str': ['effective but too-tepid biopic.',
            'if you sometimes like to go to the movies to have fun, wasabi is a good place to start.'],
    'label': array([2, 3])
    }


Add A New Metric
---------------------------------------------

If you have a new way to evaluate the model, you should construct a
metric class inheriting the :class:`.metric.MetricBase`.

Here are some necessary functions you must implement. You can click on
the link to find more details.

* ``__init__()``
* :meth:`.MetricBase.forward`
* :meth:`.MetricBase.close`

Here we give an example for calculating the average length of generated
sentences.

.. code-block:: python

    class AverageLengthMetric(MetricBase):
        def __init__(self, dataloader, gen_key="gen"):
            super().__init__()
            self.dataloader = dataloader
            self.gen_key = gen_key
            self.token_num = 0
            self.sent_num = 0

        def forward(self, data):
            gen = data[gen_key]
            for sent in gen:
                self.token_num += len(self.dataloader.trim_in_ids(sent))
                self.sent_num += 1

        def close(self):
            metric_result = super().close()
            metric_result.update({"len_avg": self.token_num / self.sent_num})
            return metric_result

There is some regulations to design an metric.

* Using :ref:`allvocabs <vocabulary_ref>` for reference.
* Dealing with ``<unk>``, which should be regarded as error or
  using some methods to do smoothing. Pay attention to the connections
  between ``<unk>`` and
  :ref:`rare vocabularies <vocabulary_ref>`.
* Record hash value. Hash value keeps the same if and only if the metric is tested
  under the same settings. :meth:`.metric.MetricBase._hash_unordered_list` records unordered information. :meth:`.metric.MetricBase._hash_ordered_data` records the ordered information.
  :meth:`.metric.MetricBase._hashvalue` returns the hash value.
  (In the example, there is no hash value because we don't have input and the setting is always the same)

