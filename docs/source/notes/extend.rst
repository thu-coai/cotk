Extending Cotk: More Data, More Metrics!
============================================

We hope ``cotk`` can be adapted to more datasets
and more tasks. Therefore, we will have a repository (TO BE ONLINE)
to collect any contribution to cotk (regardless its quality).
And our maintainer will choose modules with high-quality
and merge them to the main repository.

Add A New Dataset
----------------------------------------------

For local use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have a new dataset and want to load it with ``cotk`` and
the task is similar with the existed tasks, like :class:`.dataloader.LanguageGeneration`,
:class:`.dataloader.SingleTurnDialog` or :class:`.dataloader.MultiTurnDialog`.
The simplest way is to just adapt the format of dataset.

For example, if you want to adapt your data to like :class:`.dataloader.LanguageGeneration`,
you just have to generate 3 text file named ``mscoco_train.txt``, ``mscoco_dev.txt``, ``mscoco_test.txt``.
Each file has several lines of sentences.

Then you can load them using :class:`.dataloader.MSCOCO` with a local path.

.. note ::

    If you want to write your own ``dataloader`` without changing data format,
    see `Add A New Task`_.

Auto downloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to publish your dataset and make them can be
downloaded automatically. You can zip your data and upload to an server.
You should have an url path that every one can access it.

Using :class:`.dataloader.MSCOCO` with a url path is adequate for
the requirements.

Use a resouces name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This time, you have to change some code in ``cotk``. See the json file under
``/cotk/resource_config/``, you will find each predefined resource
corresponds to a json file, where a hash value is required for checksum.

We use the following codes to hash the zip file.

.. code-block :: python

    def _get_file_sha256(file_path):
        '''Get sha256 of given file'''
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as fin:
            for chunk in iter(lambda: fin.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

After accomplishment the config file, make a **Pull Request** and wait for update!

Add A New Task
----------------------------------------------




Add A New Metric
---------------------------------------------
