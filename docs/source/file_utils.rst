file_utils
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: cotk.file_utils.file_utils

.. autofunction:: _url_to_filename
.. autofunction:: _get_config
.. autofunction:: _http_get
.. autofunction:: _get_file_sha256
.. autofunction:: _get_hashtag
.. autofunction:: _parse_file_id
.. autofunction:: _get_resource
.. autofunction:: _download_data
.. autofunction:: _load_local_data
.. autofunction:: get_resource_file_path
.. autofunction:: import_local_resources
.. autofunction:: load_file_from_url

hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: cotk.hooks

.. autofunction:: invoke_listener
.. autofunction:: compress_dict
.. autofunction:: hook_dataloader
.. autofunction:: hook_metric
.. autofunction:: hook_metric_close
.. autofunction:: hook_wordvec

.. autoclass:: HooksListener
  :members:
  :private-members:

.. autoclass:: SimpleHooksListener
  :members:
  :private-members:

.. autofunction:: start_recorder
.. autofunction:: close_recorder

resources_processor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: cotk.file_utils.resource_processor

.. autofunction:: unzip_file

.. autoclass:: ResourceProcessor
  :members:
  :private-members:

.. autoclass:: DefaultResourceProcessor
  :members:
  :private-members:

.. autoclass:: BaseResourceProcessor
  :members:
  :private-members:

.. autoclass:: MSCOCOResourceProcessor
  :members:
  :private-members:

.. autoclass:: OpenSubtitlesResourceProcessor
  :members:
  :private-members:

.. autoclass:: UbuntuResourceProcessor
  :members:
  :private-members:

.. autoclass:: SwitchboardCorpusResourceProcessor
  :members:
  :private-members:

.. autoclass:: SSTResourceProcessor
  :members:
  :private-members:

.. autoclass:: GloveResourceProcessor
  :members:
  :private-members:

.. autoclass:: Glove50dResourceProcessor
  :members:
  :private-members:

.. autoclass:: Glove100dResourceProcessor
  :members:
  :private-members:

.. autoclass:: Glove200dResourceProcessor
  :members:
  :private-members:

.. autoclass:: Glove300dResourceProcessor
  :members:
  :private-members: