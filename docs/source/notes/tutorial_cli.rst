CLI Usage: Fast Model Reproduction
==========================================

To facilitate the communication among researchers, we equip ``cotk`` with command line tools which allow you to

- download public code from github repositories or a dashboard we built for fair comparisons among models
- reproduce others' work provided that the code complies with a small set of protocols
- publish model performance and code to the dashboard

Download public code
----------------------------------------

The command line ``cotk download`` helps download public code from github repositories or the dashboard
and create a bash file named ``run_model.sh`` which runs the code.
After running the bash file with the following command line, the code should report results
in a file named ``result.json`` under the root of the code directory.

.. code-block:: none

    bash run_model.sh

Download from public github repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command line:

.. code-block:: none

    cotk download <model>

``<model>`` should be a string that indicates the code path. It can be:

.. code-block:: none

    * url of a git repo / branch / commit.
        Example:    https://github.com/USER/REPO
                    https://github.com/USER/REPO/tree/BRANCH
                    https://github.com/USER/REPO/commit/COMMIT (full commit id is needed)
    * a string specifying a git repo / branch / commit.
        Example:    USER/REPO
                    USER/REPO/BRANCH
                    USER/REPO/COMMIT (full commit id is needed)

The command line above will create the following files under the current working directory:

.. code-block:: none

    .
    └── REPO_COMMIT
        ├── REPO-COMMIT // the root of the code directory
        │   └── ... // code files
        └── run_model.sh

.. note::

    ``run_model.sh`` will be created only if the root of the git repository includes
    a file named ``config.json`` that specifies run configurations
    (Refer to :ref:`Make Your Model Reproducible <protocols_ref>` for more details).

Download from the dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command line:

.. code-block:: none

    cotk download <model> --result <path_to_dump_result>

``<model>`` is a model ID (a sequence of digits) on the dashboard.
``<path_to_dump_result>`` is the path where you want to dump the information of the model, e.g., `dashboard_result.json`.
This result file includes evaluation results of the model as well as the run configurations for reproducing them.

The command line above will create the following files under the current working directory:

.. code-block:: none

    .
    └── <model>
        ├── REPO-COMMIT // the root of the code directory
        │   └── ... // code files
        └── run_model.sh

The ``REPO`` and ``COMMIT`` here are the git repository and commit ID associated with the model ID, respectively.

.. _protocols_ref:

Make Your Model Reproducible
-------------------------------

Suppose you have written your model under a directory named ``PROJECT``.
By complying with the following protocols, you are able to make your model easily reproduced by anyone with ``cotk``:

- There exists an entry file named ``<entry>.py`` somewhere in ``PROJECT``.
  This file defines an entry function named ``run(*argv)``.
  After running ``python <entry>.py <args>``, the entry function ``run`` takes ``<args>`` as parameter
  and reports evaluation results of the model in a file named ``result.json`` right below ``PROJECT``.
- There may exists a file named ``config.json`` right below ``PROJECT``.
  This file specifies run configurations which consist of ``entry``, ``args``, and ``working_dir``.

  - ``args`` is a :class:`list`, e.g., `["--batch_size", 32]`.
  - ``working_dir`` is the path relative to ``PROJECT`` where ``<entry>.py`` locates, e.g., `./`.

  **Note that** ``cotk`` cannot create ``run_model.sh`` with any of these run configurations missing
  if the code is downloaded from the github repository.
- ``PROJECT`` is associated with a public github repository.
  Files and run configurations —— with which you produce your public model performance —— are committed.

Publish Your Model and Compete with Others
-------------------------------------------

The command line ``cotk run`` is to publish your model to the dashboard.
Besides showing evaluation results of models, this dashboard also tells
whether any two models are referring to the same data which aims at fair comparisons
(Refer to :ref:`Metric <hash_ref>` for more details).

.. code-block:: none

    usage: cotk run [-h] [--token TOKEN] [--result RESULT] [--only-run]
                [--only-upload] [--entry [ENTRY]]
                ...

    Run model and report performance to cotk dashboard.
    
    positional arguments:
      args
    
    optional arguments:
      -h, --help       show this help message and exit
      --token TOKEN
      --result RESULT  Path where you store your model performance.
                       Default: result.json
      --only-run       Just run my model, don't collect any information or upload
                       anything.
      --only-upload    Don't run my model, just upload the existing result. (Some
                       information will be missing and this option is not
                       recommended.)
      --entry [ENTRY]  Entry of your model. Default: main

.. note::

    Under any circumstances, you should be under the ``working_dir`` and correctly specify ``entry`` and ``args``.

By setting ``--only-run``, you choose to run your model locally without publishing results to the dashboard.
``--token``, ``--result`` and ``only-upload`` can be ignored.

By not setting ``--only-run``, you choose to publish your model.

``TOKEN`` is a non-empty string for registration or identification on the dashboard.
You can save it locally by running the following command line,
which saves you from typing it every time you publish your model.

.. code-block:: none

    cotk config --token TOKEN

Without ``--only-upload``, ``cotk run`` will run your model and record the runtime information.

The information to be uploaded consists of ``working_dir``, ``entry``, ``args``, model performance in ``RESULT``,
the associated github repository and commit ID, and the runtime information (empty when ``--only-upload`` is set).
After successfully publishing your model to the dashboard, ``cotk run`` will return a URL of the online report.

.. note::

    If you are to publish your model, all the protocols in :ref:`Make Your Model Reproducible <protocols_ref>`
    should be satisfied except that ``config.json`` is unnecessary.
