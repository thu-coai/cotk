# Seq2Seq (PyTorch)

SST(Stanford Sentiment Treebank) is a benchmark task for sentiment classification. Here we use a simple GRU+MLP model for this task.

You can refer to the following web for information about SST:

https://nlp.stanford.edu/sentiment/

## Require Packages

* **python3**
* cotk
* pytorch >= 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Execute ``python run.py`` to train the model.
  * The default dataset is ``SST``. You can use ``--dataset`` to specify other ``dataloader`` class and ``--datapath`` to specify other data path (can be a local path, a url or a resources id). For example: ``--dataset SST --datapath resources://SST``
  * It doesn't use pretrained word vector by default setting. You can use ``--wvclass`` to specify ``wordvector`` class and ``--wvpath`` to specify pretrained word embeddings. For example: ``--wvclass gloves``. For example: ``--dataset Glove --datapath resources://Glove300``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``. For example: ``--restore pretrained-sst`` for loading ``./model/pretrained-sst.model``
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

```none
    usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE]
                  [--eh_size EH_SIZE] [--droprate DROPRATE]
                  [--batchnorm]
                  [--dataset DATASET] [--datapath DATAPATH] [--epoch EPOCH]
                  [--wvclass WVCLASS] [--wvpath WVPATH] [--out_dir OUT_DIR]
                  [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
                  [--cache_dir CACHE_DIR] [--cpu] [--debug] [--cache]

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           The name of your model, used for tensorboard, etc.
                            Default: runXXXXXX_XXXXXX (initialized by current
                            time)
      --restore RESTORE     Checkpoints name to load. "NAME_last" for the last
                            checkpoint of model named NAME. "NAME_best" means the
                            best checkpoint. You can also use "last" and "best",
                            defaultly use last model you run. Attention:
                            "NAME_last" and "NAME_best" are not guaranteed to work
                            when 2 models with same name run in the same time.
                            "last" and "best" are not guaranteed to work when 2
                            models run in the same time. Default: None (dont load
                            anything)
      --mode MODE           "train" or "test". Default: train
      --eh_size EH_SIZE     Size of encoder GRU
      --droprate DROPRATE   The probability to be zerod in dropout. 0 indicates
                            for dont use dropout
      --batchnorm           Use bathnorm
      --dataset DATASET     Dataloader class. Default: OpenSubtitles
      --datapath DATAPATH   Directory for data set. Default: ./data
      --epoch EPOCH         Epoch for trainning. Default: 100
      --wvclass WVCLASS     Wordvector class, none for not using pretrained
                            wordvec. Default: None
      --wvpath WVPATH       Directory for pretrained wordvector. Default:
                            ./wordvec
      --out_dir OUT_DIR     Output directory for test output. Default: ./output
      --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
      --model_dir MODEL_DIR Checkpoints directory for model. Default: ./model
      --cache_dir CACHE_DIR Checkpoints directory for cache. Default: ./cache
      --cpu                 Use cpu.
      --debug               Enter debug mode (using ptvsd).
      --cache               Use cache for speeding up load data and wordvec. (It
                            may cause problems when you switch dataset.)
```

## TensorBoard

Execute ``tensorboard --logdir=./tensorboard``, you will see the plot in tensorboard pages:

## Case Study of Model Results

Execute ``python run.py --mode test --restore best``

The following output will be in `./output/[name]_[dev|test].txt`:


## Performance

|     | dev | test  |
| --- | ---------- | ----- |
| SST | 0.411      | 0.424 |

## Author

[YILIN NIU](https://github.com/heyLinsir)
