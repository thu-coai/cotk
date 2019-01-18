# Seq2Seq -- a pytorch implementation

Seq2seq is a basic model for single turn dialog. You can refer to the following paper for details:

Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Advances in neural information processing systems* (pp. 3104-3112).

## Require Packages

* contk
* pytorch == 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Downloading dataset and save it to ``./data``. (Dataset will be released soon.)
* Execute ``python run.py`` to train the model.
  * The default dataset is ``OpenSubtitles``. You can use ``--dataset`` to specify other ``dataloader`` class.
  * It don't use pretrained word vector by default setting. You can use ``--wvclass`` to specify ``wordvector`` class. For example: ``--wvclass gloves``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``.
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

    usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE]
                  [--dataset DATASET] [--datapath DATAPATH] [--epoch EPOCH]
                  [--wvclass WVCLASS] [--wvpath WVPATH] [--out_dir OUT_DIR]
                  [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
                  [--cache_dir CACHE_DIR] [--cpu] [--debug] [--cache]

    A seq2seq model

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
                            models run in the same time. Default: None (don't load
                            anything)
      --mode MODE           "train" or "test". Default: train
      --dataset DATASET     Dataloader class. Default: OpenSubtitles
      --datapath DATAPATH   Directory for data set. Default: ./data
      --epoch EPOCH         Epoch for trainning. Default: 100
      --wvclass WVCLASS     Wordvector class, none for not using pretrained
                            wordvec. Default: None
      --wvpath WVPATH       Directory for pretrained wordvector. Default:
                            ./wordvec
      --out_dir OUT_DIR     Output directory for test output. Default: ./output
      --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
      --model_dir MODEL_DIR
                            Checkpoints directory for model. Default: ./model
      --cache_dir CACHE_DIR
                            Checkpoints directory for cache. Default: ./cache
      --cpu                 Use cpu.
      --debug               Enter debug mode (using ptvsd).
      --cache               Use cache for speeding up load data and wordvec. (It
                            may cause problems when you switch dataset.)

### For developer

* Arguments above (except ``cache``\\``debug``) are required. You should remain the same behavior (not for implementation).

* You can add more arguments if you want.

## An example of tensorboard

Execute ``tensorboard --logdir=./``, you will see the plot in tensorboard pages:

![tensorboard_plot_example](images/tensorboard_plot_example.png)

Following plot are shown in this model:

* gen/loss (``gen`` means training process)

* gen/perplexity (``=exp(gen/word_loss)``)

* gen/word_loss (``=gen/loss`` in this model)

* dev/loss
* dev/perplexity_avg_on_batch
* test/loss
* test/perplexity_avg_on_batch

And text output:

![tensorboard_plot_example](images/tensorboard_text_example.png)

Following text are shown in this model:

* args
* dev/show_str%d (``%d`` is according to ``args.show_sample`` in ``run.py``)

## An example of test output

Execute ``python run.py --mode test --restore best``

The following output will be in `./output/[name]_[dev|test].txt`:

```
perplexity:     48.194050
bleu:    0.320098
post:   my name is josie .
resp:   <unk> <unk> , pennsylvania , the <unk> state .
gen:    i' m a teacher .
post:   i put premium gasoline in her .
resp:   josie , i told you .
gen:    i don' t know .
post:   josie , dont hang up
resp:   they do it to aii the new kids .
gen:    aii right , you guys , you know what ?
post:   about playing a part .
resp:   and thats the theme of as you like it .
gen:    i don' t know .
......
```

### For developer

- You should remain similar output in this task.

## Performance

|               | Perplexity | BLEU  |
| ------------- | ---------- | ----- |
| OpenSubtitles | 51.45      | 0.165 |

## Author

[HUANG Fei](https://github.com/hzhwcmhf)
