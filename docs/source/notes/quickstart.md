## Quick Start

Let us skim through the whole package to find what you want. 

### Dataloader

Load common used dataset and preprocess for you:

* Download online resources or import from local
* Split training set, development set and test set
* Construct vocabulary list

```python
    >>> # automatically download online resources
    >>> dataloader = cotk.dataloader.MSCOCO("resources://MSCOCO_small")
    >>> # or download from a url
    >>> dl_url = cotk.dataloader.MSCOCO("http://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip#MSCOCO")
    >>> # or import form local file
    >>> dl_zip = cotk.dataloader.MSCOCO("./MSCOCO.zip#MSCOCO")
    
    >>> print("Dataset is split into:", dataloader.key_name)
    ["train", "dev", "test"]
```

Inspect vocabulary list

```python
    >>> print("Vocabulary size:", dataloader.vocab_size)
    Vocabulary size: 2588
    >>> print("Frist 10 tokens in vocabulary:", dataloader.vocab_list[:10])
    Frist 10 tokens in vocabulary: ['<pad>', '<unk>', '<go>', '<eos>', '.', 'a', 'A', 'on', 'of', 'in']
```

Convert between ids and strings

```python
    >>> print("Convert string to ids", \
    ...           dataloader.convert_tokens_to_ids(["<go>", "hello", "world", "<eos>"]))
    Convert string to string [2, 1379, 1897, 3]
    >>> print("Convert ids to string", \
    ...           dataloader.convert_ids_to_tokens([2, 1379, 1897, 3]))
```

Iterate over batches

```python
    >>> for data in dataloader.get_batch("train", batch_size=1):
    ...     print(data)
    {'sent':
        array([[ 2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 1, 1099, 4, 3]]),
        # <go> This is an old photo of people and a <unk> wagon.
     'sent_allvocabs':
        array([[ 2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3]]),
        # <go> This is an old photo of people and a horse-drawn wagon.
     'sent_length': array([14])}
    ......
```

or using ``while`` if you like

```python
    >>> dataloader.restart("train", batch_size=1):
    >>> while True:
    ...    data = dataloader.get_next_batch("train")
    ...    if data is None: break
    ...    print(data)
```

.. note::

   If you want to know more about data loader, please refer to :mod:`docs <cotk.dataloader>`.


### Metrics

We found there are different versions of the same metric in released codes on Github,
which leads to unfair compare between models. For example, whether considering
``unk``, calculating the mean of NLL across sentences or tokens in
``perplexity`` may introduce **an error of several times** and **extremely** harm the evaluation.

We provide unified metrics implementation for all models. The metric object
receives data in batch.

```python
    >>> metric = cotk.metric.SelfBleuCorpusMetric(dataloader, gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02206768072402293,
     'self-bleu hashvalue': 'c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452'}
```

We also provide standard metrics for selected dataloader.

```python
    >>> metric = dataloader.get_inference_metric(gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02206768072402293,
     'self-bleu hashvalue': 'c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452',
     'fw-bleu': 0.3831004349785445, 'bw-bleu': 0.025958979254273006, 'fw-bw-bleu': 0.04862323612604027,
     'fw-bw-bleu hashvalue': '530d449a096671d13705e514be13c7ecffafd80deb7519aa7792950a5468549e',
     'gen': [
         ['<go>', 'This', 'is', 'an', 'old', 'photo', 'of', 'people', 'and', 'a', 'horse-drawn', 'wagon', '.'],
         ['<go>', 'An', 'old', 'stone', 'castle', 'tower', 'with', 'a', 'clock', 'on', 'it', '.']
     ]}
```

``Hash value`` is provided for checking whether the same dataset is used.

.. note::

   If you want to know more about metrics, please refer to :mod:`docs <cotk.metric>`.


### Publish and Reproduce Experiments

We provide an online dashboard to manage your experiments.

First initialize a git repo in your command line.

```bash
    git init
```

Then write your model with an entry function in ``main.py``.

```python
    import cotk
    import json

    def run():
        dataloader = cotk.dataloader.MSCOCO("resources://MSCOCO_small")
        metric = dataloader.get_inference_metric()
        metric.forward({
            "gen":
                [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
                [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
        })
        json.dump(metric.close(), open("result.json", 'w'))
```

Next, commit your changes and set upstream branch in your command line.
```bash
    git add -A
    git commit -a -m "init"
    git remote add origin master https://github.com/USERNAME/REPONAME.git
    git push origin -f master
```

Finally, type ``cotk run --entry main`` to run your model.

``cotk`` will automatically collect your git repo, username, commit and ``result.json``
to the cotk dashboard (TO BE ONLINE).

FILL AN IMAGE HERE

The dashboard is a website where you can manage your experiments or share
results with others.

You can also download others' experiments in dashboard 
and try to reproduce their results.

```bash
    cotk download ID
```

The ``ID`` comes from dashboard id. 
``cotk`` will download the codes from dashboard and tell you how to run the models.

```none
INFO: Fetching REPO/USERNAME/COMMIT
13386B [00:00, 54414.25B/s]
INFO: Codes from REPO/USERNAME/COMMIT fetched.
INFO: Model running cmd written in run_model.sh
Model running cmd:  cd ./PATH && cotk run --only-run --main run
```

### Predefined Models

We have provided some baselines for the classical tasks, see :ref:`Model Zoo <model_zoo>` in docs for details.


You can also use ``cotk download thu-coai/MODEL_NAME/master`` to get the codes.
