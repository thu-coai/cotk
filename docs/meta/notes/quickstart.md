## Quick Start

Let's skim through the whole package to find what you want. 

### Dataloader

Load common used dataset and do preprocessing:

* Download online resources or import from local path
* Split training set, development set and test set
* Construct vocabulary list

```python
    >>> import cotk.dataloader
    >>> # automatically download online resources
    >>> dataloader = cotk.dataloader.MSCOCO("resources://MSCOCO_small")
    >>> # or download from a url
    >>> dl_url = cotk.dataloader.MSCOCO("http://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip#MSCOCO")
    >>> # or import from local file
    >>> dl_zip = cotk.dataloader.MSCOCO("./MSCOCO.zip#MSCOCO")

    >>> print("Dataset is split into:", dataloader.fields.keys())
    dict_keys(['train', 'dev', 'test'])
```

Inspect vocabulary list

```python
    >>> print("Vocabulary size:", dataloader.frequent_vocab_size)
    Vocabulary size: 2597
    >>> print("First 10 tokens in vocabulary:", dataloader.frequent_vocab_list[:10])
    First 10 tokens in vocabulary: ['<pad>', '<unk>', '<go>', '<eos>', '.', 'a', 'A', 'on', 'of', 'in']
```

Convert between ids and strings

```python
    >>> print("Convert string to ids", \
    ...           dataloader.convert_tokens_to_ids(["<go>", "hello", "world", "<eos>"]))
    Convert string to ids [2, 6107, 1875, 3]
    >>> print("Convert ids to string", \
    ...           dataloader.convert_ids_to_tokens([2, 1379, 1897, 3]))
	Convert ids to string ['hello', 'world']
```

Iterate over batches

```python
    >>> for data in dataloader.get_batches("train", batch_size=1):
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

Or using ``while`` (another iteration method) if you like

```python
    >>> dataloader.restart("train", batch_size=1):
    >>> while True:
    ...    data = dataloader.get_next_batch("train")
    ...    if data is None: break
    ...    print(data)
```

!! ifndef readme
.. note::

   If you want to know more about ``Dataloader``, please refer to :mod:`docs of dataloader <cotk.dataloader>`.
!! endif

!! ifdef readme
**note**: If you want to know more about ``Dataloader``, please refer to [docs of dataloader](https://thu-coai.github.io/cotk_docs/index.html#model-zoo).
!! endif

### Metrics

We found there are different versions of the same metric in different papers,
which leads to **unfair comparison between models**. For example, whether considering
``unk``, calculating the mean of NLL across sentences or tokens in
``perplexity`` may introduce huge differences.

We provide a unified implementation for metrics, where ``hashvalue`` is provided for
checking whether the same data is used. The metric object receives mini-batches.

```python
    >>> import cotk.metric
    >>> metric = cotk.metric.SelfBleuCorpusMetric(dataloader, gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02253475750490193, 'self-bleu hashvalue': 'f7d75c0d0dbf53ffba4b845d1f61487fd2d6d3c0594b075c43111816c84c65fc'}
```

!! ifndef readme
You can merge multiple metrics together by :class:`cotk.metric.MetricChain`.
!! endif
!! ifdef readme
You can merge multiple metrics together by cotk.metric.MetricChain.
!! endif


```python
    >>> metric = cotk.metric.MetricChain()
    >>> metric.add_metric(cotk.metric.SelfBleuCorpusMetric(dataloader, gen_key="gen"))
    >>> metric.add_metric(cotk.metric.FwBwBleuCorpusMetric(dataloader, reference_test_list=dataloader.get_all_batch('test')['sent_allvocabs'], gen_key="gen"))
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    100%|██████████| 1000/1000 [00:00<00:00, 5281.95it/s]
	{'self-bleu': 0.02253475750490193, 'self-bleu hashvalue': 'f7d75c0d0dbf53ffba4b845d1f61487fd2d6d3c0594b075c43111816c84c65fc', 'fw-bleu': 0.28135593382545376, 'bw-bleu': 0.027021522872801896, 'fw-bw-bleu': 0.04930753293488745, 'fw-bw-bleu hashvalue': '60a39f381e065e8df6fb5eb272984128c9aea7dee4ba50a43bfb768395a70762'}
```

We also provide recommended metrics for selected dataloader.

```python
    >>> metric = dataloader.get_inference_metric(gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    100%|██████████| 1000/1000 [00:00<00:00, 4857.36it/s]
	100%|██████████| 1250/1250 [00:00<00:00, 4689.29it/s]
	{'self-bleu': 0.02253475750490193, 'self-bleu hashvalue': 'f7d75c0d0dbf53ffba4b845d1f61487fd2d6d3c0594b075c43111816c84c65fc', 'fw-bleu': 0.3353037449663603, 'bw-bleu': 0.027327995838287513, 'fw-bw-bleu': 0.050537105917262654, 'fw-bw-bleu hashvalue': 'c254aa4008ae11b1bc4955e7cd1f7f3aad34b664178a585a218b1474970e3f23', 'gen': [['inside', 'is', 'an', 'elephant', 'shirt', 'of', 'people', 'and', 'a', 'grasslands', 'pulls', '.'], ['An', 'elephant', 'girls', 'baggage', 'sidewalk', 'with', 'a', 'clock', 'on', 'it', '.']]}
```

!! ifndef readme
.. note::

   If you want to know more about metrics, please refer to :mod:`docs of metrics <cotk.metric>`.
!! endif

!! ifdef readme
**note**: If you want to know more about metrics, please refer to [docs of metrics](https://thu-coai.github.io/cotk_docs/metric.html).
!! endif

### Predefined Models

!! ifndef readme
We have provided some baselines for the classical tasks, see :ref:`Model Zoo <model_zoo>` in docs for details.
!! endif

!! ifdef readme
We have provided some baselines for the classical tasks, see [Model Zoo](https://thu-coai.github.io/cotk_docs/index.html#model-zoo) in docs for details.
!! endif

You can also use ``cotk download thu-coai/MODEL_NAME/master`` to get the codes.
