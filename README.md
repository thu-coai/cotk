
# Conversational Toolkits

[![CodeFactor](https://www.codefactor.io/repository/github/thu-coai/cotk/badge)](https://www.codefactor.io/repository/github/thu-coai/cotk)
[![codebeat badge](https://codebeat.co/badges/dc64db27-7e25-4fea-a231-3c9baac916f8)](https://codebeat.co/projects/github-com-thu-coai-cotk-master)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/cotk/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/cotk?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/cotk.svg?branch=master)](https://travis-ci.com/thu-coai/cotk)
[![Actions Status](https://github.com/thu-coai/cotk/workflows/windows/badge.svg)](https://github.com/thu-coai/cotk/actions)
[![Actions Status](https://github.com/thu-coai/cotk/workflows/macos/badge.svg)](https://github.com/thu-coai/cotk/actions)

``cotk`` is an open-source lightweight framework for model building and evaluation.
We provides standard dataset and evaluation suites in the domain of general language generation.
It easy to use and make you focus on designing your models!

Features included:

 * Light-weight, easy to start. Don't bother your way to construct models.
 * Predefined standard datasets, in the domain of language modeling, dialog generation and more.
 * Predefined evaluation suites, test your model with multiple metrics in several lines.
 * A dashboard to show experiments, compare your and others' models fairly.
 * Long-term maintenance and consistent development.

This project is a part of ``dialtk`` (Toolkits for Dialog System by Tsinghua University), you can follow [dialtk](http://coai.cs.tsinghua.edu.cn/dialtk/) or [cotk](http://coai.cs.tsinghua.edu.cn/dialtk/cotk/) on our home page.

**Quick links**

* [Tutorial & Documents](https://thu-coai.github.io/cotk_docs/)
* [Dashboard](http://coai.cs.tsinghua.edu.cn/dashboard/)

**Index**

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install from pip](#install-from-pip)
  - [Install from source](#install-from-source)
- [Quick Start](#quick-start)
  - [Dataloader](#Dataloader)
  - [Metrics](#metrics)
  - [Publish Experiments](#publish-experiments)
  - [Reproduce Experiments](#reproduce-experiments)
  - [Predefined Models](#predefined-models)
- [Issues](#issues)
- [Contributions](#Contributions)
- [Team](#team)
- [License](#license)


## Installation

### Requirements

-  python 3
-  numpy >= 1.13
-  nltk >= 3.4
-  tqdm >= 4.30
-  checksumdir >= 1.1
-  pytorch >= 1.0.0 (optional, accelerating the calculation of some metrics)
-  transformers (optional, used for pretrained models)

We support Unix, Windows, and macOS.

### Install from pip

You can simply get the latest stable version from pip using

```bash
    pip install cotk
```

### Install from source code

* Clone the cotk repository

```bash
    git clone https://github.com/thu-coai/cotk.git
```

* Install cotk via pip

```bash
    cd cotk
    pip install -e .
```



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


**note**: If you want to know more about ``Dataloader``, please refer to [docs of dataloader](https://thu-coai.github.io/cotk_docs/index.html#model-zoo).

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

You can merge multiple metrics together by cotk.metric.MetricChain.


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


**note**: If you want to know more about metrics, please refer to [docs of metrics](https://thu-coai.github.io/cotk_docs/metric.html).

### Predefined Models


We have provided some baselines for the classical tasks, see [Model Zoo](https://thu-coai.github.io/cotk_docs/index.html#model-zoo) in docs for details.

You can also use ``cotk download thu-coai/MODEL_NAME/master`` to get the codes.


## Issues

You are welcome to create an issue if you want to request a feature, report a bug or ask a general question.

## Contributions

We welcome contributions from community. 

* If you want to make a big change, we recommend first creating an issue with your design.
* Small contributions can be directly made by a pull request.
* If you like make contributions for our library, see issues to find what we need.

## Team

`cotk` is maintained and developed by Tsinghua university conversational AI group (THU-coai). Check our [main pages](http://coai.cs.tsinghua.edu.cn/) (In Chinese).

## License

Apache License 2.0

