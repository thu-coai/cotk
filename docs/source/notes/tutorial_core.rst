GRU Language Model: Load Data and Evaluate Models
====================================================

In this tutorial, we will train a neural language model on MSCOCO dataset.
We will focus on how to use ``cotk`` rather than the neural networks,
so we assume you have known how to construct a neural network.

After reading this tutorial, you may know:

- How to use :mod:`cotk.dataloader` downloading and loading dataset.
- How to train model with the support of ``cotk``.
- How to use :mod:`cotk.metric` evaluating models.

``cotk`` does **not** rely on any deep learning framework,
so you can even use shallow models like ngram language model.
However, this tutorial constructs neural networks with
``pytorch``, so make sure you have installed the following package:

- Python >= 3.5
- cotk
- pytorch >= 1.0.0
- livelossplot (optional, just for showing loss)


You can click `here <https://github.com/thu-coai/cotk/blob/master/docs/notes/tutorial_core_1.ipynb>`__ for ipynb files. If you don't have a suitable environment,
you can also run `the code <http://colab.research.google.com/github/thu-coai/cotk/blob/master/docs/source/notes/tutorial_core_1.ipynb>`__
on google colab.

Preparing the data
----------------------------------------

``cotk`` provides :mod:`.dataloader` to download, import and preprocess data.
Therefore, we first construct a :class:`cotk.dataloader.MSCOCO` to load MSCOCO dataset.

.. code-block:: python

    from cotk.dataloader import MSCOCO
    from pprint import pprint
    dataloader = MSCOCO("resources://MSCOCO_small") # "resources://MSCOCO_small" is a predefined resources name
    print("Vocab Size:", dataloader.vocab_size)
    print("First 10 tokens:",  dataloader.vocab_list[:10])
    print("Dataset is split into:", dataloader.key_name)
    data = dataloader.get_batch("train", [0]) # get the sample of id 0
    pprint(data, width=200)
    print(dataloader.convert_ids_to_tokens(data['sent'][0]))

.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    INFO: name: MSCOCO_small
    INFO: source: default
    INFO: processor type: MSCOCO

    100%|██████████| 1020154/1020154 [00:00<00:00, 1853831.54B/s]

    INFO: resource cached at /root/.cotk_cache/9e4c0afe33d98fa249e472206a39e5553d739234d0a27e055044ae8880e314b1_unzip/mscoco
    valid vocab list length = 2588
    vocab list length = 12411
    train set. invalid rate: 0.031716, unknown rate: 0.000000, max length before cut: 55, cut word rate: 0.000022
    dev set. invalid rate: 0.034089, unknown rate: 0.000000, max length before cut: 46, cut word rate: 0.000000
    test set. invalid rate: 0.031213, unknown rate: 0.000000, max length before cut: 27, cut word rate: 0.000000
    Vocab Size: 2588
    First 10 tokens: ['<pad>', '<unk>', '<go>', '<eos>', '.', 'a', 'A', 'on', 'of', 'in']
    Dataset is split into: ['train', 'dev', 'test']
    {'sent': array([[  2,   6,  67, 651, 549,  11,   5,  65,  89,  10, 115, 349,  83,
            4,   3]]),
    'sent_allvocabs': array([[  2,   6,  67, 651, 549,  11,   5,  65,  89,  10, 115, 349,  83,
            4,   3]]),
    'sent_length': array([15])}
    ['<go>', 'A', 'blue', 'lamp', 'post', 'with', 'a', 'sign', 'for', 'the', 'yellow', 'brick', 'road', '.']


:class:`cotk.dataloader.MSCOCO` has helped us construct vocabulary list and
turn the sentences into index representation.

.. note ::
    You can also import dataset from url (http://test.com/data.zip) or
    local path (./data.zip), as long as the format of the data is suitable.

.. note ::
    You may find ``data`` contains similiar key ``sent`` and ``sent_allvocabs``.
    The difference between them is that ``sent`` only contains
    :ref:`valid vocabularies <vocab_ref>` and
    ``sent_allvocabs`` contains both :ref:`valid vocabularies <vocab_ref>` and
    :ref:`invalid vocabularies <vocab_ref>`.

Training models
-----------------------------------------

First we construct a simple GRU Language model using ``pytorch``.

.. code-block:: python

    import torch
    from torch import nn

    embedding_size = 20
    hidden_size = 20

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = nn.Embedding(dataloader.vocab_size, embedding_size)
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, dataloader.vocab_size)
            self.crossentropy = nn.CrossEntropyLoss()

        def forward(self, data):
            # data is the dict returned by ``dataloader.get_batch``
            sent = data['sent']
            sent_length = data['sent_length']
            # sent is a LongTensor whose shape is (batch_size, sent_length)
            # sent_length is a list whose size is (batch_size)

            incoming = self.embedding_layer(sent)
            # incoming: (batch_size, sent_length, embedding_size)
            incoming, _ = self.rnn(incoming)
            # incoming: (batch_size, sent_length, hidden_size)
            incoming = self.output_layer(incoming)
            # incoming: (batch_size, sent_length, dataloader.vocab_size)

            loss = []
            for i, length in enumerate(sent_length):
                if length > 1:
                    loss.append(self.crossentropy(incoming[i, :length-1], sent[i, 1:length]))
                    # every time step predict next token

            data["gen_log_prob"] = nn.LogSoftmax(dim=-1)(incoming)

            if len(loss) > 0:
                return torch.stack(loss).mean()
            else:
                return 0

If you are familiar with GRU, you can see the codes constructed a
network for predicting next token. Then, we will train our model with
the help of ``cotk``. (It may takes several minites too train the model.)

.. code-block:: python

    from livelossplot import PlotLosses
    import numpy as np

    net = LanguageModel()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    epoch_num = 100
    batch_size = 16
    plot = PlotLosses()

    for j in range(epoch_num):
        loss_arr = []
        for i, data in enumerate(dataloader.get_batches("train", batch_size)):
            # convert numpy to torch.LongTensor
            data['sent'] = torch.LongTensor(data['sent'])
            net.zero_grad()
            loss = net(data)
            loss_arr.append(loss.tolist())
            loss.backward()
            optimizer.step()
            if i >= 40:
                break # break for shorten time of an epoch
        plot.update({"loss": np.mean(loss_arr)})
        plot.draw()
        print("epoch %d/%d" % (j+1, epoch_num))

.. rst-class:: sphx-glr-script-out

 Out:

.. image:: training_loss.png


.. code-block:: none

    loss:
        training   (min:    3.147, max:    6.560, cur:    3.235)
    epoch 100/100

Evaluations
-----------------------------------------

How well our model can fit the data? ``cotk`` have provided
some standard metrics for language generation model.

Teacher Forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~

``perplexity``
is a common used metric and it need the predicted distribution
over words. Recall we have set ``data["gen_log_prob"]`` in previous
section, we use it right now.

.. code-block:: python

    metric = dataloader.get_teacher_forcing_metric(gen_log_prob_key="gen_log_prob")
    for i, data in enumerate(dataloader.get_batches("test", batch_size)):
        # convert numpy to torch.LongTensor
        data['sent'] = torch.LongTensor(data['sent'])
        with torch.no_grad():
            net(data)
        assert "gen_log_prob" in data
        metric.forward(data)
    pprint(metric.close(), width=150)

.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    test set restart, 78 batches and 2 left
    {'perplexity': 33.99345315581511,\n",
     'perplexity hashvalue': b'O\\x10\\x1c)\\x86\\xf1\\xfe\\x10\\xce\\x1d!\\x97\\xc3\\x08m6Y\\xae\\xc3\\xe6I_8\\x1dg\\xf0\\x0bM\\xbb@\\xa58'}

The codes above evaluated the model in teacher forcing mode, where every input
token is the real data. 

.. note ::

    The type of ``data['gen_log_prob']`` is ``torch.Tensor``, but most metrics can
    **not** receive a tensor input as we are trying to implement a library **not**
    depending on any deep learning framework. :class:`.metric.PerplexityMetric` just use ``torch``
    to accelerate the calculation, a :class:`numpy.ndarray` can also be accepted.

Free Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A language model can also generate sentences by sending the
generated token back to input in each step. We call it "freerun"
or "inference" mode.

``Pytorch`` doesn't provide a convenience api for freerun, here we implement a
simple version that all the prefixes will be recalculated at every step.

.. code-block:: python

    metric = dataloader.get_inference_metric(gen_key="gen")
    generate_sample_num = 1
    max_sent_length = 20

    for i in range(generate_sample_num):
        # convert numpy to torch.LongTensor
        data['sent'] = torch.LongTensor([[dataloader.go_id] for _ in range(batch_size)])
        data['sent_length'] = np.array([1 for _ in range(batch_size)])
        for j in range(max_sent_length):
            with torch.no_grad():
                net(data)
                generated_token = torch.multinomial(data['gen_log_prob'].exp()[:, -1], 1)
            data['sent'] = torch.cat([data['sent'], generated_token], dim=-1)

        metric.forward({"gen": data['sent'][:, 1:].tolist()})
    pprint(metric.close(), width=250)

Out:

.. code-block:: none

    100%|██████████| 1000/1000 [00:00<00:00, 1153.17it/s]
    {'bw-bleu': 0.054939232761090494,
     'fw-bleu': 0.2643063370185712,\n",
     'fw-bw-bleu': 0.09096938998850655,\n",
     'fw-bw-bleu hashvalue': b'0\\x18\\xdc1\\x7f\\x82\\xb6\\x01?\\x01\\x1c\\x1f\\x8c\\xcd\\x90\\xc5\\xaf\\xfe\\xd7\\x10\\xb7\\xd7\\xd0jr5\\xcfE\\\\#5B',
     'gen': [['A', 'black', 'fire', 'hydrant', 'with', 'broccoli', 'on', 'a', 'plate', '.'],
            ['A', 'small', 'cat', 'that', 'drinking', 'are', 'sitting', 'from', 'a', 'screen', '.'],
            ['This', 'was', 'hydrant', 'at', 'a', 'kitchen', 'by', 'a', 'meal', '.'],
            ['A', 'bath', 'room', 'with', 'a', 'bicycle', 'are', 'antique', '<unk>', ',', 'sink', 'and', 'patterned', 'photograph', 'and', 'fork', 'look', 'on', 'the', 'bed'],
            ['Large', 'yellow', 'kitchen', 'with', 'one', 'and', '<unk>', 'neat', ',', 'kite', '.'],
            ['A', 'young', 'girl', 'are', 'kneeling', 'a', 'tennis', 'match', 'during', 'tennis', 'of', 'her', 'bat', '.'],
            ['A', 'zebra', 'leaned', 'across', 'a', 'city', 'street', '.'],
            ['A', 'tennis', 'player', 'looks', 'by', 'a', 'wii', 'chocolate', 'strip', '.'],
            ['A', 'woman', 'sitting', 'up', 'on', 'the', 'edge', 'of', 'luggage', '.'],
            ['Old', '<unk>', 'right', 'driving', 'along', 'from', 'the', 'stairs', '.'],
            ['A', 'close', 'up', 'with', 'a', 'colorful', 'heart', 'next', 'to', 'a', 'skateboard', '.'],
            ['A', 'man', 'holding', 'a', 'skate', 'boarding', 'a', 'mountain', 'at', 'the', 'sun', 'man', 'flying', 'the', 'air', '.'],
            ['An', 'airplane', 'with', 'speakers', 'on', 'a', 'surface', '.'],
            ['A', '<unk>', '<unk>', 'on', 'the', 'two', 'conference', 'floors', '.'],
            ['A', 'girl', 'holding', 'a', 'wii', 'screen', 'with', 'a', 'hot', 'Man', 'with', 'wii', 'face', 'in', 'formation', '.'],
            ['Three', 'and', 'snowboards', 'a', 'video', 'game', 'during', 'the', 'beach', '.']],
     'self-bleu': 0.04565576967736443,
     'self-bleu hashvalue': b'\\x9f\\x11!\\xd3\\x98\\x8e\\xf4x\\x99C\\xef\\x18\\xc1\\xc0\\xb7I\\xee\\xc0\\xd8\\xee\\xe3\\xf1\"pg\\x16\\x05\\xceg\\x02%\\xf6'}

Hash value
~~~~~~~~~~~~~~~~~~

Hash value is for checking whether you use the test set correctly.
We can refer to dashboard (TO BE ONLINE) for the state of art on this dataset,
and we find our hashvalue is correct.

However, if teacher forcing is tested as following codes, we will
see a different hash value, which means the implementation is not correct.

.. code-block:: python

    metric = dataloader.get_teacher_forcing_metric(gen_log_prob_key="gen_log_prob")
    for i, data in enumerate(dataloader.get_batches("test", batch_size)):
        # convert numpy to torch.LongTensor
        data['sent'] = torch.LongTensor(data['sent'])
        with torch.no_grad():
            net(data)
        assert "gen_log_prob" in data
        metric.forward(data)
        if i >= 15: #ignore the following batches
            break
    pprint(metric.close(), width=150)

Out:

.. code-block:: none

    test set restart, 78 batches and 2 left
    {'perplexity': 31.5929983966103, 'perplexity hashvalue': b\"\\x0c\\xfd9r\\xc7\\x8b_\\xf5\\xf7\\xf90\\xd1v\\x7f\\xd8Ua\\xc8g\\xdc\\xd3MV\\xeeH\\xe0\\x86\\xed@'\\x91\\x91\"}


Additional: Word Vector
----------------------------------------

It is a common technique to use pre-trained word vector when
processing natural languages. ``cotk`` also provides a module :mod:`.wordvector`
that help you downloading and get word vectors.

.. code-block:: python

    from cotk.wordvector import Glove
    wordvec = Glove("resources://Glove50d_small")
    self.embedding_layer.weight = nn.Parameter(torch.Tensor(wordvec.load(embedding_size, dataloader.vocab_list)))

We can add these lines at the end of ``LanguageModel.__init__``.

You can find the results and codes with pretrained word vector at
`here <https://github.com/thu-coai/cotk/blob/master/docs/notes/tutorial_core_2.ipynb>`__ for ipynb files
or run `the code <http://colab.research.google.com/github/thu-coai/cotk/blob/master/docs/source/notes/tutorial_core_2.ipynb>`__
on google colab.
