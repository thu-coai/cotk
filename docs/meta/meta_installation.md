~~ location sphinx ../source/notes/installation.md

## Installation

### Requirements

-  python 3
-  numpy >= 1.13
-  nltk >= 3.4
-  tqdm >= 4.30
-  checksumdir >= 1.1
-  pytorch >= 1.0.0 (optional)
-  pytorch-pretrained-bert (optional)

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

* If you want to run the models in ``./models``, you have to additionally install [TensorFlow](https://www.tensorflow.org) or [PyTorch](https://pytorch.org/).
