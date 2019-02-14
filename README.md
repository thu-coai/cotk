# Conversational Toolkits

[![CodeFactor](https://www.codefactor.io/repository/github/thu-coai/contk/badge)](https://www.codefactor.io/repository/github/thu-coai/contk)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/contk/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/contk?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/contk.svg?branch=master)](https://travis-ci.com/thu-coai/contk)
[![codebeat badge](https://codebeat.co/badges/2b2107ed-1145-4ce1-bfcf-8b134a0c6e0a)](https://codebeat.co/projects/github-com-thu-coai-contk-master)

## Environment

* **python 3**
* numpy >= 1.13
* nltk >= 3.2
* tqdm >= 4.30
* checksumdir >= 1.1

## Document

Please refer to [document](https://thu-coai.github.io/contk_docs/)

## To Developer

### Contk Package

`./contk` is the package folder.

* All your code must followed a coding standard specified by **Pylint**. You can simply install Pylint via pip:

  ```
  pip install pylint
  ```
  In this project, you should follow the default pylint configuration that we provide and check `contk` after you update it:

  ```
  pylint contk
  ```

* Class and function docstring are always required.  

#### Some suggestions for docstring

You can have a look at docstring in other files. Or you may want to go to learn how docstring is written in other projects:

* [https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* [http://www.sphinx-doc.org/en/1.5/ext/example_google.html](http://www.sphinx-doc.org/en/1.5/ext/example_google.html)
* [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
* [https://pytorch.org/docs/stable/torch.html](https://pytorch.org/docs/stable/torch.html) (click '[source]' after functions, you can see the original docstrings)

Here are some references of docstring (it is a part of reStructuredText, like markdown).

* [http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

### Models

You can implement your model in './models'.

* Before you run a model, you have to install `contk` by  run `pip install -e .` in project root directory.
* When you run a model, your CWD(current working directory) should be model's folder (eg: Using `python run.py` in `./model/seq2seq-pytorch`).
* Code style is not so strict in your model implementation.
* But you have to explain how to use your model.
* You should provide a pretrained model file for your implementation. (But don't commit it to git repo.)


## Issues

You are welcome to create an issue if you want to request a feature, report a bug or ask a general question.

## Contributions

We welcome contributions from community. 

* If you want to make a big change, we recommend first creating an issue with your design.
* Small contributions can be directly made by a pull request.
* If you like make contributions for our library, see issues to find what we need.

## Team

`Contk` is maintained and developed by Tsinghua university conversational AI group (THU-coai). Check our [main pages](http://coai.cs.tsinghua.edu.cn/) (In Chinese).

## License

Apache License 2.0
