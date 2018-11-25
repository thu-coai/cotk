# Conversation Processing Toolkits

### Environment

python 3

(maybe some other packages... waiting for update)

### To Developer

#### cotk Package

`./cotk` is the package folder.

* All your code must followed a coding standard specified by **pylint**. 
* Package and function documents are always required. 
* Some missing document will be updated in the future.

#### models

You can implement your model in './models'.

* Before you run a model, you have to install `cotk` by  run `pip install -e .` in project root directory.
* When you run a model, your CWD(current working directory) should be model's folder (eg: Using `python run.py` in `./model/seq2seq-pytorch`).
* Code style is not so strict in your model implementation.
* But you have to explain how to use your model.
* You should provide a pretrained model file for your implementation. (But don't commit it to git repo.)

### License

MIT License