import pytest
import random
from main import main
def default_args():
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser(description='A seq2seq model')
	args = Storage()

	parser.add_argument('--name', type=str, default=None,
		help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default=None,
		help='Checkpoints name to load. \
			"NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
			You can also use "last" and "best", defaultly use last model you run. \
			Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
			"last" and "best" are not guaranteed to work when 2 models run in the same time.\
			Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')
	parser.add_argument('--dataset', type=str, default='OpenSubtitles',
		help='Dataloader class. Default: OpenSubtitles')
	parser.add_argument('--datapath', type=str, default='./data',
		help='Directory for data set. Default: ./data')
	parser.add_argument('--epoch', type=int, default=100,
		help="Epoch for trainning. Default: 100")
	parser.add_argument('--wvclass', type=str, default=None,
		help="Wordvector class, none for not using pretrained wordvec. Default: None")
	parser.add_argument('--wvpath', type=str, default=None,
		help="Directory for pretrained wordvector. Default: ./wordvec")

	parser.add_argument('--out_dir', type=str, default="./output",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true",
		help='Use cpu.')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true',
		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
	cargs = parser.parse_args(args="")


	# Editing following arguments to bypass command line.
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.mode = cargs.mode
	args.dataset = cargs.dataset
	args.datapath = cargs.datapath
	args.epochs = cargs.epoch
	args.wvclass = cargs.wvclass
	args.wvpath = cargs.wvpath
	args.out_dir = cargs.out_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = not cargs.cpu

	# The following arguments are not controlled by command line.
	args.restore_optimizer = True
	args.load_exclude_set = []
	args.restoreCallback = None

	args.batch_per_epoch = 1500
	args.embedding_size = 300
	args.eh_size = 200
	args.dh_size = 200
	args.lr = 1e-3
	args.batch_size = 30
	args.grad_clip = 5
	args.show_sample = [0]  # show which batch when evaluating at tensotboard
	args.max_sen_length = 50
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	return args

def my_args():
	args = default_args()
	args.cuda = 0
	args.name = 'test_seq2seq_pytorch'
	args.wvclass = 'Glove'
	args.epochs = 1
	args.batch_per_epoch = 5
	args.batch_size = 5
	import os
	cwd = os.path.abspath(os.path.dirname(__file__))
	if not os.path.exists(cwd + '/output'):
	    os.mkdir(cwd + '/output')
	if not os.path.exists(cwd + '/model'):
	    os.mkdir(cwd + '/model')
	path = os.path.split(cwd)[0]
	path = os.path.split(path)[0]
	args.datapath = path + '/tests/dataloader/dummy_opensubtitles'
	args.wvpath = path + '/tests/models/dummy_glove_300d.txt'
	random.seed(0)
	return args

def test_train():
	args = my_args()
	args.mode = 'train'
	main(args)

def test_test():
	args = my_args()
	args.mode = 'test'
	main(args)

def test_restore():
	args = my_args()
	args.mode = 'train'
	args.restore = 'last'
	main(args)

def test_cache():
	args = my_args()
	args.mode = 'train'
	args.cache = 1
	main(args)
