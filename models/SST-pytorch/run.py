# coding:utf-8

def run(*argv):
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser(description='A classification model with GRU encoder and MLP for prediction. \
		Dropout and batchnorm is supported.')
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

	parser.add_argument('--eh_size', type=int, default=200,
		help='Size of encoder GRU')
	parser.add_argument('--class_num', type=int, default=5,
		help='Number of classes')
	parser.add_argument('--droprate', type=float, default=0,
		help='The probability to be zerod in dropout. 0 indicates for don\'t use dropout')
	parser.add_argument('--batchnorm', action='store_true',
		help='Use bathnorm')

	parser.add_argument('--dataset', type=str, default='SST',
		help='Dataloader class. Default: SST')
	parser.add_argument('--datapath', type=str, default='resources://SST',
		help='Directory for data set. Default: resources://SST')
	parser.add_argument('--epoch', type=int, default=100,
		help="Epoch for trainning. Default: 100")
	parser.add_argument('--wvclass', type=str, default='Glove',
		help="Wordvector class, none for not using pretrained wordvec. Default: Glove")
	parser.add_argument('--wvpath', type=str, default="resources://Glove300d",
		help="Directory for pretrained wordvector. Default: resources://Glove300d")

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
	cargs = parser.parse_args(argv)


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
	load_exclude_set = []
	restoreCallback = None

	args.batch_per_epoch = 500
	args.embedding_size = 300
	args.eh_size = cargs.eh_size
	args.class_num = cargs.class_num

	args.droprate = cargs.droprate
	args.batchnorm = cargs.batchnorm

	args.lr = 1e-3
	args.batch_size = 64
	args.batch_num_per_gradient = 4
	args.grad_clip = 5
	args.show_sample = [0]  # show which batch when evaluating at tensorboard
	args.max_sent_length = 50
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	import random
	random.seed(0)

	from main import main
	main(args, load_exclude_set, restoreCallback)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
