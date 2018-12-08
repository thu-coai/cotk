# coding:utf-8

if __name__ == '__main__':
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser(description='A seq2seq model')
	args = Storage()

	parser.add_argument('--name', type=str, default=None, 
		help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default=None, 
		help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
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
	cargs = parser.parse_args()


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

	import random
	random.seed(0)

	from main import main
	main(args)
