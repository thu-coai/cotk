# coding:utf-8

if __name__ == '__main__':
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser()
	args = Storage()

	parser.add_argument('--logname', type=str, default=None)
	parser.add_argument('--restore', type=str, default=None)
	parser.add_argument('--mode', type=str, default="train")
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--cache', action='store_true')
	parser.add_argument('--dataset', type=str, default='./OpenSubtitles')
	parser.add_argument('--wordvec', type=str, default=None)
	parser.add_argument('--outdir', type=str, default="./output")
	parser.add_argument('--cpu', action="store_true")
	parser.add_argument('--epoch', type=int, default=100)
	cargs = parser.parse_args()

	args.logname = cargs.logname or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.epochs = cargs.epoch
	args.mode = cargs.mode
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.dataset = cargs.dataset
	args.wordvec = cargs.wordvec
	args.outdir = cargs.outdir
	args.cuda = not cargs.cpu

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
	args.show_sample = [0]
	args.max_sen_length = 50
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	import random
	random.seed(0)

	from main import main
	main(args)
