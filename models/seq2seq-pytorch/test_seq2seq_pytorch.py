import pytest
import random
from run import run
from main import main
import os
import json
import shutil
cwd = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(cwd)[0]
path = os.path.split(path)[0]

def setup_function(function):
	import sys
	sys.argv = ['python3']
	random.seed(0)
	import numpy as np
	np.random.seed(0)
	import torch
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	try:
		shutil.rmtree(os.path.join(cwd, 'output_test'), ignore_errors=True)
		shutil.rmtree(os.path.join(cwd, 'tensorboard_test'), ignore_errors=True)
		shutil.rmtree(os.path.join(cwd, 'model_test'), ignore_errors=True)
		shutil.rmtree(os.path.join(cwd, 'cache_test'), ignore_errors=True)
		os.mkdirs(os.path.join(cwd, 'output_test'), exist_ok=True)
		os.mkdirs(os.path.join(cwd, 'tensorboard_test'), exist_ok=True)
		os.mkdirs(os.path.join(cwd, 'model_test'), exist_ok=True)
		os.mkdirs(os.path.join(cwd, 'cache_test'), exist_ok=True)
	except Exception as e:
		pass

def teardown_function(function):
	shutil.rmtree(os.path.join(cwd, 'output_test'), ignore_errors=True)
	shutil.rmtree(os.path.join(cwd, 'tensorboard_test'), ignore_errors=True)
	shutil.rmtree(os.path.join(cwd, 'model_test'), ignore_errors=True)
	shutil.rmtree(os.path.join(cwd, 'cache_test'), ignore_errors=True)

def modify_args(args):
	args.cuda = False
	args.restore = None
	args.wvclass = 'Glove'
	args.wvpath = os.path.join(path, 'tests', 'wordvector', 'dummy_glove', '300d')
	args.out_dir = os.path.join(cwd, 'output_test')
	args.log_dir = os.path.join(cwd, 'tensorboard_test')
	args.model_dir = os.path.join(cwd, 'model_test')
	args.cache_dir = os.path.join(cwd, 'cache_test')

	args.name = 'test_seq2seq_pytorch'
	args.wvclass = 'Glove'
	args.epochs = 1
	args.batch_per_epoch = 5
	args.batch_size = 5
	args.datapath = os.path.join(path, 'tests', 'dataloader', 'dummy_opensubtitles')

def test_train(mocker):
	def side_effect_train(args, *others):
		modify_args(args)
		args.mode = 'train'
		main(args, *others)
	def side_effect_restore(args, *others):
		modify_args(args)
		args.mode = 'train'
		args.restore = 'last'
		main(args, *others)
	def side_effect_cache(args, *others):
		modify_args(args)
		args.mode = 'train'
		args.cache = True
		main(args, *others)
	mock = mocker.patch('main.main', side_effect=side_effect_train)
	run()
	mock.side_effect = side_effect_restore
	run()
	mock.side_effect = side_effect_cache
	run()

def test_test(mocker):
	def side_effect_test(args, *others):
		modify_args(args)
		args.mode = 'test'
		main(args, *others)
	mock = mocker.patch('main.main', side_effect=side_effect_test)
	run()
	old_res = json.load(open("./result.json", "r"))
	run()
	new_res = json.load(open("./result.json", "r"))
	for key in old_res:
		if key[-9:] == 'hashvalue':
			assert old_res[key] == new_res[key]
