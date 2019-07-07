import pytest
import json
import random
from main import main
import tensorflow as tf
import time

def default_args():
    import argparse
    import time

    from utils import Storage

    parser = argparse.ArgumentParser(description='A seqGAN language generation model')
    args = Storage()

    parser.add_argument('--name', type=str, default='seqGAN',
        help='The name of your model, used for variable scope and tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
    parser.add_argument('--restore', type=str, default='last',
        help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
    parser.add_argument('--mode', type=str, default="train",
        help='"train" or "test". Default: train')
    parser.add_argument('--dataset', type=str, default='MSCOCO',
        help='Dataloader class. Default: MSCOCO')
    parser.add_argument('--datapath', type=str, default='/home/data/share/mscoco',
        help='Directory for data set. Default: ./data')
    parser.add_argument('--wvclass', type=str, default=None,
        help="Wordvector class, none for not using pretrained wordvec. Default: None")
    parser.add_argument('--wvpath', type=str, default='/home/data/share/glove/glove.6B.300d.txt',
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
    args.pre_train = True
    args.global_step = 0
    args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
    args.restore = cargs.restore
    args.mode = cargs.mode
    args.dataset = cargs.dataset
    args.datapath = cargs.datapath
    args.wvclass = cargs.wvclass
    args.wvpath = cargs.wvpath
    args.out_dir = cargs.out_dir
    args.log_dir = cargs.log_dir
    args.model_dir = cargs.model_dir
    args.cache_dir = cargs.cache_dir
    args.debug = cargs.debug
    args.cache = cargs.cache
    args.cuda = not cargs.cpu

    args.sample = 1000
    args.softmax_samples = 512
    args.test_sample = None
    args.embedding_size = 300
    args.eh_size = 200
    args.dh_size = 200
    args.z_dim = 100
    args.min_kl = 10
    args.full_kl_step = 30000
    args.lr = 1e-1
    args.lr_decay = 0.995
    args.momentum = 0.9
    args.batch_size = 128
    args.grad_clip = 5.0
    args.show_sample = [0]
    args.checkpoint_steps = 1000
    args.checkpoint_max_to_keep = 5
    
    args.gen_pre_epoch_num = 25 #120 #Number of pretraining epoch
    args.dis_pre_epoch_num = 1 #pretraining times of discriminator
    args.total_adv_batch = 200 #total batch used for adversarial training
    args.gen_adv_batch_num = 120 #update times of generator in adversarial training
    args.test_per_epoch = 5
    args.rollout_num = 5  #Rollout number for reward estimation
    args.dis_adv_epoch_num = 1 #5 #update times of discriminator in adversarial training
    args.dis_dropout_keep_prob = 0.75 # dropout rate of discriminator
    args.num_classes = 2 #number of class (real and fake)    
    args.dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] #convolutional kernel size of discriminator
    args.dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] #number of filters of each conv. kernel
    args.dis_dropout_keep_prob = 0.75 # dropout rate of discriminator
    args.dis_l2_reg_lambda = 0.2 #L2 regularization strength
    args.dis_lr = 1e-4 #Learning rate of discriminator    

    return args

def my_args():
    args = default_args()
    args.cuda = 0
    args.name = 'test_seqGAN_tensorflow'
    args.wvclass = 'Glove'
    import os
    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(cwd + '/output'):
        os.mkdir(cwd + '/output')
    if not os.path.exists(cwd + '/model'):
        os.mkdir(cwd + '/model')
    path = os.path.split(cwd)[0]
    path = os.path.split(path)[0]
    args.datapath = path + '/tests/dataloader/dummy_mscoco'
    args.wvpath = path + '/tests/wordvector/dummy_glove/300d'
    args.sample = 2
    args.gen_pre_epoch_num = 1
    args.total_adv_batch = 1
    args.gen_adv_batch_num = 1
    args.rollout_num = 1
    args.teacher_forcing = False
    args.pre_train = False
    args.batch_size = 5
    #args.model_dir = "model_test"
    args.test_sample = 1

    random.seed(0)
    return args

def test_train():
    args = my_args()
    args.mode = 'train'
    main(args)
    tf.reset_default_graph()

def test_test():
    args = my_args()
    args.mode = 'test'
    main(args)
    old_res = json.load(open("./result.json", "r"))
    tf.reset_default_graph()
    main(args)
    new_res = json.load(open("./result.json", "r"))
    for key in old_res:
        if key[-9:] == 'hashvalue':
            assert old_res[key] == new_res[key]
    tf.reset_default_graph()

def test_restore():
    args = my_args()
    args.mode = 'train'
    args.restore = 'last'
    main(args)
    tf.reset_default_graph()

def test_cache():
    args = my_args()
    args.mode = 'train'
    args.cache = 1
    main(args)
    tf.reset_default_graph()
