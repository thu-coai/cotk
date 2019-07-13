import os

import json
import numpy as np
import tensorflow as tf
from cotk.dataloader import LanguageGeneration
from cotk.wordvector import WordVector, Glove
from utils import debug, try_cache, SummaryHelper

import time
import random

from discriminator import Discriminator
from generator import Generator
from rollout import rollout

def create_model(sess, data, args, embed):
    #get maximum input sequence length
    data.restart("train", batch_size=args.batch_size, shuffle=True)
    batched_data = data.get_next_batch("train")
    length = []
    while batched_data != None:
        length.append(len(batched_data["sent"][0]))
        batched_data = data.get_next_batch("train")
    sequence_length = np.max(length)

    latest_dir = '%s/checkpoint_latest' % args.model_dir
    best_dir = '%s/checkpoint_best' % args.model_dir    

    summary = create_summary(args)
    with tf.variable_scope("generator"):
        #Build generator and its rollout
        generator = Generator(args, data, embed, summary, sequence_length, latest_dir, best_dir)
        generator.build()
        rollout_gen = rollout(args, data, embed, sequence_length)

    with tf.variable_scope("discriminator"):
        #Build discriminator
        discriminator = Discriminator(args, data, embed, summary, sequence_length, latest_dir, best_dir)
        discriminator.build_discriminator()

    latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
            max_to_keep=args.checkpoint_max_to_keep, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
    best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
            max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    if tf.train.get_checkpoint_state(latest_dir) and args.restore == "last":
        print("Reading model parameters from %s" % latest_dir)
        latest_saver.restore(sess, tf.train.latest_checkpoint(latest_dir))
    else:
        if tf.train.get_checkpoint_state(best_dir) and args.restore == "best":
            print('Reading model parameters from %s' % best_dir)
            best_saver.restore(sess, tf.train.latest_checkpoint(best_dir))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.variables_initializer(tf.global_variables()))
    generator.latest_saver, generator.best_saver = latest_saver, best_saver
    discriminator.latest_saver, discriminator.best_saver = latest_saver, best_saver

    '''
    generator.print_parameters()
    print("-----------------------------")
    discriminator.print_parameters()
    print("-----------------------------")
    '''
    return generator, discriminator, rollout_gen

def create_summary(args):
    summaryHelper = SummaryHelper("%s/%s_%s" % \
            (args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

    gen_trainSummary = summaryHelper.addGroup(scalar=["loss", "rewards"],
                                                    prefix="gen_train")
    dis_trainSummary = summaryHelper.addGroup(scalar=["loss", "accuracy"],
                                                    prefix="dis_train")
    gen_scalarlist = ["loss", "rewards"]
    dis_scalarlist = ["loss", "accuracy"]
    tensorlist = []
    textlist = []
    for i in args.show_sample:
        textlist.append("show_str%d" % i)
    gen_devSummary = summaryHelper.addGroup(scalar=gen_scalarlist, tensor=tensorlist, text=textlist,
                                                    prefix="gen_dev")
    gen_testSummary = summaryHelper.addGroup(scalar=gen_scalarlist, tensor=tensorlist, text=textlist,
                                                    prefix="gen_test")
    dis_devSummary = summaryHelper.addGroup(scalar=dis_scalarlist, tensor=tensorlist, text=textlist,
                                                    prefix="dis_dev")
    dis_testSummary = summaryHelper.addGroup(scalar=dis_scalarlist, tensor=tensorlist, text=textlist,
                                                    prefix="dis_test")

    return gen_trainSummary, gen_devSummary, gen_testSummary, dis_trainSummary, dis_devSummary, dis_testSummary

def main(args):
    if args.debug:
        debug()
    if args.cuda:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_class = LanguageGeneration.load_class(args.dataset)
    wordvec_class = WordVector.load_class(args.wvclass)
    if wordvec_class == None:
        wordvec_class = Glove
    if args.cache:
        data = try_cache(data_class, (args.datapath,), args.cache_dir)
        vocab = data.vocab_list
        embed = try_cache(lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
                          (args.wvpath, args.embedding_size, vocab),
                          args.cache_dir, wordvec_class.__name__)
    else:
        data = data_class(args.datapath)
        wv = wordvec_class(args.wvpath)
        vocab = data.vocab_list
        embed = wv.load_matrix(args.embedding_size, vocab)

    embed = np.array(embed, dtype = np.float32)
    with tf.Session(config=config) as sess:
        generator, discriminator, rollout_gen = create_model(sess, data, args, embed)
        if args.mode == "train":
            if args.pre_train:
                #Start pretraining
                print('Start pre-training generator...')
                generator.pre_train_process(sess, data)

                print('Start pre-training discriminator...')
                discriminator.train_process(generator, data, sess, args.dis_pre_epoch_num)

                #Start testing
                generator.test_process(sess, data)

            #Start adversarial training
            for batch in range(args.total_adv_batch):
                print("Adversarial  training %d"%batch)
                print('Start adversarial training generator...')
                generator.adv_train_process(sess, data, rollout_gen, discriminator)
                testout = generator.pre_evaluate(sess, data, args.batch_size, "test")
                if (batch % args.test_per_epoch == 0 or batch == args.total_adv_batch - 1) and batch != 0:
                    print('total_batch: ', batch, 'test_loss: ', testout[0])
                    generator.test_process(sess, data) 

                print('Start adversarial training discriminator...')
                discriminator.train_process(generator, data, sess, args.dis_adv_epoch_num)
        else:
            print("Start testing...")
            test_res = generator.test_process(sess, data)
            for key, val in test_res.items():
                if isinstance(val, bytes):
                    test_res[key] = str(val)
            json.dump(test_res, open("./result.json", "w"))
