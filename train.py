# -----------
# This network is from Udacity's Intro to RNN project
# And was part of the assigment to build an RNN which
# The Model in the assignment was originally trained on the text of Anna Karenina. 
# 
# Original model: 
#   https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
# Original model inspired by Karpathy's Character RNN: 
#   http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
#
# Floyd command: floyed command: floyd run --env tensorflow-1.0 --gpu "python train.py floyd"
# ------------
import time
import sys
import glob
from collections import namedtuple

import numpy as np
import tensorflow as tf
import config as config

from model import get_batches
from model import CharRNN
from data import generate_data
from data import get_output_dir

output_dir = get_output_dir(sys.argv)
vocab, vocab_to_int, int_to_vocab, encoded = generate_data(output_dir)

model = CharRNN(len(vocab), batch_size=config.BATCH_SIZE, num_steps=config.NUM_STEPS,
                lstm_size=config.LSTM_SIZE, num_layers=config.NUM_LAYERS, 
                learning_rate=config.LEARNING_RATE)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    start_train = time.time()
    print("beginning training... ")
    for e in range(config.EPOCHS):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, config.BATCH_SIZE, config.NUM_STEPS):
            start = time.time()
            counter += 1
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: config.KEEP_PROB,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            end = time.time()
            if (counter % 100 == 0):
                print("Epoch: {}/{}... ".format(e+1, config.EPOCHS),
                              "Training Step: {}... ".format(counter),
                              "Training loss: {:.4f}... ".format(batch_loss),
                              "{:.4f} sec/batch".format((end-start)))
            if (counter % config.SAVE_EVERY_N == 0):
                print("Saving checkpoint")
                saver.save(sess, output_dir + "/i{}_l{}.ckpt".format(counter, config.LSTM_SIZE))

    end_train = time.time()
    print("training finished, saving checkpoint...", 
          "time elapsed: {:.4f} secs".format(end_train-start_train))
    saver.save(sess, output_dir + "/trained.ckpt".format(counter, config.LSTM_SIZE))
