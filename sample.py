# This network is from Udacity's Intro to RNN project
# https://github.com/udacity/deep-learning/tree/master/intro-to-rnns 
import time
import glob

import numpy as np
import tensorflow as tf

from collections import namedtuple

from model import get_batches
from model import CharRNN

from data import read_processed_data

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, processd_data_dir, prime="@"):
    vocab, vocab_to_int, int_to_vocab, encoded = read_processed_data(processd_data_dir)
    vocab_size = len(vocab)

    samples = [c for c in prime]
    model = CharRNN(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)

        # prime the sample
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)

def sample_checkpoint(checkpoint_dir, checkpoint, n_samples, lstm_size, prime):
    return sample(checkpoint_dir + checkpoint, n_samples, lstm_size, checkpoint_dir, prime)
