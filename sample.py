# -----------
# This network is from Udacity's Intro to RNN project
# And was part of the assigment to build an RNN which
# The Model in the assignment was originally trained on the text of Anna Karenina. 
# 
# Original model: 
#   https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
# Original model inspired by Karpathy's Character RNN: 
#   http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
# ------------
import time
import glob
import numpy as np
import tensorflow as tf
import config

from collections import namedtuple

from model import get_batches
from model import CharRNN
from data import read_processed_data

def pick_top_n(preds, vocab_size, top_n=config.PICK_TOP_N):
    """
    Picks one of top_n most likely next characters, 
    Random selection step is provided with the actual
    probabilities with which to make the weighted
    choice between the top_n characters.
    """
    p = np.squeeze(preds)
    indexes_to_retain = np.argsort(p)[-top_n:]
    for i in range(0, len(p)):
        if i not in indexes_to_retain:
            p[i] = 0
    probs = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=probs)[0]
    return c

def get_feed(model, x, new_state):
    """
    Short cut for creating the new feed dict for each character
    """
    return {model.inputs: x,
            model.keep_prob: 1.,
            model.initial_state: new_state}

def sample(checkpoint, n_samples, lstm_size, processd_data_dir, prime=config.MEASURE_SYMBOL):
    """
    First primes the RNN with the prime event, 
    Then continues generating next character for n_samples times
    """
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
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=get_feed(model, x, new_state))
        # continue generating 
        c = pick_top_n(preds, vocab_size)
        samples.append(int_to_vocab[c])
        for i in range(n_samples):
            x[0,0] = c
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=get_feed(model, x, new_state))
            c = pick_top_n(preds, vocab_size)
            samples.append(int_to_vocab[c])
    return ''.join(samples)

def sample_checkpoint(checkpoint_dir, checkpoint, n_samples, lstm_size, prime):
    return sample(checkpoint_dir + checkpoint, n_samples, lstm_size, checkpoint_dir, prime)
