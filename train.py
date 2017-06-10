# This network is from Udacity's Intro to RNN project
# https://github.com/udacity/deep-learning/tree/master/intro-to-rnns 
# floyed command: floyd run --env tensorflow-1.0 --gpu "python train.py floyd"
import time
import sys
import glob
from collections import namedtuple

import numpy as np
import tensorflow as tf

from model import get_batches
from model import CharRNN
from data import generate_data
from data import get_output_dir

output_dir = get_output_dir(sys.argv)
vocab, vocab_to_int, int_to_vocab, encoded = generate_data(output_dir)

### TRAINING ###
batch_size = 64         # Sequences per batch
num_steps = 150         # Number of sequence steps per batch
lstm_size = 2048        # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 60
save_every_n = 5000     # Save every N iterations

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    start_train = time.time()
    print("beginning training... ")
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            start = time.time()
            counter += 1
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            end = time.time()

            if (counter % 100 == 0):
                print("Epoch: {}/{}... ".format(e+1, epochs),
                              "Training Step: {}... ".format(counter),
                              "Training loss: {:.4f}... ".format(batch_loss),
                              "{:.4f} sec/batch".format((end-start)))     
            
            if (counter % save_every_n == 0):
                print("Saving checkpoint")
                saver.save(sess, output_dir + "/i{}_l{}.ckpt".format(counter, lstm_size))

    end_train = time.time()
    print("training finished, saving checkpoint...", 
          "time elapsed: {:.4f} secs".format(end_train-start_train))
    saver.save(sess, output_dir + "/trained.ckpt".format(counter, lstm_size))
