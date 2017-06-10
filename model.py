# This network is from Udacity's Intro to RNN project
# https://github.com/udacity/deep-learning/tree/master/intro-to-rnns 
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # n_seqs = 10, n_steps = 50
    # Get the batch size and number of batches we can make
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:batch_size * n_batches]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    # y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
    for n in range(0, arr.shape[1], n_steps):
        # The features
        seq_end = n + n_steps
        x = arr[:, n:seq_end] # i think this means copy list and get that list's subset
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, 0]
        yield x, y

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    ### Build the LSTM Cell
    # Use a basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    # if num layers is 3, it will generate 3 layers of shape (10, 128)
    # so per sequence of 50, we have 3 layers of 10x128
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state

def build_output(lstm_output, lstm_size, num_classes):
    ''' Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        
        lstm_output: List of output tensors from the LSTM layer
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    
    '''
    # print(lstm_output, lstm_size, num_classes)
    # output: Tensor("rnn/transpose:0", shape=(10, 50, 128), dtype=float32) lstm_size: 128 num_classes: 83
    # I don't understand the concat step
    # print("seq_output", seq_output)
    # output: seq_output Tensor("concat:0", shape=(10, 50, 128), dtype=float32)
    seq_output = tf.concat(lstm_output, axis=1)
    
    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # That is, the shape should be batch_size*num_steps rows by lstm_size columns
    # print("x", x)
    # output: x Tensor("Reshape:0", shape=(500, 128), dtype=float32)
    x = tf.reshape(seq_output, [-1, lstm_size])
    
    # Connect the RNN outputs to a softmax layer
    # weight: 128x83, bias: 83
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    # logits at this point are [500, 83], remember that 500 represents 10 (batch_size) * 50 (steps in sequence)
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets
        
    '''
    
    # print(logits, targets, lstm_size, num_classes)
    # logits: Tensor("add:0", shape=(500, 83), dtype=float32) 
    # targets: Tensor("targets:0", shape=(10, 50), dtype=int32)
    # num_classes: 83
    
    # One-hot encode targets and reshape to match logits, one row per batch_size per step
    # print(y_one_hot): Tensor("one_hot_1:0", shape=(10, 50, 83), dtype=float32)
    y_one_hot = tf.one_hot(targets, num_classes)

    # print(y_reshaped): Tensor("Reshape_1:0", shape=(500, 83), dtype=float32)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
    
    '''
    
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

class CharRNN:
   
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
