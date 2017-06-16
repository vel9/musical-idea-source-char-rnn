# RNN: An Endless Source of Musical Ideas

Please make sure to read the [detailed description](http://vel9.com/variations/variations.html) for this project which also provides the musical results and explains the approach.

I've expanded on one of the assignments in Udacity's Deep Learning Foundations nanodegree, by training the RNN on music compositions in **kern format.

Resources: 
-----------
* Original RNN model: https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
* Karpathy's Character RNN: http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
* Original approach which used **kern format for training: http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write

Additional citations are provided within the code itself, and in the detailed project description provided above.

Set up the environment:

```conda env create -f environment.yml```

Activate environment: 

```source activate muse```

Use the v24 pre-trained checkpoint to sample from the rnn with: 

```python generate.py```

To quickly check out what's been generated, grab the gen*.xml file
and upload it to the very cool [Soundslice musicxml viewer](https://www.soundslice.com/musicxml-viewer/) 

If you want to try and improve on the model, or generate fragments of different lengths, 
or provide a different prime event, make sure to play around with the params ```config.py```
