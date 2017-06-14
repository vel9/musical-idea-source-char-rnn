#Endless Source of Musical Ideas

Please make sure to read the detailed description for this project at:
http://vel9.com/variations/variations.html

Resources: 
-----------
Original model: 
  https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
Original model inspired by Karpathy's Character RNN: 
  http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
Original approach which used **kern format for training: 
  http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write

Additional citations are provided within the files themselves
------------

Set up the environment: 

'''conda env create -f environment.yml'''

Use the pre-trained checkpoint to sample from the rnn with: 

'''python generate.py'''

To quickly listen to what's been generated, grab the gen*.xml file
and upload it to the great musicxml viewer at https://www.soundslice.com/musicxml-viewer/