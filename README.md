# Building a simple neural network #

The purpose of these scripts is to easily show how to build a simple neural network. When I was looking for tutorials about simple neural networks in tensorflow, I didn't find any good explanation or any neural network simple enough to explain the tensorflow syntax. I hope this helps you to better understand how to use tensorflow.

## Online training ##

### Basic-1-1dimensional-nn.py ###

This script contains a 1-6-6-1 neural network that is trained with a sinusoidal data. The network is trained with an online training, this means that the weights change with each data point. In other words, we train with each data point at a time. Once we have covered every point of the training set we say that we have finished an epoch. Once we finish and epoch we start the process all over again until we complete the maximum epoch amount.


### Neural network with a 1-6-6-1 arquitecture ###
![nn66]

The picture below shows graphs of the data during the training. We can see how the neural network aproximates the function of the data. The red line is the data given by the neural network, the blue line is the real data.

### Neural network data outcome graphs ###
![graphs1]

## Batch training ##

### Basic-1-1dimensional-nn-batch.py ###

This script is trains the neural network with a batch training. This means that we minimize the loss function considering not only 1 data but a set of more than 1 point. 













[nn66]:		https://github.com/diegoorellanaga/Tensorflow-Tutorial-For-Dummies/blob/master/Screenshot%20from%202017-10-20%2016-38-25.png

[graphs1]:	https://github.com/diegoorellanaga/Tensorflow-Tutorial-For-Dummies/blob/master/graphs.png
