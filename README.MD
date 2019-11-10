# My first feed-forward neural network

implementations concerning the neural network implementatiom from scratch
following the Niranj Kumar's tutorial:
https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b

## Usage

I am using a `virtualenv --python=python3` and dependencies are listed in the
_requirements.txt_ file genereted by `pip freeze > requirements.txt`. 

### Single sigmoid experimentation

Here we experiment with a single sigmoid doing a liner separation of the data
using either _Cross Entropy_ or _Mean Squared Error_ as loss functions:

```sh
python sigmoid.py
```

This first example the los curve through the epochs, prints the accuracy for
training and validation data and then shows a scatter plot of the chosen data
with thin dots for right predictions and thick dots for wrong predictions.

### Specific ff neural net

In this example we create a neural network with 2 inputs, one hidden layer with
2 neurons and the output layer with one neuron, all of them using sigmoid as the
activation function. The network uses _Mean Squared Error_ as loss function.

```sh
python first_ffnet.py
```
### TODO: Generic neural net

TODO: Neural net accepting variable number of inputs and hidden layers sizes.
