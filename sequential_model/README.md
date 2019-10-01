### Getting Started with Keras Functional API

Useful for defining:

* multi-output models
* directed acyclic graphs
* models with shared layers

_layers_ in keras is a callable instance which returns a tensor/s. 
Input and output tensors along with definition of all the layers constitutes a model. 

A simple sequential model in keras can be defined as follows:

```buildoutcfg
    model = Sequential([
        Dense(32, input_shape=(100,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
```

Also, all models in keras are callable. We can invoke models on tensors. 
It should be noted however, that when invoking models we also use weights along with the architecture.

This in effect makes it very efficient to use the same model on an array of inputs. 
For example, if we have a video stream instead of an image, we can simply apply the trained model
at each time frame and generate predictions/classifications. 

Example of such a use case is presented below:

```buildoutcfg

from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

