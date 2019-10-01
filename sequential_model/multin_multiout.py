from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import numpy as np

# for the sake of reproducing these results
np.random.seed(0)

# headline input: receives sequence of 100 integers between 1 and 1000
# name assigned for this layer 'main_input'
main_input = Input(shape=(100, ), dtype='int32', name='main_input')

# now we will encode the input using an embedding layer
# we convert it into a 512 dimensional vector
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# now an lstm will transform the vector sequence into a single vector
# summarizing entire sequence
lstm_out = LSTM(32)(x)

# inserting auxiliary loss, allowing the LSTM and embedding layer to be trained smoothly
# refer to this image to get a better sense of model architecture:
# https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png
auxiliary_output =  Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# at this point, we want to use the output from lstm along with our auxiliary input
# and feed it further into a densely connected network
auxiliary_input = Input(shape=(5, ), name='aux_input')

# concatenate lstm output and auxiliary input
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# and a logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# finally defining our model in entirity
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

headline_data = np.round(np.abs(np.random.rand(12, 100) * 100))
additional_data = np.random.randn(12, 5)
headline_labels = np.random.randn(12, 1)
additional_labels = np.random.randn(12, 1)

# compiling the model with rmspro optimizer, binary crossentropy loss
# weights assigned to each loss 1. and 0.2
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': headline_labels, 'aux_output': additional_labels},
          epochs=50, batch_size=32)

model.predict({'main_input': headline_data, 'aux_input': additional_data})