import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras


def initialize_model():
    model = Sequential([
        Dense(32, input_shape=(100,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])

    print(model.summary())

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return (model)


def generate_data():
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    return (data, one_hot_labels)


model = initialize_model()

data, one_hot_labels = generate_data()

model.fit(data, one_hot_labels, epochs=10, batch_size=32)
