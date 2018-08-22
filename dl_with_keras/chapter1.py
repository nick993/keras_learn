from __future__ import print_function

import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l1, l2
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
import json
import os
from dl_with_keras.util import normalize, flatten


NUM_CLASSES = 10
BATCH_SIZE=64
HIDDEN_LAYERS = 128
EPOCHS = 100
VERBOSE = 1
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
#OPTIMIZER = SGD()
#OPTIMIZER = RMSprop()
OPTIMIZER = Adam()
MODEL_DIR = 'tmp'

def create_model(x_train_shape):
    # Sequential Model
    model = Sequential()
    model.add(Dense(HIDDEN_LAYERS, input_shape=(x_train_shape,), kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(HIDDEN_LAYERS))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    model.summary()
    return model

def save_model_json(model):
    model_json_string = model.to_json()
    with open('mnist_train.json', 'w') as outfile:
        json.dump(model_json_string, outfile)
    save_model_params(model)

def get_model_from_json(json_string):
    return model_from_json(json_string)

def save_model_params(model):
    model.save('mnist_params.h5')

((x_train, y_train), (x_test, y_test)) = mnist.load_data()
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
x_train = normalize(flatten(x_train))
x_test = normalize(flatten(x_test))

model = create_model(x_train.shape[1])

model.compile(optimizer=SGD(), loss=categorical_crossentropy, metrics=['accuracy'])

#added checkpoints after each epochs
checkpoint = ModelCheckpoint(filepath=(MODEL_DIR + '\model-{epoch:02d}.h5'), verbose=1, save_best_only=True, save_weights_only=False, monitor='val_acc', mode='max')
tensorboard_check = TensorBoard('logs', histogram_freq=0, write_graph=True, write_images=False)
history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint, tensorboard_check])

#save_model_json(model)

score = model.evaluate(x_test, y_test, VERBOSE)
print('Test Score :' + score.__str__())

# predict_classes will predict the class output
predict_1 = model.predict_classes(x_test)
print('Predict : ' + predict_1.__str__())

## todo: learn about callbacks

