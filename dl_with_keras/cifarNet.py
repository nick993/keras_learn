from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.regularizers import l2

class cifarNet:
    @staticmethod
    def build(no_classes,inp_shape):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), padding='same', input_shape=inp_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(500, kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(no_classes))
        model.add(Activation('softmax'))

        return model


