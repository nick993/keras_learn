from keras.models import Sequential
from keras.layers import MaxPooling2D, Convolution2D, Activation, Flatten, Dense

class LeNet:
    @staticmethod
    def build(inp_shape, classes):
        model = Sequential()
        model.add(Convolution2D(20, kernel_size=5, padding='same', input_shape=inp_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Convolution2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
