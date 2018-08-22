from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

data = mnist.load_data()

def reshape_data(img):
    return img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)

def pre_process_data(X_train, X_test):
    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return (X_train, X_test)

def pre_process_result(Y_train, Y_test):
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)
    return (Y_train, Y_test)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def create_checkpoint():
    filepath = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    callback_list = [checkpoint];
    return callback_list

def fit_model(model, X_train, Y_train):
    model.fit(X_train, Y_train,validation_split=0.33, batch_size=32, epochs=8, verbose=1, callbacks=create_checkpoint())


def main():

    xs = 60000
    ((X_train, Y_train), (X_test, Y_test)) = mnist.load_data()
    X_train = X_train[0:xs, :]
    Y_train = Y_train[0:xs]
    (X_train, X_test) = pre_process_data(X_train, X_test)
    (Y_train, Y_test) = pre_process_result(Y_train, Y_test)

    model = create_model()
    compile_model(model)
    fit_model(model, X_train, Y_train)

main()