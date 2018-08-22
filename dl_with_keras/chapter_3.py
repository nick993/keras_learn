import numpy as np
from keras.datasets import mnist
from dl_with_keras.util import flatten, normalize
from keras.utils import to_categorical
from dl_with_keras.lenet import LeNet
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

NUM_CLASSES = 10
INP_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
VERBOSE = 1
NO_ITER = 5

((x_train ,y_train), (x_test, y_test)) = mnist.load_data()
x_test = normalize(x_test)
x_train = normalize(x_train)
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
#print(x_train.shape)

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
model = LeNet.build(INP_SHAPE, NUM_CLASSES)
model.compile(loss = categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

model_save = ModelCheckpoint(filepath='dcnn_tmp\\weights-{epoch:02d}-{val_acc:.2f}.h5', save_weights_only=False, save_best_only=True, monitor='val_acc', mode='max')
tensorboard_chk = TensorBoard(log_dir='dcnn_logs', histogram_freq=0, write_graph=True, write_images=False)
model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, epochs=NO_ITER, callbacks=[model_save, tensorboard_chk])


score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print('Test Score : ' + score.__str__())
