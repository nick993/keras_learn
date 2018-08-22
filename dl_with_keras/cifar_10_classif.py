from keras.datasets import cifar10
from keras.utils import to_categorical
from dl_with_keras.util import normalize
from dl_with_keras.cifarNet import cifarNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model

NO_CLASSES = 10
VERBOSE = 1
INP_SHAPE = (32, 32, 3)
BATCH_SIZE = 128
VAL_SPLIT = 0.2
EPOCHS = 30

((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
x_train = normalize(x_train)
x_test = normalize(x_test)
y_train = to_categorical(y_train, NO_CLASSES)
y_test = to_categorical(y_test, NO_CLASSES)

model = cifarNet.build(NO_CLASSES, INP_SHAPE)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model_check = ModelCheckpoint(filepath='cifarnn_tmp\\weights-{epoch:02d}-{val_acc:.2f}.h5', save_weights_only=False, save_best_only=True, verbose=VERBOSE, monitor='val_acc', mode='max')
tensorbrd_chk = TensorBoard(log_dir='cifarnn', write_images=False, write_graph=True)
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_split=VAL_SPLIT, epochs=EPOCHS, callbacks=[model_check, tensorbrd_chk])

calcs = model.evaluate(x_test, y_test)
print('Pred : ' + calcs.__str__())

#to start tensorboard
