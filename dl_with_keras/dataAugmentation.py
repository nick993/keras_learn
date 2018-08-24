from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from dl_with_keras.cifarNet import cifarNet
from keras.utils import to_categorical
from dl_with_keras.util import flatten, normalize

((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = normalize(x_train)
x_test = normalize(x_test)


print('Augmenting Dataset : ....')
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)
INP_SHAPE = (32, 32, 3)
xtas = []
ytas = []
#for i in range(x_train.shape[0]):
#    num_aug = 0
#    x = x_train[i]
#    x = x.reshape((1,) + x.shape)
#    for x_aug in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
#        if num_aug > 5:
#            break
#        xtas.append(x_aug[0])
#        num_aug += 1

datagen.fit(x_train)

model = cifarNet.build(10, INP_SHAPE)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=x_train.shape[0], epochs=5, verbose=1)
score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('Acc : ' + score.__string__)

