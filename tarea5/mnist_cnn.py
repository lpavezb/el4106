'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from time import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt

batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train_valid, y_train_valid), (x_test, y_test) = fashion_mnist.load_data()
train_len = int(len(x_train_valid)*0.75)
x_train = x_train_valid[:train_len]
y_train = y_train_valid[:train_len]

x_valid = x_train_valid[train_len:]
y_valid = y_train_valid[train_len:]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_time = time()
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))
print("train time: {:.2f}".format(time() - train_time))
score_train = model.evaluate(x_train, y_train, verbose=0)
score_valid = model.evaluate(x_valid, y_valid, verbose=0)
score_test = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])
print('Validation loss:', score_valid[0])
print('Validation accuracy:', score_valid[1])
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])


def plot_training_acc_and_loss(hist):
    plt.figure()
    keys = list(hist.keys())
    keys.sort()
    for i in range(len(keys)):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(hist[keys[i]])), hist[keys[i]])
        plt.xlabel('epochs')
        plt.ylabel(keys[i])
        plt.title(keys[i])
    plt.subplots_adjust(hspace=1, wspace=0.35)
    plt.show()


plot_training_acc_and_loss(history.history)
