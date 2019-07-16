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


def main(epochs=4, second_conv=True, dropout=True):
    batch_size = 128
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train_valid, y_train_valid), (x_test, y_test) = fashion_mnist.load_data()
    train_len = int(len(x_train_valid) * 0.75)
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

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    if second_conv:
        model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # print(model.layers[0].get_weights()[0].shape)
    # exit(0)
    # print(plt.imshow(model.layers[1].get_weights()[0]))
    # plt.show()
    train_time = time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_valid, y_valid))
    total_time = time() - train_time
    print("train time: {:.2f}".format(total_time))
    score_train = model.evaluate(x_train, y_train, verbose=0)
    score_valid = model.evaluate(x_valid, y_valid, verbose=0)
    score_test = model.evaluate(x_test, y_test, verbose=0)
    data = {"time": total_time, "score_train": score_train, "score_valid": score_valid, "score_test": score_test}
    return history, data


def plot_acc_and_loss(hist):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(hist["acc"])), hist["acc"], label="training accuracy")
    plt.plot(range(len(hist["val_acc"])), hist["val_acc"], label="validation accuracy")
    plt.xlabel('epochs')
    plt.ylabel("accuracy")
    plt.title("Train and validation accuracy")
    plt.legend(loc="lower right")
    plt.subplot(2, 1, 2)
    plt.plot(range(len(hist["loss"])), hist["loss"], label="training loss")
    plt.plot(range(len(hist["val_loss"])), hist["val_loss"], label="validation loss")
    plt.xlabel('epochs')
    plt.ylabel("accuracy")
    plt.title("Train and validation loss")
    plt.legend(loc="upper right")
    plt.subplots_adjust(hspace=1, wspace=0.35)


def p8(p=1):
    print(150 * '-')
    if p == 1:
        print("original network")
        _, data4 = main(epochs=4, second_conv=True, dropout=True)
        _, data12 = main(epochs=12, second_conv=True, dropout=True)
        history30, data30 = main(epochs=30, second_conv=True, dropout=True)
        plot_acc_and_loss(history30.history)
        plt.savefig("p8_a")
    elif p == 2:
        print("network without second convolutional layer")
        _, data4 = main(epochs=4, second_conv=False, dropout=True)
        _, data12 = main(epochs=12, second_conv=False, dropout=True)
        history30, data30 = main(epochs=30, second_conv=False, dropout=True)
        plot_acc_and_loss(history30.history)
        plt.savefig("p8_b")
    else:
        print("network without dropout")
        _, data4 = main(epochs=4, second_conv=True, dropout=False)
        _, data12 = main(epochs=12, second_conv=True, dropout=False)
        history30, data30 = main(epochs=30, second_conv=True, dropout=False)
        plot_acc_and_loss(history30.history)
        plt.savefig("p8_c")

    format_str = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}"

    print(format_str.format("epochs", "train acc", "valid acc", "test acc", "train loss", "valid loss", "test loss",
                            "total time", "time/epoch"))
    print(format_str.format(4, round(data4["score_train"][1], 3), round(data4["score_valid"][1], 3),
                            round(data4["score_test"][1], 3), round(data4["score_train"][0], 3),
                            round(data4["score_valid"][0], 3), round(data4["score_test"][0], 3),
                            round(data4["time"], 3), round(data4["time"] / 4, 3)))

    print(format_str.format(12, round(data12["score_train"][1], 3), round(data12["score_valid"][1], 3),
                            round(data12["score_test"][1], 3), round(data12["score_train"][0], 3),
                            round(data12["score_valid"][0], 3), round(data12["score_test"][0], 3),
                            round(data12["time"], 3), round(data12["time"] / 12, 3)))

    print(format_str.format(30, round(data30["score_train"][1], 3), round(data30["score_valid"][1], 3),
                            round(data30["score_test"][1], 3), round(data30["score_train"][0], 3),
                            round(data30["score_valid"][0], 3), round(data30["score_test"][0], 3),
                            round(data30["time"], 3), round(data30["time"] / 30, 3)))

    print(150 * '-')


if __name__ == "__main__":
    p8()
