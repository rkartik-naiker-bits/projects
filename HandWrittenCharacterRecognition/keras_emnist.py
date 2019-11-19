# Simple CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import losses
import pandas as pd
from Constants import Constants


train = pd.read_csv(Constants.DATA_SET_PATH+'emnist-balanced-train.csv', header=None)
test = pd.read_csv(Constants.DATA_SET_PATH+'emnist-balanced-test.csv', header=None)

X_train = train.iloc[:, 1:]
X_test = test.iloc[:, 1:]

Y_train = train.iloc[:, 0]
Y_test = test.iloc[:, 0]

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
num_classes = y_train.shape[1]
print('Num classes = ', num_classes)

params_cnn_1 = Constants.params_cnn_1
params_cnn_2 = Constants.params_cnn_2
params_cnn_3 = Constants.params_cnn_3

def cnn_model_1(dropout_flag, params):

    lst = params_cnn_1[params]
    a = lst[0]
    b = lst[1]
    c = lst[2]
    d = lst[3]
    model = Sequential()
    model.add(Conv2D(a, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(b, (5, 5), input_shape=(12, 12, 1), activation='relu'))
    model.add(MaxPooling2D())


    model.add(Flatten())
    model.add(Dense(c, activation='relu'))
    if d != 0:
        model.add(Dense(d, activation='relu'))
    if dropout_flag:
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_model_2(dropout_flag, params):

    lst = params_cnn_2[params]
    a = lst[0]
    b = lst[1]
    c = lst[2]

    model = Sequential()
    model.add(Conv2D(a, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(b, activation='relu'))
    if c != 0:
        model.add(Dense(c, activation='relu'))
    if dropout_flag:
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_model_3(dropout_flag, params):

    lst = params_cnn_3[params]
    a = lst[0]
    b = lst[1]
    c = lst[2]
    d = lst[3]
    e = lst[4]
    f = lst[5]

    model = Sequential()
    model.add(Conv2D(a, (3, 3), input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2D(b, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(c, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2D(d, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    if e != 0:
        model.add(Dense(e))
    model.add(Dense(f))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout_flag:
        model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# model = cnn_model_1(True, "1")
# model = cnn_model_2(True, "1")
model = cnn_model_3(True, "3")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))