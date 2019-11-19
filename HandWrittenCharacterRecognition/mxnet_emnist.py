from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import pandas as pd
from keras.utils import np_utils
import pandas as pd
from Constants import Constants

mx.random.seed(1)

ctx=mx.cpu()
batch_size = 200
num_inputs = 784
num_outputs = 47


def load_data(file, num_features):
    npa = np.genfromtxt(file, delimiter=',')
    npa.astype(np.float32)

    X = nd.array(npa[..., 1:])
    X = X / 255.0
    Y = nd.array(npa[..., 0])

    return X, Y


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

data_shape = (num_inputs,)
label_shape = (1,)


Xtrain, Ytrain = load_data(Constants.DATA_SET_PATH+'emnist-balanced-train.csv', num_inputs)
Xtest, Ytest = load_data(Constants.DATA_SET_PATH+'emnist-balanced-test.csv', num_inputs)

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest), batch_size=batch_size, shuffle=False)

params_cnn_1 = Constants.params_cnn_1
params_cnn_2 = Constants.params_cnn_2
params_cnn_3 = Constants.params_cnn_3


def cnn_model_1(dropout_flag, param):
    net = gluon.nn.Sequential()
    with net.name_scope():
        lst = params_cnn_1[param]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]

        net.add(gluon.nn.Conv2D(channels=a, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=b, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(c, activation="relu"))
        if d != 0:
            net.add(gluon.nn.Dense(d, activation="relu"))
        if dropout_flag:
            net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(num_outputs))
        return net


def cnn_model_2(dropout_flag, param):
    net = gluon.nn.Sequential()
    with net.name_scope():
        lst = params_cnn_2[param]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        net.add(gluon.nn.Conv2D(channels=a, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(b, activation="relu"))
        if c != 0:
            net.add(gluon.nn.Dense(c, activation="relu"))
        if dropout_flag:
            net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(num_outputs))
        return net


def cnn_model_3(dropout_flag, params):
    net = gluon.nn.Sequential()
    with net.name_scope():
        lst = params_cnn_3[params]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]
        e = lst[4]
        f = lst[5]

        net.add(gluon.nn.Conv2D(channels=a, kernel_size=3))
        net.add(gluon.nn.BatchNorm(axis=-1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Conv2D(channels=b, kernel_size=3))
        net.add(gluon.nn.BatchNorm(axis=-1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=c, kernel_size=3))
        net.add(gluon.nn.BatchNorm(axis=-1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Conv2D(channels=d, kernel_size=3))
        net.add(gluon.nn.BatchNorm(axis=-1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))


        net.add(gluon.nn.Flatten())

        if e != 0:
            net.add(gluon.nn.Dense(e))

        net.add(gluon.nn.Dense(f))
        net.add(gluon.nn.BatchNorm())
        if dropout_flag:
            net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(num_outputs))
        net.add(gluon.nn.Activation('softrelu'))

        return net


#net = cnn_model_1(True, "1")
#net = cnn_model_2(True, "1")
net = cnn_model_3(True, "1")

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    #data_iterator.reset()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, num_inputs))
        data = data.reshape(batch_size, 1, 28, 28)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 10
smoothing_constant = .01

for ep in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, num_inputs))
        data = data.reshape(batch_size, 1, 28, 28)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (ep == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (ep, ", ", train_accuracy, test_accuracy))

