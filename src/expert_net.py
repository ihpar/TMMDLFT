from nc_dictionary import NCDictionary
from oh_manager import OhManager
from model_ops import load_model, save_model
from data_loader import load_whole_data
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Activation, Dense, LSTM, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping

import time


def distance(r_o, n_a, n_b, oh_manager):
    nd_o = oh_manager.int_2_nd(r_o)
    nd_a = oh_manager.int_2_nd(n_a)
    nd_b = oh_manager.int_2_nd(n_b)

    p_o = nd_o.split(':')
    n_o = int(p_o[0])
    d_o = int(p_o[1])

    p_a = nd_a.split(':')
    n_a = int(p_a[0])
    d_a = int(p_a[1])

    p_b = nd_b.split(':')
    n_b = int(p_b[0])
    d_b = int(p_b[1])
    if abs(n_o - n_a) < abs(n_o - n_b):
        return [1, 0]
    if abs(n_o - n_a) > abs(n_o - n_b):
        return [0, 1]
    if abs(d_o - d_a) < abs(d_o - d_b):
        return [1, 0]
    if abs(d_o - d_a) > abs(d_o - d_b):
        return [0, 1]
    return [0.5, 0.5]


def create_training_data(makam, model_a, model_b, oh_manager):
    ver = 'oh'
    set_size = 8
    exclude = []
    xs, ys = load_whole_data(makam, ver, set_size, exclude)
    x_train, y_train = [], []
    counter = 0
    for x, y in zip(xs, ys):
        p_a = model_a.predict(np.array([x]))[0]
        p_b = model_b.predict(np.array([x]))[0]
        a_max = np.argmax(p_a)
        b_max = np.argmax(p_b)
        r_out = np.argmax(y)

        x_data = [oh_manager.oh_2_zo(n) for n in x]
        x_mi = x_data.copy()
        x_mj = x_data.copy()

        x_mi.append(oh_manager.int_2_zo(a_max))
        x_mi.append(oh_manager.int_2_zo(b_max))
        y_data_i = distance(r_out, a_max, b_max, oh_manager)

        x_mj.append(oh_manager.int_2_zo(b_max))
        x_mj.append(oh_manager.int_2_zo(a_max))
        y_data_j = y_data_i[::-1]

        x_train.append(x_mi)
        y_train.append(y_data_i)

        x_train.append(x_mj)
        y_train.append(y_data_j)
        if counter % 10 == 0:
            print(f'd {counter}')
        counter += 1

    x_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'b_xsN')
    y_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'b_ysN')
    with open(x_file, 'w') as fx, open(y_file, 'w') as fy:
        fx.write(json.dumps(x_train))
        fy.write(json.dumps(y_train))
    print('Files created, exiting...')


def load_training_data(makam):
    x_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'b_xsN')
    y_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'b_ysN')
    with open(x_file, 'r') as fx, open(y_file, 'r') as fy:
        xs = json.load(fx)
        ys = json.load(fy)

    xs, ys = np.array(xs), np.array(ys)
    # (NumberOfExamples, TimeSteps, FeaturesPerStep)
    x_shape = xs.shape
    xs = xs.reshape((x_shape[0], 1, x_shape[1]))
    return xs, ys


def make_model(in_shape, out_shape):
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.4))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(out_shape))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model


def train_model(makam, model, model_name, x, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = model.fit(x, y, epochs=50, batch_size=16, shuffle=False, validation_split=0.1, callbacks=[es])

    save_model(makam, model_name, model)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def main():
    start = time.time()
    makam = 'hicaz'

    oh_manager = OhManager(makam)
    model_a = load_model(makam, 'sec_BW11_v61')
    model_b = load_model(makam, 'sec_BW12_v62')
    create_training_data(makam, model_a, model_b, oh_manager)

    # v0: LSTM(100), v1: LSTM(200), v2: LSTM(100)*LSTM(100)
    v = 'v4'
    x_train, y_train = load_training_data(makam)
    print(x_train.shape, y_train.shape)
    model = make_model(x_train.shape[1:], y_train.shape[1])
    train_model(makam, model, 'b_decider_' + v, x_train, y_train)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    main()
