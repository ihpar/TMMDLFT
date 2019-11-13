from nc_dictionary import NCDictionary
from oh_manager import OhManager
from model_ops import load_model
from data_loader import load_whole_data
import numpy as np
import os
import json


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

    x_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'xs')
    y_file = os.path.join(os.path.abspath('..'), 'data', makam, 'chooser', 'ys')
    with open(x_file, 'w') as fx, open(y_file, 'w') as fy:
        fx.write(json.dumps(x_train))
        fy.write(json.dumps(y_train))
    print('Files created, exiting...')


def main():
    makam = 'hicaz'
    oh_manager = OhManager(makam)

    model_a = load_model(makam, 'sec_AW6_v61')
    model_b = load_model(makam, 'sec_AW7_v62')

    create_training_data(makam, model_a, model_b, oh_manager)


if __name__ == '__main__':
    main()
