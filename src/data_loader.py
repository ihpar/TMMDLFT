import json
import os
import numpy as np


def load_data(makam, ver, idx, set_size=1):
    path = os.path.join(os.path.abspath('..'), 'data', makam, ver)
    song = json.load(open(os.path.join(path, 's_' + idx), 'r'))
    xs, ys = [], []
    for i in range(len(song) - set_size):
        x_sec = song[i:i + set_size]
        y_sec = song[i + set_size]
        xs.append(x_sec)
        ys.append(y_sec)

    return np.array(xs), np.array(ys)


def get_data_size(makam, ver):
    path = os.path.join(os.path.abspath('..'), 'data', makam, ver)
    return len(os.listdir(path))


def to_one_hot(seq, th):
    res = np.zeros(seq.shape, dtype=int)
    for i, vec in enumerate(seq):
        for j, el in enumerate(vec):
            if el > th:
                res[i][j] = 1

    return res


def main():
    x, y = load_data('hicaz', '25')
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
