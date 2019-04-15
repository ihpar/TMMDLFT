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


def apply_threshold(vec, th):
    res = np.zeros(vec.shape, dtype=int)
    for i, _ in enumerate(vec):
        if vec[i] >= th:
            res[i] = 1
    return res


def to_one_hot_ext(pred, th, thl, nd):
    step = 0.01
    th_note, th_dur = th, th
    note_seq = pred[0][:7]
    dur_seq = pred[0][7:]

    note = apply_threshold(note_seq, th_note)
    note_num = int(''.join(str(b) for b in note), 2)

    if not nd.get_note_by_num(note_num):
        print(note_num)

    while note_num == 0 or not nd.get_note_by_num(note_num):
        th_note -= step
        note = apply_threshold(note_seq, th_note)
        note_num = int(''.join(str(b) for b in note), 2)
        if th_note < thl:
            break

    dur = apply_threshold(dur_seq, th_dur)
    dur_num = int(''.join(str(b) for b in dur), 2)
    while not nd.get_dur_by_num(dur_num):
        th_dur -= step
        dur = apply_threshold(dur_seq, th_dur)
        dur_num = int(''.join(str(b) for b in dur), 2)

    res = np.array([np.concatenate((note, dur), axis=0)])

    return res


def main():
    x, y = load_data('hicaz', '25')
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
