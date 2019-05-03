import json
import os
import numpy as np
from random import randint


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


def find_candidates(seq, lo, hi):
    candidates = list()
    while lo < hi:
        candidate = list()
        mini = 9999.0
        all_zeros = True
        for n in seq:
            if n <= lo:
                candidate.append(0)
            else:
                candidate.append(1)
                all_zeros = False
                if n < mini:
                    mini = n
        if not all_zeros:
            candidates.append(candidate)
        else:
            break
        lo = mini
    return candidates


def pick_candidate(candidates):
    # TODO: fix logic!
    return np.array(candidates[randint(0, len(candidates) - 1)])


def to_one_hot_ext(pred, lo, hi, nd):
    note_seq = pred[0][:7]
    dur_seq = pred[0][7:]
    note_candidates = find_candidates(note_seq, lo, hi)
    dur_candidates = find_candidates(dur_seq, lo, hi)
    print(len(note_candidates), len(dur_candidates))

    note = pick_candidate(note_candidates)
    note_num = int(''.join(str(b) for b in note), 2)
    while not nd.get_note_by_num(note_num):
        print('Note picked silly')
        note = pick_candidate(note_candidates)
        note_num = int(''.join(str(b) for b in note), 2)

    dur = pick_candidate(dur_candidates)
    dur_num = int(''.join(str(b) for b in dur), 2)
    while not nd.get_dur_by_num(dur_num):
        dur = pick_candidate(dur_candidates)
        dur_num = int(''.join(str(b) for b in dur), 2)

    res = np.array([np.concatenate((note, dur), axis=0)])

    return res


def main():
    x, y = load_data('hicaz', '25')
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
