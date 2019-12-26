from nakarat_endings import hicaz_song_endings
from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping

import consts
from mu2_reader import *
from model_ops import load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import random
import math
from candidate_picker import CandidatePicker


def parse_notes(notes, note_dict, oh_manager):
    res = []
    for n in notes:
        parts = n.split(':')
        name = parts[0].lower()
        dur = parts[1]
        name = note_dict.get_note_by_name(name)
        dur = note_dict.get_num_by_dur(dur)
        oh = oh_manager.nd_2_oh(str(name) + ':' + str(dur))
        res.append(oh)
    return np.array(res)


def make_ending_data(makam, note_dict, oh_manager):
    xs, ys = [], []
    for hse in hicaz_song_endings:
        prevs = hse['prevs']
        end_f = hse['endings'][0]
        end_s = hse['endings'][1]
        fin = hse['fin']
        prevs = parse_notes(prevs, note_dict, oh_manager)
        end_f = parse_notes(end_f, note_dict, oh_manager)
        prevs = np.concatenate((prevs, end_f))
        seq_len = prevs.shape[0]
        for i in range(seq_len - 8):
            part = prevs[i:i + 8]
    return xs, ys


def make_model():
    model = Sequential()
    return model


def train_nakarat_ending_model(makam, model_name, xs, ys, eps=0):
    model = make_model()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    if eps == 0:
        history = model.fit(xs, ys, epochs=100, batch_size=16, shuffle=False, validation_split=0.1, callbacks=[es])
    else:
        history = model.fit(xs, ys, epochs=eps, batch_size=16, shuffle=False)

    save_model(makam, model_name, model)
    plt.plot(history.history['loss'], label='train')
    if eps == 0:
        plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def main():
    makam = 'hicaz'
    ver = 'v0'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    xs, ys = make_ending_data(makam, note_dict, oh_manager)
    # model_name = 'nakarat_ender_' + ver
    # train_nakarat_ending_model(makam, model_name, xs, ys)


if __name__ == '__main__':
    main()
