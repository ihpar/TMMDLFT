from tensorflow.python.keras.layers import LSTM, Activation, Dense
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import consts
from mu2_reader import *
from model_ops import load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
import os
import io


def make_db(makam, part_id, dir_path, note_dict, oh_manager, set_size, is_whole=False):
    songs = []

    for curr_song in hicaz_parts.hicaz_songs:
        song = curr_song['file']
        part_map = curr_song['parts_map']
        song_final = curr_song['sf']
        song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
        songs.append(song)

    x_lst, y_lst = [], []
    for song in songs:
        part = song.get_part(part_id)
        xs, ys = [], []
        for i in range(len(part) - set_size):
            x_sec = [oh_manager.int_2_oh(x) for x in part[i:i + set_size]]
            y_sec = oh_manager.int_2_oh(part[i + set_size])
            xs.append(x_sec)
            ys.append(y_sec)

        if not is_whole:
            x_lst.append(np.array(xs))
            y_lst.append(np.array(ys))
        else:
            x_lst.extend(xs)
            y_lst.extend(ys)
    if not is_whole:
        return x_lst, y_lst
    else:
        return np.array(x_lst), np.array(y_lst)


def train_whole(makam, src_model, xs, ys, target_model):
    in_shape = xs.shape[1:]
    out_shape = ys.shape[1]

    base_model = load_model(makam, src_model, False)
    new_model = Sequential()
    for i, layer in enumerate(base_model.layers):
        if i == 4:
            break
        layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(out_shape))
    new_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    new_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    # mc = ModelCheckpoint('cp_' + target_model + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = new_model.fit(xs, ys, epochs=100, batch_size=16, shuffle=True, validation_split=0.1, callbacks=[es])

    save_model(makam, target_model, new_model)
    plt.plot(history.history['loss'])
    plt.show()


def train_model(makam, src_model, xs, ys, target_model):
    in_shape = xs[0].shape[1:]
    out_shape = ys[0].shape[1]

    base_model = load_model(makam, src_model, False)
    new_model = Sequential()
    for i, layer in enumerate(base_model.layers):
        if i == 4:
            break
        layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(out_shape))
    new_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    new_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.summary()

    histories = []

    for i in range(40):
        print(f'=== Main loop: {i} ===')
        for x, y in zip(xs, ys):
            history = new_model.fit(x, y, epochs=1, batch_size=16)
            histories.extend(history.history['loss'])

    save_model(makam, target_model, new_model)
    plt.plot(histories)
    plt.show()


def get_starters(init_file, set_size, note_dict, oh_manager):
    nom_index = 2
    den_index = 3
    beat, tot = None, Fraction(0)
    starters = []
    with codecs.open(init_file, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        i = 0
        for line in lines:
            if i == set_size:
                break
            parts = line.split('\t')
            parts = [s.strip() for s in parts]
            parts[0] = int(parts[0])

            if (parts[0] in [1, 7, 9, 10, 11, 12, 24, 28]) and (
                    parts[nom_index].isdigit() and parts[den_index].isdigit()):
                # note name
                note_name = parts[1].lower().strip()
                if note_name == '':
                    note_name = 'rest'
                note_num = note_dict.get_note_by_name(note_name)
                # note dur
                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                tot += note_len
                dur = str(note_len)
                dur_alt = parts[nom_index] + '/' + parts[den_index]
                dur = note_dict.get_num_by_dur(dur)
                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)
                combine = oh_manager.nd_2_oh(str(note_num) + ':' + str(dur))
                starters.append(combine)
                i += 1
    return np.array(starters), tot


def compose(makam, time_sig, measure_cnt, init_file, model_name, set_size, note_dict, oh_manager, song_title):
    starters, tot = get_starters(init_file, set_size, note_dict, oh_manager)
    target_dur = time_sig * measure_cnt - tot
    song = np.array([np.copy(starters)])
    xpy = song.shape[1]
    model = load_model(makam, model_name)

    while target_dur > 0:
        part = song[:, -xpy:, :]
        prediction = model.predict(part)
        shape = prediction.shape
        p_inner = np.copy(prediction[0])
        max_index = np.argmax(p_inner)
        print('Max prob: ', p_inner[max_index])

        p_inner = np.zeros(shape[1])
        p_inner[max_index] = 1.0
        n_d = oh_manager.oh_2_nd(p_inner)
        parts = n_d.split(':')
        note = int(parts[0])
        dur = int(parts[1])
        dur = note_dict.get_dur_by_num(dur)
        target_dur -= Fraction(dur)
        song = np.append(song, np.array([[p_inner]]), axis=1)

    lines = consts.mu2_header
    lines[0] = '9	4	Pay	Payda	Legato%	Bas	Çek	Söz-1	Söz-2	0.444444444'
    lines[2] = '51		9	4				Agiraksak		'
    lines[1] = lines[1].replace('{makam}', makam)
    lines[7] = lines[7].replace('{song_title}', song_title)
    for row in song[0]:
        n_d = oh_manager.oh_2_nd(row)
        parts = n_d.split(':')
        note = int(parts[0])
        dur = int(parts[1])
        note = note_dict.get_note_by_num(note)
        if not note:
            raise Exception('Note N/A')
        note = note.capitalize()

        dur = note_dict.get_dur_by_num(dur).split('/')

        if note == 'Rest':
            lines.append('9		{num}	{denom}	95	96	64	.		0.5'
                         .replace('{num}', dur[0])
                         .replace('{denom}', dur[1]))
        else:
            lines.append('9	{nn}	{num}	{denom}	95	96	64	.		0.5'
                         .replace('{nn}', note)
                         .replace('{num}', dur[0])
                         .replace('{denom}', dur[1]))

    file_name = song_title + '.mu2'
    song_path = os.path.join(os.path.abspath('..'), 'songs', makam, file_name)
    with io.open(song_path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{file_name} is saved to disk!')


def main():
    makam = 'hicaz'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    set_size = 8
    time_sig = Fraction(9, 4)
    ver = '61'

    # xs, ys = make_db(makam, 'A', dir_path, note_dict, oh_manager, set_size)
    # xs = [[[n1,n2,n3,..,n8],[n2,n3,...,n9]], song:[8s:[],8s:[],...]]
    # ys = [[n1,n2,...,nm], song:[outs]]
    # train_model(makam, 'lstm_v' + ver, xs, ys, 'sec_A40_v' + ver)

    xs, ys = make_db(makam, 'A', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    train_whole(makam, 'lstm_v' + ver, xs, ys, 'sec_B0_v' + ver)
    # compose(makam, time_sig, 4, 'init-hicaz-0.mu2', 'sec_A40_v61', set_size, note_dict, oh_manager, 't_sec_A40_v61')


if __name__ == '__main__':
    main()
