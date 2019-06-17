import data_loader as dl
from model_ops import *
import numpy as np
import matplotlib.pyplot as plt
import note_dictionary as nd
import dur_dict as dd
from fractions import Fraction
import io
import time
import consts
import json
from nc_dictionary import NCDictionary
import random
import math

from oh_manager import OhManager
from probability_calculator import ProbabilityCalculator, dd_int


def train_by_all(makam, model, ver, set_size, exclude, main_epochs):
    file_cnt = int(dl.get_data_size(makam, ver) / 3)
    histories = []
    train_set = []
    for i in range(file_cnt):
        if i in exclude:
            continue
        train_set.append(i)

    for e in range(main_epochs):
        random.shuffle(train_set)
        for i in train_set:
            print(f'Training on Song {i}: Main {e}')
            print('==============================================================')
            x_train, y_train = dl.load_data(makam, ver, str(i), set_size)
            # history = model.fit(x_train, y_train, epochs=1) # v till 51
            history = model.fit(x_train, y_train, epochs=1, batch_size=16)
            histories.append(history.history['loss'])

    return histories


def make_song(model, starting, total_dur, batch_size):
    gen_song = np.copy(starting)
    xpy = gen_song.shape[1]

    for i in range(total_dur):
        part = gen_song[:, -xpy:, :]
        prediction = np.array([dl.to_one_hot(model.predict(part), 0.8)])
        gen_song = np.append(gen_song, prediction, axis=1)

    return gen_song


def song_to_mus2_data(song):
    note_dict = nd.NoteDictionary()
    first_v = song[0][0]
    curr_v = first_v.dot(2 ** np.arange(first_v.size)[::-1])

    counter = 0
    notes = [curr_v]
    durs = []
    for vset in song:
        for v in vset:
            ternary = v.dot(2 ** np.arange(v.size)[::-1])
            if ternary != curr_v:
                note = note_dict.get_note_by_num(ternary)
                if not note[3]:
                    raise Exception('Note DNE in dict!')
                notes.append(note)
                durs.append(Fraction(counter, 192))

                curr_v = ternary
                counter = 0

            counter += 1

    durs.append(counter)
    return notes, durs


def data_to_mus2(song, makam, song_title, initiator):
    note_dict = NCDictionary()
    lines = consts.mu2_header
    lines[1] = lines[1].replace('{makam}', makam)
    lines[7] = lines[7].replace('{song_title}', song_title)
    for row in song[0]:
        note = int(''.join(str(b) for b in row[:7]), 2)
        dur = int(''.join(str(b) for b in row[7:]), 2)
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

    file_name = song_title + '_' + initiator + '.mu2'
    path = os.path.join(os.path.abspath('..'), 'songs', makam, file_name)
    with io.open(path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{file_name} is saved to disk!')
    return lines


def trainer(makam, ver, model_name, exclude, set_size, main_epochs):
    x_train, y_train = dl.load_data(makam, ver, '25', set_size)
    model = build_model(x_train.shape[1:], y_train.shape[1])

    start = time.time()
    histories = train_by_all(makam, model, ver, set_size, exclude, main_epochs)

    path = os.path.join(os.path.abspath('..'), 'models', makam, model_name + '_histories.json')
    with open(path, 'w') as hs:
        hs.write(json.dumps(histories))

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    save_model(makam, model_name, model)


def whole_train(makam, ver, model_name, exclude, set_size, epochs):
    x_train, y_train = dl.load_whole_data(makam, ver, set_size, exclude)

    model = build_whole_model(x_train.shape[1:], y_train.shape[1])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('cp_' + model_name + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    start = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, shuffle=False, validation_split=0.1,
                        callbacks=[es, mc])
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    save_model(makam, model_name, model)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def plot_loss(makam, model_name):
    path = os.path.join(os.path.abspath('..'), 'models', makam, model_name + '_histories.json')
    with open(path, 'r') as fp:
        histories = json.load(fp)
        plot_data = []
        for history in histories:
            plot_data.extend(history)

        plt.plot(plot_data)
        plt.show()


def make_song_ext(model, prob_calc, lower, upper, x, total):
    note_dict = NCDictionary()
    song = np.copy(x)
    xpy = song.shape[1]

    for i in range(total):
        part = song[:, -xpy:, :]
        # 0.79 best with 79.69% acc
        prediction = np.array([dl.to_one_hot_ext(part, model.predict(part), lower, upper, note_dict, prob_calc)])
        song = np.append(song, prediction, axis=1)

    return song


def make_oh_song(model, starter_notes, song_len, lo, hi):
    song = np.copy(starter_notes)
    xpy = song.shape[1]
    chose_cnt = 0
    for i in range(song_len):
        part = song[:, -xpy:, :]
        prediction = model.predict(part)
        shape = prediction.shape
        p_inner = np.copy(prediction[0])
        max_index = np.argmax(p_inner)
        print('Max prob: ', p_inner[max_index])
        if p_inner[max_index] < lo:
            p_cp = np.copy(p_inner)
            p_cp[max_index] = 0
            p_cp_max_index = np.argmax(p_cp)
            second_best = p_cp[p_cp_max_index]
            if second_best > hi:
                chose_cnt += 1
                print('Second best: ', second_best)
                max_index = random.choice([max_index, p_cp_max_index])
        p_inner = np.zeros(shape[1])
        p_inner[max_index] = 1.0
        song = np.append(song, np.array([[p_inner]]), axis=1)

    print('Chose cnt: ', chose_cnt)
    return song


def make_mus2_oh(song, makam, song_title, initiator):
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    lines = consts.mu2_header
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

    file_name = song_title + '_' + initiator + '.mu2'
    path = os.path.join(os.path.abspath('..'), 'songs', makam, file_name)
    with io.open(path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{file_name} is saved to disk!')


def main():
    makam = 'hicaz'
    model_name = 'lstm_v60'
    # ver = 'v3'
    # ver = 'oh'  # v 60, 61, 62, 63
    ver = 'flt'  # v 70

    # set_size = 8  # v 41
    # set_size = 4  # v 44
    # set_size = 16  # v 45
    # set_size = 6  # v 46, 47, 48
    set_size = 8  # v 50, 51, 60, 61, 62, 63, 70
    # exclude = [4, 14, 21, 32, 36, 55, 66, 88, 91, 94, 101, 109, 130]
    exclude = [4, 14, 32, 55, 66, 88, 91, 94, 109, 130]  # v 50, 51, 60, 61, 62, 63, 70
    # main_epochs = 64  # v 44, 45, 46
    # main_epochs = 96  # v 47
    # main_epochs = 128  # v 48, 49
    # main_epochs = 200  # v 51
    # epochs = 500  # v 50
    epochs = 500  # v 60, 61, 70
    # main_epochs = 50  # v 62
    # main_epochs = 100  # v 63
    whole_train(makam, ver, model_name, exclude, set_size, epochs)  # v 50, 60, 61, 70
    # trainer(makam, ver, model_name, exclude, set_size, main_epochs)  # v 62, 63
    # plot_loss(makam, model_name)

    '''
    # pc = ProbabilityCalculator(makam, set_size)
    initiator = str(exclude[3])
    model = load_model(makam, model_name)
    x_test, y_test = dl.load_data(makam, ver, initiator, set_size)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # accu = scores[1]
    # upper = min(1.0, accu * 1.1)
    # lower = max(0.4, accu * accu * accu)
    song_len = 128
    starter_notes = [x_test[0]]
    # 0.79 -> 0.7
    # song = make_song_ext(model, pc, lower, upper, starter_notes, song_len)
    # _ = data_to_mus2(song, makam, model_name, initiator)
    song = make_oh_song(model, starter_notes, song_len, 0.5, 0.1)  # ver oh
    make_mus2_oh(song, makam, model_name, initiator)  # ver oh
    # chose_cnt = 55 (v60.55), 29 (v61.55), 17 (v62.55), 2 (v63.55)
    '''


if __name__ == '__main__':
    main()
