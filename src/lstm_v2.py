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


def train_by_all(makam, model, ver, set_size, exclude, main_epochs):
    file_cnt = int(dl.get_data_size(makam, ver) / 3)
    histories = []

    for e in range(main_epochs):
        for i in range(file_cnt):
            if i in exclude:
                continue
            print(f'Training on Song {i}: Main {e}')
            print('==============================================================')
            x_train, y_train = dl.load_data(makam, ver, str(i), set_size)
            history = model.fit(x_train, y_train, epochs=1)
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


def data_to_mus2(song, makam, song_title):
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

    path = os.path.join(os.path.abspath('..'), 'songs', makam, song_title + '.mu2')
    with io.open(path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{song_title}.mu2 is saved to disk!')
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


def plot_loss(makam, model_name):
    path = os.path.join(os.path.abspath('..'), 'models', makam, model_name + '_histories.json')
    with open(path, 'r') as fp:
        histories = json.load(fp)
        plot_data = []
        for history in histories:
            plot_data.extend(history)

        plt.plot(plot_data)
        plt.show()


def make_song_ext(model, x, total):
    note_dict = NCDictionary()
    song = np.copy(x)
    xpy = song.shape[1]

    for i in range(total):
        part = song[:, -xpy:, :]
        # 0.79 best with 79.69% acc
        prediction = np.array([dl.to_one_hot_ext(model.predict(part), 0.75, 0.2, note_dict)])
        song = np.append(song, prediction, axis=1)

    return song


def main():
    makam = 'hicaz'
    model_name = 'lstm_v45'
    ver = 'v3'

    # set_size = 8  # v 41
    # set_size = 4  # v 44
    set_size = 16  # v 45
    exclude = [4, 14, 21, 32, 36, 55, 66, 88, 91, 94, 101, 109, 130]
    main_epochs = 64

    trainer(makam, ver, model_name, exclude, set_size, main_epochs)
    plot_loss(makam, model_name)

    '''
    model = load_model(makam, model_name)
    x_test, y_test = dl.load_data(makam, ver, '36', set_size)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    song = make_song_ext(model, [x_test[0]], 256)
    lines = data_to_mus2(song, makam, model_name)
    '''


if __name__ == '__main__':
    main()
