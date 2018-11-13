import data_loader as dl
from model_ops import *
import numpy as np
import matplotlib.pyplot as plt
import note_dictionary as nd
from fractions import Fraction
import io
import time
import consts
import json


def train_by_all(makam, model, ver, set_size, exclude, main_epochs):
    file_cnt = int(dl.get_data_size(makam, ver) / 3)
    histories = []

    for e in range(main_epochs):
        for i in range(file_cnt):
            if i in exclude:
                continue
            print(f'Training on Song {i}')
            print('==============================================================')
            x_train, y_train = dl.load_data(makam, ver, str(i), set_size)
            history = model.fit(x_train, y_train, epochs=12)
            histories.append(history.history['loss'])

    return histories


def make_song(model, starting, total_dur, batch_size):
    gen_song = np.copy(starting)
    xpy = gen_song.shape[1]

    for i in range(total_dur):
        part = gen_song[-1 * xpy * batch_size:]
        prediction = dl.to_one_hot(model.predict(part), 0.2)
        for j in range(batch_size):
            gen_song = np.append(gen_song, np.array([prediction[j * xpy:(j + 1) * xpy]]), axis=0)

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


def data_to_mus2(notes, durs, makam, song_title):
    lines = consts.mu2_header

    lines.append('9	La4  	1	4	95	96	64			0.25')

    path = os.path.join(os.getcwd(), 'songs', makam, song_title + '.mu2')
    with io.open(path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{song_title}.mu2 is saved to disk!')


def trainer(makam, ver, model_name, exclude, set_size, main_epochs):
    x_train, y_train = dl.load_data(makam, ver, '25', set_size)
    model = build_model(x_train.shape[1:], y_train.shape[1])

    start = time.time()
    histories = train_by_all(makam, model, ver, set_size, exclude, main_epochs)

    path = os.path.join(os.getcwd(), 'models', makam, model_name + '_histories.json')
    with open(path + '_histories', 'w') as hs:
        hs.write(json.dumps(histories))

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    save_model(makam, model_name, model)


def plot_loss(makam, model_name):
    path = os.path.join(os.getcwd(), 'models', makam, model_name + '_histories')
    with open(path, 'r') as fp:
        histories = json.load(fp)
        plot_data = []
        for history in histories:
            plot_data.extend(history)

        plt.plot(plot_data)
        plt.show()


def main():
    makam = 'hicaz'
    model_name = 'note_dur_lstm'
    ver = 'v2'
    set_size = 6
    exclude = [2, 12, 20, 30, 35, 55, 67, 88, 91, 93, 101, 103, 130]
    main_epochs = 12

    trainer(makam, ver, model_name, exclude, set_size, main_epochs)
    plot_loss(makam, model_name)


if __name__ == '__main__':
    main()
