from tensorflow.python.keras.layers import LSTM, Activation, Dense
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import RMSprop
from mu2_reader import *
from model_ops import load_model
import numpy as np


def make_db(makam, part_id, dir_path, note_dict, oh_manager, set_size):
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
        x_lst.append(np.array(xs))
        y_lst.append(np.array(ys))
    return x_lst, y_lst


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
    # base_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.summary()

    '''
    weights = []
    for layer in new_model.layers:
        weights.append(layer.get_weights())
    '''

    for x, y in zip(xs, ys):
        scores = new_model.evaluate(x, y, verbose=0)
        print("new: %s: %.2f%%" % (new_model.metrics_names[1], scores[1] * 100))

        scores = base_model.evaluate(x, y, verbose=0)
        print("base: %s: %.2f%%" % (base_model.metrics_names[1], scores[1] * 100))

        new_model.fit(x, y, epochs=1, batch_size=16)

    '''
    for i, layer in enumerate(new_model.layers):
        print(i, layer.name)
        new_model_weights = layer.get_weights()
        if all([np.array_equal(a1, a2) for a1, a2 in zip(new_model_weights, weights[i])]):
            print('Not changed')
        else:
            print('Changed!!!!')
    '''

def main():
    makam = 'hicaz'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    set_size = 8
    xs, ys = make_db(makam, 'A', dir_path, note_dict, oh_manager, set_size)
    # xs = [[[n1,n2,n3,..,n8],[n2,n3,...,n9]], song:[8s:[],8s:[],...]]
    # ys = [[n1,n2,...,nm], song:[outs]]
    train_model(makam, 'lstm_v61', xs, ys, 'intro_v61')


if __name__ == '__main__':
    main()
