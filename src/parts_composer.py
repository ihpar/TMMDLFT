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
            x_sec = np.array([oh_manager.int_2_oh(x) for x in part[i:i + set_size]])
            y_sec = np.array(oh_manager.int_2_oh(part[i + set_size]))
            xs.append(x_sec)
            ys.append(y_sec)
        x_lst.append(xs)
        y_lst.append(ys)
    return np.array(x_lst), np.array(y_lst)


def train_model(makam, src_model, xs, ys, target_model):
    model = load_model(makam, src_model)
    for x, y in zip(xs, ys):
        scores = model.evaluate(x, y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        break


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
