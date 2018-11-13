from fractions import Fraction
import json
import os
import consts
from dur_dict import DurDictionary


def get_roll(note_key, note_dur, min_dur):
    roll = []
    binr = [int(x) for x in bin(note_key)[2:].zfill(10)]
    count = int(note_dur / min_dur)
    for i in range(count):
        roll.append(binr)
    return roll


def corpus_2_roll(makam):
    min_dur = consts.min_dur
    song_list = []
    with open(makam + 'corpus_dur.txt', 'r') as f:
        songs = f.readlines()
        for i, song in enumerate(songs):
            song_roll = []
            notes = song.split(' ')
            for j, note in enumerate(notes):
                parts = note.split(':')
                note_key = int(parts[0])
                note_dur = Fraction(parts[1])
                roll = get_roll(note_key, note_dur, min_dur)
                song_roll.extend(roll)

            song_list.append(song_roll)
    return song_list


def build_in_out(song, in_len):
    song_x = []
    song_y = []
    for i in range(len(song)):
        if i + in_len >= len(song):
            break
        xs = song[i:i + in_len]
        ys = song[i + in_len]
        song_x.append(xs)
        song_y.append(ys)
    return song_x, song_y


def get_paths(makam, ver, song, xs, ys, idx):
    dire = os.path.join(os.path.abspath('..'), 'data', makam, ver)
    res = [os.path.join(dire, song + idx), os.path.join(dire, xs + idx), os.path.join(dire, ys + idx)]
    return res


def corpus_2_extended_hot(makam):
    song_list = []
    durs = DurDictionary(makam)

    with open(makam + 'corpus_dur.txt', 'r') as f:
        songs = f.readlines()
        for song in songs:
            song_data = []
            notes = song.split(' ')
            for note in notes:
                parts = note.split(':')
                note_key = int(parts[0])
                note_dur = Fraction(parts[1])
                binr = [int(x) for x in bin(note_key)[2:].zfill(10)]
                binr.extend(durs.dur_dict[note_dur])
                song_data.append(binr)

            song_list.append(song_data)

    return song_list


def main():
    corpus = corpus_2_extended_hot('hicaz--')
    ver = 'v2'

    for i, song in enumerate(corpus):
        print(f'Saving song data {i}')
        song_x, song_y = build_in_out(song, 6)
        si = str(i)
        paths = get_paths('hicaz', ver, 's_', 'x_', 'y_', si)
        with open(paths[0], 'w') as s, open(paths[1], 'w') as x, open(paths[2], 'w') as y:
            s.write(json.dumps(song))
            x.write(json.dumps(song_x))
            y.write(json.dumps(song_y))


if __name__ == '__main__':
    main()
