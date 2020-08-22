import os
import numpy as np
import codecs
from fractions import Fraction
from sklearn.model_selection import LeaveOneOut
import sys

import hicaz_parts
import nihavent_parts
from dur_translator import DurTranslator
from nc_dictionary import NCDictionary
from note_translator import NoteTranslator


# from oh_manager import OhManager


def parse_song_in_mu2(song_path, note_dict, nt=None, dt=None):
    nom_index = 2
    den_index = 3
    song = []

    with codecs.open(song_path, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            parts = [s.strip() for s in parts]

            if (parts[0] in [str(x) for x in [1, 4, 7, 9, 10, 11, 12, 23, 24, 28]]) and (parts[nom_index].isdigit() and parts[den_index].isdigit()):
                note_name = parts[1].lower().strip()
                if note_name == '':
                    note_name = 'rest'

                if nt:
                    note_num = nt.get_note_num_by_name(note_name)
                else:
                    note_num = note_dict.get_note_by_name(note_name)

                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                dur = str(note_len)
                dur_alt = parts[nom_index] + '/' + parts[den_index]

                if dt:
                    dur = dt.get_dur_num_by_name(dur_alt)
                else:
                    dur = note_dict.get_num_by_dur(dur)

                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)

                song.append(str(note_num) + ':' + str(dur))
    return song


def get_base_data(makam, note_dict=None, nt=None, dt=None):
    min_song_len = sys.maxsize
    broad_list = []
    note_nums, dur_nums = [], []
    with open(makam + '--nc_corpus.txt', 'r') as crp:
        songs = crp.read().splitlines()
        for song in songs:
            song_entity = {'notes': [], 'durs': []}
            notes = song.split(' ')
            for note in notes:
                parts = [int(x) for x in note.split(':')]
                '''
                if makam == 'hicaz':
                    nota = note_dict.get_note_by_num(parts[0])
                    dur = note_dict.get_dur_by_num(parts[1])
                else:
                    nota = nt.get_note_name_by_num(parts[0])
                    dur = dt.get_dur_name_by_num(parts[1])
                '''
                note_nums.append(parts[0])
                dur_nums.append(parts[1])

                song_entity['notes'].append(parts[0])
                song_entity['durs'].append(parts[1])

            min_song_len = min(min_song_len, len(song_entity['notes']))
            broad_list.append(song_entity)

    added_songs = []
    if makam == 'hicaz':
        for song in hicaz_parts.hicaz_songs:
            if song['file'].startswith('bes-hicaz-'):
                added_songs.append(song['file'])
    elif makam == 'nihavent':
        for song in nihavent_parts.nihavent_songs:
            added_songs.append(song['file'])

    for song in added_songs:
        song = os.path.join(os.path.abspath('..'), 'songs', 'added', makam, song)
        song = parse_song_in_mu2(song, note_dict, nt, dt)
        song_entity = {'notes': [], 'durs': []}

        for note in song:
            parts = [int(x) for x in note.split(':')]
            note_nums.append(parts[0])
            dur_nums.append(parts[1])
            song_entity['notes'].append(parts[0])
            song_entity['durs'].append(parts[1])

        min_song_len = min(min_song_len, len(song_entity['notes']))
        broad_list.append(song_entity)

    note_nums = np.array(note_nums)
    dur_nums = np.array(dur_nums)
    return note_nums, dur_nums, broad_list, min_song_len


# noinspection DuplicatedCode
def get_gen_data(generated_songs_path, note_dict=None, nt=None, dt=None):
    min_song_len = sys.maxsize
    broad_list = []

    note_nums, dur_nums = [], []
    gen_songs = [os.path.join(generated_songs_path, f) for f in os.listdir(generated_songs_path) if os.path.isfile(os.path.join(generated_songs_path, f))]
    for song in gen_songs:
        song = parse_song_in_mu2(song, note_dict, nt, dt)
        song_entity = {'notes': [], 'durs': []}
        for note in song:
            parts = [int(x) for x in note.split(':')]
            note_nums.append(parts[0])
            dur_nums.append(parts[1])
            song_entity['notes'].append(parts[0])
            song_entity['durs'].append(parts[1])

        min_song_len = min(min_song_len, len(song_entity['notes']))
        broad_list.append(song_entity)

    note_nums = np.array(note_nums)
    dur_nums = np.array(dur_nums)
    return note_nums, dur_nums, broad_list, min_song_len


def abs_measurement(makam, generated_songs_path):
    oh_manager, nt, dt, note_dict = None, None, None, None
    if makam == 'hicaz':
        note_dict = NCDictionary()
    else:
        nt = NoteTranslator(makam)
        dt = DurTranslator(makam)

    # oh_manager = OhManager(makam)
    note_nums_src, dur_nums_src, broad_list, min_len = get_base_data(makam, note_dict, nt, dt)
    mean_src = np.mean(note_nums_src)
    std_src = np.std(note_nums_src)
    print('src mean:', mean_src, 'src std:', std_src)

    note_nums_gen, dur_nums_gen, gen_broad_list, gen_min_len = get_gen_data(generated_songs_path, note_dict, nt, dt)
    mean_gen = np.mean(note_nums_gen)
    std_gen = np.std(note_nums_gen)
    print('gen mean:', mean_gen, 'gen std:', std_gen)


def exhaustive_cross_val_inter():
    num_samples = 100

    set1_eval = {'total_used_pitch': np.zeros((num_samples, 1))}
    metrics_list = set1_eval.keys()

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))

    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


def main():
    makam = 'nihavent'
    gen_dir = 'nihavent'
    generated_songs_path = os.path.join(os.path.abspath('..'), 'songs', gen_dir)
    # abs_measurement(makam, generated_songs_path)
    exhaustive_cross_val_inter()


if __name__ == '__main__':
    main()
