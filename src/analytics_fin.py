import os
import numpy as np
import codecs
from fractions import Fraction
from sklearn.model_selection import LeaveOneOut
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
import random

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

    print('base min:', min_len, 'gen min:', gen_min_len)


def get_different_pitch_count(song):
    unique = set(song['notes'])
    elem_count = len(unique)
    return elem_count


def get_different_dur_count(song):
    unique = set(song['durs'])
    elem_count = len(unique)
    return elem_count


def utils_c_dist(a, b):
    c_dist = np.zeros(len(b))
    for i in range(0, len(b)):
        c_dist[i] = np.linalg.norm(a - b[i])
    return c_dist


def utils_kl_dist(a, b, num_sample=1000):
    pdf_a = stats.gaussian_kde(a)  # Representation of a kernel-density estimate using Gaussian kernels.
    pdf_b = stats.gaussian_kde(b)
    sample_a = np.linspace(np.min(a), np.max(a), num_sample)
    sample_b = np.linspace(np.min(b), np.max(b), num_sample)
    return stats.entropy(pdf_a(sample_a), pdf_b(sample_b))


def utils_overlap_area(a, b):
    pdf_a = stats.gaussian_kde(a)
    pdf_b = stats.gaussian_kde(b)
    return integrate.quad(lambda x: min(pdf_a(x), pdf_b(x)), np.min((np.min(a), np.min(b))), np.max((np.max(a), np.max(b))))[0]


def abs_rel_pdfs(feature, makam, generated_songs_path):
    oh_manager, nt, dt, note_dict = None, None, None, None
    if makam == 'hicaz':
        note_dict = NCDictionary()
    else:
        nt = NoteTranslator(makam)
        dt = DurTranslator(makam)

    num_samples = 20
    note_nums_src, dur_nums_src, base_broad_list, min_len = get_base_data(makam, note_dict, nt, dt)
    song_idx = range(len(base_broad_list))
    chosen = random.sample(song_idx, num_samples)
    print('chosen:', chosen)
    note_nums_gen, dur_nums_gen, gen_broad_list, gen_min_len = get_gen_data(generated_songs_path, note_dict, nt, dt)

    set1_eval = {feature: np.zeros((num_samples, 1))}  # base set
    set2_eval = {feature: np.zeros((num_samples, 1))}  # gen set

    for i in range(num_samples):
        if feature == 'total_used_pitch':
            set1_eval[feature][i] = get_different_pitch_count(base_broad_list[chosen[i]])
            set2_eval[feature][i] = get_different_pitch_count(gen_broad_list[i])
        elif feature == 'total_used_note':
            set1_eval[feature][i] = get_different_dur_count(base_broad_list[chosen[i]])
            set2_eval[feature][i] = get_different_dur_count(gen_broad_list[i])

    print('\n' + feature + ':')
    print('------------------------')
    print(' base_set')
    print('  mean: ', np.mean(set1_eval[feature], axis=0))
    print('  std: ', np.std(set1_eval[feature], axis=0))

    print('------------------------')
    print(' gen_set')
    print('  mean: ', np.mean(set2_eval[feature], axis=0))
    print('  std: ', np.std(set2_eval[feature], axis=0))

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))

    set1_intra = np.zeros((num_samples, 1, num_samples - 1))
    set2_intra = np.zeros((num_samples, 1, num_samples - 1))

    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][0] = utils_c_dist(set1_eval[feature][test_index], set1_eval[feature][train_index])
        set2_intra[test_index[0]][0] = utils_c_dist(set2_eval[feature][test_index], set2_eval[feature][train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, 1, num_samples))

    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][0] = utils_c_dist(set1_eval[feature][test_index], set2_eval[feature])

    plot_set1_intra = np.transpose(set1_intra, (1, 0, 2)).reshape(1, -1)
    plot_set2_intra = np.transpose(set2_intra, (1, 0, 2)).reshape(1, -1)
    plot_sets_inter = np.transpose(sets_inter, (1, 0, 2)).reshape(1, -1)

    titles = {
        'total_used_pitch': 'Total Used Pitches',
        'total_used_note': 'Total Used Durations'
    }

    sns.kdeplot(plot_set1_intra[0], label='intra base set')
    sns.kdeplot(plot_sets_inter[0], label='inter sets')
    sns.kdeplot(plot_set2_intra[0], label='intra gen set')

    plt.title(titles[feature])
    plt.xlabel('Euclidean distance')
    plt.show()

    print('\n' + feature + ':')
    print('------------------------')
    print(' base set')
    print('  Kullback–Leibler Divergence:', utils_kl_dist(plot_set1_intra[0], plot_sets_inter[0]))
    print('  Overlap Area:', utils_overlap_area(plot_set1_intra[0], plot_sets_inter[0]))

    print('------------------------')
    print(' gen set')
    print('  Kullback–Leibler Divergence:', utils_kl_dist(plot_set2_intra[0], plot_sets_inter[0]))
    print('  Overlap Area:', utils_overlap_area(plot_set2_intra[0], plot_sets_inter[0]))


def main():
    makams = ['hicaz', 'nihavent']
    gen_dirs = ['hicaz-sarkilar', 'nihavent']
    curr_makam = 0
    features = ['total_used_pitch',
                'total_used_note']
    curr_feature = 1

    generated_songs_path = os.path.join(os.path.abspath('..'), 'songs', gen_dirs[curr_makam])
    # abs_measurement(makams[curr_makam], generated_songs_path)

    abs_rel_pdfs(features[curr_feature], makams[curr_makam], generated_songs_path)


if __name__ == '__main__':
    main()
