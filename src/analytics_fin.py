import os
import numpy as np
import codecs
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import random
import added_ss

from fractions import Fraction
from sklearn.model_selection import LeaveOneOut
from scipy import stats, integrate
from dur_translator import DurTranslator
from nc_dictionary import NCDictionary
from note_translator import NoteTranslator
from analytics_utils import ND, ST, PCH


# noinspection DuplicatedCode
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


def get_songs_data(makam, note_dict=None, nt=None, dt=None, st=ST.base):
    min_song_len = sys.maxsize
    broad_list = []
    note_nums, dur_nums = [], []
    if st == ST.base:
        songs = added_ss.added_songs[makam]
        d_path = os.path.join(os.path.abspath('..'), 'mu2')
    else:
        songs = added_ss.added_songs[makam + '_gen']
        d_path = os.path.join(os.path.abspath('..'), 'songs', makam)

    for bs in songs:
        song = parse_song_in_mu2(os.path.join(d_path, bs['file']), note_dict, nt, dt)
        song_entity = {'notes': [], 'durs': [], 'parts': {}}
        for note in song:
            parts = [int(x) for x in note.split(':')]
            note_nums.append(parts[0])
            dur_nums.append(parts[1])

            song_entity['notes'].append(parts[0])
            song_entity['durs'].append(parts[1])

        song_entity['parts'] = bs['parts_map'].copy()
        min_song_len = min(min_song_len, len(song_entity['notes']))
        broad_list.append(song_entity)

    note_nums = np.array(note_nums)
    dur_nums = np.array(dur_nums)
    return note_nums, dur_nums, broad_list, min_song_len


def abs_measurement(makam):
    oh_manager, nt, dt, note_dict = None, None, None, None
    if makam == 'hicaz':
        note_dict = NCDictionary()
    else:
        nt = NoteTranslator(makam)
        dt = DurTranslator(makam)

    # oh_manager = OhManager(makam)
    note_nums_src, dur_nums_src, broad_list, min_len = get_songs_data(makam, note_dict, nt, dt, ST.base)
    mean_src = np.mean(note_nums_src)
    std_src = np.std(note_nums_src)
    print('src mean:', mean_src, 'src std:', std_src)

    note_nums_gen, dur_nums_gen, gen_broad_list, gen_min_len = get_songs_data(makam, note_dict, nt, dt, ST.generated)
    mean_gen = np.mean(note_nums_gen)
    std_gen = np.std(note_nums_gen)
    print('gen mean:', mean_gen, 'gen std:', std_gen)

    print('base min:', min_len, 'gen min:', gen_min_len)


def get_different_pitch_count(song):
    unique = set(song['notes'])
    elem_count = len(unique)
    return elem_count


def get_pitch_range(song):
    notes = [x for x in song['notes'] if x > 1]
    maxi = max(notes)
    mini = min(notes)
    pitch_range = maxi - mini
    return pitch_range


def get_avg_pitch_shift(song):
    notes = [x for x in song['notes'] if x > 1]
    intervals = np.diff(notes)
    return np.mean(abs(intervals))


def get_avg_ioi(song, note_dict=None, dt=None):
    durs = song['durs']
    if dt:
        durs = [Fraction(dt.get_dur_name_by_num(x)) for x in durs]
    else:
        durs = [Fraction(note_dict.get_dur_by_num(x)) for x in durs]

    intervals = np.diff(durs)
    return np.mean(abs(intervals))


def get_related_measures(song, bars, makam, note_dict=None, dt=None):
    time_sig = Fraction(9, 8)
    if makam == 'nihavent':
        time_sig = Fraction(8, 8)

    pm = song['parts']
    pa = my_flatten(pm['I'] + pm['A'])
    pa = pa[:bars[0]]
    pb = my_flatten(pm['B'])
    pb = pb[:bars[1]]
    pc = my_flatten(pm['C'])
    pc = pc[:bars[2]]

    notes = song['notes']
    durs = song['durs']

    measures = []
    measure = {'notes': [], 'durs': []}
    total = Fraction(0)

    measure_no = 0
    res = []
    for n, d in zip(notes, durs):
        if note_dict:
            dl = Fraction(note_dict.get_dur_by_num(d))
        else:
            dl = Fraction(dt.get_dur_name_by_num(d))
        measure['notes'].append(n)
        measure['durs'].append(d)
        total += dl
        if total >= time_sig:
            measures.append(measure)
            if (measure_no in pa) or (measure_no in pb) or (measure_no in pc):
                res.append(measure)
            measure = {'notes': [], 'durs': []}
            total = Fraction(0)
            measure_no += 1
    return res


def get_count_per_bar(song, bars, makam, note_dict=None, dt=None, nd=ND.note):
    num_bars = sum(bars)
    res = np.zeros((num_bars, 1))
    measures = get_related_measures(song, bars, makam, note_dict, dt)
    for i, measure in enumerate(measures):
        if nd == ND.note:
            cnt = len(set(measure['notes']))
        else:
            cnt = len(set(measure['durs']))
        res[i] = cnt
    return res


def get_bar_pch(song, pch, bars, makam, note_dict=None, nt=None, dt=None):
    num_bars = sum(bars)
    measures = get_related_measures(song, bars, makam, note_dict, dt)
    bc = pch.get_note_bin_count()
    res = np.zeros((num_bars, bc))
    for i, measure in enumerate(measures):
        pch.init_note_histogram()
        notes = measure['notes']
        for note in notes:
            pch.add_note(note)
        res[i] = pch.get_note_histogram()
    return res


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


# noinspection PyTypeChecker
def utils_overlap_area(a, b):
    pdf_a = stats.gaussian_kde(a)
    pdf_b = stats.gaussian_kde(b)
    lim = 200
    return integrate.quad(lambda x: min(pdf_a(x), pdf_b(x)), np.min((np.min(a), np.min(b))), np.max((np.max(a), np.max(b))), limit=lim)[0]


def my_flatten(il):
    res = []
    for i in il:
        if type(i) is list:
            res.extend(i)
        else:
            res.append(i)
    return res


def is_choosable(song_obj, bars):
    res = False
    pm = song_obj['parts']
    pa = my_flatten(pm['I'] + pm['A'])
    pb = my_flatten(pm['B'])
    pc = my_flatten(pm['C'])
    if (len(pa) >= bars[0]) and (len(pb) >= bars[1]) and (len(pc) >= bars[2]):
        res = True
    return res


def choose_songs(base_broad_list, bars, num_samples):
    song_idx = list(range(len(base_broad_list)))
    chosen_songs = random.sample(song_idx, num_samples)
    song_idx = [x for x in song_idx if x not in chosen_songs]
    for i, song_ix in enumerate(chosen_songs):
        new_song = song_ix
        while not is_choosable(base_broad_list[new_song], bars):
            new_song = random.choice(song_idx)
            song_idx.remove(new_song)
        chosen_songs[i] = new_song

    return chosen_songs


def get_pch(song_obj, pch):
    pch.init_note_histogram()
    for note in song_obj['notes']:
        pch.add_note(note)
    return pch.get_note_histogram()


def get_dch(song_obj, pch):
    pch.init_dur_histogram()
    for dur in song_obj['durs']:
        pch.add_dur(dur)
    return pch.get_dur_histogram()


def get_pctm(song_obj, pch):
    pch.init_note_transition_matrix()
    notes = song_obj['notes']
    note_len = len(notes) - 1
    for i in range(note_len):
        curr = notes[i]
        nex = notes[i + 1]
        pch.add_tuple(curr, nex)
    return pch.get_note_transition_matrix()


def get_nltm(song_obj, pch):
    pch.init_dur_transition_matrix()
    durs = song_obj['durs']
    dur_len = len(durs) - 1
    for i in range(dur_len):
        curr = durs[i]
        nex = durs[i + 1]
        pch.add_dur_tuple(curr, nex)
    return pch.get_dur_transition_matrix()


def abs_rel_pdfs(feature, makam, titles):
    oh_manager, nt, dt, note_dict = None, None, None, None
    if makam == 'hicaz':
        note_dict = NCDictionary()
    else:
        nt = NoteTranslator(makam)
        dt = DurTranslator(makam)

    num_samples = 20
    note_nums_src, dur_nums_src, base_broad_list, min_len = get_songs_data(makam, note_dict, nt, dt, ST.base)
    song_idx = range(len(base_broad_list))

    bars = [4, 4, 2]
    num_bars = sum(bars)

    if feature in ['bar_used_pitch', 'bar_used_note', 'bar_pitch_class_histogram']:
        chosen = choose_songs(base_broad_list, bars, num_samples)
    else:
        chosen = random.sample(song_idx, num_samples)
    print('chosen:', chosen)

    note_nums_gen, dur_nums_gen, gen_broad_list, gen_min_len = get_songs_data(makam, note_dict, nt, dt, ST.generated)
    pch = None

    if feature == 'bar_used_pitch' or feature == 'bar_used_note':
        set1_eval = np.zeros((num_samples, num_bars, 1))  # base set
        set2_eval = np.zeros((num_samples, num_bars, 1))  # gen set
    elif feature == 'total_pitch_class_histogram' or feature == 'total_note_length_histogram':
        pch = PCH(makam)
        if feature == 'total_pitch_class_histogram':
            bc = pch.get_note_bin_count()
        else:
            bc = pch.get_dur_bin_count()

        set1_eval = np.zeros((num_samples, bc))  # base set
        set2_eval = np.zeros((num_samples, bc))  # gen set
    elif feature == 'bar_pitch_class_histogram':
        pch = PCH(makam)
        bc = pch.get_note_bin_count()
        set1_eval = np.zeros((num_samples, num_bars, bc))  # base set
        set2_eval = np.zeros((num_samples, num_bars, bc))  # gen set
    elif feature == 'pitch_class_transition_matrix':
        pch = PCH(makam)
        bc = pch.get_note_bin_count()
        set1_eval = np.zeros((num_samples, bc, bc))  # base set
        set2_eval = np.zeros((num_samples, bc, bc))  # gen set
    elif feature == 'note_length_transition_matrix':
        pch = PCH(makam)
        bc = pch.get_dur_bin_count()
        set1_eval = np.zeros((num_samples, bc, bc))  # base set
        set2_eval = np.zeros((num_samples, bc, bc))  # gen set
    else:
        set1_eval = np.zeros((num_samples, 1))  # base set
        set2_eval = np.zeros((num_samples, 1))  # gen set

    for i in range(num_samples):
        if feature == 'total_used_pitch':
            set1_eval[i] = get_different_pitch_count(base_broad_list[chosen[i]])
            set2_eval[i] = get_different_pitch_count(gen_broad_list[i])
        elif feature == 'bar_used_pitch':
            set1_eval[i] = get_count_per_bar(base_broad_list[chosen[i]], bars, makam, note_dict, dt, ND.note)
            set2_eval[i] = get_count_per_bar(gen_broad_list[i], bars, makam, note_dict, dt, ND.note)
        elif feature == 'total_used_note':
            set1_eval[i] = get_different_dur_count(base_broad_list[chosen[i]])
            set2_eval[i] = get_different_dur_count(gen_broad_list[i])
        elif feature == 'bar_used_note':
            set1_eval[i] = get_count_per_bar(base_broad_list[chosen[i]], bars, makam, note_dict, dt, ND.dur)
            set2_eval[i] = get_count_per_bar(gen_broad_list[i], bars, makam, note_dict, dt, ND.dur)
        elif feature == 'total_pitch_class_histogram':
            set1_eval[i] = get_pch(base_broad_list[chosen[i]], pch)
            set2_eval[i] = get_pch(gen_broad_list[i], pch)
        elif feature == 'bar_pitch_class_histogram':
            set1_eval[i] = get_bar_pch(base_broad_list[chosen[i]], pch, bars, makam, note_dict, nt, dt)
            set2_eval[i] = get_bar_pch(gen_broad_list[i], pch, bars, makam, note_dict, nt, dt)
        elif feature == 'total_note_length_histogram':
            set1_eval[i] = get_dch(base_broad_list[chosen[i]], pch)
            set2_eval[i] = get_dch(gen_broad_list[i], pch)
        elif feature == 'pitch_class_transition_matrix':
            set1_eval[i] = get_pctm(base_broad_list[chosen[i]], pch)
            set2_eval[i] = get_pctm(gen_broad_list[i], pch)
        elif feature == 'note_length_transition_matrix':
            set1_eval[i] = get_nltm(base_broad_list[chosen[i]], pch)
            set2_eval[i] = get_nltm(gen_broad_list[i], pch)
        elif feature == 'pitch_range':
            set1_eval[i] = get_pitch_range(base_broad_list[chosen[i]])
            set2_eval[i] = get_pitch_range(gen_broad_list[i])
        elif feature == 'avg_pitch_shift':
            set1_eval[i] = get_avg_pitch_shift(base_broad_list[chosen[i]])
            set2_eval[i] = get_avg_pitch_shift(gen_broad_list[i])
        elif feature == 'avg_IOI':
            set1_eval[i] = get_avg_ioi(base_broad_list[chosen[i]], note_dict, dt)
            set2_eval[i] = get_avg_ioi(gen_broad_list[i], note_dict, dt)

    no_ax_set = ['bar_used_pitch',
                 'bar_used_note',
                 'total_pitch_class_histogram',
                 'total_note_length_histogram',
                 'bar_pitch_class_histogram',
                 'pitch_class_transition_matrix',
                 'note_length_transition_matrix']

    print('\n' + titles[feature] + ':')
    print('------------------------')
    print(' Base Set')
    if feature in no_ax_set:
        print('  mean: ', np.mean(set1_eval))
        print('  std: ', np.std(set1_eval))
    else:
        print('  mean: ', np.mean(set1_eval, axis=0))
        print('  std: ', np.std(set1_eval, axis=0))

    print('------------------------')
    print(' Gen Set')
    if feature in no_ax_set:
        print('  mean: ', np.mean(set2_eval))
        print('  std: ', np.std(set2_eval))
    else:
        print('  mean: ', np.mean(set2_eval, axis=0))
        print('  std: ', np.std(set2_eval, axis=0))

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))

    set1_intra = np.zeros((num_samples, 1, num_samples - 1))
    set2_intra = np.zeros((num_samples, 1, num_samples - 1))

    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][0] = utils_c_dist(set1_eval[test_index], set1_eval[train_index])
        set2_intra[test_index[0]][0] = utils_c_dist(set2_eval[test_index], set2_eval[train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, 1, num_samples))

    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][0] = utils_c_dist(set1_eval[test_index], set2_eval)

    plot_set1_intra = np.transpose(set1_intra, (1, 0, 2)).reshape(1, -1)
    plot_set2_intra = np.transpose(set2_intra, (1, 0, 2)).reshape(1, -1)
    plot_sets_inter = np.transpose(sets_inter, (1, 0, 2)).reshape(1, -1)

    sns.kdeplot(plot_set1_intra[0], label='intra base set')
    sns.kdeplot(plot_sets_inter[0], label='inter sets')
    sns.kdeplot(plot_set2_intra[0], label='intra gen set')

    plt.title(titles[feature] + ' (' + makam.capitalize() + ')')
    plt.xlabel('Euclidean distance')
    plt.show()

    print('\n' + titles[feature] + ' (' + makam.capitalize() + ')' + ':')
    print('------------------------')
    print(' Base Set')
    print('  Kullback–Leibler Divergence:', utils_kl_dist(plot_set1_intra[0], plot_sets_inter[0]))
    print('  Overlap Area:', utils_overlap_area(plot_set1_intra[0], plot_sets_inter[0]))

    print('------------------------')
    print(' Gen Set')
    print('  Kullback–Leibler Divergence:', utils_kl_dist(plot_set2_intra[0], plot_sets_inter[0]))
    print('  Overlap Area:', utils_overlap_area(plot_set2_intra[0], plot_sets_inter[0]))


def main():
    makams = ['hicaz', 'nihavent']
    curr_makam = 1
    features = ['total_used_pitch',
                'bar_used_pitch',
                'total_used_note',
                'bar_used_note',
                'total_pitch_class_histogram',
                'bar_pitch_class_histogram',
                'total_note_length_histogram',
                'pitch_class_transition_matrix',
                'pitch_range',
                'avg_pitch_shift',
                'avg_IOI',
                'note_length_transition_matrix']

    titles = {
        features[0]: 'Total Used Pitches',
        features[1]: 'Pitches Per Bar',
        features[2]: 'Total Used Durations',
        features[3]: 'Durations Per Bar',
        features[4]: 'Total Pitch Class Histogram',
        features[5]: 'Pitch Class Histogram Per Bar',
        features[6]: 'Total Duration Class Histogram',
        features[7]: 'Pitch Class Transition Matrix',
        features[8]: 'Pitch Range',
        features[9]: 'Average Pitch Shift',
        features[10]: 'Average Inter-Onset-Interval',
        features[11]: 'Note Length Transition Matrix'
    }

    curr_feature = 11

    # abs_measurement(makams[curr_makam])
    abs_rel_pdfs(features[curr_feature], makams[curr_makam], titles)


if __name__ == '__main__':
    main()
