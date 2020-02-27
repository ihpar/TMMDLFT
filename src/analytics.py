import os
import codecs
from fractions import Fraction
from nc_dictionary import NCDictionary
# from oh_manager import OhManager
from collections import defaultdict
import matplotlib.pyplot as plt
import hicaz_parts


def get_notes(files, note_dict):
    nom_index = 2
    den_index = 3
    res = []
    for song_path in files:
        with codecs.open(song_path, 'r', encoding='utf8') as sf:
            lines = sf.read().splitlines()
            i = 1
            for line in lines:
                parts = line.split('\t')
                parts = [s.strip() for s in parts]
                parts[0] = int(parts[0])

                if (parts[0] in [1, 7, 9, 10, 11, 12, 23, 24, 28]) and (
                        parts[nom_index].isdigit() and parts[den_index].isdigit()):
                    # note name
                    note_name = parts[1].lower().strip()
                    if note_name == '':
                        note_name = 'rest'
                    note_num = note_dict.get_note_by_name(note_name)
                    # note dur
                    note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                    dur = str(note_len)
                    dur_alt = parts[nom_index] + '/' + parts[den_index]
                    dur = note_dict.get_num_by_dur(dur)
                    if not dur:
                        dur = note_dict.get_num_by_dur(dur_alt)
                    res.append([note_num, note_len])
                i += 1

    return res


def plot_weights(file_list_a, file_list_b, note_dict, chart_title, legend_a, legend_b, out_file=''):
    corpus_notes, composer_notes = get_notes(file_list_a, note_dict), get_notes(file_list_b, note_dict)
    corpus_od = defaultdict(int)
    for note in corpus_notes:
        corpus_od[note[0]] += note[1]
    key_max = max(corpus_od.keys(), key=(lambda ki: corpus_od[ki]))
    corpus_max = corpus_od[key_max]

    composer_od = defaultdict(int)
    for note in composer_notes:
        composer_od[note[0]] += note[1]
    key_max = max(composer_od.keys(), key=(lambda ki: composer_od[ki]))
    composer_max = composer_od[key_max]

    corpus_xs, composer_xs = [], []
    corpus_ys, composer_ys = [], []
    cor_dict, com_dict = {}, {}

    th = 0.005

    for k, v in sorted(corpus_od.items()):
        normalized = float(v) / corpus_max
        if normalized < th:
            continue
        corpus_xs.append(k)
        corpus_ys.append(normalized)
        cor_dict[k] = normalized

    for k, v in sorted(composer_od.items()):
        normalized = float(v) / composer_max
        if normalized < th:
            continue
        composer_xs.append(k)
        composer_ys.append(normalized)
        com_dict[k] = normalized

    uni = list(set().union(corpus_xs, composer_xs))
    uni.sort()

    cor_ys, comp_ys = [], []
    note_names = []

    for el in uni:
        if el in cor_dict:
            cor_ys.append(cor_dict[el])
        else:
            cor_ys.append(0)

        if el in com_dict:
            comp_ys.append(com_dict[el])
        else:
            comp_ys.append(0)

        note_name = note_dict.get_note_by_num(el)
        if note_name == 'rest':
            note_name = 'sus'
        note_names.append(note_name)

    print(note_names)

    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    ax.set_title(chart_title)
    ax.plot(note_names, cor_ys, label=legend_a)
    ax.plot(note_names, comp_ys, label=legend_b)

    plt.grid()
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    plt.show()


def plot_freqs(seq_len, top, file_list_a, file_list_b, note_dict, chart_title, legend_a, legend_b, out_file=''):
    notes_a, notes_b = get_notes(file_list_a, note_dict), get_notes(file_list_b, note_dict)
    a_dict, b_dict = defaultdict(int), defaultdict(int)

    for i in range(len(notes_a) - seq_len):
        sect = notes_a[i:i + seq_len]
        rep = [str(x[0]) for x in sect]
        rep = '-'.join(rep)
        a_dict[rep] += 1

    for i in range(len(notes_b) - seq_len):
        sect = notes_b[i:i + seq_len]
        rep = [str(x[0]) for x in sect]
        rep = '-'.join(rep)
        b_dict[rep] += 1

    a_max_value, b_max_value = max(a_dict.values()), max(b_dict.values())
    i = 0
    a_dict_cut, b_dict_cut = {}, {}
    for w in sorted(a_dict, key=a_dict.get, reverse=True):
        a_dict_cut[w] = a_dict[w]
        i += 1
        if i == top:
            break

    i = 0
    for w in sorted(b_dict, key=b_dict.get, reverse=True):
        b_dict_cut[w] = b_dict[w]
        i += 1
        if i == top:
            break

    uni = list(set().union(a_dict_cut.keys(), b_dict_cut.keys()))
    a_lst, b_lst, note_names = [], [], []

    for el in uni:
        if el in a_dict:
            a_lst.append(a_dict[el] / a_max_value)
        else:
            a_lst.append(0)

        if el in b_dict:
            b_lst.append(b_dict[el] / b_max_value)
        else:
            b_lst.append(0)

        parts = el.split('-')
        notes = []
        for part in parts:
            note_name = note_dict.get_note_by_num(int(part))
            notes.append(note_name)

        note_names.append('-'.join(notes))

    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    ax.set_title(chart_title)
    ax.plot(note_names, a_lst, label=legend_a)
    ax.plot(note_names, b_lst, label=legend_b)

    plt.grid()
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    plt.show()


def main():
    makam = 'hicaz'
    note_dict = NCDictionary()
    # oh_manager = OhManager(makam)

    '''
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    corpus_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f)) and (f.startswith('hicaz--') or f.startswith('bes-hicaz-'))]

    dir_path = os.path.join(os.path.abspath('..'), 'songs', 'hicaz-sarkilar')
    composer_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if os.path.isfile(os.path.join(dir_path, f))]

    # plot_weights(corpus_files, composer_files, note_dict, 'Nota Ağırlıkları', 'SymbTr', 'Oto Besteci', out_file='agirlik_genel.png')

    corpus_files = []
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    for song in hicaz_parts.hicaz_songs:
        corpus_files.append(os.path.join(dir_path, song['file']))

    # plot_weights(corpus_files, composer_files, note_dict, 'Aksak Şarkı Nota Ağırlıkları', 'SymbTr', 'Oto Besteci', out_file='agirlik_sarki.png')
    '''

    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    corpus_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f)) and (f.startswith('hicaz--') or f.startswith('bes-hicaz-'))]

    '''
    corpus_files = []
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    for song in hicaz_parts.hicaz_songs:
        corpus_files.append(os.path.join(dir_path, song['file']))
    '''
    dir_path = os.path.join(os.path.abspath('..'), 'songs', 'hicaz-sarkilar')
    composer_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if os.path.isfile(os.path.join(dir_path, f))]

    plot_freqs(5, 10, corpus_files, composer_files, note_dict, 'Beşli Dizi Frekansları', 'SymbTr', 'Oto Besteci', out_file='freq_genel_5.png')
    # plot_freqs(5, 10, corpus_files, composer_files, note_dict, 'Aksak Şarkı Beşli Dizi Frekansları', 'SymbTr', 'Oto Besteci', out_file='freq_aksak_5.png')


if __name__ == '__main__':
    main()
