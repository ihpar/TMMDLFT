import os
import codecs
from fractions import Fraction
from nc_dictionary import NCDictionary
from oh_manager import OhManager
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def get_notes(files, note_dict, oh_manager):
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
                    res.append([note_num, note_name, note_len])
                i += 1

    return res


def plot_weights(corpus_files, composer_files, note_dict, oh_manager):
    corpus_notes = get_notes(corpus_files, note_dict, oh_manager)
    corpus_od = defaultdict(int)
    for note in corpus_notes:
        corpus_od[note[1]] += note[2]
    key_max = max(corpus_od.keys(), key=(lambda ki: corpus_od[ki]))
    corpus_max = corpus_od[key_max]

    composer_notes = get_notes(composer_files, note_dict, oh_manager)
    composer_od = defaultdict(int)
    for note in composer_notes:
        composer_od[note[1]] += note[2]
    key_max = max(composer_od.keys(), key=(lambda ki: composer_od[ki]))
    composer_max = composer_od[key_max]

    xs = []
    ys = []

    for k, v in sorted(corpus_od.items()):
        if v < 100:
            continue
        xs.append(k)
        ys.append(float(v) / corpus_max)

    print(xs)

    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    ax.set_title('Hicaz Makamı Genel Nota Ağırlıkları')
    ax.plot(xs, ys)
    plt.xticks(rotation=90)

    plt.tight_layout()
    # plt.savefig('coding_error.png')
    plt.show()


def main():
    makam = 'hicaz'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)

    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    corpus_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f)) and (f.startswith('hicaz--') or f.startswith('bes-hicaz-'))]

    dir_path = os.path.join(os.path.abspath('..'), 'songs', 'hicaz-sarkilar')
    composer_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if os.path.isfile(os.path.join(dir_path, f))]

    plot_weights(corpus_files, composer_files, note_dict, oh_manager)


if __name__ == '__main__':
    main()
