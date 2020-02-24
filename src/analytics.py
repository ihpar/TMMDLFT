import os
import codecs
from fractions import Fraction
from nc_dictionary import NCDictionary
from oh_manager import OhManager


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
                    res.append([note_num, dur])
                i += 1

    return res


def plot_weights(files, note_dict, oh_manager):
    notes = get_notes(files, note_dict, oh_manager)

    print(notes)


def main():
    makam = 'hicaz'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)

    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and (f.startswith('hicaz--') or f.startswith('bes-hicaz-'))]

    plot_weights(files, note_dict, oh_manager)


if __name__ == '__main__':
    main()
