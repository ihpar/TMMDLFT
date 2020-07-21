import os
import codecs
from fractions import Fraction
import json

from note_dictionary import NoteDictionary
from note_translator import NoteTranslator
from dur_translator import DurTranslator
from oh_manager import OhManager


def extract_mu2(f):
    notes, durs = [], []

    num_idx = 2
    den_idx = 3
    with codecs.open(f, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for i, line in enumerate(lines):
            try:
                parts = line.split('\t')
                parts = [s.strip() for s in parts]
                p_type = int(parts[0])

                if (p_type in [1, 7, 9, 10, 11, 12, 23, 24, 28]) and (parts[num_idx].isdigit() and parts[den_idx].isdigit()):
                    note = parts[1].lower().strip()
                    if note == '':
                        note = 'rest'
                    dur = parts[num_idx] + '/' + parts[den_idx]
                    notes.append(note)
                    durs.append(dur)
            except:
                print(f, i)

    return notes, durs


def extract_txt(f):
    notes, durs = [], []

    num_idx = 6
    den_idx = 7
    with codecs.open(f, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for i, line in enumerate(lines):
            try:
                parts = line.split('\t')
                parts = [s.strip() for s in parts]
                p_type = -1
                if parts[1].isdigit():
                    p_type = int(parts[1])

                if (p_type in [1, 7, 9, 10, 11, 12, 23, 24, 28]) and (parts[num_idx].isdigit() and parts[den_idx].isdigit()):
                    note = parts[2].lower().strip()
                    if note == 'es':
                        note = 'rest'
                    dur = parts[num_idx] + '/' + parts[den_idx]
                    notes.append(note)
                    durs.append(dur)
            except:
                print(f, i)

    return notes, durs


def extract_all_notes_and_durations(makam, dir_path, dirs):
    """
    Reads all songs from given makam in mu2 and txt formats and creates sorted
    note name and dur files
    """
    mu2_dir = os.path.join(dir_path, dirs[0])
    txt_dir = os.path.join(dir_path, dirs[1])
    mu2_files = [os.path.join(mu2_dir, f) for f in os.listdir(mu2_dir) if os.path.isfile(os.path.join(mu2_dir, f)) and f.startswith(makam + '--')]
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if os.path.isfile(os.path.join(txt_dir, f)) and f.startswith(makam + '--')]

    notes, durs = set([]), set([])
    for f in mu2_files:
        ns, ds = extract_mu2(f)
        for n in ns:
            notes.add(n)
        for d in ds:
            durs.add(d)

    for f in txt_files:
        ns, ds = extract_txt(f)
        for n in ns:
            notes.add(n)
        for d in ds:
            durs.add(d)

    note_dict = NoteDictionary()
    notes_dict = {}
    for note in notes:
        dict_note = note_dict.get_num_by_name(note)
        if not dict_note[3]:
            raise Exception('Irregular note')

        note_num = dict_note[0] * dict_note[1] * dict_note[2]
        if note_num not in notes_dict:
            notes_dict[note_num] = []
        notes_dict[note_num].append(note)

    for k, v in sorted(notes_dict.items()):
        print(k, v)

    durs_dict = {}
    for dur in durs:
        dur_fr = Fraction(dur)
        str_rep = str(dur_fr)
        if Fraction(str_rep) not in durs_dict:
            durs_dict[Fraction(str_rep)] = []
        durs_dict[Fraction(str_rep)].append(dur)

    for k, v in sorted(durs_dict.items()):
        print(k, v)

    with open(makam + '_sorted_note_corpus.txt', 'w') as f:
        for k, v in sorted(notes_dict.items()):
            f.write(','.join(v) + '\n')

    with open(makam + '_sorted_dur_corpus.txt', 'w') as f:
        for k, v in sorted(durs_dict.items()):
            f.write(','.join(v) + '\n')


def build_nd_tuple_corpus(makam, dir_path, dirs):
    """
    Reads all songs in mu2 and txt format for given makam and
    creates sorted note-duration file
    """
    nt = NoteTranslator(makam)
    dt = DurTranslator(makam)

    mu2_dir = os.path.join(dir_path, dirs[0])
    txt_dir = os.path.join(dir_path, dirs[1])
    mu2_files = [os.path.join(mu2_dir, f) for f in os.listdir(mu2_dir) if os.path.isfile(os.path.join(mu2_dir, f)) and f.startswith(makam + '--')]
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if os.path.isfile(os.path.join(txt_dir, f)) and f.startswith(makam + '--')]

    nd = set([])
    for f in mu2_files:
        ns, ds = extract_mu2(f)
        for n, d in zip(ns, ds):
            note_num = nt.get_note_num_by_name(n)
            dur_num = dt.get_dur_num_by_name(d)
            combined = str(note_num) + ':' + str(dur_num)
            nd.add(combined)

    for f in txt_files:
        ns, ds = extract_txt(f)
        for n, d in zip(ns, ds):
            note_num = nt.get_note_num_by_name(n)
            dur_num = dt.get_dur_num_by_name(d)
            combined = str(note_num) + ':' + str(dur_num)
            nd.add(combined)

    s_able = []
    for c in sorted(nd):
        parts = c.split(':')
        s_able.append([int(x) for x in parts])

    with open(makam + '--ndsc.txt', 'w') as f:
        for el in sorted(s_able):
            el_str = ':'.join([str(x) for x in el])
            f.write(el_str + '\n')


def create_training_data(makam, dir_path, dirs):
    nt = NoteTranslator(makam)
    dt = DurTranslator(makam)
    oh_manager = OhManager(makam)

    mu2_dir = os.path.join(dir_path, dirs[0])
    mu2_files = [os.path.join(mu2_dir, f) for f in os.listdir(mu2_dir) if os.path.isfile(os.path.join(mu2_dir, f)) and f.startswith(makam + '--')]

    dir_path = os.path.join(os.path.abspath('..'), 'data', makam, 'oh')

    for i, f in enumerate(mu2_files):
        ns, ds = extract_mu2(f)
        oh_list = []
        for n, d in zip(ns, ds):
            note_num = nt.get_note_num_by_name(n)
            dur_num = dt.get_dur_num_by_name(d)
            nd = str(note_num) + ':' + str(dur_num)
            oh = oh_manager.nd_2_oh(nd).tolist()
            oh_list.append(oh)

        with open(os.path.join(dir_path, 's_' + str(i)), 'w') as fc:
            fc.write(json.dumps(oh_list))


def test_training_file(makam, ver, f_name):
    nt = NoteTranslator(makam)
    dt = DurTranslator(makam)
    oh_manager = OhManager(makam)

    f_path = os.path.join(os.path.abspath('..'), 'data', makam, ver, f_name)
    with open(f_path, 'r') as cf:
        notes = json.load(cf)
        for note in notes:
            nd = oh_manager.oh_2_nd(note)
            parts = nd.split(':')
            note_name = nt.get_note_name_by_num(int(parts[0]))
            note_dur = dt.get_dur_name_by_num(int(parts[1]))
            print(note_name, note_dur)


def create_nc_corpus(makam, ver):
    oh_manager = OhManager(makam)

    dir_path = os.path.join(os.path.abspath('..'), 'data', makam, ver)
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    songs = []
    for f in files:
        song = []
        with open(os.path.join(dir_path, f), 'r') as cf:
            notes = json.load(cf)
            for note in notes:
                nd = oh_manager.oh_2_nd(note)
                song.append(nd)
        songs.append(song)

    with open(makam + '--nc_corpus.txt', 'w') as tf:
        for song in songs:
            line = ' '.join(song)
            tf.write(line + '\n')


def main():
    makam = 'nihavent'
    dirs = ['mu2', 'txt']
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master'
    # first extract all unique notes and durs and create sorted files
    # extract_all_notes_and_durations(makam, dir_path, dirs)

    # build unique sorted note-dur tuple corpus
    # build_nd_tuple_corpus(makam, dir_path, dirs)

    # turn whole SymbTr into training data
    # create_training_data(makam, dir_path, dirs)
    # test_training_file(makam, 'oh', 's_0')
    # create_nc_corpus(makam, 'oh')


if __name__ == '__main__':
    main()
