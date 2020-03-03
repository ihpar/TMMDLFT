import os
import codecs
from fractions import Fraction
import pandas as pd


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

    print(notes)
    print(durs)


def main():
    makam = 'nihavent'
    dirs = ['mu2', 'txt']
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master'
    extract_all_notes_and_durations(makam, dir_path, dirs)


if __name__ == '__main__':
    main()
