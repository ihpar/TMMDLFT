import codecs
import os
import pandas as pd
import numpy as np
import math
from fractions import Fraction
from PIL import Image

from nc_dictionary import NCDictionary
from oh_manager import OhManager


def make_pic_from_mu2(fp, note_dict, oh_manager, edge):
    lp_index = 8
    rp_index = 9
    nom_index = 3
    den_index = 4

    notes = []
    n_s, n_e = 0, 0

    doc = codecs.open(os.path.abspath(fp), 'rU', 'UTF-8')
    df = pd.read_csv(doc, sep='\t')

    i, maxi = -1, 0
    for row in df.itertuples():
        row_key = row[1]
        if row_key == 9:
            if row[lp_index] == '[':
                n_s = i + 1
            if row[rp_index] == ']':
                n_e = i

            if not (np.isnan(row[nom_index]) or np.isnan(row[den_index])):
                i += 1
                note = 'rest' if pd.isnull(row[2]) else row[2].lower().strip()
                dur = str(Fraction(int(row[nom_index]), int(row[den_index])))
                dur_alt = str(int(row[nom_index])) + '/' + str(int(row[den_index]))
                note_num = note_dict.get_note_by_name(note)
                dur = note_dict.get_num_by_dur(dur)
                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)

                combine = oh_manager.nd_2_int(str(note_num) + ':' + str(dur))
                maxi = max(combine, maxi)
                notes.append(combine)

    in_edge = math.ceil(math.sqrt(i + 1))
    notes = np.pad(notes, (0, (in_edge * in_edge) - len(notes)), 'constant').reshape(in_edge, in_edge)
    offset = math.floor((edge - in_edge) / 2)
    top_bot = (offset, edge - in_edge - offset)
    left_right = (offset, edge - in_edge - offset)
    notes = np.pad(notes, (top_bot, left_right), 'constant').reshape(edge, edge)
    t_s, l_s = divmod(n_s, in_edge)
    t_e, l_e = divmod(n_e, in_edge)
    s_r, s_c = (top_bot[0] + t_s), (left_right[0] + l_s)  # start row, col
    e_r, e_c = (top_bot[0] + t_e), (left_right[0] + l_e)  # end row, col
    # print('s:', notes[s_r, s_c], 'e:', notes[e_r, e_c])
    frame = np.array([s_r, left_right[0], in_edge, (t_e - t_s)])  # (top, left, width, height)

    # if n_s == 0 or n_e == 0:
    if n_e == 0:
        frame = None

    return notes, frame, maxi


def count_len(fp):
    nom_index = 3
    den_index = 4
    doc = codecs.open(os.path.abspath(fp), 'rU', 'UTF-8')
    df = pd.read_csv(doc, sep='\t')

    tot = 0
    for row in df.itertuples():
        row_key = row[1]
        if row_key == 9:
            if not (np.isnan(row[nom_index]) or np.isnan(row[den_index])):
                tot += 1
    return tot


def build_corpus(makam):
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    dp = os.path.join(os.path.normpath(os.path.expanduser('~/Desktop')), 'SymbTr-master', 'mu2')
    files = [os.path.join(dp, f) for f in os.listdir(dp) if
             os.path.isfile(os.path.join(dp, f)) and f.startswith(makam + '--')]
    max_len = 0
    for fp in files:
        length = count_len(fp)
        if length > max_len:
            max_len = length

    edge = math.ceil(math.sqrt(max_len))

    maxi_norm = 0
    corpus = []
    frames = []
    for fp in files:
        notes, frame, maxi = make_pic_from_mu2(fp, note_dict, oh_manager, edge)
        if frame is None:
            continue
        corpus.append(notes)
        frames.append(frame)
        maxi_norm = max(maxi, maxi_norm)

    for i, n in enumerate(corpus):
        # corpus[i] = n / maxi_norm
        corpus[i] = (n / maxi_norm) * 255  # map to 0-255 interval for images

    i = 0
    r_dir = os.path.join(os.path.abspath(__file__ + "/../../"), 'data', makam, 'pic')
    for c, f in zip(corpus, frames):
        fr_file = os.path.join(r_dir, 'f_' + str(i) + '.npy')
        np.save(fr_file, f)
        im = Image.fromarray(c)
        im = im.convert('L')
        im.save(os.path.join(r_dir, 's_' + str(i) + '.png'))
        i += 1


def place_nakarat(makam, edge, coords, song):
    song_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'songs', makam, song + '.mu2')
    length = count_len(song_path)
    in_edge = math.ceil(math.sqrt(length))
    offset = math.floor((edge - in_edge) / 2)
    c_top, c_left, c_width, c_height = coords[0], coords[2], coords[2], coords[3]
    start_idx = (c_top - offset) * in_edge + c_left
    end_idx = (c_top + c_height - offset) * in_edge + c_width
    print(start_idx, end_idx)

    rows_list = []
    nom_index = 3
    den_index = 4
    i = 0
    with codecs.open(song_path, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            if int(parts[0]) == 9 and (parts[nom_index].isdigit() and parts[den_index].isdigit()):
                if i == start_idx:
                    rows_list.append('9							[		0.5')
                if i == end_idx:
                    rows_list.append('9								]	0.5')
                i += 1
            rows_list.append(line)

    song_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'songs', makam, 'n_' + song + '.mu2')
    with codecs.open(song_path, 'w', encoding='utf8') as sf:
        for line in rows_list:
            sf.write(line + '\n')


def main():
    makam = 'hicaz'
    build_corpus(makam)
    edge = 26  # hicaz

    '''
    # region for new guesses
    maxi = 404
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    song_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'songs', makam, 'lstm_v61_88.mu2')
    notes, _, _ = make_pic_from_mu2(song_path, note_dict, oh_manager, edge)
    notes = (notes / maxi) * 255
    im = Image.fromarray(notes)
    im = im.convert('L')
    im.save(os.path.join(os.path.abspath(__file__ + "/../../"), 'songs', makam, 'lstm_v61_88.png'))
    
    # [10  6 14  4] <-> [top left width height] : lstm_v61_66.png
    # [ 6  7 12 11] : last train lstm_v61_66.png
    coords = np.array([10, 6, 14, 4])
    place_nakarat(makam, edge, coords, 'lstm_v61_66')
    '''


if __name__ == '__main__':
    main()
