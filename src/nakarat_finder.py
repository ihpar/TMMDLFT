import codecs
import os
import pandas as pd
import numpy as np
import math

from nc_dictionary import NCDictionary
from oh_manager import OhManager


def make_pic_from_mu2(fp, note_dict, oh_manager, edge):
    print(fp)
    lp_index = 8
    rp_index = 9
    nom_index = 3
    den_index = 4
    lns_index = 5

    notes = []
    n_s, n_e = 0, 0

    doc = codecs.open(os.path.abspath(fp), 'rU', 'UTF-8')
    df = pd.read_csv(doc, sep='\t')

    for row in df.itertuples():
        row_key = row[1]
        if row_key == 9:
            if not (np.isnan(row[nom_index]) or np.isnan(row[den_index])):
                note = 'rest' if pd.isnull(row[2]) else row[2].lower().strip()
                dur = str(int(row[nom_index])) + '/' + str(int(row[den_index]))
                note = note_dict.get_note_by_name(note)
                dur = note_dict.get_num_by_dur(dur)
                combine = oh_manager.nd_2_int(str(note) + ':' + str(dur))
                notes.append(combine)


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


def main():
    makam = 'hicaz'
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

    for fp in files:
        make_pic_from_mu2(fp, note_dict, oh_manager, edge)
        break


if __name__ == '__main__':
    main()
