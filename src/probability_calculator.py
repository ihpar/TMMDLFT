from collections import defaultdict
from nltk import ngrams
import pickle
import numpy as np


def dd_int():
    return defaultdict(int)


class ProbabilityCalculator:
    def __init__(self, makam):
        self.makam_name = makam

    def build_ngrams(self, ng):
        note_dict = defaultdict(dd_int)
        dur_dict = defaultdict(dd_int)
        total = 0
        with open(self.makam_name + '--nc_corpus.txt', 'r') as f:
            songs = f.readlines()
            for song in songs:
                notes = song.split(' ')
                xgrams = ngrams(notes, ng)
                for ngram in xgrams:
                    note_list = [(n.split(':'))[0] for n in ngram]
                    key_notes = ','.join(note_list[:-1])
                    value_note = note_list[-1]
                    note_dict[key_notes][value_note] += 1

                    dur_list = [(n.split(':'))[1] for n in ngram]
                    key_durs = ','.join(dur_list[:-1])
                    value_dur = dur_list[-1]
                    dur_dict[key_durs][value_dur] += 1

                    total += 1

        note_file = self.makam_name + '_notes_' + str(ng) + '_grams'
        with open(note_file, 'wb') as target:
            pickle.dump(note_dict, target)

        dur_file = self.makam_name + '_durs_' + str(ng) + '_grams'
        with open(dur_file, 'wb') as target:
            pickle.dump(dur_dict, target)

    def search_note_seq(self, prev, nex):
        res = defaultdict(dd_int)
        for i in range(len(prev)):
            ss = prev[i:]
            ng = len(ss) + 1
            note_file = self.makam_name + '_notes_' + str(ng) + '_grams'

            with open(note_file, 'rb') as src:
                dd = pickle.load(src)
                key = ','.join([str(e) for e in ss])
                for n in nex:
                    res[n][ng] = dd[key][str(n)]

        return res

    def order(self, sr):
        h = len(sr)
        w = len(list(sr.values())[0])
        mat = [[0 for _ in range(w)] for _ in range(h)]
        res = []
        i = 0
        for k, v in sr.items():
            j = 0
            for ik, iv in v.items():
                mat[i][j] = iv
                j += 1
            i += 1
        print(mat)
        return res


def main():
    pc = ProbabilityCalculator('hicaz')
    # pc.build_ngrams(7)
    sr = pc.search_note_seq([63, 56, 51, 56, 63, 67], [63, 71])
    dummy = {63: {7: 47, 6: 88, 5: 185, 4: 457, 3: 917, 2: 2783},
             71: {7: 0, 6: 0, 5: 2, 4: 2, 3: 8, 2: 93}}
    ordered = pc.order(sr)
    for k in ordered:
        print(k)


if __name__ == '__main__':
    main()
