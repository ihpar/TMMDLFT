from collections import defaultdict
from nltk import ngrams
import pickle
import numpy as np


def dd_int():
    return defaultdict(int)


class ProbabilityCalculator:
    def __init__(self, makam, set_size):
        self.makam_name = makam
        self.set_size = set_size
        self.note_ngram_dicts = []
        self.dur_ngram_dicts = []
        for i in range(2, self.set_size + 2):
            note_file = self.makam_name + '_notes_' + str(i) + '_grams'
            with open(note_file, 'rb') as src:
                dd = pickle.load(src)
                self.note_ngram_dicts.append(dd)

            dur_file = self.makam_name + '_durs_' + str(i) + '_grams'
            with open(dur_file, 'rb') as src:
                dd = pickle.load(src)
                self.dur_ngram_dicts.append(dd)

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
            key = ','.join([str(e) for e in ss])
            for n in nex:
                res[n][ng] = self.note_ngram_dicts[-1 * (i + 1)][key][str(n)]

        return res

    def search_dur_seq(self, prev, nex):
        res = defaultdict(dd_int)
        for i in range(len(prev)):
            ss = prev[i:]
            ng = len(ss) + 1
            key = ','.join([str(e) for e in ss])
            for n in nex:
                res[n][ng] = self.dur_ngram_dicts[-1 * (i + 1)][key][str(n)]

        return res

    def order(self, sr):
        h = len(sr)
        w = len(list(sr.values())[0])
        mat = np.array([[0 for _ in range(w)] for _ in range(h)])
        res = []
        done = []
        keys = []
        i = 0
        for k, v in sr.items():
            keys.append(k)
            j = 0
            for ik, iv in v.items():
                mat[i][j] = iv
                j += 1
            i += 1

        for j in range(w):
            s_col = {}
            col = mat[:, j]
            for i, e in enumerate(col):
                s_col[i] = e
            for r in sorted(s_col, key=s_col.get, reverse=True):
                if s_col[r] > 0 and r not in done:
                    res.append(r)
                    done.append(r)
            if len(done) == h:
                break

        res_keys = []
        for i in res:
            res_keys.append(keys[i])

        for i in keys:
            if i not in res_keys:
                res_keys.append(i)

        return res_keys

    def sort_note_probabilities(self, prev, nex):
        sr = self.search_note_seq(prev, nex)
        ordered = self.order(sr)
        return ordered

    def sort_dur_probabilities(self, prev, nex):
        sr = self.search_dur_seq(prev, nex)
        ordered = self.order(sr)
        return ordered


def main():
    pc = ProbabilityCalculator('hicaz', 6)
    # print(pc.sort_note_probabilities([63, 56, 51, 56, 63, 67], [63, 71, 76]))
    # print(pc.sort_dur_probabilities([12, 12, 12, 12, 12, 12], [11, 21, 4, 6]))
    # pc.build_ngrams(9)


if __name__ == '__main__':
    main()
