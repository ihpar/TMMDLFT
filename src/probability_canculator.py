from collections import defaultdict
from nltk import ngrams
import pickle


def dd_int():
    return defaultdict(int)


class ProbabilityCalculator:
    def __init__(self):
        pass

    def build_ngrams(self, makam, src, ng):
        note_dict = defaultdict(dd_int)
        dur_dict = defaultdict(dd_int)
        total = 0
        with open(src, 'r') as f:
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

        note_file = makam + '_notes_' + str(ng) + '_grams'
        with open(note_file, 'wb') as target:
            pickle.dump(note_dict, target)

        dur_file = makam + '_durs_' + str(ng) + '_grams'
        with open(dur_file, 'wb') as target:
            pickle.dump(dur_dict, target)


def main():
    pc = ProbabilityCalculator()
    pc.build_ngrams('hicaz', 'hicaz--nc_corpus.txt', 7)


if __name__ == '__main__':
    main()
