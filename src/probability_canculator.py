import os
from collections import defaultdict


class ProbabilityCalculator:
    def __init__(self):
        pass

    def build_ngrams(self, makam, ng):
        odic = defaultdict(int)
        total = 0.0
        with open(makam + 'corpus.txt', 'r') as f:
            songs = f.readlines()
            for song in songs:
                notes = song.split(' ')
                for note in notes:
                    parts = note.split(':')
                    note_key = int(parts[0])
                    note_dur = int(parts[1])
                    odic[note_key] += 1
                    total += 1

        for e in sorted(odic, key=odic.get, reverse=True):
            print(e, odic[e] / total)


def main():
    pc = ProbabilityCalculator()
    pc.build_ngrams('hicaz--nc_', 6)


if __name__ == '__main__':
    main()
