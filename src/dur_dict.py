from fractions import Fraction
import numpy as np


class DurDictionary:
    def __init__(self, makam):
        self.__durs = set()
        self.dur_dict = {}
        self.dur_dict_rev = {}
        self.__build_durs__(makam)

    def __build_durs__(self, makam):
        with open(makam + 'corpus_dur.txt', 'r') as f:
            songs = f.readlines()
            for song in songs:
                notes = song.split(' ')
                for note in notes:
                    parts = note.split(':')
                    note_dur = Fraction(parts[1])
                    self.__durs.add(note_dur)

            for i, dur in enumerate(sorted(self.__durs)):
                binr = [int(x) for x in bin(i + 1)[2:].zfill(6)]
                self.dur_dict[dur] = binr
                self.dur_dict_rev[i + 1] = dur

    def get_dur_by_num(self, num):
        if num not in self.dur_dict_rev:
            return False
        return self.dur_dict_rev[num]

    def __repr__(self):
        return "<DurDictionary di:%s rev_di:%s>" % (self.dur_dict, self.dur_dict_rev)

    def __str__(self):
        res = ''
        for k, v in self.dur_dict.items():
            res += str(k).ljust(5) + '\t'
            res += ' '.join(str(e) for e in v) + '\n'
        return res
