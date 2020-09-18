import os
import numpy as np
import added_ss
from dur_translator import DurTranslator
from nc_dictionary import NCDictionary
from note_translator import NoteTranslator


class ST:
    base = 1
    generated = 2


class ND:
    note = 1
    dur = 2


def parse_note(note_str):
    note_body, note_octave, note_acc, acc_amt = '', '', '', ''

    if len(note_str) >= 2:
        acc = note_str[-2]
        if (acc == 'b' or acc == '#') and note_str[-1].isnumeric():
            note_acc = acc
            acc_amt = note_str[-1]
            note_str = note_str[:-2]

    if note_str == 'rest':
        note_body = note_str
    else:
        note_octave = note_str[-1]
        note_body = note_str[:-1]

    return note_body, note_octave, note_acc, acc_amt


class PCH:
    def __init__(self, makam):
        self.makam = makam
        self.base_d_path = os.path.join(os.path.abspath('..'), 'mu2')
        self.gen_d_path = os.path.join(os.path.abspath('..'), 'songs', makam)
        self.b_songs = added_ss.added_songs[makam]
        self.g_songs = added_ss.added_songs[makam + '_gen']

        self.note_dict, self.nt, self.dt = None, None, None

        self.note_nums, self.dur_nums = set(), set()
        self.octaves = ['3', '4', '5', '6']
        self.note_collection, self.dur_collection = [], []
        self.note_histogram = {}
        self.dur_histogram = {}

        self.pctm, self.nltm = None, None

        if makam == 'hicaz':
            self.note_dict = NCDictionary()
        else:
            self.nt = NoteTranslator(makam)
            self.dt = DurTranslator(makam)

        self.build_self()

    def build_self(self):
        from analytics_fin import parse_song_in_mu2
        for bs in self.b_songs:
            song = parse_song_in_mu2(os.path.join(self.base_d_path, bs['file']), self.note_dict, self.nt, self.dt)
            for note in song:
                parts = [int(x) for x in note.split(':')]
                self.note_nums.add(parts[0])
                self.dur_nums.add(parts[1])

        for gs in self.g_songs:
            song = parse_song_in_mu2(os.path.join(self.gen_d_path, gs['file']), self.note_dict, self.nt, self.dt)
            for note in song:
                parts = [int(x) for x in note.split(':')]
                self.note_nums.add(parts[0])
                self.dur_nums.add(parts[1])

        oi_names = set()
        for n in self.note_nums:
            if self.nt:
                note_name = self.nt.get_note_name_by_num(n)
            else:
                note_name = self.note_dict.get_note_by_num(n)

            note_body, note_octave, note_acc, acc_amt = parse_note(note_name)
            oi_name = note_body + note_acc + acc_amt
            for oc in self.octaves:
                n_name = note_body + oc + note_acc + acc_amt
                if self.nt:
                    note_name = self.nt.get_note_num_by_name(n_name)
                else:
                    note_name = self.note_dict.get_note_by_name(n_name)

                if note_name and (oi_name not in oi_names):
                    self.note_collection.append(n)
                    oi_names.add(oi_name)
                    break
        self.note_collection.sort()

        for d in self.dur_nums:
            self.dur_collection.append(d)
        self.dur_collection.sort()

    def translate_note_num(self, note_num):
        if self.nt:
            note_name = self.nt.get_note_name_by_num(note_num)
        else:
            note_name = self.note_dict.get_note_by_num(note_num)
        note_body, note_octave, note_acc, acc_amt = parse_note(note_name)

        for oc in self.octaves:
            n_name = note_body + oc + note_acc + acc_amt
            if self.nt:
                nn = self.nt.get_note_num_by_name(n_name)
            else:
                nn = self.note_dict.get_note_by_name(n_name)

            if nn in self.note_collection:
                return nn

        return -1

    def add_note(self, note_num):
        nn = self.translate_note_num(note_num)
        if nn == -1:
            return

        if nn in self.note_histogram:
            self.note_histogram[nn] += 1
        else:
            self.note_histogram[nn] = 1

    def get_m_idx(self, note_num):
        nn = self.translate_note_num(note_num)
        if nn == -1:
            return -1
        idx = 0
        for nc in self.note_collection:
            if nc == nn:
                return idx
            idx += 1
        return idx

    def get_dm_idx(self, dur_num):
        idx = 0
        for dc in self.dur_collection:
            if dc == dur_num:
                return idx
            idx += 1
        return -1

    def add_tuple(self, n1, n2):
        ix1 = self.get_m_idx(n1)
        ix2 = self.get_m_idx(n2)

        if ix1 == -1 or ix2 == -1:
            return

        self.pctm[ix1][ix2] += 1

    def add_dur_tuple(self, d1, d2):
        ix1 = self.get_dm_idx(d1)
        ix2 = self.get_dm_idx(d2)
        if ix1 == -1 or ix2 == -1:
            return
        self.nltm[ix1][ix2] += 1

    def add_dur(self, dur_num):
        if dur_num in self.dur_histogram:
            self.dur_histogram[dur_num] += 1
        else:
            self.dur_histogram[dur_num] = 1

    def get_note_histogram(self):
        res = np.zeros(len(self.note_collection))
        for i, n in enumerate(self.note_collection):
            if n in self.note_histogram:
                res[i] = self.note_histogram[n]
            else:
                res[i] = 0
        res = res / sum(res)
        return res

    def get_dur_histogram(self):
        res = np.zeros(len(self.dur_collection))
        for i, d in enumerate(self.dur_collection):
            if d in self.dur_histogram:
                res[i] = self.dur_histogram[d]
            else:
                res[i] = 0
        res = res / sum(res)
        return res

    def init_note_histogram(self):
        self.note_histogram = {}

    def init_dur_histogram(self):
        self.dur_histogram = {}

    def init_note_transition_matrix(self):
        bc = self.get_note_bin_count()
        self.pctm = np.zeros((bc, bc))

    def init_dur_transition_matrix(self):
        bc = self.get_dur_bin_count()
        self.nltm = np.zeros((bc, bc))

    def get_note_bin_count(self):
        return len(self.note_collection)

    def get_dur_bin_count(self):
        return len(self.dur_collection)

    def get_note_transition_matrix(self):
        trans_mat = np.copy(self.pctm)
        trans_mat = trans_mat / sum(sum(trans_mat))
        return trans_mat

    def get_dur_transition_matrix(self):
        trans_mat = np.copy(self.nltm)
        trans_mat = trans_mat / sum(sum(trans_mat))
        return trans_mat
