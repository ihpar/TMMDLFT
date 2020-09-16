import os

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


class PCH:
    def __init__(self, makam):
        self.makam = makam
        self.base_d_path = os.path.join(os.path.abspath('..'), 'mu2')
        self.gen_d_path = os.path.join(os.path.abspath('..'), 'songs', makam)
        self.b_songs = added_ss.added_songs[makam]
        self.g_songs = added_ss.added_songs[makam + '_gen']

        self.note_dict, self.nt, self.dt = None, None, None

        self.note_nums, self.dur_nums = set(), set()

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

        octaves = set()
        for n in self.note_nums:
            note_body, note_octave, note_acc, acc_amt = '', '', '', ''
            if self.nt:
                note_name = self.nt.get_note_name_by_num(n)
            else:
                note_name = self.note_dict.get_note_by_num(n)

            if len(note_name) >= 2:
                acc = note_name[-2]
                if (acc == 'b' or acc == '#') and note_name[-1].isnumeric():
                    note_acc = acc
                    acc_amt = note_name[-1]
                    note_name = note_name[:-2]

            if note_name == 'rest':
                note_body = note_name
            else:
                note_octave = note_name[-1]
                note_body = note_name[:-1]

            # print(note_body, note_octave, note_acc, acc_amt)
            octaves.add(note_octave)
            oi_note = note_body + note_acc + acc_amt
            print(oi_note)

        print(octaves)


def main():
    pch = PCH('nihavent')


if __name__ == '__main__':
    main()
