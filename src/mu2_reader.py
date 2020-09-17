import codecs
from os import path
from fractions import Fraction

from nc_dictionary import NCDictionary
from oh_manager import OhManager
from songobj import SongObj
import hicaz_parts


# noinspection DuplicatedCode
def decompose_mu2(dp, fn, part_map, song_final, note_dict, oh_manager, nt=None, dt=None):
    nom_index = 2
    den_index = 3
    beat, tot = None, Fraction(0)
    measure_no = 0
    song_path = path.join(dp, fn)
    song = SongObj(fn, part_map, song_final)
    i = 0
    with codecs.open(song_path, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for line in lines:
            i += 1
            parts = line.split('\t')
            parts = [s.strip() for s in parts]
            parts[0] = int(parts[0])

            if parts[0] == 51:
                beat = Fraction(int(parts[nom_index]), int(parts[den_index]))
                song.set_time_sign(beat)

            elif parts[0] == 52:
                song.set_tempo(int(parts[4]))

            elif (parts[0] in [1, 4, 7, 9, 10, 11, 12, 23, 24, 28]) and (
                    parts[nom_index].isdigit() and parts[den_index].isdigit()):
                # note name
                note_name = parts[1].lower().strip()
                if note_name == '':
                    note_name = 'rest'

                if nt:
                    note_num = nt.get_note_num_by_name(note_name)
                else:
                    note_num = note_dict.get_note_by_name(note_name)

                # note dur
                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                dur = str(note_len)
                dur_alt = parts[nom_index] + '/' + parts[den_index]

                if dt:
                    dur = dt.get_dur_num_by_name(dur_alt)
                else:
                    dur = note_dict.get_num_by_dur(dur)

                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)
                # repr
                try:
                    combine = oh_manager.nd_2_int(str(note_num) + ':' + str(dur))
                    song.insert_note(combine, measure_no)
                except KeyError as e:
                    print(f'Ex in {fn}: {i}, {note_name},{str(note_len)}')
                    print(str(e))
                    raise Exception('No correspondence')
                tot += note_len
                if tot == beat:
                    # print(i, line)
                    measure_no += 1
                    tot = Fraction(0)
                if tot > beat:
                    print(i, line)
                    print(song_path)
                    raise Exception('--Broken--')

    return song


def main():
    makam = 'hicaz'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)

    for curr_song in hicaz_parts.hicaz_songs:
        song = curr_song['file']
        part_map = curr_song['parts_map']
        song_final = curr_song['sf']
        decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)


if __name__ == '__main__':
    main()
