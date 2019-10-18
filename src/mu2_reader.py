import codecs
from os import path
from fractions import Fraction

from nc_dictionary import NCDictionary
from oh_manager import OhManager
from songobj import SongObj
from hicaz_parts import hicaz_songs


def decompose_mu2(dp, fn, part_map, note_dict, oh_manager):
    nom_index = 2
    den_index = 3
    beat, tot = None, Fraction(0)
    measure_no = 0
    song_path = path.join(dp, fn)
    song = SongObj(fn, part_map)

    with codecs.open(song_path, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            parts[0] = int(parts[0])

            if parts[0] == 51:
                beat = Fraction(int(parts[nom_index]), int(parts[den_index]))
                song.set_time_sign(beat)

            elif parts[0] == 52:
                song.set_tempo(int(parts[4]))

            elif (parts[0] == 9 or parts[0] == 24) and (parts[nom_index].isdigit() and parts[den_index].isdigit()):
                # note name
                note_name = parts[1].lower().strip()
                if note_name == '':
                    note_name = 'rest'
                note_num = note_dict.get_note_by_name(note_name)
                # note dur
                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                dur = str(note_len)
                dur_alt = parts[nom_index] + '/' + parts[den_index]
                dur = note_dict.get_num_by_dur(dur)
                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)
                # repr
                combine = oh_manager.nd_2_int(str(note_num) + ':' + str(dur))
                song.insert_note(combine, measure_no)

                tot += note_len
                if tot == beat:
                    measure_no += 1
                    tot = Fraction(0)
                if tot > beat:
                    print('Broken')
                    break

    print(song)


def main():
    makam = 'hicaz'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    curr_song = hicaz_songs[1]
    song = curr_song['file']
    part_map = curr_song['parts_map']
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    decompose_mu2(dir_path, song, part_map, note_dict, oh_manager)


if __name__ == '__main__':
    main()
