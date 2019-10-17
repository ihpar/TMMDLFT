import codecs
from os import path
from fractions import Fraction
from songobj import SongObj


def decompose_mu2(dp, fn, part_map):
    nom_index = 2
    den_index = 3
    beat, tot = None, Fraction(0)
    measure_no = 0
    I, A, B, C = part_map['I'], part_map['A'], part_map['B'], part_map['C']
    song_path = path.join(dp, fn)
    song = SongObj(fn)
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

            elif parts[0] == 9 and (parts[nom_index].isdigit() and parts[den_index].isdigit()):
                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                tot += note_len
                if tot == beat:
                    measure_no += 1
                    tot = Fraction(0)
                    print(parts)


def main():
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    song = 'hicaz--sarki--agiraksak--bak_ne--haci_arif_bey.mu2'
    part_map = {'I': [0, 1, 2, [3, 4]], 'A': [5, 6, 7, 8], 'B': [9, 10, 11, 12], 'C': [13, 14, 15, 16]}
    decompose_mu2(dir_path, song, part_map)


if __name__ == '__main__':
    main()
