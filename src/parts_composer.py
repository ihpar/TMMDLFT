from mu2_reader import *


def compose_beginning():
    pass


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
