from mu2_reader import decompose_mu2


class CandidatePicker:
    def __init__(self, makam, songs_arr, parts, dir_path, note_dict, oh_manager, set_size):
        self.songs = []

        for curr_song in songs_arr:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            self.songs.append(song)

    def pick_candidate(self, prev_notes, candidates):
        pass
