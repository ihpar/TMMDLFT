from mu2_reader import decompose_mu2


class CandidatePicker:
    def __init__(self, makam, songs_arr, parts, dir_path, note_dict, oh_manager, set_size):
        self.songs = []
        self.probabilities = {}
        self.br_probabilities = {}

        for curr_song in songs_arr:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            self.songs.append(song)

    def pick_candidate(self, prev_notes, candidates):
        probs = []
        for cand in candidates:
            prob = self.calc_prob(prev_notes, cand)
            probs.append(prob)
        max_idx = probs.index(max(probs))
        return candidates[max_idx]

    def calc_prob(self, prev_notes, candidate):
        # prev_notes = [[0,0,1,...,0], [0,0,0,...,0], ..., [0,1,0,...,0]]
        # candidate = 135
        prev_notes_nums = []
        for n in prev_notes:
            prev_notes_nums.append(n.index(1))
        prev_notes_nums.append(candidate)
        str_rep = self.str_rep(prev_notes_nums)
        if str_rep in self.probabilities:
            return self.probabilities[str_rep]
        if str_rep in self.br_probabilities:
            return self.br_probabilities[str_rep]
        return 0

    def str_rep(self, arr):
        return '-'.join([str(elem) for elem in arr])
