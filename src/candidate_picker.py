from mu2_reader import decompose_mu2
import numpy as np


class CandidatePicker:
    def __init__(self, makam, songs_arr, parts, dir_path, note_dict, oh_manager, set_size):
        self.songs = []
        self.probabilities = {}  # only in sarki form
        self.br_probabilities = {}  # broad corpus

        for curr_song in songs_arr:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            self.songs.append(song)
        part_ids = ['I', 'A', 'B', 'C']
        total = 0
        for part_id in part_ids:
            for song in self.songs:
                part = song.get_part(part_id)
                for i in range(len(part) - set_size):
                    sec = self.str_rep(part[i:i + set_size + 1])
                    if sec in self.probabilities:
                        self.probabilities[sec] += 1
                    else:
                        self.probabilities[sec] = 1
                    total += 1

        for k in self.probabilities:
            self.probabilities[k] = self.probabilities[k] / total

        total = 0
        with open(makam + '--nc_corpus', 'r') as crp:
            songs = crp.read().splitlines()
            for song in songs:
                notes = song.split(' ')
                for i in range(len(notes) - set_size):
                    sec = self.str_rep([oh_manager.nd_2_int(note) for note in notes[i:i + set_size + 1]])
                    if sec in self.br_probabilities:
                        self.br_probabilities[sec] += 1
                    else:
                        self.br_probabilities[sec] = 1
                    total += 1
        for k in self.br_probabilities:
            self.br_probabilities[k] = self.br_probabilities[k] / total

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
        for n in prev_notes[0]:
            prev_notes_nums.append(np.argmax(n))
        prev_notes_nums.append(candidate)
        s_rep = self.str_rep(prev_notes_nums)
        if s_rep in self.probabilities:
            return 1 + self.probabilities[s_rep]
        if s_rep in self.br_probabilities:
            return self.br_probabilities[s_rep]
        return 0

    def str_rep(self, arr):
        return '-'.join([str(elem) for elem in arr])
