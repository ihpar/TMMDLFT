from mu2_reader import decompose_mu2
import numpy as np
import random


class CandidatePicker:
    def __init__(self, makam, songs_arr, parts, dir_path, note_dict, oh_manager, set_size):
        self.songs = []
        self.probabilities = {}  # only in sarki form
        self.br_probabilities = {}  # broad corpus
        self.set_size = set_size

        for curr_song in songs_arr:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            self.songs.append(song)

        total = 0
        for part_id in parts:
            for song in self.songs:
                part = song.get_part(part_id)
                for i in range(len(part) - int(set_size / 2)):
                    sec = self.str_rep(part[i:i + int(set_size / 2) + 1])
                    if sec in self.probabilities:
                        self.probabilities[sec] += 1
                    else:
                        self.probabilities[sec] = 1
                    total += 1

        for k in self.probabilities:
            self.probabilities[k] = self.probabilities[k] / total

        total = 0
        with open(makam + '--nc_corpus.txt', 'r') as crp:
            songs = crp.read().splitlines()
            for song in songs:
                notes = song.split(' ')
                for i in range(len(notes) - int(set_size / 2)):
                    sec = self.str_rep([oh_manager.nd_2_int(note) for note in notes[i:i + int(set_size / 2) + 1]])
                    if sec in self.br_probabilities:
                        self.br_probabilities[sec] += 1
                    else:
                        self.br_probabilities[sec] = 1
                    total += 1
        for k in self.br_probabilities:
            self.br_probabilities[k] = self.br_probabilities[k] / total

    def pick_candidate(self, prev_notes, candidates, n_probs):
        probs = []
        for i, cand in enumerate(candidates):
            prob = self.calc_prob(prev_notes, cand, n_probs[i])
            probs.append(prob)
        max_prob = max(probs)
        res_set = []
        for i, prob in enumerate(probs):
            if prob == max_prob:
                res_set.append(candidates[i])
        return random.choice(res_set)

    def calc_prob(self, prev_notes, candidate, n_prob):
        # prev_notes = [[0,0,1,...,0], [0,0,0,...,0], ..., [0,1,0,...,0]]
        # candidate = 135
        prev_notes_nums = []
        for i, n in enumerate(prev_notes[0]):
            if i < int(self.set_size / 2):
                continue
            prev_notes_nums.append(np.argmax(n))
        prev_notes_nums.append(candidate)
        s_rep = self.str_rep(prev_notes_nums)
        if s_rep in self.probabilities:
            return 2 + self.probabilities[s_rep]
        if s_rep in self.br_probabilities:
            return 1 + self.br_probabilities[s_rep]
        return 0

    def str_rep(self, arr):
        return '-'.join([str(elem) for elem in arr])
