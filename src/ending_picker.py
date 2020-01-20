from mu2_reader import decompose_mu2
import numpy as np
import random


class EndingPicker:
    def __init__(self, makam, songs_arr, dir_path, note_dict, oh_manager, set_size):
        self.songs = []
        self.probabilities = {}  # only in sarki form
        self.br_probabilities = {}  # broad corpus
        self.set_size = set_size

        for curr_song in songs_arr:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            part = song.get_part('B')
