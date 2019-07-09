from oh_manager import OhManager
import random


class FltManager:
    def __init__(self, makam):
        self.__flt_2_nd = {}
        self.__nd_2flt = {}
        self.__div_factor = 1
        self.__uniques = set()
        song_list = []
        self.__oh_mgr = OhManager(makam)

        with open(makam + '--nc_corpus.txt', 'r') as crp:
            songs = crp.read().splitlines()
            for song in songs:
                song_data = []
                notes = song.split(' ')
                for note in notes:
                    int_ver = self.__oh_mgr.nd_2_int(note)
                    self.__uniques.add(int_ver)
                    song_data.append(int_ver)

                song_list.append(song_data)

        self.__div_factor = max(self.__uniques)
        self.__uniques = [n / self.__div_factor for n in self.__uniques]

    def round_num(self, num):
        if num in self.__uniques:
            return num

        closest = 0
        min_diff = 1
        for f in self.__uniques:
            if abs(f - num) < min_diff:
                min_diff = abs(f - num)
                closest = f

        closest_2 = 0
        min_diff = 1
        for f in self.__uniques:
            if abs(f - num) < min_diff:
                min_diff = abs(f - num)
                if f != closest:
                    closest_2 = f

        if abs(closest - closest_2) < 0.05 and random.randint(0, 2) == 1:
            closest = closest_2

        return closest

    def flt_2_nd(self, flt):
        ix = int(flt * self.__div_factor)
        return self.__oh_mgr.int_2_nd(ix)
