import numpy as np


class OhManager:
    def __init__(self, makam):
        self.__nd_2_num = {}
        self.__num_2_nd = {}
        self.__total = 0
        with open(makam + '--ndsc.txt', 'r') as f:
            nds = f.read().splitlines()
            for i, nd in enumerate(nds):
                self.__nd_2_num[nd] = i
                self.__num_2_nd[i] = nd
                self.__total += 1

    def nd_2_int(self, nd):
        return self.__nd_2_num[nd]

    def int_2_nd(self, num):
        return self.__num_2_nd[num]

    def nd_2_oh(self, nd):
        num = self.nd_2_int(nd)
        res = np.zeros(self.__total)
        res[num] = 1
        return res

    def oh_2_nd(self, oh):
        num = np.argmax(oh)
        return self.int_2_nd(num)

    def oh_2_zo(self, oh):
        num = np.argmax(oh)
        return num / self.__total

    def int_2_zo(self, num):
        return num / self.__total

    def int_2_oh(self, num):
        res = np.zeros(self.__total)
        res[num] = 1
        return res
