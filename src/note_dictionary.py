class NoteDictionary:
    def __init__(self):
        self.__alphabet = ['do', 're', 'mi', 'fa', 'sol', 'la', 'si']
        self.__flat_alphabet = ['re', 'mi', 'fa', 'sol', 'la', 'si', 'do']
        self.__intervals = [9, 9, 4, 9, 9, 9, 4]
        self.__koma_alp = [
            ['', 'b9'],
            ['#1', 'b8'],
            ['#2', 'b7'],
            ['#3', 'b6'],
            ['#4', 'b5'],
            ['#5', 'b4'],
            ['#6', 'b3'],
            ['#7', 'b2'],
            ['#8', 'b1'],
            ['#9', '']
        ]
        self.__sharp_koma_alp = self.__koma_alp[1:]
        self.__flat_koma_alp = self.__koma_alp[:-1]
        self.__sharp_dictionary = {}
        self.__flat_dictionary = {}
        self.__natural_dictionary = {}

        self.__r_sharp_dictionary = {}
        self.__r_flat_dictionary = {}
        self.__r_natural_dictionary = {}

        self.__build_self()

    def __build_self(self):
        idx = 2
        idy = 1
        idn = 1
        self.__sharp_dictionary[0] = self.Note('rest', '', '')
        self.__sharp_dictionary[1] = self.Note('do', 0, '')

        self.__flat_dictionary[0] = self.Note('rest', '', '')
        self.__natural_dictionary[0] = self.Note('rest', '', '')

        for octave_no in range(0, 9):
            # sharps
            for note_no in range(0, len(self.__alphabet)):
                k_idx = 0

                for koma_no in range(0, len(self.__sharp_koma_alp)):
                    if k_idx == self.__intervals[note_no]:
                        break
                    root = self.__alphabet[note_no]
                    self.__sharp_dictionary[idx] = self.Note(root, octave_no, self.__sharp_koma_alp[koma_no][0])
                    idx += 1
                    k_idx += 1

            # flats
            for note_no in range(0, len(self.__flat_alphabet)):
                for koma_no in reversed(range(0, self.__intervals[note_no])):
                    root = self.__alphabet[(note_no + 1) % len(self.__alphabet)]
                    octave = octave_no
                    if (note_no + 1) == len(self.__alphabet):
                        octave += 1
                    self.__flat_dictionary[idy] = self.Note(root, octave, 'b' + str(koma_no + 1))
                    idy += 1

            # naturals
            for note_no in range(0, len(self.__alphabet)):
                self.__natural_dictionary[idn] = self.Note(self.__alphabet[note_no], octave_no, '')
                idn += self.__intervals[note_no]

        self.__flat_dictionary[idy] = self.Note('re', 9, 'b9')

        # build reverse dicts
        for k, v in self.__sharp_dictionary.items():
            if k in self.__natural_dictionary:
                self.__r_natural_dictionary[self.__natural_dictionary[k].name] = k

            self.__r_sharp_dictionary[v.name] = k
            self.__r_flat_dictionary[self.__flat_dictionary[k].name] = k

    def print_self(self):
        for k, v in self.__sharp_dictionary.items():
            if k in self.__natural_dictionary:
                print(str(k) + '\t' + v.name + '\t' + self.__flat_dictionary[k].name + '\t' +
                      self.__natural_dictionary[k].name)
            else:
                print(str(k) + '\t' + v.name + '\t' + self.__flat_dictionary[k].name)

    def get_num_by_name(self, note_name):
        ret = [-1, -1, -1, False]
        if note_name in self.__r_flat_dictionary:
            ret[0] = self.__r_flat_dictionary[note_name]
            ret[3] = True
        if note_name in self.__r_sharp_dictionary:
            ret[1] = self.__r_sharp_dictionary[note_name]
            ret[3] = True
        if note_name in self.__r_natural_dictionary:
            ret[2] = self.__r_natural_dictionary[note_name]
            ret[3] = True

        return ret

    def get_note_by_num(self, num):
        ret = [False, False, False, False]
        if num in self.__flat_dictionary:
            ret[0] = self.__flat_dictionary[num]
            ret[3] = True
        if num in self.__sharp_dictionary:
            ret[1] = self.__sharp_dictionary[num]
            ret[3] = True
        if num in self.__natural_dictionary:
            ret[2] = self.__natural_dictionary[num]
            ret[3] = True

        return ret

    class Note:
        def __init__(self, root, octave, ariza):
            self.root_note = root
            self.octave = octave
            self.ariza = ariza
            self.name = root + str(octave) + ariza
