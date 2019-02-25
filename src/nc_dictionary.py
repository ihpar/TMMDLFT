class NCDictionary:
    def __init__(self):
        self.__src_file = 'sorted_note_corpus.txt'
        self.__dur_src_file = 'sorted_dur_corpus.txt'
        self.__note_dict_1 = {}
        self.__note_dict_2 = {}
        self.__note_dict_rev_1 = {}
        self.__note_dict_rev_2 = {}
        self.__dur_dict = {}
        self.__dur_dict_rev = {}
        self.__build_self()

    def __build_self(self):
        with open(self.__src_file, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
            for i, line in enumerate(lines):
                notes = line.split(',')
                self.__note_dict_1[i + 1] = notes[0]
                self.__note_dict_rev_1[notes[0]] = i + 1
                if len(notes) > 1:
                    self.__note_dict_2[i + 1] = notes[1]
                    self.__note_dict_rev_2[notes[1]] = i + 1

        with open(self.__dur_src_file, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
            for i, line in enumerate(lines):
                self.__dur_dict[i + 1] = line
                self.__dur_dict_rev[line] = i + 1

    def get_note_by_name(self, note_name):
        if note_name in self.__note_dict_rev_1:
            return self.__note_dict_rev_1[note_name]
        if note_name in self.__note_dict_rev_2:
            return self.__note_dict_rev_2[note_name]
        return None

    def get_note_by_num(self, note_num):
        if note_num in self.__note_dict_1:
            return self.__note_dict_1[note_num]
        if note_num in self.__note_dict_2:
            return self.__note_dict_2[note_num]
        return None

    def get_dur_by_num(self, num):
        if num in self.__dur_dict:
            return self.__dur_dict[num]
        return None

    def get_num_by_dur(self, dur):
        if dur in self.__dur_dict_rev:
            return self.__dur_dict_rev[dur]
        return None
