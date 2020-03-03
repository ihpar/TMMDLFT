class DurTranslator:
    def __init__(self, makam):
        self.__makam = makam
        self.__dur_to_int = {}
        self.__int_to_dur = {}
        with open(self.__makam + '_sorted_dur_corpus.txt', 'r') as f:
            lines = f.read().splitlines()
            i = 1
            for line in lines:
                durs = line.split(',')
                for dur in durs:
                    self.__dur_to_int[dur] = i
                    if i not in self.__int_to_dur:
                        self.__int_to_dur[i] = []
                    self.__int_to_dur[i].append(dur)
                i = i + 1

    def get_dur_num_by_name(self, dur_name):
        if dur_name in self.__dur_to_int:
            return self.__dur_to_int[dur_name]
        else:
            return None

    def get_dur_name_by_num(self, dur_num):
        if dur_num in self.__int_to_dur:
            return self.__int_to_dur[dur_num][0]
        else:
            return None
