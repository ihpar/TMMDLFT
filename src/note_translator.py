class NoteTranslator:
    def __init__(self, makam):
        self.__makam = makam
        self.__note_to_int = {}
        self.__int_to_note = {}
        with open(self.__makam + '_sorted_note_corpus.txt', 'r') as f:
            lines = f.read().splitlines()
            i = 1
            for line in lines:
                notes = line.split(',')
                for note in notes:
                    self.__note_to_int[note] = i
                    if i not in self.__int_to_note:
                        self.__int_to_note[i] = []
                    self.__int_to_note[i].append(note)
                i = i + 1

    def get_note_num_by_name(self, note_name):
        if note_name in self.__note_to_int:
            return self.__note_to_int[note_name]
        else:
            return None

    def get_note_name_by_num(self, note_num):
        if note_num in self.__int_to_note:
            return self.__int_to_note[note_num][0]
        else:
            return None
