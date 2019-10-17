class Measure:
    def __init__(self, m_no, rep=False):
        self._no = m_no
        self._rep = rep
        self._notes = []

    def print_self(self):
        print(self._no, self._rep, self._notes)


class PartObj:
    def __init__(self, pid, m_list):
        self._pid = pid
        self._measures = []
        for e in m_list:
            if isinstance(e, list):
                self._measures.append(Measure(e[0], True))
                self._measures.append(Measure(e[1], True))
            else:
                self._measures.append(Measure(e))

    def print_self(self):
        print(self._pid)
        for measure in self._measures:
            measure.print_self()


class SongObj:
    def __init__(self, song_name, tempo=None, time_signature=None):
        self._name = song_name
        self._tempo = tempo
        self._time_sign = time_signature
        self._parts = []

    def set_tempo(self, tempo):
        self._tempo = tempo

    def set_time_sign(self, time_signature):
        self._time_sign = time_signature

    def init_measures(self, parts_map):
        for k in parts_map:
            self._parts.append(PartObj(k, parts_map[k]))

    def in_list(self, my_list, item):
        if item in my_list:
            return True
        else:
            return any(self.in_list(sublist, item) for sublist in my_list if isinstance(sublist, list))

    def print_self(self):
        print(self._name, self._time_sign, self._tempo)
        for part in self._parts:
            part.print_self()

    def insert_note(self, note, curr_m_no):
        pass
