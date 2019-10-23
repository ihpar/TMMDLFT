class Measure:
    def __init__(self, m_no, rep=False):
        self._no = m_no
        self._rep = rep
        self._notes = []

    def __str__(self):
        res = '\n' + '\t' + 'No: ' + str(self._no) + ', Rep: ' + str(self._rep) + ', Notes: ' + str(self._notes)
        return res

    def insert_note(self, note):
        self._notes.append(note)

    def get_note(self, note_idx):
        return self._notes[note_idx]

    def get_notes(self):
        return self._notes


class PartObj:
    def __init__(self, pid, m_list):
        self._pid = pid
        self._measures = []
        self._rev_measures = {}
        i = 0
        for e in m_list:  # m_list = [4, 5, 6, [7, 8, 20]]
            if isinstance(e, list):
                self._measures.append(Measure(e[0], True))
                self._measures.append(Measure(e[1], True))
                self._rev_measures[e[0]] = i
                i += 1
                self._rev_measures[e[1]] = i
                if len(e) == 3:
                    self._measures.append(Measure(e[2], True))
                    i += 1
                    self._rev_measures[e[2]] = i
            else:
                self._measures.append(Measure(e))
                self._rev_measures[e] = i
            i += 1

    def __str__(self):
        res = self._pid + ': '
        for measure in self._measures:
            res += str(measure)
        if not self._measures:
            res += '\n' + '\tEmpty'
        return res

    def insert_note(self, note, curr_m_no):
        idx = self._rev_measures[curr_m_no]
        (self._measures[idx]).insert_note(note)

    def get_note(self, measure_no, note_idx):
        idx = self._rev_measures[measure_no]
        return (self._measures[idx]).get_note(note_idx)

    def flat_measures(self):
        res = []
        for measure in self._measures:
            res = res + measure.get_notes()
        return res


class SongObj:
    def __init__(self, song_name, parts_map, final, tempo=None, time_signature=None):
        self._name = song_name
        self._tempo = tempo
        self._time_sign = time_signature
        self._parts = []
        self._rev_parts = {}
        self._part_ids = {}
        self.init_measures(parts_map)
        self._final = final

    def __str__(self):
        res = ''
        for part in self._parts:
            res += str(part) + '\n'
        res += '\n' + self._name + ' - ' + str(self._time_sign) + ' - ' + str(self._tempo) + '\n'
        if self._final:
            res += 'fin: ' + str(self.get_note(self._final[0], self._final[1])) + '\n'
        else:
            res += 'fin: []\n'
        return res

    def set_tempo(self, tempo):
        self._tempo = tempo

    def set_time_sign(self, time_signature):
        self._time_sign = time_signature

    def init_measures(self, parts_map):
        i = 0
        for k in parts_map:
            m_list = parts_map[k]
            # create part object and add to parts list
            self._parts.append(PartObj(k, m_list))
            self._part_ids[k] = i
            # create a reverse mapping based on measure numbers
            for el in m_list:  # m_list = [1, 2, 3, [4, 5]]
                if isinstance(el, list):
                    self._rev_parts[el[0]] = i
                    self._rev_parts[el[1]] = i
                    if len(el) == 3:
                        self._rev_parts[el[2]] = i
                else:
                    self._rev_parts[el] = i
            i += 1

    def in_list(self, my_list, item):
        if item in my_list:
            return True
        else:
            return any(self.in_list(sublist, item) for sublist in my_list if isinstance(sublist, list))

    def insert_note(self, note, curr_m_no):
        idx = self._rev_parts[curr_m_no]
        (self._parts[idx]).insert_note(note, curr_m_no)

    def get_note(self, measure_no, note_idx):
        idx = self._rev_parts[measure_no]
        return (self._parts[idx]).get_note(measure_no, note_idx)

    def get_part(self, part_id):
        return self._parts[self._part_ids[part_id]].flat_measures()
