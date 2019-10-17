class SongObj:
    def __init__(self, song_name, tempo=None, time_signature=None):
        self._name = song_name
        self._tempo = tempo
        self._time_sign = time_signature
        self._I = []
        self._A = []
        self._B = []
        self._C = []

    def set_tempo(self, tempo):
        self._tempo = tempo

    def set_time_sign(self, time_signature):
        self._time_sign = time_signature
