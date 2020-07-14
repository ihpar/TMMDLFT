from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping

import consts
import dur_translator
import nihavent_parts
import note_translator
from mu2_reader import *
from model_ops import load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import random
import math
from candidate_picker import CandidatePicker
from nakarat_ender import make_second_rep

cnt_pa, cnt_pb = 0, 0


def get_flattened_parts(makam, part_id, dir_path, note_dict, oh_manager):
    parts = []
    for curr_song in hicaz_parts.hicaz_songs:
        song = curr_song['file']
        part_map = curr_song['parts_map']
        song_final = curr_song['sf']
        song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
        part = song.get_part(part_id)
        parts.append(part)
    return parts


def make_db(makam, part_id, dir_path, note_dict, oh_manager, set_size, is_whole=False):
    songs = []
    nt = note_translator.NoteTranslator(makam)
    dt = dur_translator.DurTranslator(makam)

    if makam == 'hicaz':
        for curr_song in hicaz_parts.hicaz_songs:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager)
            songs.append(song)
    elif makam == 'nihavent':
        for curr_song in nihavent_parts.nihavent_songs:
            song = curr_song['file']
            part_map = curr_song['parts_map']
            song_final = curr_song['sf']
            song = decompose_mu2(dir_path, song, part_map, song_final, note_dict, oh_manager, nt, dt)
            songs.append(song)

    x_lst, y_lst = [], []
    for song in songs:
        part = song.get_part(part_id)
        xs, ys = [], []
        for i in range(len(part) - set_size):
            x_sec = [oh_manager.int_2_oh(x) for x in part[i:i + set_size]]
            y_sec = oh_manager.int_2_oh(part[i + set_size])
            xs.append(x_sec)
            ys.append(y_sec)
        if not xs:
            continue
        if not is_whole:
            x_lst.append(np.array(xs))
            y_lst.append(np.array(ys))
        else:
            x_lst.extend(xs)
            y_lst.extend(ys)

    if not is_whole:
        return x_lst, y_lst
    else:
        return np.array(x_lst), np.array(y_lst)


def train_whole(makam, src_model, xs, ys, target_model, eps=0):
    out_shape = ys.shape[1]

    base_model = load_model(makam, src_model, False)
    new_model = Sequential()
    for i, layer in enumerate(base_model.layers):
        if i == 4:
            break
        if i < 2:
            layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(out_shape))
    new_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    new_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    if eps == 0:
        history = new_model.fit(xs, ys, epochs=100, batch_size=16, shuffle=False, validation_split=0.1, callbacks=[es])
    else:
        history = new_model.fit(xs, ys, epochs=eps, batch_size=16, shuffle=False)

    save_model(makam, target_model, new_model)
    plt.plot(history.history['loss'], label='train')
    if eps == 0:
        plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def train_model(makam, src_model, xs, ys, target_model, eps):
    out_shape = ys[0].shape[1]

    base_model = load_model(makam, src_model, False)
    new_model = Sequential()
    for i, layer in enumerate(base_model.layers):
        if i == 4:
            break
        layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(out_shape))
    new_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    new_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    new_model.summary()

    histories = []

    for i in range(eps):
        print(f'=== Main loop: {i} ===')
        for x, y in zip(xs, ys):
            history = new_model.fit(x, y, epochs=1, batch_size=16)
            histories.extend(history.history['loss'])

    save_model(makam, target_model, new_model)
    plt.plot(histories)
    plt.show()


def get_starters(init_file, set_size, note_dict, oh_manager):
    nom_index = 2
    den_index = 3
    beat, tot = None, Fraction(0)
    starters = []
    with codecs.open(init_file, 'r', encoding='utf8') as sf:
        lines = sf.read().splitlines()
        i = 0
        for line in lines:
            if i == set_size:
                break
            parts = line.split('\t')
            parts = [s.strip() for s in parts]
            parts[0] = int(parts[0])

            if (parts[0] in [1, 7, 9, 10, 11, 12, 24, 28]) and (
                    parts[nom_index].isdigit() and parts[den_index].isdigit()):
                # note name
                note_name = parts[1].lower().strip()
                if note_name == '':
                    note_name = 'rest'
                note_num = note_dict.get_note_by_name(note_name)
                # note dur
                note_len = Fraction(int(parts[nom_index]), int(parts[den_index]))
                tot += note_len
                dur = str(note_len)
                dur_alt = parts[nom_index] + '/' + parts[den_index]
                dur = note_dict.get_num_by_dur(dur)
                if not dur:
                    dur = note_dict.get_num_by_dur(dur_alt)
                combine = oh_manager.nd_2_oh(str(note_num) + ':' + str(dur))
                starters.append(combine)
                i += 1
    return np.array(starters), tot


def get_starters_by_part(init_part, set_size, note_dict, oh_manager, models, lo, hi, cp, time_sig):
    starter = init_part[0][-set_size:]
    song = np.array([np.copy(starter)])
    total = Fraction(0)
    xpy = song.shape[1]
    model_a = models[0]
    model_b = models[1]
    decider = models[2]
    m_remainder = time_sig
    for i in range(set_size):
        if m_remainder == Fraction(0):
            m_remainder = time_sig
        part = song[:, -xpy:, :]
        p_a = get_prediction(model_a, part, lo, hi, cp)
        p_b = get_prediction(model_b, part, lo, hi, cp)
        chosen = choose_prediction(part, p_a, p_b, decider, oh_manager)
        n_d = oh_manager.int_2_nd(chosen)
        parts = n_d.split(':')
        note_num = int(parts[0])
        dur = Fraction(note_dict.get_dur_by_num(int(parts[1])))
        if dur > m_remainder:
            dur = m_remainder
            dur_num = note_dict.get_num_by_dur(str(dur))
            chosen = oh_manager.nd_2_int(str(note_num) + ':' + str(dur_num))

        m_remainder -= dur
        total += dur

        p_inner = np.zeros(part.shape[2])
        p_inner[chosen] = 1.0

        song = np.append(song, np.array([[p_inner]]), axis=1)

    return song[:, -xpy:, :][0], total


def compose(makam, time_sig, measure_cnt, init_file, model, set_size, lo, hi, cp, note_dict, oh_manager, by_part=False, totil=0):
    if not by_part:
        starters, tot = get_starters(init_file, set_size, note_dict, oh_manager)
    else:
        starters, tot = init_file, totil

    if tot > (time_sig * measure_cnt):
        raise Exception('Starter notes duration exceeded time limit!')

    song = np.array([np.copy(starters)])
    xpy = song.shape[1]

    tot_certain, tot_rand = 0, 1

    elapsed_measures = math.floor(tot / time_sig)
    measure_remainder = time_sig - (tot - (elapsed_measures * time_sig))
    target_dur = (time_sig * measure_cnt) - tot

    while target_dur > 0:
        if measure_remainder == Fraction(0):
            measure_remainder = time_sig

        part = song[:, -xpy:, :]
        prediction = model.predict(part)
        shape = prediction.shape
        p_inner = np.copy(prediction[0])
        max_index = np.argmax(p_inner)

        if p_inner[max_index] < hi:
            index_candidates = [max_index]
            n_probs = [p_inner[max_index]]
            has_candidates = True
            while has_candidates:
                p_inner[max_index] = 0
                max_index = np.argmax(p_inner)
                if p_inner[max_index] > lo:
                    index_candidates.append(max_index)
                    n_probs.append(p_inner[max_index])
                else:
                    has_candidates = False
            max_index = cp.pick_candidate(part, index_candidates, n_probs)
            # max_index = random.choice(index_candidates)
            tot_rand += 1
        else:
            tot_certain += 1

        n_d = oh_manager.int_2_nd(max_index)
        parts = n_d.split(':')
        note_num = int(parts[0])
        dur = int(parts[1])
        dur = Fraction(note_dict.get_dur_by_num(dur))
        if dur > measure_remainder:
            dur = measure_remainder

        dur_num = note_dict.get_num_by_dur(str(dur))
        n_d = str(note_num) + ':' + str(dur_num)
        n_d_num = oh_manager.nd_2_int(n_d)
        p_inner = np.zeros(shape[1])
        # p_inner[max_index] = 1.0
        p_inner[n_d_num] = 1.0

        song = np.append(song, np.array([[p_inner]]), axis=1)

        measure_remainder -= dur
        target_dur -= dur

    print(f'Certain: {tot_certain}, Rand: {tot_rand}, Ratio: {tot_certain / tot_rand}')
    return song


def print_notes(notes, oh_manager, note_dict):
    for note in notes:
        nd = oh_manager.oh_2_nd(note)
        parts = nd.split(':')
        note_name = note_dict.get_note_by_num(int(parts[0]))
        note_dur = note_dict.get_dur_by_num(int(parts[1]))
        print(f'{note_name}, {note_dur}')


def compose_v2(makam, time_sig, measure_cnt, init_file, models, set_size, lo, hi, cp, note_dict, oh_manager, by_part=False):
    global cnt_pa, cnt_pb
    model_a = models[0]
    model_b = models[1]
    decider = models[2]

    if not by_part:
        starters, tot = get_starters(init_file, set_size, note_dict, oh_manager)
    else:
        starters, tot = get_starters_by_part(init_file, set_size, note_dict, oh_manager, models, lo, hi, cp, time_sig)

    # print_notes(starters, oh_manager, note_dict)

    if tot > (time_sig * measure_cnt):
        raise Exception('Starter notes duration exceeded time limit!')

    song = np.array([np.copy(starters)])
    xpy = song.shape[1]

    elapsed_measures = math.floor(tot / time_sig)
    measure_remainder = time_sig - (tot - (elapsed_measures * time_sig))
    target_dur = (time_sig * measure_cnt) - tot

    # print('---------------------------------------------')
    # print(f'total:{str(tot)}, elapsed:{str(elapsed_measures)}, m.rem:{str(measure_remainder)}, target:{str(target_dur)}')
    # print('---------------------------------------------')

    while target_dur > 0:
        if measure_remainder == Fraction(0):
            measure_remainder = time_sig

        part = song[:, -xpy:, :]
        p_a = get_prediction(model_a, part, lo, hi, cp)
        p_b = get_prediction(model_b, part, lo, hi, cp)

        chosen = choose_prediction(part, p_a, p_b, decider, oh_manager)

        n_d = oh_manager.int_2_nd(chosen)
        parts = n_d.split(':')
        note_num = int(parts[0])
        dur = int(parts[1])
        dur = Fraction(note_dict.get_dur_by_num(dur))
        dur_cpy = dur

        if dur > measure_remainder:
            dur = measure_remainder

        dur_num = note_dict.get_num_by_dur(str(dur))
        n_c_d = str(note_num) + ':' + str(dur_num)
        try:
            n_d_num = oh_manager.nd_2_int(n_c_d)
            p_inner = np.zeros(part.shape[2])
            p_inner[n_d_num] = 1.0

            song = np.append(song, np.array([[p_inner]]), axis=1)
        except KeyError as e:
            print(n_d, dur_cpy, dur, note_num)
            print(f'Key Error: {str(e)}')
            return []

        measure_remainder -= dur
        target_dur -= dur

    print(f'PA:{cnt_pa}, PB:{cnt_pb}')
    cnt_pa, cnt_pb = 0, 0
    return song


def get_prediction(model, part, lo, hi, cp):
    prediction = model.predict(part)
    p_inner = np.copy(prediction[0])
    max_index = np.argmax(p_inner)

    if p_inner[max_index] < hi:
        index_candidates = [max_index]
        n_probs = [p_inner[max_index]]
        has_candidates = True
        while has_candidates:
            p_inner[max_index] = 0
            max_index = np.argmax(p_inner)
            if p_inner[max_index] > lo:
                index_candidates.append(max_index)
                n_probs.append(p_inner[max_index])
            else:
                has_candidates = False
        max_index = cp.pick_candidate(part, index_candidates, n_probs)
    return max_index


def choose_prediction(part, p_a, p_b, decider, oh_manager):
    global cnt_pa, cnt_pb
    inp = []
    for nd in part[0]:
        inp.append(oh_manager.oh_2_zo(nd))
    inp.append(oh_manager.int_2_zo(p_a))
    inp.append(oh_manager.int_2_zo(p_b))
    inp = np.array([inp])
    inp = inp.reshape((inp.shape[0], 1, inp.shape[1]))
    pred = decider.predict(inp)[0]
    if abs(pred[0] - pred[1]) < 0.2:
        chosen = random.choice([0, 1])
        if chosen == 0:
            cnt_pa += 1
            return p_a
        else:
            cnt_pb += 1
            return p_b
    if np.argmax(pred) == 0:
        cnt_pa += 1
        return p_a
    cnt_pb += 1
    return p_b


def get_mu2_str(note, nom, denom):
    if note == 'Rest':
        return '9		{num}	{denom}	100					0.0'.replace('{num}', nom).replace('{denom}', denom)
    else:
        return '9	{nn}	{num}	{denom}	95	96	64	 		0.0'.replace('{nn}', note).replace('{num}', nom).replace('{denom}', denom)


def song_2_mus(song, makam, title, oh_manager, note_dict, time_sig, mcs, second_rep):
    lines = consts.mu2_header.copy()
    lines[0] = '9	8	Pay	Payda	Legato%	Bas	Çek	Söz-1	Söz-2	0.0'
    lines[2] = '51		9	8				Aksak		'
    lines[1] = lines[1].replace('{makam}', makam)
    lines[7] = lines[7].replace('{song_title}', title)
    m_tot = Fraction(0)
    m_cnt = 0
    mzs = [int(x) for x in mcs.split(',')]
    has_second_rep = second_rep.size > 0
    pb_len = mzs[1] - mzs[0]
    for row in song[0]:
        n_d = oh_manager.oh_2_nd(row)
        parts = n_d.split(':')
        note = int(parts[0])
        dur = int(parts[1])
        note = note_dict.get_note_by_num(note)
        if not note:
            raise Exception('Note N/A')
        note = note.capitalize()

        dur = note_dict.get_dur_by_num(dur).split('/')
        lines.append(get_mu2_str(note, dur[0], dur[1]))

        m_tot += Fraction(int(dur[0]), int(dur[1]))
        if m_tot >= time_sig:
            m_tot = Fraction(0)
            m_cnt += 1
            if m_cnt not in mzs:
                # if m_cnt % 2 == 0:
                #     lines.append('21									0.0')
                lines.append('14									0.0')

            if m_cnt == mzs[0]:
                lines.append('9								:	0.0')
                lines.append('9								)	0.0')
                # lines.append('21									0.0')
                lines.append('14									0.0')
                lines.append('9							[		0.0')
                lines.append('9							(		0.0')
            if has_second_rep and m_cnt == mzs[1] - 1:
                # first rep
                lines.append('9								)	0.0')
                lines.append('9							(1		0.0')
            if has_second_rep and m_cnt == mzs[1]:
                print('Has second rep')
                # 2nd rep
                lines.append('9								:	0.0')
                lines.append('9								1)	0.0')
                lines.append('9							(2		0.0')
                # add 2nd rep here
                for s_r in second_rep:
                    n_d = oh_manager.oh_2_nd(s_r)
                    parts = n_d.split(':')
                    note = note_dict.get_note_by_num(int(parts[0])).capitalize()
                    dur = note_dict.get_dur_by_num(int(parts[1])).split('/')
                    lines.append(get_mu2_str(note, dur[0], dur[1]))
            if m_cnt == mzs[1]:
                if has_second_rep:
                    lines.append('9								2)	0.0')
                else:
                    lines.append('9								:	0.0')
                    lines.append('9								)	0.0')
                lines.append('9								]	0.0')
                # lines.append('21									0.0')
                lines.append('9							(		')
            if m_cnt == mzs[2]:
                lines.append('9								:	0.0')
                lines.append('9								)	0.0')
                lines.append('9								$	0.0')
    file_name = title + '.mu2'
    song_path = os.path.join(os.path.abspath('..'), 'songs', makam, file_name)
    with io.open(song_path, 'w', encoding='utf-8') as song_file:
        for line in lines:
            song_file.write(line + '\n')

    print(f'{file_name} is saved to disk!')


def make_ab_db(makam, part_ids, dir_path, note_dict, oh_manager, set_size):
    x_lst, y_lst = [], []
    parts_a = get_flattened_parts(makam, part_ids[0], dir_path, note_dict, oh_manager)
    parts_b = get_flattened_parts(makam, part_ids[1], dir_path, note_dict, oh_manager)
    if len(parts_a) != len(parts_b):
        raise Exception('Part lengths must be equal!')

    for a, b in zip(parts_a, parts_b):
        a_lst = a[-8:]
        b = a_lst + b
        xs, ys = [], []
        for i in range(len(b) - set_size):
            x_sec = [oh_manager.int_2_oh(x) for x in b[i:i + set_size]]
            y_sec = oh_manager.int_2_oh(b[i + set_size])
            xs.append(x_sec)
            ys.append(y_sec)
        x_lst.extend(xs)
        y_lst.extend(ys)
    return np.array(x_lst), np.array(y_lst)


def compose_ending(makam, enders, part, time_sig, measure_cnt, note_dict, oh_manager, lo, hi):
    second_rep = np.array([])
    perfect_end = False
    perfect_note = 'La4'
    if makam == 'hicaz':
        perfect_note = 'La4'
    for row in reversed(part[0]):
        n_d = oh_manager.oh_2_nd(row)
        parts = n_d.split(':')
        note = note_dict.get_note_by_num(int(parts[0])).capitalize()
        if note == 'Rest':
            continue
        if note == perfect_note:
            perfect_end = True
        break

    if not perfect_end:
        second_rep = make_second_rep(makam, enders, part, time_sig, measure_cnt, note_dict, oh_manager, lo, hi)

    return second_rep


def main():
    # makam = 'hicaz'
    makam = 'nihavent'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    dir_path = 'E:\\Akademik\\Tik5\\nihavent_sarkilar\\nihavent-ekler'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)
    set_size = 8
    # time_sig = Fraction(9, 8)
    time_sig = Fraction(8, 8)
    ver = '62'
    sep = 'CW2'

    # xs = [[[n1,n2,n3,..,n8],[n2,n3,...,n9]], song:[8s:[],8s:[],...]]
    # ys = [[n1,n2,...,nm], song:[outs]]
    xi, yi = make_db(makam, 'I', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    xa, ya = make_db(makam, 'A', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    # xb, yb = make_db(makam, 'B', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    # xc, yc = make_db(makam, 'C', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    xs = np.concatenate((xi, xa), axis=0)
    # xs = np.concatenate((xs, xb), axis=0)
    # xs = np.concatenate((xs, xc), axis=0)
    ys = np.concatenate((yi, ya), axis=0)
    # ys = np.concatenate((ys, yb), axis=0)
    # ys = np.concatenate((ys, yc), axis=0)
    # IABCW1 (freeze 1st, new dense, val_split: 0.1),
    # IABCW2 (unfreeze all, new dense, val_split: 0.1, batch=32)
    # AW11,12 (freeze 1st, new dense, val_split: 0.1, batch=16)
    ### train_whole(makam, 'lstm_v' + ver, xs, ys, 'sec_' + sep + '_v' + ver, eps=10)

    '''
    # nakarat train begin
    xs, ys = make_ab_db(makam, ['A', 'B'], dir_path, note_dict, oh_manager, set_size)
    # xc, yc = make_db(makam, 'C', dir_path, note_dict, oh_manager, set_size, is_whole=True)
    # xs = np.concatenate((xs, xc), axis=0)
    # ys = np.concatenate((ys, yc), axis=0)
    # B0_v61, B1_v61, AH20_v62, AH40_v62, sec_AW_v61, AW5 (freeze 1st), AW6 (freeze 1st), AW7 (freeze 1st, keep dense)
    # AW8 (freeze 1st, new dense), AW9 (freeze 1st, keep dense, val_split: 0.1->0.25),
    # AW10 (freeze 1st, keep dense, val_split: 0.1)
    # BW1/2 (freeze 1st, keep dense, val_split: 0.1), BW3 (freeze 1st, new dense, val_split: 0.1)
    # BW4 (unfreeze all, new dense, val_split: 0.1, batch=8), BW5 (unfreeze all, new dense, val_split: 0.1, batch=32)
    # BW6 (unfreeze all, new dense, val_split: 0.1, batch=64)
    # BW7 (freeze 1st, new dense, val_split: 0, batch=16), BW8 (freeze 1st , 2nd, new dense, val_split: 0, batch=16)
    # BW9,10 (freeze 1st, new dense, val_split: 0, batch=16)
    # BW11,12 (freeze 1st, new dense, val_split: 0, batch=16, epoch=12,10)
    train_whole(makam, 'lstm_v' + ver, xs, ys, 'sec_' + sep + '_v' + ver, eps=10)
    # nakarat train end
    '''
    '''
    # C train begin
    xs, ys = make_ab_db(makam, ['B', 'C'], dir_path, note_dict, oh_manager, set_size)
    # CW1,2 (freeze 1st, new dense, val_split: 0.1, batch=16)
    train_whole(makam, 'lstm_v' + ver, xs, ys, 'sec_' + sep + '_v' + ver, eps=10)
    # C train end
    '''

    '''
    measure_cnt = 4
    lo = 0.1
    hi = 0.4

    models_a = [load_model(makam, 'sec_AW9_v61'), load_model(makam, 'sec_AW10_v62'), load_model(makam, 'b_decider_v_ia7')]
    models_b = [load_model(makam, 'sec_BW11_v61'), load_model(makam, 'sec_BW12_v62'), load_model(makam, 'b_decider_v_b8')]
    models_c = [load_model(makam, 'sec_CW1_v61'), load_model(makam, 'sec_CW2_v62'), load_model(makam, 'b_decider_v_c9')]
    enders = ['nakarat_end_v2', 'nakarat_end_v1']

    for i in range(0, 1):
        init = str(i)

        song_name = 'Hicaz_Aksak_Tester_' + init
        initiator = 'init-hicaz-' + init + '.mu2'
        # compose(makam, time_sig, measure_cnt, initiator, model, set_size, lo, hi, cp, note_dict, oh_manager, song_name)
        cp = CandidatePicker(makam, hicaz_parts.hicaz_songs, ['I', 'A'], dir_path, note_dict, oh_manager, set_size)
        part_a = compose_v2(makam, time_sig, measure_cnt, initiator, models_a, set_size, lo, hi, cp, note_dict, oh_manager)
        if len(part_a) == 0:
            continue

        cp = CandidatePicker(makam, hicaz_parts.hicaz_songs, ['B'], dir_path, note_dict, oh_manager, set_size)
        part_b = compose_v2(makam, time_sig, measure_cnt, part_a, models_b, set_size, lo, hi, cp, note_dict, oh_manager, by_part=True)
        second_rep = compose_ending(makam, enders, part_b, time_sig, measure_cnt, note_dict, oh_manager, lo, hi)
        if len(part_b) == 0:
            continue

        cp = CandidatePicker(makam, hicaz_parts.hicaz_songs, ['C'], dir_path, note_dict, oh_manager, set_size)
        part_c = compose_v2(makam, time_sig, measure_cnt, part_b, models_c, set_size, lo, hi, cp, note_dict, oh_manager, by_part=True)
        if len(part_c) == 0:
            continue

        song = np.append(part_a, part_b, axis=1)
        song = np.append(song, part_c, axis=1)
        song_2_mus(song, makam, song_name, oh_manager, note_dict, time_sig, '4,8,12', second_rep)
    '''


if __name__ == '__main__':
    main()
