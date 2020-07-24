# from nakarat_endings import hicaz_song_endings
import dur_translator
import note_translator
from ending_picker import EndingPicker
from my_endings import my_hicaz_song_endings, my_nihavent_song_endings

from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping

from mu2_reader import *
from model_ops import load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
import hicaz_parts
import random


def parse_notes(notes, note_dict, oh_manager, nt=None, dt=None):
    res = []
    for n in notes:
        parts = n.split(':')
        name = parts[0].lower()
        dur = parts[1]
        if nt:
            name = nt.get_note_num_by_name(name)
        else:
            name = note_dict.get_note_by_name(name)

        if dt:
            dur = dt.get_dur_num_by_name(dur)
        else:
            dur = note_dict.get_num_by_dur(dur)

        oh = oh_manager.nd_2_oh(str(name) + ':' + str(dur))
        res.append(oh)
    return np.array(res)


def make_ending_data(makam, note_dict, oh_manager, set_size):
    nt, dt = None, None
    if makam == 'nihavent':
        nt = note_translator.NoteTranslator(makam)
        dt = dur_translator.DurTranslator(makam)

    xs, ys, endings = [], [], []
    if makam == 'hicaz':
        endings = my_hicaz_song_endings.copy()
    elif makam == 'nihavent':
        endings = my_nihavent_song_endings.copy()

    for se in endings:
        try:
            prevs = se['prevs']
            if not prevs:
                continue
            # end_f = hse['endings'][0]
            end_s = se['endings'][1]
            # fin = hse['fin']
            if makam == 'hicaz':
                prevs = parse_notes(prevs, note_dict, oh_manager)
                end_s = parse_notes(end_s, note_dict, oh_manager)
            elif makam == 'nihavent':
                prevs = parse_notes(prevs, note_dict, oh_manager, nt, dt)
                end_s = parse_notes(end_s, note_dict, oh_manager, nt, dt)

            prevs = np.concatenate((prevs, end_s))
            seq_len = prevs.shape[0]
            for i in range(seq_len - set_size):
                x = prevs[i:i + set_size]
                y = prevs[i + set_size]
                xs.append(x)
                ys.append(y)
        except KeyError as e:
            print(se['file'] + '\n' + str(e))
    return np.array(xs), np.array(ys)


def make_model(makam, base_model, out_shape):
    base_model = load_model(makam, base_model, False)
    model = Sequential()
    for i, layer in enumerate(base_model.layers):
        if i == 4:
            break
        if i < 2:
            layer.trainable = False
        model.add(layer)

    model.add(Dense(out_shape))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def train_nakarat_ending_model(makam, base_model, model_name, xs, ys, eps=0):
    out_shape = ys.shape[1]
    model = make_model(makam, base_model, out_shape)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    if eps == 0:
        history = model.fit(xs, ys, epochs=100, batch_size=16, shuffle=False, validation_split=0.1, callbacks=[es])
    else:
        history = model.fit(xs, ys, epochs=eps, batch_size=16, shuffle=False)

    save_model(makam, model_name, model)
    plt.plot(history.history['loss'], label='train')
    if eps == 0:
        plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def pick_chosen(pred_a, pred_b, threshold):
    candidates = []
    chosen_a = np.argmax(pred_a)
    candidates.append(chosen_a)
    pred_a[chosen_a] = 0

    chosen_b = np.argmax(pred_b)
    candidates.append(chosen_b)
    pred_b[chosen_b] = 0
    can_continue = True
    while can_continue:
        can_continue = False
        chosen_a = np.argmax(pred_a)
        if pred_a[chosen_a] > threshold:
            candidates.append(chosen_a)
            pred_a[chosen_a] = 0
            can_continue = True

        chosen_b = np.argmax(pred_b)
        if pred_b[chosen_b] > threshold:
            candidates.append(chosen_b)
            pred_b[chosen_b] = 0
            can_continue = True

    return random.choice(candidates)


def make_ending_predictions(last_notes, nakarat_ender_model_a, nakarat_ender_model_b, time_sig, oh_manager, note_dict, threshold):
    print('making ending predictions')
    tot = Fraction(0)
    xpy = last_notes.shape[1]
    predictions = []
    while tot < time_sig:
        part = last_notes[:, -xpy:, :]
        y_a = nakarat_ender_model_a.predict(part)
        y_b = nakarat_ender_model_b.predict(part)
        chosen = pick_chosen(y_a[0], y_b[0], threshold)

        n_d = oh_manager.int_2_nd(chosen)
        parts = n_d.split(':')
        note_num = int(parts[0])
        dur = Fraction(note_dict.get_dur_by_num(int(parts[1])))
        remaining = time_sig - tot
        if dur > remaining:
            dur = remaining
        tot += dur
        dur_num = note_dict.get_num_by_dur(str(dur))
        n_c_d = str(note_num) + ':' + str(dur_num)
        try:
            n_d_num = oh_manager.nd_2_int(n_c_d)
            p_inner = np.zeros(part.shape[2])
            p_inner[n_d_num] = 1.0

            last_notes = np.append(last_notes, np.array([[p_inner]]), axis=1)
            predictions.append(p_inner)
        except KeyError as e:
            print(n_d, parts[1], dur, note_num)
            print(f'Key Error: {str(e)}')
            raise Exception('Nakarat ending err')

    return np.array(predictions)


def get_remaining(remainder, oh_manager, note_dict):
    res = []
    num = remainder.numerator
    den = remainder.denominator
    note_num = note_dict.get_note_by_name('rest')

    if num == 1 or (num % 2) == 0:
        note_dur = note_dict.get_num_by_dur(str(Fraction(num, den)))
        oh = oh_manager.nd_2_oh(str(note_num) + ':' + str(note_dur))
        res.append(oh)
    else:
        print('Fixing remaining rests')
        one_fr = Fraction(1, den)
        rem_fr = Fraction(num - 1, den)

        note_dur = note_dict.get_num_by_dur(str(one_fr))
        oh = oh_manager.nd_2_oh(str(note_num) + ':' + str(note_dur))
        res.append(oh)
        while (rem_fr.numerator != 1) and (rem_fr.numerator % 2 != 0):
            num = rem_fr.numerator
            den = rem_fr.denominator
            one_fr = Fraction(1, den)
            note_dur = note_dict.get_num_by_dur(str(one_fr))
            oh = oh_manager.nd_2_oh(str(note_num) + ':' + str(note_dur))
            res.append(oh)
            rem_fr = Fraction(num - 1, den)

        note_dur = note_dict.get_num_by_dur(str(rem_fr))
        oh = oh_manager.nd_2_oh(str(note_num) + ':' + str(note_dur))
        res.append(oh)

    return res


def end_if_can(predictions, makam, oh_manager, note_dict, time_sig):
    print('checking if can end')
    perfect_note = 'La4'
    if makam == 'hicaz':
        perfect_note = 'La4'

    can_end = False
    i = 0
    for row in reversed(predictions):
        n_d = oh_manager.oh_2_nd(row)
        parts = n_d.split(':')
        note = note_dict.get_note_by_num(int(parts[0])).capitalize()
        if note == perfect_note:
            can_end = True
            break
        i += 1

    if not can_end:
        print('cannot end yet')
        return can_end, predictions

    print('can end')
    new_ending = []
    tot = Fraction(0)
    for j, row in enumerate(predictions):
        if j < (len(predictions) - i):
            n_d = oh_manager.oh_2_nd(row)
            parts = n_d.split(':')
            dur = Fraction(note_dict.get_dur_by_num(int(parts[1])))
            tot += dur

            new_ending.append(row)
        else:
            break

    remainder = time_sig - tot
    if remainder > 0:
        remaining_parts = get_remaining(remainder, oh_manager, note_dict)
        for row in remaining_parts:
            new_ending.append(row)

    return can_end, np.array(new_ending)


def fixed_ending(fixed_note_lst, fixed_dur_list, oh_manager, note_dict):
    res = []
    for note, dur in zip(fixed_note_lst, fixed_dur_list):
        note_num = note_dict.get_note_by_name(note)
        dur_num = note_dict.get_num_by_dur(dur)
        res.append(oh_manager.nd_2_oh(str(note_num) + ':' + str(dur_num)))
    return np.array(res)


def make_second_rep(makam, enders, part, time_sig, measure_cnt, note_dict, oh_manager, lo, hi):
    # ending_picker = EndingPicker(makam, hicaz_parts.hicaz_songs, 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2', note_dict, oh_manager, 4)
    nakarat_ender_model_a, nakarat_ender_model_b = load_model(makam, enders[0]), load_model(makam, enders[1])
    tot = Fraction(0)
    m_no = 0
    measures = []
    for i in range(measure_cnt):
        measures.append([])

    for r in part[0]:
        n_d = oh_manager.oh_2_nd(r)
        parts = n_d.split(':')
        dur = Fraction(note_dict.get_dur_by_num(int(parts[1])))
        tot += dur
        measures[m_no].append(n_d)
        if tot == time_sig:
            tot = Fraction(0)
            m_no += 1

    last_notes = []
    n_cnt = 0
    broken = False
    for m_no in reversed(range(measure_cnt - 1)):
        curr_measure = measures[m_no]
        for n in reversed(curr_measure):
            n_cnt += 1
            last_notes.append(n)
            if n_cnt == 8:
                broken = True
                break
        if broken:
            break

    last_notes.reverse()
    x = np.array([[oh_manager.nd_2_oh(n) for n in last_notes]])
    song_can_end, correctly_ended = False, False
    predictions = []
    while not song_can_end:
        try:
            predictions = make_ending_predictions(x, nakarat_ender_model_a, nakarat_ender_model_b, time_sig, oh_manager, note_dict, hi)
            song_can_end, predictions = end_if_can(predictions, makam, oh_manager, note_dict, time_sig)
            if song_can_end:
                correctly_ended = True
        except:
            print('An exc')
        hi = hi - 0.1
        if hi <= lo:
            break
    if not correctly_ended:
        print('FIXED ENDING')
        fixed_note_lst = ['la4', 'la4', 'la4', 'la4', 'rest', 'rest']
        fixed_dur_list = ['1/4', '1/8', '1/8', '1/4', '1/4', '1/8']
        if makam == 'hicaz':
            fixed_note_lst = ['la4', 'la4', 'la4', 'la4', 'rest', 'rest']
            fixed_dur_list = ['1/4', '1/8', '1/8', '1/4', '1/4', '1/8']
        predictions = fixed_ending(fixed_note_lst, fixed_dur_list, oh_manager, note_dict)
    return predictions


def main():
    # makam = 'hicaz'
    makam = 'nihavent'
    ver = 'v1'
    model_name = 'nakarat_end_' + ver
    base_model = 'sec_BW1_v101'
    set_size = 8
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)

    xs, ys = make_ending_data(makam, note_dict, oh_manager, set_size)
    train_nakarat_ending_model(makam, base_model, model_name, xs, ys, eps=15)


if __name__ == '__main__':
    main()
