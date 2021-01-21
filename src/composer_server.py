import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from parts_composer import compose_zemin, compose_nakarat, compose_meyan, song_2_mus
import eventlet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lorem_ip_man'
socketio = SocketIO(app, async_mode='eventlet')
eventlet.monkey_patch()


@socketio.on('start_composing')
def start_composing(data):
    try:
        makam = data['makam']
        notes = data['notes']

        socketio.sleep(0)
        res = compose_zemin(makam, notes)
        if res['type'] == 'error':
            emit('composition_status', {'type': 'error', 'section': 'zemin'})
            return

        emit('composition_status', {'type': 'composed', 'section': 'zemin'})

        makam = res['makam']
        dir_path = res['dir_path']
        set_size = res['set_size']
        measure_cnt = res['measure_cnt']
        note_dict = res['note_dict']
        oh_manager = res['oh_manager']
        time_sig = res['time_sig']
        part_a = res['part_a']

        socketio.sleep(0)
        res = compose_nakarat(makam, dir_path, set_size, measure_cnt, note_dict, oh_manager, time_sig, part_a)
        if res['type'] == 'error':
            emit('composition_status', {'type': 'error', 'section': 'nakarat'})
            return

        emit('composition_status', {'type': 'composed', 'section': 'nakarat'})

        part_b = res['part_b']
        second_rep = res['second_rep']

        socketio.sleep(0)
        res = compose_meyan(makam, dir_path, set_size, measure_cnt, note_dict, oh_manager, time_sig, part_b)
        if res['type'] == 'error':
            emit('composition_status', {'type': 'error', 'section': 'meyan'})
            return

        emit('composition_status', {'type': 'composed', 'section': 'meyan'})

        part_c = res['part_c']

        song = np.append(part_a, part_b, axis=1)
        song = np.append(song, part_c, axis=1)

        socketio.sleep(0)
        song_2_mus(song, makam, 'song_name', oh_manager, note_dict, time_sig, '4,8,12', second_rep, to_browser=True)

        emit('composition_status', {'type': 'composed', 'section': 'all'})

    except Exception as e:
        emit('composition_status', {'type': 'error', 'section': 'start_composing', 'msg': str(e)})


@socketio.on('disconnect')
def disconnected():
    print('disconnected')


@socketio.on('connect')
def connected():
    print('connected')


@socketio.on('message')
def handle_message(msg):
    print('message:', msg)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, debug=True)
