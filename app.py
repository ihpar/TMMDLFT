from flask import Flask, render_template, url_for
from flask_socketio import SocketIO, send, emit
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lorem_ip'
socketio = SocketIO(app)


@socketio.on('start_composing')
def start_composing(data):
    makam = data['makam']
    notes = data['notes']
    print(makam)
    for combo in notes:
        parts = combo.split(':')
        note = parts[0]
        dur = parts[1]
        print(note, dur)

    emit('composition_status', {'det': 1})
    time.sleep(4)
    emit('composition_status', {'ket': ['a', 2, 3.14]})
    time.sleep(4)
    emit('composition_status', {'suc': {'met': -3}})


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
