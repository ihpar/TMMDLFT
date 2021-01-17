from flask import Flask, render_template, url_for
from flask_socketio import SocketIO, send, emit

from parts_composer import gui_composer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lorem_ip'
socketio = SocketIO(app)


@socketio.on('start_composing')
def start_composing(data):
    try:
        makam = data['makam']
        notes = data['notes']

        emit('composition_status', {'type': 'status', 'msg': gui_composer(makam, notes)})
    except Exception as e:
        emit('composition_status', {'type': 'error', 'msg': str(e)})


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
