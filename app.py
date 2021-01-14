from flask import Flask, render_template, url_for
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lorem_ip'
socketio = SocketIO(app)


@socketio.on('client_connected')
def handle_client_connect_event(data):
    print("yay! client connected")
    print('data is:', str(data))


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
    socketio.run(app)
