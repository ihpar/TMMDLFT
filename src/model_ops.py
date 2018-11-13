import os

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop


def build_model(in_shape, out_shape):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def save_model(makam, model_name, model):
    path = os.path.join(os.path.abspath('..'), 'models', makam)
    json_path = os.path.join(path, model_name + '.json')
    w_path = os.path.join(path, model_name + '.h5')

    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(w_path)
    print("Saved model to disk")


def load_model(makam, model_name):
    path = os.path.join(os.path.abspath('..'), 'models', makam)
    json_path = os.path.join(path, model_name + '.json')
    w_path = os.path.join(path, model_name + '.h5')

    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(w_path)
    print(f'Model loaded from {json_path} and {w_path}')
    optimizer = RMSprop(lr=0.01)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return loaded_model
