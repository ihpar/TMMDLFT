import os

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizers import RMSprop, SGD


def build_model(in_shape, out_shape):
    model = Sequential()
    '''
    v 41
    model.add(LSTM(256, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)
    '''
    # v. 44, 45, 46
    '''
    model.add(LSTM(512, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)
    '''
    '''
    # v. 47
    model.add(LSTM(256, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)
    '''
    '''
    # v. 48
    model.add(LSTM(256, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)
    '''

    '''
    # v. 49
    model.add(LSTM(512, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)
    '''
    '''
    # v. 51
    model.add(LSTM(500, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(out_shape))
    model.add(Activation('sigmoid'))
    optimizer = RMSprop(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    '''
    # v. 62
    model = Sequential()
    model.add(LSTM(600, return_sequences=True, input_shape=in_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(600, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(out_shape))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
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
    print(f'Model loaded from {json_path}')
    optimizer = RMSprop(lr=0.001)
    loaded_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return loaded_model


def build_whole_model(x_shape, y_shape):
    '''
    # v 50
    model = Sequential()
    model.add(LSTM(500, return_sequences=True, input_shape=x_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(y_shape))
    model.add(Activation('sigmoid'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    '''
    '''
    # v 60
    model = Sequential()
    model.add(LSTM(420, return_sequences=True, input_shape=x_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(420, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(420, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(420, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    '''
    '''
    # v 61
    model = Sequential()
    model.add(LSTM(600, return_sequences=True, input_shape=x_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(600, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    '''

    # v 70
    model = Sequential()
    model.add(LSTM(600, return_sequences=True, input_shape=x_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(600, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(y_shape))
    # model.add(Activation('softmax'))
    optimizer = SGD(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model
