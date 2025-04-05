import numpy as np
import pandas as pd

import keras
from keras import optimizers
from keras.models import Sequential
# from keras.layers import Activation, Dense,Dropout
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler


class MyLSTM:
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, args):
        self.model = Sequential()
        self.is_model_created = False
        self.hidden_dim = args.hidden_dim
        self.epochs = args.epochs


    def create_model(self, sequence_length=20, feature_count=8):

        # 1D Convolution to extract local patterns
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, feature_count)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))

        # LSTM to capture sequence-level patterns
        self.model.add(LSTM(self.hidden_dim, return_sequences=False))
        self.model.add(Dropout(0.2))

        # Dense layer for binary classification
        self.model.add(Dense(1))

        self.model.compile(
            loss='mean_squared_error',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

    def fit(self, X, y):
        # data_x = np.array(data_x)
        # train_x = data_x[:, 1:-1]
        # train_y = data_x[:, -1]
        train_x = X
        train_y = y

        if self.is_model_created == False:
            self.create_model(train_x.shape[1])
            self.is_model_created = True

        n_samples, timesteps, n_features = train_x.shape
        train_x_2d = train_x.reshape(n_samples * timesteps, n_features)

        train_x_2d = self.sc_in.fit_transform(train_x_2d)

        train_x = train_x_2d.reshape(n_samples, timesteps, n_features)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        # train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        print("train_X shape:", train_x.shape)
        self.model.fit(train_x, train_y, epochs=self.epochs, verbose=1, shuffle=False, batch_size=50)

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x = self.sc_in.transform(test_x)
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        pred_y = self.model.predict(test_x)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y

