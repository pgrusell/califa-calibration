'''
This class will implement a NN that, given a 60-Co spectra, would be
able to identify the peaks in channels with the same precision as if
it was fitted to a double Gaussian plus linear background.

The input data should be:

:x: Number of counts in every bin. The binning is assumed to be normalized
and homogeneously sampled. To do this from a .root file used the upacker.py
module.

:y: This contains the normalized position of the peaks in channels of the
60-Co.

:norm: This contains the factors that scaled the bins
for each crystal and can be used to predict the actual value of the
centroids in the same units as they are in CALIFA.
'''

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Dense, BatchNormalization, LeakyReLU, Input, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@tf.keras.utils.register_keras_serializable()
def peak_loss(y_true, y_pred):
    sorted_true = tf.sort(y_true, axis=1)
    sorted_pred = tf.sort(y_pred, axis=1)
    error = sorted_true - sorted_pred
    penalty = tf.reduce_mean(tf.where(tf.abs(error) < 0.05,
                                      0.5 * tf.square(error),
                                      0.05 * (tf.abs(error) - 0.025)))
    order_penalty = tf.reduce_mean(tf.nn.relu(
        sorted_pred[:, 0] - sorted_pred[:, 1]))
    return penalty + 0.1 * order_penalty


class Calibration:

    def __init__(self, x: np.ndarray = None, y: np.ndarray = None, norm: np.ndarray = None, epochs: int = 0) -> None:

        self.x = x
        self.y = y

        if np.any(self.y):
            self.y.sort(axis=1)
            self.n_out_neurons = len(y[0])

        if np.any(self.x):
            self.n_input_neurons = len(x[0])
            self.bins = np.linspace(0, 1, self.n_input_neurons)
        else:
            self.bins = np.linspace(0, 1, 250)

        self.norm = norm
        self.epochs = epochs

        # Normalize x-data
        self.x = self.x / np.max(self.x, axis=1, keepdims=True)

    def load_model(self, model_path: str = 'model.keras'):
        self.model = load_model(model_path)

    def make(self):
        self._build_model()
        self._build_test_set()
        self.train_model(self.x_train, self.y_train, self.epochs)

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.n_input_neurons, 1)),

            Conv1D(64, kernel_size=5, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.05),
            MaxPooling1D(pool_size=2),

            LSTM(64, return_sequences=True),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.05),

            Conv1D(128, kernel_size=3, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.05),
            MaxPooling1D(pool_size=2),

            LSTM(32),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.05),

            Dense(128), LeakyReLU(), BatchNormalization(), Dropout(0.2),
            Dense(64), LeakyReLU(), BatchNormalization(), Dropout(0.2),
            Dense(2)
        ])

        model.compile(optimizer='adam', loss=peak_loss)
        self.model = model

    def _build_test_set(self, test_rate: float = 0.15):
        '''
        Method to split the input data set in train and test data sets
        '''

        x, y, norm = np.copy(self.x), np.copy(self.y), np.copy(self.norm)
        n_samples = len(x)
        indxs = np.random.permutation(n_samples)

        split_ndx = int(test_rate * n_samples)

        self.x_train = x[indxs][:split_ndx]
        self.y_train = y[indxs][:split_ndx]
        self.norm_train = norm[indxs][:split_ndx]

        self.x_test = x[indxs][split_ndx:]
        self.y_test = y[indxs][split_ndx:]
        self.norm_test = norm[indxs][split_ndx:]

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, model_path: str = 'model.keras'):

        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=1e-6)

        early_stop = EarlyStopping(
            monitor='loss', patience=20, restore_best_weights=True)

        self.model.fit(self.x_train, self.y_train, epochs=self.epochs,
                       batch_size=32, callbacks=[reduce_lr])

        self.model.save(model_path)

    def _gaussian(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    def _refine_peak(self, x, y, pred_pos, window=0.03):
        '''
        This method will refine the prediction from the NN by just fitting the
        predicted peak to a window of the spectra near it
        '''
        mask = (x > pred_pos - window) & (x < pred_pos + window)
        x_fit = x[mask]
        y_fit = y[mask]

        if len(y_fit) > 5:
            y_fit = savgol_filter(y_fit, window_length=5, polyorder=2)

        if len(x_fit) < 5:
            return pred_pos

        try:
            popt, _ = curve_fit(self._gaussian, x_fit, y_fit, p0=[
                                max(y_fit), pred_pos, 0.01])
            return popt[1]
        except:
            return pred_pos

    def _refine_prediction(self, x, y_pred):
        return np.array([
            self._refine_peak(self.bins, x, y_pred[0]),
            self._refine_peak(self.bins, x, y_pred[1])
        ])

    def predict_value(self, x: np.ndarray) -> np.ndarray:

        # Normalize the data
        x = x / np.max(x, axis=1, keepdims=True)

        if len(x.shape) == 1:  # just predict over one point
            x = np.expand_dims(x, axis=0)

        y_pred = self.model.predict(x, verbose=False)
        y_pred_refined = np.zeros_like(y_pred)

        for i, (xi, yi) in enumerate(zip(x, y_pred)):
            y_pred_refined[i] = self._refine_prediction(xi, yi)

        # Sort the values for them to be position of the first peak,
        # position of the second peak.
        y_pred_refined.sort(axis=1)

        return y_pred_refined
