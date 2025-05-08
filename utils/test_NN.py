import keras
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import calibrate_NN


class Test:

    def __init__(self, calibration: calibrate_NN.Calibration, x_test: np.ndarray, y_test: np.ndarray) -> None:

        self.calibration = calibration
        self.model = calibration.model
        self.x_test = x_test
        self.y_test = y_test
        self.y_test

        # Make the predictions
        self.y_pred = calibration.predict_value(x_test)

    def make_test(self):

        self.loss_test()
        self.plot_results()
        self.plot_error()

    def loss_test(self) -> str:
        test_loss = self.model.evaluate(self.x_test, self.y_test)
        print(f'Lost on the test set is: {test_loss:.6f}')

    def plot_results(self) -> Figure:
        '''
        This plot will represent some of the x_data examples with the predicted
        peaks and the true peaks
        '''
        fig, axs = plt.subplots(4, 3)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        for ax in axs.flatten():
            i = np.random.randint(0, len(self.x_test))
            x = self.x_test[i]
            y = self.y_test[i]

            ax.plot(self.calibration.bins, x)
            ax.vlines(self.y_pred[i][0], 0, max(x),
                      linestyle='--', color='grey')
            ax.vlines(self.y_pred[i][1], 0, max(x),
                      linestyle='--', color='grey')

            ax.vlines(y[0], 0, max(x),
                      linestyle='--', color='green')
            ax.vlines(y[1], 0, max(x),
                      linestyle='--', color='red')

            ax.set_yticks([])
            ax.set_xticks([])

    def plot_error(self) -> Figure:
        '''
        This method will study the error between the prediction for each peak
        and its real value
        '''

        error = self.y_pred - self.y_test
        m_error = np.mean(error, axis=0)
        std_error = np.std(error, axis=0)

        fig, ax = plt.subplots()

        for i in [0, 1]:

            ax.hist(error[:, i], bins=np.linspace(-1, 1, 100),
                    label=f'mean = {m_error[i]: .3f}\nstd = {std_error[i]: .3f}', edgecolor='black', alpha=0.3)

        ax.legend()

    def plot_correlation(self) -> Figure:
        '''
        This method plots the correlation between the predicted peaks and the real
        ones in order to avoid underfitting
        '''

        fig, ax = plt.subplots()

        for i in [0, 1]:
            ax.hist2d(self.y_pred[:, i], self.y_test[:, i], bins=(
                np.linspace(0, 1, 20), np.linspace(0, 1, 20)))
