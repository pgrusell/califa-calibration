'''
This class transforms the input, histograms inside a root file and
its calibration parameters from a .txt file into np.ndarrays that 
can be read by keras
'''

import os
import ROOT
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Unpacker:

    def __init__(self, file_name: str, n_bins: int = 250, bad_crystals: list = [], min_bins: int = 1, max_bins: int = 2000, noise=False) -> None:
        '''
        :file_name: input file name. It has to be inside the data folder
        which has to contain both file_in.txt and file_in.root.
        :n_points: number of bins in the common spectra
        '''

        self.file_name = file_name
        self.n_bins = n_bins

        self.min_bins = min_bins
        self.max_bins = max_bins

        self.noise = noise

        self.bad_crystals = bad_crystals

    def make(self, is_y_data: str = True):
        '''
        Method to unpack the data
        '''

        # Dictionary to store the information
        self.db = {}

        # Get file paths
        self._get_file_paths(is_y_data)

        # Load x-dataset
        self._load_x_data()

        # Load y-dataset
        if is_y_data:
            self._load_y_data()

        # Get the final datasets (and the un-normalization parameters)
        self._normalize_data(is_y_data)

        # Save new x-data values in the db
        for i, (key, val) in enumerate(self.db.items()):
            self.db[key]['x-data'] = self.x_data[i]
            if is_y_data:
                self.db[key]['y-data'] = self.y_data[i]

    def _get_file_paths(self, is_y_data: str = True) -> None:
        '''
        This method finds the absolute path of the .root and .txt file with
        the name file_name in the data directory
        '''
        data_directory = os.path.abspath("../data")
        data_files = os.listdir(data_directory)

        for file in data_files:
            if file.startswith(self.file_name):
                if is_y_data:
                    if file.endswith('.txt'):
                        self.y_data_path = f'{data_directory}/{file}'
                if file.endswith('.root'):
                    self.x_data_path = f'{data_directory}/{file}'

    def _load_x_data(self) -> None:
        '''
        This method saves the spectra (bins and counts) for each crystal.
        '''
        file = ROOT.TFile.Open(self.x_data_path, "READ")

        for key in file.GetListOfKeys():
            h_title = str(key).split(' ')[1]

            # if is an histogram
            if h_title.startswith('fh'):

                if h_title.startswith('fh1'):

                    crystal_id = int(h_title[23:])

                else:

                    crystal_id = int(h_title[22:])

                if self.bad_crystals.count(crystal_id) > 0:
                    continue

                if crystal_id > 2544:
                    continue

                histo = file.Get(h_title)

                x = []
                count = []

                maxBins = np.min([histo.GetNbinsX(), self.max_bins])
                # for i in range(1, int(histo.GetNbinsX())):
                for i in range(self.min_bins, maxBins):
                    x.append(histo.GetBinCenter(i))
                    count.append(histo.GetBinContent(i))

                self.db[crystal_id] = {'x-data': np.array([x, count])}

    def _load_y_data(self) -> None:
        '''
        This method takes the position of the centroids in channels
        for each crystal
        '''
        data = np.loadtxt(self.y_data_path)

        for line in data:
            crystal_id = line[0]

            if self.bad_crystals.count(crystal_id) > 0:
                continue

            if crystal_id > 2544:
                continue

            mean_1 = line[1]
            mean_2 = line[3]

            self.db[crystal_id]['y-data'] = np.array([mean_1, mean_2])

    def _normalize_data(self, is_y_data: str = True) -> None:
        '''
        This method will transform the data of the spectrum to make it
        gain-independent. 
        '''

        n_samples = len(self.db)
        common_bins = np.linspace(0, 1, self.n_bins)
        norm = np.zeros([n_samples, 2])
        x_data = np.zeros([n_samples, self.n_bins])
        y_data = np.zeros([n_samples, 2])

        for i, (key, val) in enumerate(self.db.items()):

            bins, counts = val['x-data']
            if is_y_data:
                y = val['y-data']

            # Normalize the bins
            bins_min = bins.min()
            bins_max = (bins - bins_min).max()
            bins_norm = np.copy((bins - bins_min) / bins_max)

            # Create an interpolation function
            f_interp = sc.interpolate.interp1d(bins_norm, counts)

            if self.noise:
                x_data[i][:] = np.array([f_interp(
                    bini) + np.random.normal(0, 2*np.sqrt(abs(f_interp(bini)))) for bini in common_bins])
            else:
                x_data[i][:] = np.array([f_interp(bini)
                                        for bini in common_bins])
            norm[i][:] = bins_min, bins_max  # store for un-normalization

            if is_y_data:
                y_data[i][:] = np.copy((y - bins_min) / bins_max)

        self.x_data = x_data
        if is_y_data:
            self.y_data = y_data
        self.norm = norm
        self.common_bins = common_bins

    def display_data(self) -> Figure:
        '''
        Function to show a plot of the spectra with its maxima
        '''

        fig, axs = plt.subplots(4, 5)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        for ax in axs.flatten():

            i = np.random.randint(0, len(self.x_data))

            ax.plot(self.common_bins, self.x_data[i])
            ax.vlines(self.y_data[i][0], 0, 5000, linestyle='--', color='grey')
            ax.vlines(self.y_data[i][1], 0, 5000, linestyle='--', color='grey')
            ax.set_yticks([], [])
            ax.set_xticks([], [])

        plt.show()
