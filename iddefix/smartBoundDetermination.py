#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: MaltheRaschke
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

sys.path.append('../garfield/')

class SmartBoundDetermination:

    def __init__(self, frequency_data, impedance_data, minimum_peak_height=2):
        self.frequency_data = frequency_data
        self.impedance_data = impedance_data
        self.minimum_peak_height = minimum_peak_height
        self.peaks = None
        self.peaks_height = None
        self.minus_3dB_points = None
        self.upper_lower_bounds = None
        self.parameterBounds = self.run()

    def find(self, frequency_data=None, impedance_data=None, minimum_peak_height=None):
        """
        This function determines the bounds for the fitting parameters. The bounds are determined by the impedance data
        and the frequency data.

        Parameters
        ----------
        frequency_data : numpy array
            Array containing the frequency data [Hz].
        impedance_data : numpy array
            Array containing the impedance data [Ohm].
        minimum_peak_height : float
            Minimum peak height for the peak finding algorithm.

        Returns
        -------
        parameterBounds : list
            List containing the bounds for the fitting parameters.
        """

        # Use instance attributes if no arguments are provided
        if frequency_data is None:
            frequency_data = self.frequency_data
        if impedance_data is None:
            impedance_data = self.impedance_data
        if minimum_peak_height is None:
            minimum_peak_height = self.minimum_peak_height


        # Find the peaks of the impedance data
        peaks, peaks_height = find_peaks(impedance_data, height=minimum_peak_height, threshold=None)

        # Store peaks and peaks_height as instance attributes
        self.peaks = peaks
        self.peaks_height = peaks_height

        Nres = len(peaks)
        initial_Qs = np.zeros(Nres)
        self.minus_3dB_points = np.zeros(Nres)
        self.upper_lower_bounds = np.zeros(Nres)

        for i, (peak, height) in enumerate(zip(peaks, peaks_height['peak_heights'])):
            minus_3dB_point = height * np.sqrt(1/2)
            self.minus_3dB_points[i] = minus_3dB_point
            idx_crossings = np.argwhere(np.diff(np.sign(impedance_data - minus_3dB_point))).flatten()

            upper_lower_bound = np.min(np.abs(frequency_data[idx_crossings] - frequency_data[peak]))
            self.upper_lower_bounds[i] = upper_lower_bound

            initial_Qs[i] = frequency_data[peak]/(upper_lower_bound*2)
            
        parameterBounds = []
        
        for i in range(Nres):
            # Add the fixed bounds
            Rs_bounds = (peaks_height['peak_heights'][i]*0.8, peaks_height['peak_heights'][i]*10)
            Q_bounds = (initial_Qs[i]/2 , initial_Qs[i]*5)
            freq_bounds = (frequency_data[peaks[i]]-0.01e9, frequency_data[peaks[i]]+0.01e9)
        
            parameterBounds.extend([Rs_bounds, Q_bounds, freq_bounds])

        return parameterBounds
    
    def inspect(self):
        plt.figure()
        plt.plot(self.frequency_data, self.impedance_data)

        if self.peaks is not None:   
            for i , (peak, minus_3dB_point, upper_lower_bound) in enumerate(zip(self.peaks, self.minus_3dB_points, self.upper_lower_bounds)):
                plt.plot(self.frequency_data[peak], self.impedance_data[peak], 'x', color='black')
                plt.vlines(self.frequency_data[peak], ymin=minus_3dB_point, ymax=self.impedance_data[peak], color='r', linestyle='--')
                plt.hlines(minus_3dB_point, xmin=self.frequency_data[peak] - upper_lower_bound, xmax=self.frequency_data[peak] + upper_lower_bound, color='g', linestyle='--')
                plt.text(self.frequency_data[peak], self.impedance_data[peak], f'#{i}', fontsize=9)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Impedance [Ohm]')
        plt.title('Smart Bound Determination')
        plt.show()
        
        return None