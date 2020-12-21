import sys
import pip
from resampling_header import *
import numpy as np
from PyQt5.QtWidgets import QApplication

# MAIN

# create square wave signal
length = 1           # seconds
fs = 100             # Hz - Sampling Frequency
f = 5                # Hz - Frequency of the signal
amplitude = 1        # Amplitude (units e.g Volts)
k_max_signal = 7    # The "depth" of the Fourier Series

# Create vector from 0 to 1 - stepsize = 1/fs
# Calculate evenly spaced numbers over a specified interval.
t = np.linspace(0, length, fs, endpoint=False)

# Fourier series of a square signal
x = myFourierSeries(fs, amplitude, t.shape[0], k_max_signal, f)

# Our Application
app = QtWidgets.QApplication(sys.argv)
mainWindow = MainWindow()

# Add a function to our Application - y, x, color, name, sample frequency, duration in seconds
mainWindow.addFunction(t, x, 'r', 'square signal', fs, length)

app.exec_()
