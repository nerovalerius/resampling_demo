import sys
import pip
import random

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
matplotlib.use('Qt5Agg')

import numpy as np
from scipy import signal

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QGridLayout,QPushButton, QApplication, QSlider)
from PyQt5.QtCore import Qt


class MySignal(object):
    x = None
    y = None
    color = ''
    f_s = 0

    
    def __init__(self, y, x, color, f_s):
        self.x = x
        self.y = y
        self.color = color
        self.f_s = f_s


class MplCanvas(FigureCanvasQTAgg):


    # Layout - 3 Rows, 2 Colums

    #   | Original Time Domain    | Original FFT Linear Spectrum    |
    #   | Upsampled Time Domain   | Upsampled FFT Linear Spectrum   |  
    #   | Downsampled Time Domain | Downsampled FFT Linear Spectrum |
    #   
    #   | Upsampling Slider       | Downsampling Slider             |


    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.original_signal_plot = fig.add_subplot(321)                    
        self.original_fft_plot = fig.add_subplot(322)

        self.upsampled_signal_plot = fig.add_subplot(323)                 
        self.upsampled_fft_plot = fig.add_subplot(324)

        self.downsampled_signal_plot = fig.add_subplot(325)                 
        self.downsampled_fft_plot = fig.add_subplot(326)

        super(MplCanvas, self).__init__(fig)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
  
    def initUI(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Set Window Size and Title
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Up- and Downsampling Demonstration - Armin Niedermüller')

        # Define a grid layout
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # Define the Plot with its subplots
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # Create slider
        upsample_slider = QSlider(Qt.Horizontal)
        upsample_slider.setRange(0,10)
        upsample_slider.setSingleStep(1)
        upsample_slider.setValue(1)
        upsample_slider.setTickInterval(1)
        upsample_slider.setTickPosition(QSlider.TicksBothSides)

        # Create slider
        downsample_slider = QSlider(Qt.Horizontal)
        downsample_slider.setRange(0,10)
        downsample_slider.setSingleStep(1)
        downsample_slider.setValue(1)
        downsample_slider.setTickInterval(1)
        downsample_slider.setTickPosition(QSlider.TicksBothSides)

        # Create Labels
        downsampleLabel = QtWidgets.QLabel()
        downsampleLabel.setText('Downsampling')
        upsampleLabel = QtWidgets.QLabel()
        upsampleLabel.setText('Upsampling')

        # Connect the sliders to our plots - if the slider value changes, the plot is updated
        upsample_slider.valueChanged[int].connect(self.upsamplePlot)
        downsample_slider.valueChanged[int].connect(self.downsamplePlot)

        # Create a Grid Layout and put the single widgets into it
        grid_layout.addWidget(self.canvas, 1, 1)
        grid_layout.addWidget(upsampleLabel, 2, 1)
        grid_layout.addWidget(upsample_slider, 3, 1)
        grid_layout.addWidget(downsampleLabel, 4, 1)
        grid_layout.addWidget(downsample_slider, 5, 1)


       
        # Dictionary with Functions
        self.signals = dict()

        self.show()

    def addFunction(self, y, x, color, name, f_s):
        
        mySignal = MySignal(y, x, color, f_s)

        # Add function to our list
        self.signals[name] = mySignal

    def upsamplePlot(self, value):
     
        # Upsampling
        # 0 er Array erstellen
        l_upsampling = value
        np.zeros(self.signals['square signal'].f_s * l_upsampling)

        # our function
        x = self.signals['square signal'].x
        

        # Calculate FFT
        X = np.fft.fft(x)
        freq = np.fft.fftfreq(len(x), 1/self.signals['square signal'].f_s)


        # Add zeros
        X_upsampled = np.insert(X, int(X.shape[0]/2), np.zeros(self.signals['square signal'].f_s * l_upsampling)) 
        print(X_upsampled.shape)

        # Inverse FFT
        x_upsampled = np.fft.ifft(X_upsampled)
       
        if l_upsampling != 0:
            x_upsampled *= l_upsampling
            
        # Create vector from 0 to 1 - stepsize = 1/fs
        t_upsampled = np.linspace(0, length,
                                 self.signals['square signal'].f_s
                                 + self.signals['square signal'].f_s * l_upsampling)

        upsampledSignal = MySignal(t_upsampled, x_upsampled, 'r', self.signals['square signal'].f_s)


        # Add function to our list
        self.signals['upsampled square signal'] = upsampledSignal
            
        # Calculate x values for new f>-domain
        X_x = np.linspace(- int(X_upsampled.shape[0]/2), int(X_upsampled.shape[0]/2),
                         self.signals['square signal'].f_s
                         + self.signals['square signal'].f_s * l_upsampling)

        # Clear the plot after an update
        self.canvas.upsampled_signal_plot.cla() 

        # Draw a new plot
        self.canvas.upsampled_signal_plot.plot(self.signals['upsampled square signal'].y,
                                                self.signals['upsampled square signal'].x,
                                                self.signals['upsampled square signal'].color) 

        # Trigger the canvas to update and redraw.
        self.canvas.draw()


    def downsamplePlot(self, value):
     
            # Upsampling
            # 0 er Array erstellen
            l_upsampling = value
            np.zeros(self.signals['square signal'].f_s * l_upsampling)

            # our function
            x = self.signals['square signal'].x
        

            # Calculate FFT
            X = np.fft.fft(x)
            freq = np.fft.fftfreq(len(x), 1/self.signals['square signal'].f_s)


            # Add zeros
            X_upsampled = np.insert(X, int(X.shape[0]/2), np.zeros(self.signals['square signal'].f_s * l_upsampling)) 
            print(X_upsampled.shape)

            # Inverse FFT
            x_upsampled = np.fft.ifft(X_upsampled)
       
            if l_upsampling != 0:
                x_upsampled *= l_upsampling
            
            # Create vector from 0 to 1 - stepsize = 1/fs
            t_upsampled = np.linspace(0, length,
                                     self.signals['square signal'].f_s
                                     + self.signals['square signal'].f_s * l_upsampling)

            upsampledSignal = MySignal(t_upsampled, x_upsampled, 'r', self.signals['square signal'].f_s)


            # Add function to our list
            self.signals['upsampled square signal'] = upsampledSignal
            
            # Calculate x values for new f>-domain
            X_x = np.linspace(- int(X_upsampled.shape[0]/2), int(X_upsampled.shape[0]/2),
                             self.signals['square signal'].f_s
                             + self.signals['square signal'].f_s * l_upsampling)

            self.canvas.downsampled_signal_plot.cla() 
            self.canvas.downsampled_signal_plot.plot(self.signals['upsampled square signal'].y,
                                                    self.signals['upsampled square signal'].x,
                                                    self.signals['upsampled square signal'].color) 

            # Trigger the canvas to update and redraw.
            self.canvas.draw()







def myFourierSeries(fs_signal, h_signal, len_signal, k_max_signal, frequency):
    """
    Parameters
    ----------
    fs_signal:    int,    Sampling frequency of the signal
    h_signal:     double, Amplitude
    len_signal:   int,    Signal length in seconds
    k_max_signal: int,    variable Fourier Series length

    Returns
    ---------
    f:            array[double], Fourier Series of a square signal
    """

    # Generate evenly spaced timestamps
    #x = np.linspace(0, len_signal, fs_signal, endpoint=False) 
    time = np.arange(0, len_signal, 1/fs)   # Create vector from 0 to 1 - stepsize = 1/fs
    
    # This will be our resulting signal
    f = np.zeros([len_signal])

    # Go over all samples for our signal and calculate its value via fourier series
    for x in range(0, len_signal):

      # The inner sum - see RHS of formula
      sum = 0
      for k in range(1, k_max_signal+1, 2):
        sum += k**(-1) * np.sin(2 * np.pi * k * time[x] * frequency)
            
      # The scalar in front of the sum - see RHS of formula
      f[x] =  sum * 4 * h_signal * np.pi**(-1)
    
    return f 




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





app = QtWidgets.QApplication(sys.argv)

mainWindow = MainWindow()

mainWindow.addFunction(t, x, 'b', 'square signal', fs)

app.exec_()
