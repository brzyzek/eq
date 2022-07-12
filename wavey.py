# c. 06/30/2022
# tom brzyzek

# Audio Importing Tool.

import numpy as np
import scipy.io.wavfile as spwav
import scipy.signal as spsig
import matplotlib
import matplotlib.pyplot as plt
import os

class Wavey:
    def __init__(self):
        self.sample_rate = 44100                                    # sample rate of song data
        # self.timefig = 0
        # self.timeax = 0
        # self.timefig = 0
        # self.freqax = 0


    def import_song(self, filename):
        song_path = filename
        song_samp_rate, song_data = spwav.read(song_path)
        
        # Matrix Transpose Handling:
        try:
            y = song_data[0,:].size
            if song_data[0,:].size == 1 or song_data[0,:].size == 2:
                song_data = np.transpose(song_data)

        except IndexError:
            song_data = song_data

        song_time = song_data.size / song_samp_rate

        # Mono/Stereo Handling. If data is 2 rows(stereo), then take average over the 2 channels.
        try:
            song_data.shape[1]
            song_data = song_data[0,:] + song_data[1,:] / 2

        except IndexError:
            song_data = song_data


        # Resample Handling.
        if not song_samp_rate == self.sample_rate:
            resamp_data = spsig.resample(song_data,int(self.sample_rate*song_time))
            song_time_arr = np.arange(0, resamp_data.size) / song_samp_rate
            return (np.array(resamp_data), np.array(song_time_arr))

        else:
            song_time_arr = np.arange(0, song_data.size) / song_samp_rate
            return (np.array(song_data), np.array(song_time_arr))


    def fft(self, time_data):
        freq_data = np.fft.rfft(time_data)
        freq_arr = np.fft.rfftfreq(time_data.size, d=1./self.sample_rate)
        return freq_data, freq_arr

    def init_time_plot(self):
        self.timefig = matplotlib.figure.Figure(figsize=(6.677,1.468), layout='constrained')
        self.timeax = self.timefig.figure.subplots()
        self.format_plot(self.timefig, self.timeax)
        return self.timefig

    def update_time_plot(self, time_data, time_samples):
        self.timeax.cla()
        self.format_plot(self.timefig, self.timeax)
        self.timeax.plot(time_samples, time_data, label="original time data")
        self.timeax.set_xlim([0,time_samples[-1]])

    def init_freq_plot(self):
        self.freqfig = matplotlib.figure.Figure(figsize=(6.677,1.468), layout='constrained')
        self.freqax = self.freqfig.figure.subplots()
        self.format_plot(self.freqfig, self.freqax)
        return self.freqfig

    def update_freq_plot(self, freq_data, freq_samples):
        self.freqax.cla()
        self.format_plot(self.freqfig, self.freqax)
        self.freqax.plot(freq_samples, freq_data, label="original freq data")
        self.freqax.set_xlim([0,freq_samples[-1]])
        
    def format_plot(self, figure, axis):
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.yaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        axis.xaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(40.0))
        axis.grid(True)
        figure.set_facecolor('silver')

    def reset_plot(self, figure, axis):
        axis.cla()
        self.format_plot(figure, axis)
        



# x = Wavey()
# fileURL = "./songs/africa-toto.wav"
# time_signal, time_sample = x.import_song(fileURL)
# freq_signal, freq_sample = x.fft(time_signal)
# x.time_plot(time_signal, time_sample)
# x.freq_plot(freq_signal, freq_sample)