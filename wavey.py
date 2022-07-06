# c. 06/30/2022
# tom brzyzek

# Audio Importing Tool.

import numpy as np
import scipy.io.wavfile as spwav
import scipy.signal as spsig
import os

class Wavey:
    def __init__(self):
        self.sample_rate = 44100                                    # sample rate of song data


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
        song_time_arr = np.arange(0, song_data.size) / song_samp_rate

        # Mono/Stereo Handling. If data is 2 rows(stereo), then take average over the 2 channels.
        try:
            song_data.shape[1]
            song_data = song_data[0,:] + song_data[1,:] / 2

        except IndexError:
            song_data = song_data


        # Resample Handling.
        if not song_samp_rate == self.sample_rate:
            resamp_data = spsig.resample(song_data,int(self.sample_rate*song_time))
            return (np.array(resamp_data), np.array(song_time_arr))

        else:
            return (np.array(song_data), np.array(song_time_arr))


    def fft(self, time_data):
        freq_data = np.fft.fft(time_data)
        return freq_data