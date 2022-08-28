# c. 06/30/2022
# tom brzyzek

# Waveform Generation and Adjustment Toolset

from ast import Pass
import numpy as np
import scipy.io.wavfile as spwav
import scipy.signal as spsig
import matplotlib
import matplotlib.pyplot as plt
import audio_dspy as adsp
from playsound import playsound
import multiprocessing

class Signal:
    def __init__(self):
        self.sample_rate = 0                                   # sample rate of song data


    def import_song(self, filename):
        song_path = filename
        song_samp_rate, song_data = spwav.read(song_path)
        self.sample_rate = song_samp_rate
        
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

        song_time_arr = np.arange(0, song_data.size) / song_samp_rate
        return (np.array(song_data), np.array(song_time_arr))

    def export_song(self, song_data, filename):
        spwav.write(filename, self.sample_rate, song_data)


    def fft(self, time_data):
        freq_data = np.fft.rfft(time_data)
        freq_arr = np.fft.rfftfreq(time_data.size, d=1./self.sample_rate)
        return freq_data, freq_arr

    def normalize(self, data):
        return data / np.max(np.abs(data))

    def play_audio(self, audio_array):
        spwav.write('output.wav', self.sample_rate, audio_array)
        self.p = multiprocessing.Process(target = playsound, args=('output.wav',))
        self.p.start()
    
    def stop_audio(self):
        spwav.write('output.wav', self.sample_rate, np.array([]))
        self.p = multiprocessing.Process(target = playsound, args=('output.wav',))
        self.p.start()
        self.p.terminate()
        
class SignalPlot(Signal):
    def __init__(self):
        Pass
        
    def init_time_plot(self):
        self.timefig = matplotlib.figure.Figure(figsize=(6.677,1.468), layout='constrained')
        self.timeax = self.timefig.figure.subplots()
        self.format_plot(self.timefig, self.timeax)
        return self.timefig

    def update_time_plot(self, time_data, time_samples, resetFlag=True):
        if resetFlag == True:
            self.timeax.cla()
        self.format_plot(self.timefig, self.timeax)
        self.timeax.plot(time_samples, time_data, label="original time data", alpha=0.5)
        self.timeax.set_xlim([0,int(time_samples[-1])])

    def init_freq_plot(self):
        self.freqfig = matplotlib.figure.Figure(figsize=(6.677,1.468), layout='constrained')
        self.freqax = self.freqfig.figure.subplots()
        self.format_plot(self.freqfig, self.freqax)
        return self.freqfig

    def update_freq_plot(self, freq_data, freq_samples, resetFlag=True):
        if resetFlag == True:
            self.freqax.cla()
        self.format_plot(self.freqfig, self.freqax)
        self.freqax.set_xscale('log')
        self.freqax.plot(freq_samples, freq_data, label="original freq data", alpha=0.5)
        self.freqax.set_xlim([1,int(freq_samples[-1])])
        
    def format_plot(self, figure, axis):
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.yaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        axis.xaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(40.0))
        axis.grid(True)
        figure.set_facecolor('silver')

    def init_filt_pz_plot(self):
        self.filtfig = matplotlib.figure.Figure(figsize=(1,1), layout='constrained')
        self.filtax = self.filtfig.figure.subplots()
        
        self.filtax .yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.filtax .yaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        self.filtax .xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.filtax .xaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        self.filtax .grid(True)
        self.filtfig.set_facecolor('silver')
        self.filtax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

        circ = matplotlib.patches.Circle((0, 0), radius=1, fill=False, ls='dashed')
        self.filtax.add_patch(circ)
        return self.filtfig

    def update_filt_pz_plot(self, pole_data, zero_data, resetFlag=True):
        if resetFlag == True:
            self.filtax.cla()
        pole_x = np.real(pole_data)
        pole_y = np.imag(pole_data)
        zero_x = np.real(zero_data)
        zero_y = np.imag(zero_data)

        self.filtax .yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.filtax .yaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        self.filtax .xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.filtax .xaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(10.0))
        self.filtax .grid(True)
        self.filtfig.set_facecolor('silver')
        self.filtax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

        circ = matplotlib.patches.Circle((0, 0), radius=1, fill=False, ls='solid')
        self.filtax.add_patch(circ)
        self.filtax.scatter(pole_x, pole_y, label="Poles", marker='x', alpha=0.5)
        self.filtax.scatter(zero_x, zero_y, label="Zeros", marker='o', alpha=0.5)


    def init_filt_mag_plot(self):
        self.filtfig = matplotlib.figure.Figure(figsize=(1,1), layout='constrained') #DEVC: determine figsize
        self.filtax = self.filtfig.figure.subplots()
        
        self.format_plot(self.filtfig, self.filtax)
        return self.filtfig

    def update_filt_mag_plot(self, w, h, sample_rate, resetFlag=True):
        if resetFlag == True:
            self.filtax.cla()
        self.format_plot(self.filtfig, self.filtax)

        self.filtax.set_xscale('log')
        self.filtax.plot(w*sample_rate/np.pi, np.abs(h), label="Magnitude", alpha=0.5)
   
        # self.filtax.set_xlim([1,int(w[-1])]) DEVC: Need to add a scaling factor to w so that we mimic the correct frequencies. 

    def init_filt_phase_plot(self):
        self.filtfig = matplotlib.figure.Figure(figsize=(1,1), layout='constrained') #DEVC: determine figsize
        self.filtax = self.filtfig.figure.subplots()
        
        self.format_plot(self.filtfig, self.filtax)
        return self.filtfig

    def update_filt_phase_plot(self, w, h, sample_rate, resetFlag=True):
        if resetFlag == True:
            self.filtax.cla()
        self.format_plot(self.filtfig, self.filtax)

        self.filtax.set_xscale('log')
        self.filtax.plot(w*sample_rate/np.pi, np.angle(h), label="Phase", alpha=0.5)
        # self.filtax.set_xlim([1,int(w[-1])]) DEVC: Need to add a scaling factor to w so that we mimic the correct frequencies. 

    def reset_plot(self, figure, axis):
        axis.cla()
        self.format_plot(figure, axis)

class EQ:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.EQ = adsp.EQ(self.sample_rate)
        # self.filter_arr = []
        self.pole_arr = []
        self.zero_arr = []
        self.w = []
        self.h = []

    def add_filter(self, type, fc, Q = 1, gain = 0):
        if gain == 0:
            Pass
        else:
            match type:
                case 'bell':
                    self.EQ.add_bell(fc, Q, gain)
                case 'LPF':
                    self.EQ.add_LPF(fc, Q)
                case 'HPF':
                    self.EQ.add_HPF(fc, Q)
                case 'lowshelf':
                    self.EQ.add_lowshelf(fc, Q, gain)
                case 'highshelf':
                    self.EQ.add_highshelf(fc, Q, gain)
                case 'notch':
                    self.EQ.add_notch(fc, Q)
                case _:
                    raise ValueError(type)

    def add_filter_by_coeffs(self, b_coeffs, a_coeffs):
        if np.size(b_coeffs) != np.size(a_coeffs):
            raise ValueError
        
        order = np.size(b_coeffs - 1)
        filt = adsp.Filter(order, self.sample_rate)
        filt.set_coefs(b_coeffs, a_coeffs)
        self.EQ.add_filter(filt)

    def process(self, signal):
            return self.EQ.process_block(signal)
    
    def remove_filter(self, id):
        self.EQ.filters.pop(id)

    def reset(self):
        self.EQ.reset()
        self.EQ.filters = []
        # self.filter_arr = []
        self.pole_arr = []
        self.zero_arr = []
        self.w = []
        self.h = []


    def get_poles_zeros(self):
        for filter in self.EQ.filters:
            # self.filter_arr.append(filter)
            b_coefs = filter.b_coefs
            self.zero_arr.append(np.roots(b_coefs))
            a_coefs  = filter.a_coefs
            self.pole_arr.append(np.roots(a_coefs))
        self.zero_arr = np.concatenate(self.zero_arr, axis=0 )
        self.pole_arr = np.concatenate(self.pole_arr, axis=0 )

    def get_impulse_response(self):
        for filter in self.EQ.filters:
            b_coefs = filter.b_coefs
            a_coefs  = filter.a_coefs
            filt_w, filt_h = spsig.freqz(b_coefs, a_coefs , worN=512)
            if self.w == []:
                self.w = filt_w
            else:
                self.w = self.w * filt_w

            if self.h == []:
                self.h = filt_h
            else:
                self.h = self.h * filt_h

            

        

# Preliminary Test Block
# x = Signal(44100)
# fileURL = "./songs/starwars_intro_3-star_wars.wav"
# time_signal, time_sample = x.import_song(fileURL)
# freq_signal, freq_sample = x.fft(time_signal)
# # x.time_plot(time_signal, time_sample)
# # x.freq_plot(freq_signal, freq_sample)

# q = EQ(44100)
# # Notes: Gain in Linear Form. Calculate dB from it. (6 dB -> 2 -6dB -> .5)
# Q = 4.5
# gaindB= 1
# gain = 10**(gaindB/20)
# q.add_filter('LPF', 5000, 4.25, 1)
# y = q.process(time_signal)
# q.EQ.plot_eq_curve()
# plt.show()

# x.play_audio(y)

# x = EQ(44100)
# # Notes: Gain in Linear Form. Calculate dB from it. (6 dB -> 2 -6dB -> .5)
# Q = 4.5
# gaindB= 1
# gain = 10**(gaindB/20)
# print(gaindB)
# # x.add_filter('lowshelf', 100, 40.25, gain)
# x.add_filter('bell', 50, 4.25, gain)
# # x.add_filter('bell', 2000, 4.25, gain)
# # x.add_filter('bell', 200, 4.25, gain)
# # x.EQ.print_eq_info()
# # x.reset()
# # x.add_filter('bell', 200, 4.25, gain)
# # t = np.arange(0, 99)
# # N = 1024
# # z = np.random.rand(N) * 2 - 1
# # x.reset()
# # y = x.process(z)
# # y_fft = np.fft.fft(y)
# # x.remove_filter(2)
# # print(z)
# # print(y)
# # plt.plot(np.abs(np.fft.rfft(z)))
# # plt.plot(np.abs(np.fft.rfft(y)))
# x.EQ.plot_eq_curve()
# plt.show()

#  PLOT TESTING (FILTER)
# x = Signal()
# fileURL = "./songs/starwars_intro_3-star_wars.wav"
# time_signal, time_sample = x.import_song(fileURL)
# freq_signal, freq_sample = x.fft(time_signal)
# # x.time_plot(time_signal, time_sample)
# # x.freq_plot(freq_signal, freq_sample)

# q = EQ(x.sample_rate)
# # Notes: Gain in Linear Form. Calculate dB from it. (6 dB -> 2 -6dB -> .5)
# Q = 4.5
# gaindB= 1
# gain = 10**(gaindB/20)
# q.add_filter('LPF', 5000, 4.25, 1)
# q.add_filter('LPF', 5400, 4.25, 1)
# # y = q.process(time_signal)
# # q.EQ.plot_eq_curve()
# # plt.show()

# q.get_poles_zeros()
# print(q.pole_arr)
# print(q.zero_arr)

# xP = SignalPlot()
# xP.init_filt_pz_plot()
# xP.update_filt_pz_plot(q.pole_arr, q.zero_arr)
# plt.show()
 
