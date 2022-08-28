# c. 06/08/2022
# tom brzyzek

# Project Main file for Equilizer

from re import L
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import QFile
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui import Ui_Equilizer
from waveform import Signal, SignalPlot, EQ


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Equilizer()
        self.ui.setupUi(self)
        self.audio = Signal()
        self.audioPlot = SignalPlot()
        self.sample_rate = self.audio.sample_rate
        self.EQ = EQ(self.sample_rate)
        self.fileURL = ""
        self.time_signal = np.array([])
        self.freq_signal = np.array([])
        self.time_sample = np.array([])
        self.freq_sample = np.array([])
        self.filtered_time_signal = np.array([])
        self.filtered_freq_signal = np.array([])

        self._initFilters()
        self._initButtons()
        self._initPlots()


    def _initButtons(self):

        # File Browsing Buttons:
        self.ui.pushButtonFileBrowse.clicked.connect(self._FileExplorer)
        self.ui.pushButtonFileImport.clicked.connect(self._importFunction)
        self.ui.pushButtonReset.clicked.connect(self._resetFunction)
        self.ui.pushButtonEQ.clicked.connect(self._processFilters)
        self.ui.pushButtonPlayOriginal.clicked.connect(self._playOriginal)
        self.ui.pushButtonPlayFiltered.clicked.connect(self._playFiltered)
        self.ui.pushButtonStop.clicked.connect(self._stopPlaying)
        self.ui.pushButtonSaveFiltered.clicked.connect(self._saveFiltered)


    def _FileExplorer(self):
        foldername = os.path.expanduser('~')

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setOption(QFileDialog.ReadOnly, True)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setDirectory(foldername)
        dialog.exec()
        self.ui.lineEditFileBrowse.setText(dialog.selectedFiles()[0])

    def _importFunction(self):
        self.fileURL = self.ui.lineEditFileBrowse.text()

        if self.fileURL == "":
            self._resetFunction()

        else:
            self.time_signal, self.time_sample = self.audio.import_song(self.fileURL)
            self.freq_signal, self.freq_sample = self.audio.fft(self.time_signal)
            self.time_signal = self.audio.normalize(self.time_signal)
            self.freq_signal = self.audio.normalize(self.freq_signal)
            self._updatePlots(self.time_signal, self.time_sample, self.freq_signal, self.freq_sample, resetFlag=True)
            self.sample_rate = self.audio.sample_rate
            self.EQ = EQ(self.sample_rate)

    def _resetFunction(self):
        self.ui.lineEditFileBrowse.setText(None)

        self.ui.verticalSlider25Hz.setValue(0)
        self.ui.verticalSlider31Hz.setValue(0)
        self.ui.verticalSlider40Hz.setValue(0)
        self.ui.verticalSlider50Hz.setValue(0)
        self.ui.verticalSlider63Hz.setValue(0)
        self.ui.verticalSlider80Hz.setValue(0)
        self.ui.verticalSlider100Hz.setValue(0)
        self.ui.verticalSlider125Hz.setValue(0)
        self.ui.verticalSlider160Hz.setValue(0)
        self.ui.verticalSlider200Hz.setValue(0)
        self.ui.verticalSlider250Hz.setValue(0)
        self.ui.verticalSlider315Hz.setValue(0)
        self.ui.verticalSlider400Hz.setValue(0)
        self.ui.verticalSlider500Hz.setValue(0)
        self.ui.verticalSlider630Hz.setValue(0)
        self.ui.verticalSlider800Hz.setValue(0)
        self.ui.verticalSlider1kHz.setValue(0)
        self.ui.verticalSlider1k25Hz.setValue(0)
        self.ui.verticalSlider1k6Hz.setValue(0)
        self.ui.verticalSlider2kHz.setValue(0)
        self.ui.verticalSlider2k5Hz.setValue(0)
        self.ui.verticalSlider3k15Hz.setValue(0)
        self.ui.verticalSlider4kHz.setValue(0)
        self.ui.verticalSlider5kHz.setValue(0)
        self.ui.verticalSlider6k3Hz.setValue(0)
        self.ui.verticalSlider8kHz.setValue(0)
        self.ui.verticalSlider10kHz.setValue(0)
        self.ui.verticalSlider12k5Hz.setValue(0)
        self.ui.verticalSlider16kHz.setValue(0)
        self.ui.verticalSlider20kHz.setValue(0)

        self.audioPlot.reset_plot(self.audioPlot.timefig, self.audioPlot.timeax)
        self.timeFig.draw()
        self.audioPlot.reset_plot(self.audioPlot.freqfig, self.audioPlot.freqax)
        self.freqFig.draw()
        self.audioPlot.reset_plot(self.audioPlot.filtfig, self.audioPlot.filtax)
        self.filtFig.draw()

        self.EQ.reset()

        self._stopPlaying()


    def _initPlots(self):
        self.timeFig = FigureCanvas(self.audioPlot.init_time_plot())
        self.ui.timePlotLayout.addWidget(self.timeFig)
        self.freqFig = FigureCanvas(self.audioPlot.init_freq_plot())
        self.ui.frequencyPlotLayout.addWidget(self.freqFig)
        self.filtFig = FigureCanvas(self.audioPlot.init_filt_plot())
        self.ui.filterPlotLayout.addWidget(self.filtFig)

    def _initFilters(self):
        """
        Filter Parameters Matrix
        Columns: [type, center frequency, Q-factor]
        Rows: Filters in Ascending Order by Frequency
        """

        filtPar25 = ['lowshelf' , 28, 4.5] # Calculate Q Factor, issue with filters at low frequencies.
        filtPar31 = ['bell', 31.75, 4.23333333]
        filtPar40 = ['bell', 40.25, 4.23684211]
        filtPar50 = ['bell', 50.75, 4.41304348]
        filtPar63 = ['bell', 64, 4.26666667]
        filtPar80 = ['bell', 80.75, 4.36486486]
        filtPar100 = ['bell', 101.25, 4.5]
        filtPar125 = ['bell', 127.5, 4.25]
        filtPar160 = ['bell', 161.25, 4.3]
        filtPar200 = ['bell', 202.5, 4.5]
        filtPar250 = ['bell', 253.75, 4.41304348] 
        filtPar315 = ['bell', 320, 4.26666667]
        filtPar400 = ['bell', 403.75, 4.36486486]
        filtPar500 = ['bell', 507.5, 4.41304348 ]
        filtPar630 = ['bell', 640, 4.26666667 ]
        filtPar800 = ['bell', 807.5, 4.36486486]
        filtPar1k = ['bell', 1012.5, 4.5]
        filtPar1k25 = ['bell', 1275, 4.25]
        filtPar1k6 = ['bell', 1612.5, 4.3]
        filtPar2k = ['bell', 2025, 4.5]
        filtPar2k5 = ['bell', 2537.5, 4.41304348]
        filtPar3k15 = ['bell', 3200, 4.26666667]
        filtPar4k = ['bell', 4037.5, 4.36486486]
        filtPar5k = ['bell', 5075, 4.41304348]
        filtPar6k3 = ['bell', 6400, 4.26666667]
        filtPar8k = ['bell', 8075, 4.36486486]
        filtPar10k = ['bell', 10125, 4.5]
        filtPar12k5 = ['bell', 12750, 4.25]
        filtPar16k = ['bell', 16125, 4.3]
        filtPar20k = ['highshelf', 18000, 4.5]

        self.filtPar = [filtPar25, filtPar31, filtPar40, filtPar50, filtPar63, filtPar80, filtPar100, filtPar125, filtPar160, filtPar200,
                   filtPar250, filtPar315, filtPar400, filtPar500, filtPar630, filtPar800, filtPar1k, filtPar1k25, filtPar1k6, filtPar2k,
                   filtPar2k5, filtPar3k15, filtPar4k, filtPar5k, filtPar6k3, filtPar8k, filtPar10k, filtPar12k5, filtPar16k, filtPar20k]

        self.filtSliders = [self.ui.verticalSlider25Hz, self.ui.verticalSlider31Hz, self.ui.verticalSlider40Hz, self.ui.verticalSlider50Hz, 
                            self.ui.verticalSlider63Hz, self.ui.verticalSlider80Hz, self.ui.verticalSlider100Hz, self.ui.verticalSlider125Hz, 
                            self.ui.verticalSlider160Hz, self.ui.verticalSlider200Hz, self.ui.verticalSlider250Hz, self.ui.verticalSlider315Hz, 
                            self.ui.verticalSlider400Hz, self.ui.verticalSlider500Hz, self.ui.verticalSlider630Hz, self.ui.verticalSlider800Hz, 
                            self.ui.verticalSlider1kHz, self.ui.verticalSlider1k25Hz, self.ui.verticalSlider1k6Hz, self.ui.verticalSlider2kHz, 
                            self.ui.verticalSlider2k5Hz, self.ui.verticalSlider3k15Hz, self.ui.verticalSlider4kHz, self.ui.verticalSlider5kHz, 
                            self.ui.verticalSlider6k3Hz, self.ui.verticalSlider8kHz, self.ui.verticalSlider10kHz, self.ui.verticalSlider12k5Hz, 
                            self.ui.verticalSlider16kHz, self.ui.verticalSlider20kHz]


    def _processFilters(self):
        self.EQ.reset()
        i = 0
        for filt in self.filtSliders:
            val = filt.value()
            if val != 0:
                self.EQ.add_filter(self.filtPar[i][0], self.filtPar[i][1], self.filtPar[i][2], 10**(val/20))
            i += 1

        # print(self.EQ.EQ.filters) #DEVR
        # DEVC: Can use the below to return the a_coefs, b_coefs, and __z to use when plotting.
        # for filter in self.EQ.EQ.filters:
        #     print(filter.a_coefs)
        self._updatePlots(self.time_signal, self.time_sample, self.freq_signal, self.freq_sample, resetFlag=True)
        if self.EQ.EQ.filters != []:
            self.filtered_time_signal = self.EQ.process(self.time_signal)
            self.filtered_freq_signal, filt_freq_sample = self.audio.fft(self.filtered_time_signal)
            self.filtered_freq_signal = self.audio.normalize(self.filtered_freq_signal)
            self.EQ.get_poles_zeros()
            self._updatePlots(self.filtered_time_signal, self.time_sample, self.filtered_freq_signal, self.freq_sample, resetFlag=False)

    def _updatePlots(self, time_signal, time_interval, freq_signal, freq_interval, resetFlag=True):
            self.audioPlot.update_time_plot(time_signal, time_interval, resetFlag)
            self.timeFig.draw()
            self.audioPlot.update_freq_plot(freq_signal, freq_interval, resetFlag)
            self.freqFig.draw()
            self.audioPlot.update_filt_plot(self.EQ.zero_arr, self.EQ.pole_arr, resetFlag)
            self.filtFig.draw()

    def _playOriginal(self):
        self.audio.play_audio(self.time_signal)

    def _playFiltered(self):
        self.audio.play_audio(self.filtered_time_signal)

    def _stopPlaying(self):
        self.audio.stop_audio()

    def _saveFiltered(self):
        if self.filtered_time_signal.size > 0:
            file_location = QFileDialog.getSaveFileName(self, 'Save File')
            self.audio.export_song(self.filtered_time_signal, file_location[0])
        else:
            QMessageBox.about(self, "Error", "No filtered data to save.")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()


    sys.exit(app.exec())
