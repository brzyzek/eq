# c. 06/08/2022
# tom brzyzek

# Project Main file for Equilizer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QFile
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui import Ui_Equilizer
from wavey import Wavey


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Equilizer()
        self.ui.setupUi(self)
        self.audio = Wavey()
        self.fileURL = ""
        self.time_signal = np.array([])
        self.freq_signal = np.array([])
        self.time_sample = np.array([])
        self.freq_sample = np.array([])

        self._initButtons()
        self._initPlots()


    def _initButtons(self):

        # File Browsing Buttons:
        self.ui.pushButtonFileBrowse.clicked.connect(self._FileExplorer)
        self.ui.pushButtonFileImport.clicked.connect(self._importFunction)
        self.ui.pushButtonReset.clicked.connect(self._resetFunction)


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
            self.audio.update_time_plot(self.time_signal, self.time_sample)
            self.timeFig.draw()
            self.audio.update_freq_plot(self.freq_signal, self.freq_sample)
            self.freqFig.draw()


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

        self.audio.reset_plot(self.audio.timefig, self.audio.timeax)
        self.timeFig.draw()
        self.audio.reset_plot(self.audio.freqfig, self.audio.freqax)
        self.freqFig.draw()


    def _initPlots(self):
        self.timeFig = FigureCanvas(self.audio.init_time_plot())
        self.ui.timePlotLayout.addWidget(self.timeFig)
        self.freqFig = FigureCanvas(self.audio.init_freq_plot())
        self.ui.frequencyPlotLayout.addWidget(self.freqFig)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()


    sys.exit(app.exec())
