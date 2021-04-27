from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem,QApplication, QWidget, QPushButton,  QLineEdit, QCheckBox, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QComboBox, QSizePolicy, qApp, QLabel,QPlainTextEdit
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
import os
import numpy as np
import sys

class App(QDialog):
    def __init__(self):
        super().__init__()
        print('started')
        self.dirs = dict()
        self.handles = dict()
        self.title = 'BCI imaging pipeline control - jClust'
        self.left = 20 # 10
        self.top = 30 # 10
        self.width = 1400 # 1024
        self.height = 900  # 768
        #self.microstep_size = 0.09525 # microns per step
        
        self.base_directory = ''
        self.setups = ['DOM3_MMIMS',
                       'KayvonScope']
        self.subjects = ['BCI_03',
                         'BCI_04',
                         'BCI_05',
                         'BCI_06',
                         'BCI_07',
                         'BCI_08',
                         'BCI_09']
        
        self.initUI()
        
# =============================================================================
#         self.timer  = QTimer(self)
#         self.timer.setInterval(1000)          # Throw event timeout with an interval of 1000 milliseconds
#         self.timer.timeout.connect(self.updatelocation) # each time timer counts a second, call self.blink
#         
#         self.timer_bpod  = QTimer(self)
#         self.timer_bpod.setInterval(5000)          # Throw event timeout with an interval of 1000 milliseconds
#         self.timer_bpod.timeout.connect(self.updatebpodplot) # each time timer counts a second, call self.blink
# =============================================================================
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.createGridLayout()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox_exp_details)
        windowLayout.addWidget(self.horizontalGroupBox_progress)
        self.setLayout(windowLayout)
        self.show()
    def createGridLayout(self):
        
        
        self.horizontalGroupBox_exp_details = QGroupBox("Experiment details / controls")
        layout = QGridLayout()
        
        self.handles['setup_select'] = QComboBox(self)
        self.handles['setup_select'].setFocusPolicy(Qt.NoFocus)
        self.handles['setup_select'].addItems(self.setups)
        #self.handles['setup_select'].currentIndexChanged.connect(lambda: self.update_exp_details())  
        layout.addWidget(QLabel('Setup'),0,0)
        layout.addWidget(self.handles['setup_select'],1, 0)
        
        self.handles['subject_select'] = QComboBox(self)
        self.handles['subject_select'].setFocusPolicy(Qt.NoFocus)
        self.handles['subject_select'].addItems(self.subjects)
        #self.handles['subject_select'].currentIndexChanged.connect(lambda: self.update_exp_details())  
        layout.addWidget(QLabel('Subject'),0,2)
        layout.addWidget(self.handles['subject_select'],1, 2)
        
        self.handles['session_select'] = QLineEdit(self)
        self.handles['session_select'].setText('session comes here')
        self.handles['session_select'].returnPressed.connect(lambda: self.update_exp_details())#self.update_arduino_vals()) 
        layout.addWidget(QLabel('Session'),0,3)
        layout.addWidget(self.handles['session_select'],1, 3)
        
        self.handles['refimage_start'] = QPushButton('Generate reference image')
        self.handles['refimage_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['refimage_start'].clicked.connect(self.generate_reference_image)
        layout.addWidget(self.handles['refimage_start'],0,4,1,2)
        layout.addWidget(QLabel('Movie#:'),1,4)
        self.handles['refimage_movienum'] = QComboBox(self)
        self.handles['refimage_movienum'].setFocusPolicy(Qt.NoFocus)
        self.handles['refimage_movienum'].addItems(np.asarray(np.arange(20)+1,str))
        layout.addWidget(self.handles['refimage_movienum'],1, 5)
        
        self.handles['motioncorr_start'] = QPushButton('Start motion correction')
        self.handles['motioncorr_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['motioncorr_start'].clicked.connect(self.do_motion_correction)
        layout.addWidget(self.handles['motioncorr_start'],0,6)
        
        self.handles['motioncorr_auto'] = QCheckBox(self)
        self.handles['motioncorr_auto'].setText('auto')
        #self.handles['motioncorr_auto'].stateChanged.connect(self.auto_updatelocation)
        layout.addWidget(self.handles['motioncorr_auto'],1, 6)
        
        self.handles['concatenate_start'] = QPushButton('Concatenate movies')
        self.handles['concatenate_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['concatenate_start'].clicked.connect(self.concatenate_movies)
        layout.addWidget(self.handles['concatenate_start'],0,7)
        
        self.handles['concatenate_auto'] = QCheckBox(self)
        self.handles['concatenate_auto'].setText('auto')
        #self.handles['motioncorr_auto'].stateChanged.connect(self.auto_updatelocation)
        layout.addWidget(self.handles['concatenate_auto'],1, 7)
        
        self.handles['celldetect_start'] = QPushButton('Segment ROIs')
        self.handles['celldetect_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['celldetect_start'].clicked.connect(self.detect_cells)
        layout.addWidget(self.handles['celldetect_start'],0,8,1,2)
        
        layout.addWidget(QLabel('Core#:'),1,8)
        self.handles['celldetect_corenum'] = QComboBox(self)
        self.handles['celldetect_corenum'].setFocusPolicy(Qt.NoFocus)
        self.handles['celldetect_corenum'].addItems(np.asarray(np.arange(1,20)+1,str))
        layout.addWidget(self.handles['celldetect_corenum'],1, 9)
        
        self.handles['regmetrics_start'] = QPushButton('Registration metrics')
        self.handles['regmetrics_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['regmetrics_start'].clicked.connect(self.detect_cells)
        layout.addWidget(self.handles['regmetrics_start'],0,10,1,2)
        
        layout.addWidget(QLabel('Core#:'),1,10)
        self.handles['regmetrics_corenum'] = QComboBox(self)
        self.handles['regmetrics_corenum'].setFocusPolicy(Qt.NoFocus)
        self.handles['regmetrics_corenum'].addItems(np.asarray(np.arange(1,20)+1,str))
        layout.addWidget(self.handles['regmetrics_corenum'],1, 11)
        
        
        self.handles['S2P_start'] = QPushButton('Start suite2p')
        self.handles['S2P_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['S2P_start'].clicked.connect(self.start_s2p)
        layout.addWidget(self.handles['S2P_start'],0,12)
        self.horizontalGroupBox_exp_details.setLayout(layout)
        
        self.horizontalGroupBox_progress = QGroupBox("Progress")
        layout = QGridLayout()
        self.handles['progress_table'] = QTableWidget()
        self.handles['progress_table'].setRowCount(10)
        self.handles['progress_table'].setColumnCount(5)
        layout.addWidget(self.handles['progress_table'],0,0)
        self.horizontalGroupBox_progress.setLayout(layout)
        
    def update_exp_details(self):
        print('change the json file and start updating')
    def generate_reference_image(self):
        print('reference image is being generated')
    def do_motion_correction(self):
        print('doing motion correction')
    def concatenate_movies(self):
        print('concatenating')
    def detect_cells(self):
        print('detecting cells')
    
        
    def start_s2p(self):
        print('starting s2p')
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())        