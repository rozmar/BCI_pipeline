from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem,QApplication, QWidget, QPushButton,  QLineEdit, QCheckBox, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QComboBox, QSizePolicy, qApp, QLabel,QPlainTextEdit
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5 import QtGui
import os
import stat
import numpy as np
import sys
import json
from utils import utils_io
from pathlib import Path
import time
import threading

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100,shareaxis = None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        if  not shareaxis:
            self.axes = fig.add_subplot(111)
        else:
            self.axes = fig.add_subplot(111,sharex = shareaxis, sharey = shareaxis)
        super(MplCanvas, self).__init__(fig)


class App(QDialog):
    def __init__(self):
        super().__init__()
        print('started')
        self.target_movie_directory_base = '/groups/svoboda/svobodalab/users/rozmar/BCI_suite2p/'#'/groups/svoboda/home/rozsam/Data/BCI_data/'
        self.handles = dict()
        self.title = 'BCI imaging pipeline control - JaneliaCluser'
        self.left = 20 # 10
        self.top = 30 # 10
        self.width = 1400 # 1024
        self.height = 900  # 768
        #self.microstep_size = 0.09525 # microns per step
        
        self.base_directory = ''
        self.setups = ['DOM3-MMIMS',
                       'KayvonScope']
        self.subjects = ['BCI_03',
                         'BCI_04',
                         'BCI_05',
                         'BCI_06',
                         'BCI_07',
                         'BCI_08',
                         'BCI_09',
                         'BCI_10']
        self.s2p_params = {'max_reg_shift':50, # microns
                              'max_reg_shift_NR': 20, # microns
                              'block_size': 200, # microns
                              'smooth_sigma':0.5, # microns
                              'smooth_sigma_time':0, #seconds,
                              'overwrite': False,
                              'num_workers':4,
                              'z_stack_name':''} # folder where the suite2p output is saved
        
        self.initUI()
        
        self.timer  = QTimer(self)
        self.timer.setInterval(10000)          # Throw event timeout with an interval of 1000 milliseconds
        self.timer.timeout.connect(self.autoupdateprogress) # each time timer counts a second, call self.blink
        self.timer.start()
        self.plotinprogress = False
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
        self.handles['setup_select'].currentIndexChanged.connect(lambda: self.find_available_sessions())  
        layout.addWidget(QLabel('Setup'),0,0)
        layout.addWidget(self.handles['setup_select'],1, 0)
        
        self.handles['subject_select'] = QComboBox(self)
        self.handles['subject_select'].setFocusPolicy(Qt.NoFocus)
        self.handles['subject_select'].addItems(self.subjects)
        self.handles['subject_select'].currentIndexChanged.connect(lambda: self.find_available_sessions())  
        layout.addWidget(QLabel('Subject'),0,2)
        layout.addWidget(self.handles['subject_select'],1, 2)
        
        self.handles['session_select'] = QComboBox(self)
        self.handles['session_select'].setFocusPolicy(Qt.NoFocus)
        #self.handles['session_select'].addItems(self.subjects)
        self.handles['session_select'].currentIndexChanged.connect(lambda: self.update_session())  
        layout.addWidget(self.handles['session_select'],1, 3,1,2)
        
        self.handles['session'] = QLineEdit(self)
        self.handles['session'].setText('2021-02-13')
        self.handles['session'].returnPressed.connect(lambda: self.update_exp_details())#self.update_arduino_vals()) 
        layout.addWidget(QLabel('Session'),0,3,1,2)
        layout.addWidget(QLabel('Create new / start copy:'),2,2)
        layout.addWidget(self.handles['session'],2, 4)
        
        self.handles['active_experiment_text'] = QLabel('Active session:')
        layout.addWidget(self.handles['active_experiment_text'],0,5,1,5)
        
        

        self.handles['zstack_select'] = QComboBox(self)
        self.handles['zstack_select'].setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.handles['zstack_select'],2,5,1,2)
        
        self.handles['zstack_set'] = QPushButton('Set as Zstack')
        self.handles['zstack_set'].setFocusPolicy(Qt.NoFocus)
        self.handles['zstack_set'].clicked.connect(self.set_zstack)
        layout.addWidget(self.handles['zstack_set'],1,5,1,2)
        
# =============================================================================
#         self.handles['zstack_auto'] = QCheckBox(self)
#         self.handles['zstack_auto'].setText('auto')
#         #self.handles['motioncorr_auto'].stateChanged.connect(self.auto_updatelocation)
#         layout.addWidget(self.handles['zstack_auto'],2, 6)
# =============================================================================
        
        
        self.handles['refimage_start'] = QPushButton('Generate reference image')
        self.handles['refimage_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['refimage_start'].clicked.connect(self.generate_reference_image)
        layout.addWidget(self.handles['refimage_start'],1,7,1,2)
        layout.addWidget(QLabel('Movie#:'),2,7)
        self.handles['refimage_movienum'] = QComboBox(self)
        self.handles['refimage_movienum'].setFocusPolicy(Qt.NoFocus)
        self.handles['refimage_movienum'].addItems(np.asarray(np.arange(20)+1,str))
        layout.addWidget(self.handles['refimage_movienum'],2, 8)
        
        self.handles['motioncorr_start'] = QPushButton('Start motion correction')
        self.handles['motioncorr_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['motioncorr_start'].clicked.connect(self.do_motion_correction)
        layout.addWidget(self.handles['motioncorr_start'],1,9)
        
        self.handles['motioncorr_auto'] = QCheckBox(self)
        self.handles['motioncorr_auto'].setText('auto')
        #self.handles['motioncorr_auto'].stateChanged.connect(self.auto_updatelocation)
        layout.addWidget(self.handles['motioncorr_auto'],2, 9)
        
        self.handles['concatenate_start'] = QPushButton('Concatenate movies')
        self.handles['concatenate_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['concatenate_start'].clicked.connect(self.concatenate_movies)
        layout.addWidget(self.handles['concatenate_start'],1,10)
        
        self.handles['concatenate_auto'] = QCheckBox(self)
        self.handles['concatenate_auto'].setText('auto')
        #self.handles['motioncorr_auto'].stateChanged.connect(self.auto_updatelocation)
        layout.addWidget(self.handles['concatenate_auto'],2, 10)
        
        self.handles['celldetect_start'] = QPushButton('Segment ROIs')
        self.handles['celldetect_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['celldetect_start'].clicked.connect(self.detect_cells)
        layout.addWidget(self.handles['celldetect_start'],1,11,1,2)
        
        layout.addWidget(QLabel('Core#:'),2,11)
        self.handles['celldetect_corenum'] = QComboBox(self)
        self.handles['celldetect_corenum'].setFocusPolicy(Qt.NoFocus)
        self.handles['celldetect_corenum'].addItems(np.asarray(np.arange(1,20)+1,str))
        layout.addWidget(self.handles['celldetect_corenum'],2, 12)
        
        self.handles['regmetrics_start'] = QPushButton('Registration metrics')
        self.handles['regmetrics_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['regmetrics_start'].clicked.connect(self.registration_metrics)
        layout.addWidget(self.handles['regmetrics_start'],1,13,1,2)
        
        layout.addWidget(QLabel('Core#:'),2,13)
        self.handles['regmetrics_corenum'] = QComboBox(self)
        self.handles['regmetrics_corenum'].setFocusPolicy(Qt.NoFocus)
        self.handles['regmetrics_corenum'].addItems(np.asarray(np.arange(1,20)+1,str))
        layout.addWidget(self.handles['regmetrics_corenum'],2, 14)
        
        
        self.handles['S2P_start'] = QPushButton('Start suite2p')
        self.handles['S2P_start'].setFocusPolicy(Qt.NoFocus)
        self.handles['S2P_start'].clicked.connect(self.start_s2p)
        layout.addWidget(self.handles['S2P_start'],1,15)
        self.horizontalGroupBox_exp_details.setLayout(layout)
        
        self.horizontalGroupBox_progress = QGroupBox("Progress")
        layout = QGridLayout()
        self.handles['progress_table'] = QTableWidget()
        self.handles['progress_table'].setRowCount(10)
        self.handles['progress_table'].setColumnCount(4)
        self.handles['progress_table'].setHorizontalHeaderLabels(['Movie name','Registered','Concatenated','Cell detection'])
        layout.addWidget(self.handles['progress_table'],0,0,50,1)
        
        
        self.handles['meanimage_plot'] = MplCanvas(self, width=5, height=4, dpi=100)
        #self.handles['meanimage_plot'].axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.handles['meanimage_toolbar'] = NavigationToolbar(self.handles['meanimage_plot'], self)
        layout.addWidget(self.handles['meanimage_toolbar'],0,1)
        layout.addWidget(self.handles['meanimage_plot'],1,1,40,1)
        
        self.handles['displacement_plot'] = MplCanvas(self, width=5, height=2, dpi=100)
        #self.handles['meanimage_plot'].axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.handles['displacement_toolbar'] = NavigationToolbar(self.handles['displacement_plot'], self)
        layout.addWidget(self.handles['displacement_toolbar'],41,1)
        layout.addWidget(self.handles['displacement_plot'],42,1)
        self.handles['displacement_plot_Z_ax'] = self.handles['displacement_plot'].axes.twinx()
        self.horizontalGroupBox_progress.setLayout(layout)
        
    @pyqtSlot()
    def autoupdateprogress(self):
        #elf.plotinprogress = False
        if self.plotinprogress:
            print('not ready')
            return
        else:
            self.plotinprogress = True
        try:
            with open(os.path.join(self.target_movie_directory,'s2p_params.json'), "r") as read_file:
                s2p_params = json.load(read_file)
        except:
            return
        if self.handles['zstack_select'].currentText() != s2p_params['z_stack_name']:
            self.handles['zstack_set'].setStyleSheet("background-color : red")
        else:
            self.handles['zstack_set'].setStyleSheet("background-color : green")
            
        
        
        try:    
            try:
                roifindjson_file = os.path.join(self.target_movie_directory,'_concatenated_movie','roifind_progress.json')
                with open(roifindjson_file, "r") as read_file:
                    roifind_progress_dict = json.load(read_file)
                if not roifind_progress_dict['roifind_finished']:
                    roifindcolor = 'red'
                else:
                    roifindcolor = 'green'
            except:
                roifindcolor = 'green'
                
            try:
                concatenated_movie_filelist_json = os.path.join(self.target_movie_directory,'_concatenated_movie','filelist.json')
                with open(concatenated_movie_filelist_json, "r") as read_file:
                    filelist_dict = json.load(read_file)
                if filelist_dict['concatenation_underway']:
                    concatenationcolor = 'red'
                    roifindcolor = 'red'
                else:
                    concatenationcolor = 'green'
            except:
                concatenationcolor = 'green'

                
            self.handles['concatenate_start'].setStyleSheet("background-color : {}".format(concatenationcolor))
            self.handles['celldetect_start'].setStyleSheet("background-color : {}".format(roifindcolor))
            self.handles['meanimage_plot'].axes.clear()
            
            try:
                meanimage_dict = np.load(os.path.join(self.target_movie_directory,'mean_image.npy'),allow_pickle = True).tolist()
                refImg = meanimage_dict['refImg']
                self.handles['meanimage_plot'].axes.imshow(refImg)
            except:
                self.handles['meanimage_plot'].axes.imshow(np.zeros([10,10]))
            self.handles['meanimage_plot'].draw()                
            self.handles['displacement_plot'].axes.clear()
            self.handles['displacement_plot_Z_ax'].clear()
            try:
                x = np.arange(len(filelist_dict['xoff_mean_list']))
                self.handles['displacement_plot'].axes.errorbar(x, np.asarray(filelist_dict['xoff_mean_list']),yerr = np.asarray(filelist_dict['xoff_std_list']),fmt = '-',label = 'X offset')
                self.handles['displacement_plot'].axes.errorbar(x, np.asarray(filelist_dict['yoff_mean_list']),yerr = np.asarray(filelist_dict['yoff_std_list']),fmt = '-',label = 'Y offset')
                try:
                    self.handles['displacement_plot'].axes.errorbar(x, np.asarray(filelist_dict['zoff_mean_list']),yerr = np.asarray(filelist_dict['zoff_std_list']),fmt = 'r-',label = 'Z offset')
                except:
                    pass
                self.handles['displacement_plot'].axes.legend()
                
            except:
                self.handles['displacement_plot'].axes.plot(10,10,'ko')
                pass
            self.handles['displacement_plot'].draw()
            if self.handles['concatenate_auto'].isChecked() and concatenationcolor == 'green':
                self.concatenate_movies()
                
            self.update_progress_table()
# =============================================================================
#             try:
#                 self.update_progress_table()
#             except:
#                 print('could not plot')
#                 self.handles['progress_table'].setRowCount(1)
#                 self.handles['progress_table'].setItem(0,0, QTableWidgetItem('no data'))
#                 self.handles['progress_table'].setItem(0,1, QTableWidgetItem('no data'))
#                 self.handles['progress_table'].setItem(0,2, QTableWidgetItem('no data'))
#                 self.handles['progress_table'].setItem(0,3, QTableWidgetItem('no data'))
# =============================================================================
                
            if self.handles['motioncorr_auto'].isChecked():
                self.do_motion_correction()   
            copyfile_json_file = os.path.join(self.target_movie_directory_base,'copyfile.json')
            with open(copyfile_json_file, "r") as read_file:
                copyfile_params = json.load(read_file)
            self.handles['active_experiment_text'].setText('Current Setup: {}, Subject: {}, Session:{}'.format(copyfile_params['setup'],copyfile_params['subject'],copyfile_params['session']))
        except:
            pass
        self.plotinprogress = False
        print('update done {}'.format(time.time()))
        
            
    def update_progress_table(self):
        file_dict = np.load(os.path.join(self.target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
        try:
            stack_names = file_dict['copied_stacks']
        except:
            stack_names = []
        stack_names_in_gui = [self.handles['zstack_select'].itemText(i) for i in range(self.handles['zstack_select'].count())]
        if stack_names != stack_names_in_gui:
            self.handles['zstack_select'].clear()
            self.handles['zstack_select'].addItems(stack_names)
            print('zstacks refreshed')
        
        
        concatenated_movie_filelist_json = os.path.join(self.target_movie_directory,'_concatenated_movie','filelist.json')
        cell_detect_json = os.path.join(self.target_movie_directory,'_concatenated_movie','roifind_progress.json')
        try:
            with open(concatenated_movie_filelist_json, "r") as read_file:
                concatenated_filelist_dict = json.load(read_file)
            concatenation_started = True
        except:
            concatenation_started = False
            
        try:
            with open(cell_detect_json, "r") as read_file:
                cell_detect_dict = json.load(read_file)
            cell_detect_started = True
        except:
            cell_detect_started = False            
            
        self.handles['progress_table'].setRowCount(len(file_dict['copied_files']))
        for i,file in enumerate(file_dict['copied_files'][::-1]):
            self.handles['progress_table'].setItem(i,0, QTableWidgetItem(file))
            if not concatenation_started:
                self.handles['progress_table'].setItem(i,2, QTableWidgetItem('Nope'))
                self.handles['progress_table'].item(i,2).setBackground(QtGui.QColor('red'))
            else:
                if file in concatenated_filelist_dict['file_name_list']:
                    self.handles['progress_table'].setItem(i,2, QTableWidgetItem('Done'))
                    self.handles['progress_table'].item(i,2).setBackground(QtGui.QColor('green'))
                else:
                    self.handles['progress_table'].setItem(i,2, QTableWidgetItem('Nope'))
                    self.handles['progress_table'].item(i,2).setBackground(QtGui.QColor('red'))
            
            if not cell_detect_started:
                self.handles['progress_table'].setItem(i,3, QTableWidgetItem('Nope'))
                self.handles['progress_table'].item(i,3).setBackground(QtGui.QColor('red'))
            elif cell_detect_dict['roifind_finished']:
                if file in cell_detect_dict['roifind_source_movies']:
                    self.handles['progress_table'].setItem(i,3, QTableWidgetItem('Done - included'))
                    self.handles['progress_table'].item(i,3).setBackground(QtGui.QColor('green'))
                else:
                    self.handles['progress_table'].setItem(i,3, QTableWidgetItem('Done-not included'))
                    self.handles['progress_table'].item(i,3).setBackground(QtGui.QColor('white'))
            elif not cell_detect_dict['roifind_finished']:
                if file in cell_detect_dict['roifind_source_movies']:
                    self.handles['progress_table'].setItem(i,3, QTableWidgetItem('In progress - included'))
                    self.handles['progress_table'].item(i,3).setBackground(QtGui.QColor('yellow'))
                else:
                    self.handles['progress_table'].setItem(i,3, QTableWidgetItem('In progress -not included'))
                    self.handles['progress_table'].item(i,3).setBackground(QtGui.QColor('yellow'))
                    
            dir_now = os.path.join(self.target_movie_directory,file[:-4])
            tiff_now = os.path.join(self.target_movie_directory,file[:-4],file)
            reg_json_file = os.path.join(self.target_movie_directory,file[:-4],'reg_progress.json')
            if 'reg_progress.json' in os.listdir(dir_now):
                with open(reg_json_file, "r") as read_file:
                    reg_dict = json.load(read_file)
                    registration_started = reg_dict['registration_started']
            else:
                registration_started = False
            if registration_started:
                if 'registration_finished' in reg_dict.keys():
                    if reg_dict['registration_finished']:
                        registration_finished = True
                    else:
                        registration_finished = False
                        
                else:
                    registration_finished = False
                self.handles['progress_table'].setItem(i,0, QTableWidgetItem(file))
            else:
                registration_finished = False
            if registration_finished:
                regtime = int(float(reg_dict['registration_finished_time'])-float(reg_dict['registration_started_time']))
                self.handles['progress_table'].setItem(i,1, QTableWidgetItem('Done in {} s'.format(regtime)))
                self.handles['progress_table'].item(i,1).setBackground(QtGui.QColor('green'))
            elif registration_started:
                starttime = time.strftime('%H:%M:%S', time.gmtime(float(reg_dict['registration_started_time'])))
                self.handles['progress_table'].setItem(i,1, QTableWidgetItem('Started at {}'.format(starttime)))
                self.handles['progress_table'].item(i,1).setBackground(QtGui.QColor('white'))
            else:
                self.handles['progress_table'].setItem(i,1, QTableWidgetItem('Not started'))
                self.handles['progress_table'].item(i,1).setBackground(QtGui.QColor('red'))
    
    def find_available_sessions(self):
        setup = self.handles['setup_select'].currentText()
        subject = self.handles['subject_select'].currentText()
        try:
            sessions = os.listdir(os.path.join(self.target_movie_directory_base,setup,subject))
        except:
            sessions = []
# =============================================================================
#         try:
#             self.handles['session_select'].currentIndexChanged.disconnect()
#         except:
#             pass
# =============================================================================
        self.handles['session_select'].clear()
        self.handles['session_select'].addItems(sessions)
        #self.handles['subject_select'].currentIndexChanged.connect(lambda: self.update_session())
        if len(sessions)>0:
            self.update_session()
    
    def update_session(self):
        setup = self.handles['setup_select'].currentText()
        subject = self.handles['subject_select'].currentText()
        session = self.handles['session_select'].currentText()
        self.handles['session'].setText(session)
        self.target_movie_directory = os.path.join(self.target_movie_directory_base,setup,subject,session)
        print(self.target_movie_directory )
# =============================================================================
#         with open(os.path.join(self.target_movie_directory,'s2p_params.json'), "r") as read_file:
#             s2p_params = json.load(read_file)
#         self.s2p_params = s2p_params
# =============================================================================
        #self.autoupdateprogress()
    
    def update_exp_details(self):
        setup = self.handles['setup_select'].currentText()
        subject = self.handles['subject_select'].currentText()
        session = self.handles['session'].text()
        self.target_movie_directory = os.path.join(self.target_movie_directory_base,setup,subject,session)
        sp2_params_file = os.path.join(self.target_movie_directory,'s2p_params.json')
        Path(self.target_movie_directory).mkdir(parents = True,exist_ok = True)
        os.chmod(self.target_movie_directory, 0o777 )
        setup = self.handles['setup_select'].currentText()
        subject = self.handles['subject_select'].currentText()
        try:
            sessions = os.listdir(os.path.join(self.target_movie_directory_base,setup,subject))
        except:
            sessions = []
# =============================================================================
#         try:            
#             self.handles['session_select'].currentIndexChanged.disconnect()
#         except:
#             pass
# =============================================================================
            #self.handles['session_select'].currentIndexChanged.disconnect()
        self.handles['session_select'].clear()
        self.handles['session_select'].addItems(sessions)
        #self.handles['subject_select'].currentIndexChanged.connect(lambda: self.update_session())
        
        
        with open(sp2_params_file, "w") as data_file:
            json.dump(self.s2p_params, data_file, indent=2)
        #% Check for new .tiff files in a given directory and copy them when they are finished - should be run every few seconds
        copyfile_json_file = os.path.join(self.target_movie_directory_base,'copyfile.json')
        copyfile_params = {'setup':setup,
                           'subject':subject,
                           'session':session}
        with open(copyfile_json_file, "w") as data_file:
                json.dump(copyfile_params, data_file, indent=2)
        try:
            self.update_progress_table()
        except:
            pass
        
    def set_zstack(self) :
        self.s2p_params['z_stack_name'] =self.handles['zstack_select'].currentText()
        sp2_params_file = os.path.join(self.target_movie_directory,'s2p_params.json')
        with open(sp2_params_file, "w") as data_file:
            json.dump(self.s2p_params, data_file, indent=2)
        self.autoupdateprogress()
        
        
    def generate_reference_image(self): #TODO save when it starts and if it is finished yet.
        reference_movie_dir = os.path.join(self.target_movie_directory,'_reference_image')
        Path(reference_movie_dir).mkdir(parents = True,exist_ok = True)
        os.chmod(reference_movie_dir, 0o777)
        reference_movie_json = os.path.join(self.target_movie_directory,'_reference_image','refimage_progress.json')
        try:
            with open(reference_movie_json, "r") as read_file:
                refimage_dict = json.load(read_file)
            if refimage_dict['ref_image_finished'] == False:
                print('reference image search is already on the way, canceling')
                return
        except:
            pass
        refimage_dict = {'ref_image_started':True,
                         'ref_image_finished':False,
                         'ref_image_started_time': str(time.time())}
        with open(reference_movie_json, "w") as data_file:
            json.dump(refimage_dict, data_file, indent=2)
        trial_num_to_use = int(self.handles['refimage_movienum'].currentText())
        #%
# =============================================================================
#         if not os.path.exists(os.path.join(self.target_movie_directory,'mean_image.npy')):
# =============================================================================
        cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                'conda activate suite2p',
                                'cd ~/Scripts/Python/BCI_pipeline/',
                                'python cluster_helper.py {} "\'{}\'" {}'.format('utils_imaging.generate_mean_image_from_trials',self.target_movie_directory,trial_num_to_use)]
        with open("/groups/svoboda/home/rozsam/Scripts/runBCI.sh","w") as shfile:
            #shfile.writelines(cluster_command_list) 
            for L in cluster_command_list:
                shfile.writelines(L+'\n') 
        bash_command = "bsub -n 1 -J BCI_job 'sh /groups/svoboda/home/rozsam/Scripts/runBCI.sh > ~/Scripts/BCI_output.txt'"
        os.system(bash_command)
# =============================================================================
#         else:
#             print('reference image is already present')
# =============================================================================
        print('reference image is being generated')
    def do_motion_correction(self):
        file_dict = np.load(os.path.join(self.target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
        
        for file in file_dict['copied_files']:
            if not os.path.exists(os.path.join(self.target_movie_directory,'mean_image.npy')):
                print('no reference image!!')
                break
            
            dir_now = os.path.join(self.target_movie_directory,file[:-4])
            #tiff_now = os.path.join(self.target_movie_directory,file[:-4],file)
            reg_json_file = os.path.join(self.target_movie_directory,file[:-4],'reg_progress.json')
            if 'reg_progress.json' in os.listdir(dir_now):
                with open(reg_json_file, "r") as read_file:
                    reg_dict = json.load(read_file)
            else:
                reg_dict = {'registration_started':False}
                
            if reg_dict['registration_started']:
                continue
            print('starting {}'.format(file))
            reg_dict['registration_started'] = True
            with open(reg_json_file, "w") as data_file:
                json.dump(reg_dict, data_file, indent=2)
            #%
            cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                    'conda activate suite2p',
                                    'cd ~/Scripts/Python/BCI_pipeline/',
                                    "python cluster_helper.py {} '\"{}\"' '\"{}\"'".format('utils_imaging.register_trial',self.target_movie_directory,file)]
            cluster_output_file = os.path.join(dir_now,'s2p_registration_output.txt')
            bash_command = r"bsub -n 1 -J BCI_register_{} -o /dev/null '{} > {}'".format(file,' && '.join(cluster_command_list),cluster_output_file)
            os.system(bash_command)
            
        print('doing motion correction')
    def concatenate_movies(self):
        concatenated_movie_dir = os.path.join(self.target_movie_directory,'_concatenated_movie')
        try:
            concatenated_movie_filelist_json = os.path.join(self.target_movie_directory,'_concatenated_movie','filelist.json')
            with open(concatenated_movie_filelist_json, "r") as read_file:
                filelist_dict = json.load(read_file)
            if filelist_dict['concatenation_underway']:
                print('concatenation is already running, aborting')
                return None
        except:
            pass # concatenation json file is not present
        Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
        os.chmod(concatenated_movie_dir, 0o777 )
# =============================================================================
#         #utils_io.concatenate_suite2p_files(target_movie_directory)
#         cluster_command_list = ['eval "$(conda shell.bash hook)"',
#                                 'conda activate suite2p',
#                                 'cd ~/Scripts/Python/BCI_pipeline/',
#                                 "python cluster_helper.py {} '\"{}\"'".format('utils_io.concatenate_suite2p_files',self.target_movie_directory)]
#         cluster_output_file = os.path.join(os.path.join(self.target_movie_directory,'_concatenated_movie'),'s2p_concatenation_output.txt')
#         bash_command = r"bsub -n 1 -J BCI_concatenate_files '{} > {}'".format(' && '.join(cluster_command_list),cluster_output_file)
#         os.system(bash_command)
# =============================================================================
        
        os.system("python cluster_helper.py {} '{}' &".format('utils_io.concatenate_suite2p_files',self.target_movie_directory))
        
        
        print('concatenating')
    def detect_cells(self):
        core_num_to_use = int(self.handles['celldetect_corenum'].currentText())
        concatenated_movie_dir = os.path.join(self.target_movie_directory,'_concatenated_movie')
        full_movie_dir = concatenated_movie_dir
        try:
            roifindjson_file = os.path.join(full_movie_dir,'roifind_progress.json')
            with open(roifindjson_file, "r") as read_file:
                roifind_progress_dict = json.load(read_file)
            if roifind_progress_dict['roifind_started'] and not roifind_progress_dict['roifind_finished']:
                print('roi segmentation is already running, aborting')
                return None
        except:
            pass
        #%
        cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                'conda activate suite2p',
                                'cd ~/Scripts/Python/BCI_pipeline/',
                                "python cluster_helper.py {} '\"{}\"'".format('utils_imaging.find_ROIs',full_movie_dir)]
        cluster_output_file = os.path.join(full_movie_dir,'s2p_ROI_finding_output.txt')
        bash_command = r"bsub -n {} -J BCI_ROIfind '{} > {}'".format(core_num_to_use,' && '.join(cluster_command_list),cluster_output_file)
        os.system(bash_command) # -o /dev/null
        print('detecting cells')
        
    def registration_metrics(self):
        core_num_to_use = int(self.handles['regmetrics_corenum'].currentText())
        concatenated_movie_dir = os.path.join(self.target_movie_directory,'_concatenated_movie')
        full_movie_dir = concatenated_movie_dir
        cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                'conda activate suite2p',
                                'cd ~/Scripts/Python/BCI_pipeline/',
                                "python cluster_helper.py {} '\"{}\"'".format('utils_imaging.registration_metrics',full_movie_dir)]
        cluster_output_file = os.path.join(full_movie_dir,'s2p_registration_metrics_output.txt')
        bash_command = r"bsub -n {} -J BCI_registration_metric '{} > {}'".format(core_num_to_use,' && '.join(cluster_command_list),cluster_output_file)
        os.system(bash_command) # -o /dev/null
    def start_s2p(self):
        concatenated_movie_dir = os.path.join(self.target_movie_directory,'_concatenated_movie')
        os.system('cd {}; python -m suite2p &'.format(concatenated_movie_dir))
        print('starting s2p')
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())        