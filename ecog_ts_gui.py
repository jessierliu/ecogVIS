# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtCore, QtGui, Qt

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QMessageBox, QHBoxLayout,
    QTextEdit, QApplication, QPushButton, QVBoxLayout, QGroupBox, QFormLayout, QDialog,
    QRadioButton, QGridLayout, QComboBox, QInputDialog)

import pyqtgraph as pg
from Function.subFunctions import ecogTSGUI
import os
import datetime
import numpy as np


model = None
intervalAdd_ = False
intervalDel_ = False
annotationAdd_ = False
annotationDel_ = False
annotationColor_ = 'red'
class Application(QWidget):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    def __init__(self, filename, parent = None):
        global model

        # e.g.: /home/User/freesurfer_subjects/Subject_XX.nwb
        pathName = filename
        self.temp = []
        self.cursor_ = False
        self.channel_ = False
        self.error = None
        super(Application, self).__init__()
        self.keyPressed.connect(self.on_key)
        self.active_mode = 'default'
        self.init_gui()
        self.setWindowTitle('')
        self.show()
        #self.Maximized()

        # Run the main file
        parameters = {}
        parameters['pars'] = {'Figure': [self.win1, self.win2, self.win3]}
        parameters['editLine'] = {'qLine0': self.qline0, 'qLine1': self.qline1, 'qLine2': self.qline2, 'qLine3': self.qline3,
                  'qLine4': self.qline4}

        model = ecogTSGUI(self, pathName, parameters)



    def log_error(self, error):
        pwd = os.getcwd()
        folder = os.path.join(pwd, 'error_log')
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(os.path.join(folder, 'error.log'), 'a') as file:
            file.write(datetime.datetime.today().strftime('%Y-%m-%d  %H:%M:%S') + ' ' + error + '\n')
        file.close()


    def keyPressEvent(self, event):
        super(Application, self).keyPressEvent(event)
        self.keyPressed.emit(event)


    def on_key(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            model.channel_Scroll_Up('unit')
        elif event.key() == QtCore.Qt.Key_PageUp:
            model.channel_Scroll_Up('page')
        elif event.key() == QtCore.Qt.Key_Down:
            model.channel_Scroll_Down('unit')
        elif event.key() == QtCore.Qt.Key_PageDown:
            model.channel_Scroll_Down('page')
        elif event.key() == QtCore.Qt.Key_Left:
            model.time_scroll(scroll=-1/3)
        elif event.key() == QtCore.Qt.Key_Right:
            model.time_scroll(scroll=1/3)

        event.accept()


    def init_gui(self):
        '''
        create a horizontal box layout
        '''
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        groupbox1 = QGroupBox('Channels Plot')
        vb = CustomViewBox()
        self.win1 = pg.PlotWidget(viewBox = vb)   #middle signals plot
        self.win2 = pg.PlotWidget(border = 'k')   #upper horizontal bar
        self.win3 = pg.PlotWidget()               #lower audio plot
        self.win1.setBackground('w')
        self.win2.setBackground('w')
        self.win3.setBackground('w')
        self.win1.setMouseEnabled(x = False, y = False)
        self.win2.setMouseEnabled(x = False, y = False)
        self.win3.setMouseEnabled(x = False, y = False)

        self.figure1 = self.win1.plot(x = [], y = [])

        self.win2.hideAxis('left')
        self.win2.hideAxis('bottom')
        self.win3.hideAxis('left')
        self.win3.hideAxis('bottom')

        form5layout = QGridLayout() #QVBoxLayout()
        form5layout.setSpacing(0.0)
        form5layout.setRowStretch(0, 1)
        form5layout.setRowStretch(1, 8)
        form5layout.setRowStretch(2, 1)
        form5layout.addWidget(self.win2)
        form5layout.addWidget(self.win1)
        form5layout.addWidget(self.win3)

        groupbox1.setLayout(form5layout)

        vbox.addWidget(groupbox1)


        '''
        Another vertical box layout
        '''
        vbox1 = QVBoxLayout()
        panel1 = QGroupBox('Panel')
        panel1.setFixedWidth(200)
        panel1.setFixedHeight(200)
        grid1 = QGridLayout()

        # Annotation buttons
        qlabelAnnotations = QLabel('Annotation:')
        self.combo1 = QComboBox()
        self.combo1.addItem('red')
        self.combo1.addItem('green')
        self.combo1.addItem('blue')
        self.combo1.addItem('yellow')
        self.combo1.activated.connect(self.AnnotationColor)
        self.push1_1 = QPushButton('Add')
        self.push1_1.clicked.connect(self.AnnotationAdd)
        self.push1_1.setCheckable(True)
        self.push1_2 = QPushButton('Del')
        self.push1_2.clicked.connect(self.AnnotationDel)
        self.push1_2.setCheckable(True)
        self.push1_3 = QPushButton('Save')
        self.push1_3.clicked.connect(self.AnnotationSave)

        # Custom intervals buttons
        qlabelIntervals = QLabel('Intervals:')
        self.combo2 = QComboBox()
        self.combo2.addItem("Invalid")
        self.combo2.addItem("add custom")
        self.combo2.activated.connect(self.IntervalType)
        self.push2_1 = QPushButton('Add')
        self.push2_1.clicked.connect(self.IntervalAdd)
        self.push2_1.setCheckable(True)
        self.push2_2 = QPushButton('Del')
        self.push2_2.clicked.connect(self.IntervalDel)
        self.push2_2.setCheckable(True)
        self.push2_3 = QPushButton('Save')
        self.push2_3.clicked.connect(self.IntervalSave)

        # Get channel buttons
        qlabelChannels = QLabel('Channels:')
        self.combo3 = QComboBox()
        self.combo3.addItem("Do")
        self.combo3.addItem("a")
        self.combo3.addItem("Checkbox")

        # Buttons layout
        grid1.addWidget(qlabelAnnotations, 0, 0, 1, 3)
        grid1.addWidget(self.combo1, 0, 3, 1, 3)
        grid1.addWidget(self.push1_1, 1, 0, 1, 2)
        grid1.addWidget(self.push1_2, 1, 2, 1, 2)
        grid1.addWidget(self.push1_3, 1, 4, 1, 2)
        grid1.addWidget(qlabelIntervals, 2, 0, 1, 3)
        grid1.addWidget(self.combo2, 2, 3, 1, 3)
        grid1.addWidget(self.push2_1, 3, 0, 1, 2)
        grid1.addWidget(self.push2_2, 3, 2, 1, 2)
        grid1.addWidget(self.push2_3, 3, 4, 1, 2)
        grid1.addWidget(qlabelChannels, 4, 0, 1, 3)
        grid1.addWidget(self.combo3, 4, 3, 1, 3)
        panel1.setLayout(grid1)

        panel2 = QGroupBox('Signal Type')
        panel2.setFixedWidth(200)
        panel2.setFixedHeight(100)
        form2 = QFormLayout()
        self.rbtn1 = QRadioButton('raw ECoG')
        self.rbtn1.setChecked(True)
        self.rbtn2 = QRadioButton('High Gamma')
        self.rbtn2.setChecked(False)
        form2.addWidget(self.rbtn1)
        form2.addWidget(self.rbtn2)
        panel2.setLayout(form2)

        panel3 = QGroupBox('Plot Controls')
        panel3.setFixedWidth(200)
        form3 = QGridLayout()
        self.enableButton = QPushButton('Enable')
        self.enableButton.setFixedWidth(100)
        self.enableButton.clicked.connect(self.enable)
        qlabel1 = QLabel('Top')
        qlabel2 = QLabel('Bottom')
        qlabel3 = QLabel('Interval \nstart(s)')
        qlabel4 = QLabel('Time \nspan')
        qlabel5 = QLabel('Vertical\nScale')
        self.qline0 = QLineEdit('16')
        self.qline0.setEnabled(False)
        self.qline1 = QLineEdit('1')
        self.qline1.setEnabled(False)
        self.qline2 = QLineEdit('0.01')
        self.qline2.setEnabled(False)
        self.qline3 = QLineEdit('2')
        self.qline3.setEnabled(False)
        self.qline4 = QLineEdit('1')
        self.qline4.setEnabled(False)

        self.qline0.returnPressed.connect(self.channelDisplayed)
        self.qline1.returnPressed.connect(self.channelDisplayed)
        self.qline2.returnPressed.connect(self.interval_start)
        self.qline3.returnPressed.connect(self.time_window_size)
        self.qline4.returnPressed.connect(self.verticalScale)

        self.pushbtn1_1 = QPushButton('^')
        self.pushbtn1_1.clicked.connect(self.scroll_up)
        self.pushbtn1_2 = QPushButton('^^')
        self.pushbtn1_2.clicked.connect(self.scroll_up_page)
        self.pushbtn2_1 = QPushButton('v')
        self.pushbtn2_1.clicked.connect(self.scroll_down)
        self.pushbtn2_2 = QPushButton('vv')
        self.pushbtn2_2.clicked.connect(self.scroll_down_page)

        self.pushbtn3 = QPushButton('<<')
        self.pushbtn3.clicked.connect(self.page_backward)
        self.pushbtn4 = QPushButton('<')
        self.pushbtn4.clicked.connect(self.scroll_backward)
        self.pushbtn5 = QPushButton('>>')
        self.pushbtn5.clicked.connect(self.page_forward)
        self.pushbtn6 = QPushButton('>')
        self.pushbtn6.clicked.connect(self.scroll_forward)
        self.pushbtn7 = QPushButton('*2')
        self.pushbtn7.clicked.connect(self.time_window_enlarge)
        self.pushbtn8 = QPushButton('/2')
        self.pushbtn8.clicked.connect(self.time_window_reduce)
        self.pushbtn9 = QPushButton('*2')
        self.pushbtn9.clicked.connect(self.verticalScaleIncrease)
        self.pushbtn10 = QPushButton('/2')
        self.pushbtn10.clicked.connect(self.verticalScaleDecrease)

        form3.addWidget(self.enableButton, 0, 1)
        form3.addWidget(qlabel1, 1, 0)
        form3.addWidget(self.qline0, 1, 1)
        form3.addWidget(self.pushbtn1_1, 1, 2)
        form3.addWidget(self.pushbtn1_2, 1, 3)
        form3.addWidget(qlabel2, 2, 0)
        form3.addWidget(self.qline1, 2, 1)
        form3.addWidget(self.pushbtn2_1, 2, 2)
        form3.addWidget(self.pushbtn2_2, 2, 3)

        form3.addWidget(qlabel3, 4, 0, 1, 2)
        form3.addWidget(self.qline2, 4, 2, 1, 2)
        form3.addWidget(self.pushbtn3, 5, 0)
        form3.addWidget(self.pushbtn4, 5, 1)
        form3.addWidget(self.pushbtn6, 5, 2)
        form3.addWidget(self.pushbtn5, 5, 3)
        form3.addWidget(qlabel4, 6, 0)
        form3.addWidget(self.qline3, 6, 1)
        form3.addWidget(self.pushbtn7, 6, 2)
        form3.addWidget(self.pushbtn8, 6, 3)
        form3.addWidget(qlabel5, 7, 0)
        form3.addWidget(self.qline4, 7, 1)
        form3.addWidget(self.pushbtn9, 7, 2)
        form3.addWidget(self.pushbtn10, 7, 3)
        form3.addWidget(QLabel(), 8, 0)
        panel3.setLayout(form3)
        vbox1.addWidget(panel1)
        vbox1.addWidget(panel2)
        vbox1.addWidget(panel3)

        hbox.addLayout(vbox1)   #add panels first
        hbox.addLayout(vbox)    #add plots second
        self.setLayout(hbox)


    def check_status(self):
        global intervalAdd_
        global intervalDel_
        global annotationAdd_
        global annotationDel_

        #self.active_mode = 'default', 'intervalAdd', 'intervalDel'
        if self.active_mode != 'intervalAdd':
            self.push2_1.setChecked(False)
            intervalAdd_ = False

        if self.active_mode != 'intervalDel':
            self.push2_2.setChecked(False)
            intervalDel_ = False

        if self.active_mode != 'annotationAdd':
            self.push1_1.setChecked(False)
            annotationAdd_ = False

        if self.active_mode != 'annotationDel':
            self.push1_2.setChecked(False)
            annotationDel_ = False

        if self.channel_:
            self.channel_ = False

        if self.cursor_:
            self.cursor_ = False


    def AnnotationColor(self):
        global annotationColor_
        annotationColor_ = str(self.combo1.currentText())


    def AnnotationAdd(self):
        global annotationAdd_
        if self.push1_1.isChecked():  #if button is pressed down
            self.active_mode = 'annotationAdd'
            self.check_status()
            annotationAdd_ = True
        else:
            self.active_mode = 'default'
            self.check_status()

    def AnnotationDel(self):
        global annotationDel_
        if self.push1_2.isChecked():  #if button is pressed down
            self.active_mode = 'annotationDel'
            self.check_status()
            annotationDel_ = True
        else:
            self.active_mode = 'default'
            self.check_status()

    def AnnotationSave(self):
        self.active_mode = 'default'
        self.check_status()
        try:
            model.AnnotationSave()
        except Exception as ex:
            self.log_error(str(ex))



    def IntervalType(self):
        global intervalType_

        item = str(self.combo2.currentText())
        if item == 'add custom':
            text, ok = QInputDialog.getText(None, 'Add custom intervals',
                                            'Interval class name:')
            curr_ind = self.combo2.currentIndex()
            self.combo2.setItemText(curr_ind, text)
            self.combo2.addItem('add custom')
            self.combo2.setCurrentIndex(curr_ind)
            intervalType_ = text
        else:
            intervalType_ = item


    def IntervalAdd(self):
        global intervalAdd_

        if self.push2_1.isChecked():  #if button is pressed down
            self.active_mode = 'intervalAdd'
            self.check_status()
            intervalAdd_ = True
        else:
            self.active_mode = 'default'
            self.check_status()


    def IntervalDel(self):
        global intervalDel_
        if self.push2_2.isChecked():  #if button is pressed down
            self.active_mode = 'intervalDel'
            self.check_status()
            intervalDel_ = True
        else:
            self.active_mode = 'default'
            self.check_status()


    def IntervalSave(self):
        self.active_mode = 'default'
        self.check_status()
        try:
            model.pushSave()
        except Exception as ex:
            self.log_error(str(ex))



    def GetChannel(self):
        self.check_status()
        self.channel = self.win1.scene().sigMouseClicked.connect(self.get_channel)
        self.channel_ = True

    def get_channel(self, event):
        mousePoint = self.win1.plotItem.vb.mapSceneToView(event.scenePos())
        model.getChannel(mousePoint)



    ## Lower-Left buttons
    def enable(self):
        if self.enableButton.text() == 'Enable':
            self.enableButton.setText('Disable')
            self.qline0.setEnabled(True)
            self.qline1.setEnabled(True)
            self.qline2.setEnabled(True)
            self.qline3.setEnabled(True)
            self.qline4.setEnabled(True)
        else:
            self.enableButton.setText('Enable')
            self.qline0.setEnabled(False)
            self.qline1.setEnabled(False)
            self.qline2.setEnabled(False)
            self.qline3.setEnabled(False)
            self.qline4.setEnabled(False)

    def scroll_up(self):
        model.channel_Scroll_Up('unit')

    def scroll_up_page(self):
        model.channel_Scroll_Up('page')

    def scroll_down(self):
        model.channel_Scroll_Down('unit')

    def scroll_down_page(self):
        model.channel_Scroll_Down('page')

    def page_backward(self):
        model.time_scroll(scroll=-1)

    def scroll_backward(self):
        model.time_scroll(scroll=-1/3)

    def page_forward(self):
        model.time_scroll(scroll=1)

    def scroll_forward(self):
        model.time_scroll(scroll=1/3)

    def verticalScale(self):
        model.refreshScreen()

    def horizontalScaleIncrease(self):
        model.horizontalScaleIncrease()

    def horizontalScaleDecrease(self):
        model.horizontalScaleDecrease()

    def verticalScaleIncrease(self):
        model.verticalScaleIncrease()

    def verticalScaleDecrease(self):
        model.verticalScaleDecrease()

    def time_window_size(self):
        model.updateCurXAxisPosition()

    def time_window_enlarge(self):
        model.time_window_resize(2.)

    def time_window_reduce(self):
        model.time_window_resize(0.5)

    def interval_start(self):
        model.updateCurXAxisPosition()

    def channelDisplayed(self):
        model.nChannels_Displayed()



class CustomViewBox(pg.ViewBox):
    def __init__(self):
        pg.ViewBox.__init__(self)


    def mouseClickEvent(self, ev):
        global intervalDel_
        global annotationAdd_
        global annotationDel_
        global annotationColor_

        if intervalDel_:
            mousePoint = self.mapSceneToView(ev.scenePos())
            x = mousePoint.x()
            try:
                model.deleteInterval(round(x, 3))
            except Exception as ex:
                print(str(ex))

        if annotationAdd_:
            mousePoint = self.mapSceneToView(ev.scenePos())
            x = mousePoint.x()
            y = mousePoint.y()
            try:
                text, ok = QInputDialog.getText(None, 'Annotations', 'Enter your annotation:')
                model.AnnotationAdd(x=x, y=y, color=annotationColor_, text=text)
            except Exception as ex:
                print(str(ex))

        if annotationDel_:
            mousePoint = self.mapSceneToView(ev.scenePos())
            x = mousePoint.x()
            y = mousePoint.y()
            try:
                model.AnnotationDel(x=x, y=y)
            except Exception as ex:
                print(str(ex))



    def mouseDragEvent(self, ev):
        global intervalType_
        global intervalAdd_

        if intervalAdd_:
            if ev.button() == QtCore.Qt.RightButton:
                ev.ignore()
            else:
                pg.ViewBox.mouseDragEvent(self, ev)
                if ev.isStart():    #first click before dragging
                    a = self.mapSceneToView(ev.scenePos())
                    self.pos1 = a.x()              #initial x mark
                    model.DrawMarkTime(self.pos1)     #temporary line on x mark
                elif ev.isFinish():  #release from dragging
                    model.RemoveMarkTime()   #remove line on x mark
                    a = self.mapSceneToView(ev.scenePos())
                    self.pos2 = a.x()              #final x mark
                    if self.pos1 is not None:
                        if self.pos2 < self.pos1:     #marking from right to left
                            BadInterval = [round(self.pos2, 3), round(self.pos1, 3)]
                        else:               #marking from left to right
                            BadInterval = [round(self.pos1, 3), round(self.pos2, 3)]
                        model.addBadTimeSeg(BadInterval)
                        model.refreshScreen()



def main(filename):
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = Application(filename)

    sys.exit(app.exec_())
