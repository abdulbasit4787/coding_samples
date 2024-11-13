from scipy.interpolate import CubicSpline
import pyqtgraph as pg
import time
import numpy as np
from scipy.signal import butter, filtfilt, lfilter_zi, lfilter
from pyqtgraph import BarGraphItem, LegendItem
import csv, logging, os, sys, struct, json, serial, tempfile, threading, queue, datetime
from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QIcon
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread, Qt, QTime, QDateTime


logging.basicConfig(level=logging.WARNING)

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

class BatteryWidget(QWidget):
    def __init__(self, parent=None):
        super(BatteryWidget, self).__init__(parent)
        self.battery_level = 1.0  # Battery level (0 to 1)
        self.battery_voltage = 4.2  # Battery voltage
        self.setMinimumSize(50, 25)
        self.setMaximumSize(50, 25)
        self.flash_on = False  # State for flashing effect
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self.toggle_flash)
        self.flash_timer.start(500)  # Flash every 500 ms

    def setBatteryLevel(self, level):
        self.battery_level = level
        self.update()

    def setBatteryVoltage(self, voltage):
        self.battery_voltage = voltage
        self.update()

    def toggle_flash(self):
        self.flash_on = not self.flash_on
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the battery body
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.drawRect(1, 1, self.width() - 7, self.height() - 2)

        # Draw the battery head
        painter.drawRect(self.width() - 6, int(self.height() / 4), 5, int(self.height() / 2))

        if self.battery_voltage < 3.7:
            if self.flash_on:
                # Draw flashing red slash
                pen = QPen(Qt.red, 4)
                painter.setPen(pen)
                painter.drawLine(1, self.height() - 2, self.width() - 7, 1)
        else:
            # Determine the battery color
            if self.battery_voltage < 3.8:
                color = QColor(Qt.red)
            else:
                color = QColor(Qt.green)

            # Draw the battery level
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            level_width = int((self.width() - 10) * self.battery_level)
            painter.drawRect(2, 2, level_width, self.height() - 4)

        painter.end()

class StickyMenu(QtWidgets.QMenu):
    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.pos())
        if action:
            action.trigger()
            event.accept()  # Prevent the menu from closing
        else:
            super().mouseReleaseEvent(event)  # Default behavior

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Worker(QObject):
    data_ready = pyqtSignal(list, list, list)

    def __init__(self, ui_mainwindow):
        super().__init__()
        self.ui_mainwindow = ui_mainwindow
        self.is_running = True
        self.data_buffer = ""

        """self.aabb_count_timer = QTimer()
        self.aabb_count_timer.timeout.connect(self.print_aabb_count)
        self.aabb_count_timer.start(1000)  # Trigger every second"""

    def run(self):
        try:
            self.ui_mainwindow.ser = serial.Serial(
                port=self.ui_mainwindow.get_current_serial_port(),
                baudrate=self.ui_mainwindow.get_current_baudrate(),
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            if not self.ui_mainwindow.ser.is_open:
                print("Failed to open serial port.")
                return
            else:
                self.ui_mainwindow.connection_btn.setIcon(QIcon(resource_path('logos/wifi.PNG')))
                self.read_and_process_data()
        except Exception as e:
            print(f"Failed to open serial port: {e}")
            self.ui_mainwindow.ser = None
            return

    def read_and_process_data(self):
        try:
            if self.ui_mainwindow.ser is None or not self.ui_mainwindow.ser.is_open:
                print("Serial port is not open.")
                return

            if self.ui_mainwindow.ser and self.ui_mainwindow.ser.is_open:
                if not hasattr(self.ui_mainwindow, 'commands_sent'):
                    self.ui_mainwindow.initial_command_sequence(self.ui_mainwindow.ser)
                    self.ui_mainwindow.ensure_command_response(self.ui_mainwindow.ser, f"AT+CONNECT=,{self.ui_mainwindow.device_id}")  # 00:3C:84:20:E9:B1")
                    self.ui_mainwindow.ensure_command_response(self.ui_mainwindow.ser, "AT+EXIT")
                    self.ui_mainwindow.commands_sent = True

            while self.ui_mainwindow.ser and self.ui_mainwindow.ser.is_open and self.is_running:
                new_data = self.ui_mainwindow.ser.read(self.ui_mainwindow.ser.in_waiting)  #(10000)
                if new_data:
                    self.data_buffer += new_data.hex().upper()
                    self.process_data_buffers()

        except Exception as e:
            print(f"Error during data reception: {e}")

    """def print_aabb_count(self):
        count = self.data_buffer.count('AABB')
        print(f"Number of AABB in the last second: {count}")
        self.data_buffer = ""  # Clear buffer after counting"""

    def process_data_buffers(self):
        formatted_data_list = []
        hand_grip_data = []
        battery_data = []

        while True:
            start_index_aabb = self.data_buffer.find("AABB")

            end_index = start_index_aabb + 78
            if end_index > len(self.data_buffer):
                break

            data_segment = self.data_buffer[start_index_aabb + 4 : end_index]
            calculated_checksum, received_checksum = self.ui_mainwindow.calculate_and_verify_checksum(data_segment)

            if calculated_checksum == received_checksum or calculated_checksum - 1 == received_checksum:
                serial_number_hex = data_segment[:8]
                serial_number_hex = ''.join(reversed([serial_number_hex[i:i + 2] for i in range(0, len(serial_number_hex), 2)]))
                serial_number = int(serial_number_hex, 16)
                aabb_data_segment = data_segment[8:56]
                formatted_data = self.ui_mainwindow.process_aabb_data(aabb_data_segment)
                for data in formatted_data:
                    data.insert(0, serial_number)
                formatted_data_list.extend(formatted_data)

                aacc_data_segment = data_segment[56:64]
                hand_grip = self.ui_mainwindow.process_aacc_data(aacc_data_segment)
                hand_grip_data.extend(hand_grip)

                aadd_data_segment = data_segment[64:72]
                battery = self.ui_mainwindow.process_aadd_data(aadd_data_segment)
                battery_data.extend(battery)

                self.data_buffer = self.data_buffer[end_index:]

            else:
                self.data_buffer = self.data_buffer[end_index:]

        self.data_ready.emit(formatted_data_list, hand_grip_data, battery_data)


    def stop(self):
        self.is_running = False
        if self.ui_mainwindow.ser and self.ui_mainwindow.ser.is_open:
            self.ui_mainwindow.ser.close()

class Ui_MainWindow(object):

    def __init__(self):
        self.data_buffers_v1 = [[] for _ in range(8)]
        self.data_buffers_v2 = [[] for _ in range(8)]
        self.data_buffers_v3 = [[] for _ in range(8)]
        self.data_buffers_v4 = []
        self.plot_curves_v1 = []
        self.plot_curves_v2 = []
        self.plot_curves_v3 = []
        self.bar_curve = []
        self.legend_items = []
        self.is_recording = False
        self.recording_timer = None
        self.last_image_time = None
        self.is_paused = False
        self.layout_actions = []
        self.connection = None
        self.ser = None
        self.b_lowpass = None
        self.a_lowpass = None
        self.zi_lowpass = None
        self.b_highpass = None
        self.a_highpass = None
        self.zi_highpass = None
        self.b_bandpass = None
        self.a_bandpass = None
        self.zi_bandpass = None
        self.b_bandstop = None
        self.a_bandstop = None
        self.zi_bandstop = None
        self.user_name = ""
        self.phone_number = ""
        self.gender = ""
        self.selected_folder = ""
        self.current_display_value = 0
        self.legend_item4 = None
        self.timer_started = False
        self.dots = [None] * 8
        self.text_items = [None] * 8
        self.dots2 = [None] * 8
        self.text_items2 = [None] * 8
        self.save_path = None
        self.device_id = ""
        self.record_id = 1
        self.folder_selected = False
        self.lower_range = 0.0
        self.higher_range = 0.0

    def setupUi(self, MainWindow):
        self.main_window = MainWindow
        self.MainWindow = MainWindow  # Store the MainWindow reference
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1751, 920)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Setting up a grid layout for centralwidget to manage dynamic resizing
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        # Setup for the first graph in graph_layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.graph_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widgets = [pg.PlotWidget(background='w') for _ in range(8)]
        for i, plot_widget in enumerate(self.plot_widgets):
            self.graph_layout.addWidget(plot_widget)
            plot_item = plot_widget.getPlotItem()
            plot_item.setLabels(bottom=('Time (s)',), left=('mV'))
            color = pg.intColor(i, hues=8)
            plot_item.showGrid(x=True, y=True, alpha=0.5)
            plot_curve = plot_item.plot(pen=color)
            self.plot_curves_v1.append(plot_curve)
            legend = LegendItem(offset=(70, 20))
            legend.setParentItem(plot_item.graphicsItem())
            self.legend_items.append(legend)
            legend.addItem(plot_curve, f'Channel {i + 1}')
            plot_widget.scene().sigMouseClicked.connect(lambda event, index=i: self.on_plot_click(event, index))
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.update_plot)
        self.timer1.start(50)  # Update interval in milliseconds

        # Setup for the second graph in joint_plot_layout
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.joint_plot_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.joint_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget2.setBackground('w')
        self.joint_plot_layout.addWidget(self.plot_widget2)
        plot_item2 = self.plot_widget2.getPlotItem()
        plot_item2.setLabels(bottom=('Time (s)',), left=('mV'))
        self.legend2 = LegendItem(offset=(70, 20))
        self.legend2.setParentItem(plot_item2.graphicsItem())
        for i in range(8):
            color = pg.intColor(i, hues=8)
            plot_curve = plot_item2.plot(pen=color)
            self.plot_curves_v2.append(plot_curve)
            self.legend2.addItem(plot_curve, f'Channel {i + 1}')
        plot_item2.showGrid(x=True, y=True, alpha=0.5)
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.update_plot2)
        self.timer2.start(50)  # Update interval in milliseconds

        # Setup for the third graph in extra_plot_layout
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.extra_plot_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.extra_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget3.setBackground('w')
        self.extra_plot_layout.addWidget(self.plot_widget3)
        plot_item3 = self.plot_widget3.getPlotItem()
        plot_item3.setLabels(bottom=('Time (s)',), left=('Amplitude'))
        legend3 = LegendItem(offset=(70, 20))
        legend3.setParentItem(plot_item3.graphicsItem())
        for i in range(8):
            color = pg.intColor(i, hues=8)
            plot_curve = plot_item3.plot(pen=color)
            self.plot_curves_v3.append(plot_curve)
            legend3.addItem(plot_curve, f'Channel {i + 1}')
        plot_item3.showGrid(x=True, y=True, alpha=0.5)
        plot_item3.ctrl.fftCheck.setChecked(True)
        self.plot_widget2.scene().sigMouseClicked.connect(lambda event, index=i: self.on_plot_click2(event, index))
        self.timer3 = QTimer()
        self.timer3.timeout.connect(self.update_plot3)
        self.timer3.start(50)  # Update interval in milliseconds

        # Setup for the new hand grip layout
        self.verticalLayoutWidget_9 = QtWidgets.QWidget(self.centralwidget)
        self.hand_grip_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_9)
        self.hand_grip_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget4 = pg.PlotWidget()
        self.plot_widget4.setBackground('w')
        self.hand_grip_layout.addWidget(self.plot_widget4)
        plot_item4 = self.plot_widget4.getPlotItem()
        self.legend4 = LegendItem(offset=(70, 20))
        self.legend4.setParentItem(plot_item4.graphicsItem())
        self.bar_item = BarGraphItem(x=[0.5], height=[0], width=0.5, brush='r')
        plot_item4.addItem(self.bar_item)
        self.legend_item4 = self.legend4.addItem(self.bar_item, 'Force (kg)')
        for sample in self.legend4.items:
            for item in sample:
                if isinstance(item, pg.graphicsItems.LabelItem.LabelItem):
                    item.setText(
                        f'<div style="font-size: 15pt; color: black; line-height: 1.2; vertical-align: baseline;">{item.text}</div>')
        plot_item4.setLabels(left=('Force (kg)'))
        plot_item4.setYRange(0, 52, padding=0)
        plot_item4.setXRange(0, 1, padding=0)
        plot_item4.showGrid(x=False, y=True, alpha=0.5)
        plot_item4.hideAxis('bottom')
        self.plot_widget4.getViewBox().setMenuEnabled(False)
        self.plot_widget4.scene().contextMenu = None
        plot_item4.setMouseEnabled(x=False, y=False)
        plot_item4.disableAutoRange()
        self.timer4 = QTimer()
        self.timer4.timeout.connect(self.update_plot4)
        self.timer4.start(1)


        # Connection Button setup
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.connection_btn_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.connection_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.connection_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.connection_btn.setObjectName("connection_btn")
        self.connection_btn.setIcon(QIcon(resource_path('logos/no-wifi.PNG')))
        self.connection_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.connection_btn.setMinimumWidth(120)
        self.connection_btn_layout.addWidget(self.connection_btn)
        self.connection_btn.clicked.connect(self.start_connection_thread)
        self.connection_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #555; /* Set border color and width */
                border-radius: 10px;    /* Set the roundness of the corners */
                padding: 5px;           /* Optional: Adjust padding */
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #fff, stop:1 #ccc); /* Gradient background */
                color: black;           /* Set text color */
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #ccc, stop:1 #fff); /* Change for hover */
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #aaa, stop:1 #eee); /* Change for pressed */
            }
        """)

        # Recorder Button setup
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.recorder_btn_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.recorder_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.recorder_btn = QtWidgets.QPushButton("Recorder", self.verticalLayoutWidget_5)
        self.recorder_btn.setObjectName("recorder_btn")
        self.recorder_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.recorder_btn.setMinimumWidth(120)
        self.recorder_btn_layout.addWidget(self.recorder_btn)
        self.recorder_btn.clicked.connect(self.toggle_recording)
        self.recorder_btn.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #555; /* Set border color and width */
                        border-radius: 10px;    /* Set the roundness of the corners */
                        padding: 5px;           /* Optional: Adjust padding */
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #fff, stop:1 #ccc); /* Gradient background */
                        color: black;           /* Set text color */
                    }
                    QPushButton:hover {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #ccc, stop:1 #fff); /* Change for hover */
                    }
                    QPushButton:pressed {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                    }
                """)

        # Clear Data Button setup
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.centralwidget)
        self.clear_data_btn_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.clear_data_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.clear_data_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_6)
        self.clear_data_btn.setObjectName("clear_data_btn")
        self.clear_data_btn.setIcon(QIcon(resource_path('logos/bin.PNG')))
        self.clear_data_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.clear_data_btn.setMinimumWidth(120)
        self.clear_data_btn_layout.addWidget(self.clear_data_btn)
        self.clear_data_btn.clicked.connect(self.clear_plots)
        self.clear_data_btn.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #555; /* Set border color and width */
                        border-radius: 10px;    /* Set the roundness of the corners */
                        padding: 5px;           /* Optional: Adjust padding */
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #fff, stop:1 #ccc); /* Gradient background */
                        color: black;           /* Set text color */
                    }
                    QPushButton:hover {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #ccc, stop:1 #fff); /* Change for hover */
                    }
                    QPushButton:pressed {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                    }
                """)

        # Pause/Resume Button setup
        self.verticalLayoutWidget_11 = QtWidgets.QWidget(self.centralwidget)
        self.pause_btn_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_11)
        self.pause_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.pause_btn_layout.setObjectName("pause_btn_layout")
        self.pause_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_11)
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.pause_btn.setMinimumWidth(120)
        self.pause_btn_layout.addWidget(self.pause_btn)
        self.pause_btn.clicked.connect(self.toggle_pause_resume)
        self.pause_btn.setIcon(QIcon(resource_path('logos/pause_play.PNG')))
        self.pause_btn.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #555; /* Set border color and width */
                        border-radius: 10px;    /* Set the roundness of the corners */
                        padding: 5px;           /* Optional: Adjust padding */
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #fff, stop:1 #ccc); /* Gradient background */
                        color: black;           /* Set text color */
                    }
                    QPushButton:hover {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #ccc, stop:1 #fff); /* Change for hover */
                    }
                    QPushButton:pressed {
                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                    }
                """)

        # Label 41 setup
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.centralwidget)
        self.label_41_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.label_41_layout.setContentsMargins(0, 0, 0, 0)
        self.label_41 = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.label_41.setObjectName("label_41")
        self.label_41_layout.addWidget(self.label_41)

        # Label 40 setup
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.centralwidget)
        self.label_40_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.label_40_layout.setContentsMargins(0, 0, 0, 0)
        self.label_40 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.label_40.setObjectName("label_40")
        self.label_40_layout.addWidget(self.label_40)

        # Battery icon setup
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.addWidget(self.label_40)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.battery_widget = BatteryWidget(self.verticalLayoutWidget_8)
        self.horizontalLayout.addWidget(self.battery_widget)

        # Treatment method setup
        self.treatment_method_layout = QtWidgets.QHBoxLayout()
        self.treatment_method_input = QtWidgets.QLineEdit(self.centralwidget)
        self.treatment_method_input.setPlaceholderText("Treatment Method")
        self.treatment_method_input.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.treatment_method_input.setMinimumWidth(150)
        self.treatment_method_layout.addWidget(self.treatment_method_input)

        # Range Calculator button
        self.verticalLayoutWidget_12 = QtWidgets.QWidget(self.centralwidget)
        self.range_calculator_layout = QtWidgets.QHBoxLayout(self.verticalLayoutWidget_12)
        self.range_calculator_layout.setContentsMargins(0, 0, 0, 0)
        self.range_calculator_button = QtWidgets.QPushButton(self.verticalLayoutWidget_12)
        self.range_calculator_button.setText("Range Calculator")
        self.range_calculator_button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.range_calculator_button.setMinimumWidth(120)
        self.range_calculator_layout.addWidget(self.range_calculator_button)
        self.range_calculator_button.clicked.connect(self.get_barrange_data)
        self.range_calculator_button.setStyleSheet("""
                                    QPushButton {
                                        border: 1px solid #555; /* Set border color and width */
                                        border-radius: 10px;    /* Set the roundness of the corners */
                                        padding: 5px;           /* Optional: Adjust padding */
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #fff, stop:1 #ccc); /* Gradient background */
                                        color: black;           /* Set text color */
                                    }
                                    QPushButton:hover {
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #ccc, stop:1 #fff); /* Change for hover */
                                    }
                                    QPushButton:pressed {
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                                    }
                                """)

        # Logout button setup
        self.verticalLayoutWidget_10 = QtWidgets.QWidget(self.centralwidget)
        self.logout_layout = QtWidgets.QHBoxLayout(self.verticalLayoutWidget_10)
        self.logout_layout.setContentsMargins(0, 0, 0, 0)
        self.logout_button = QtWidgets.QPushButton(self.verticalLayoutWidget_10)
        self.logout_button.setText("Logout")
        self.logout_button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.logout_button.setMinimumWidth(120)
        self.logout_layout.addWidget(self.logout_button)
        self.logout_button.clicked.connect(self.logout_function)
        self.logout_button.setStyleSheet("""
                            QPushButton {
                                border: 1px solid #555; /* Set border color and width */
                                border-radius: 10px;    /* Set the roundness of the corners */
                                padding: 5px;           /* Optional: Adjust padding */
                                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                  stop:0 #fff, stop:1 #ccc); /* Gradient background */
                                color: black;           /* Set text color */
                            }
                            QPushButton:hover {
                                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                  stop:0 #ccc, stop:1 #fff); /* Change for hover */
                            }
                            QPushButton:pressed {
                                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                  stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                            }
                        """)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuData_Management = QtWidgets.QMenu(self.menubar)
        self.menuFilters = QtWidgets.QMenu(self.menubar)

        # Setup "Device SR", "Window Size", and "Layout" menus within "Data Management"
        self.deviceSRMenu = StickyMenu("Device SR", self.centralwidget)
        self.windowSizeMenu = StickyMenu("Window Size", self.centralwidget)
        self.layoutMenu = StickyMenu("Layout", self.centralwidget)
        self.deviceSRActionGroup = QtWidgets.QActionGroup(self.deviceSRMenu)
        self.deviceSRActionGroup.setExclusive(True)
        for rate in ["500", "1000", "950"]:
            action = QtWidgets.QAction(rate, self.deviceSRMenu, checkable=True)
            self.deviceSRMenu.addAction(action)
            self.deviceSRActionGroup.addAction(action)
            if rate == "950":  # Default selection
                action.setChecked(True)

        # Window Size options
        self.windowSizeActionGroup = QtWidgets.QActionGroup(self.windowSizeMenu)
        self.windowSizeActionGroup.setExclusive(True)
        for size in ["100", "500", "950", "1900", "2850", "2000", "3000", "3200", "4800", "6000", "9000"]:
            action = QtWidgets.QAction(size, self.windowSizeMenu, checkable=True)
            self.windowSizeMenu.addAction(action)
            self.windowSizeActionGroup.addAction(action)
            if size == "950":  # Default selection
                action.setChecked(True)

        # Layout options
        self.layoutGraph1 = QtWidgets.QAction("Graph 1", self.layoutMenu, checkable=True)
        self.layoutGraph2 = QtWidgets.QAction("Graph 2", self.layoutMenu, checkable=True)
        self.layoutFFT = QtWidgets.QAction("FFT", self.layoutMenu, checkable=True)
        self.layout_actions = [self.layoutGraph1, self.layoutGraph2, self.layoutFFT]
        self.verticalLayoutWidget.show()
        self.verticalLayoutWidget_2.show()
        # self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 0, 10, 7)
        # self.verticalLayoutWidget_2.hide()
        self.verticalLayoutWidget_3.show()
        # for action in self.layout_actions:
        #    action.setCheckable(True)
        #    action.triggered.connect(self.adjust_layout_visibility)
        # self.layoutMenu.addAction(self.layoutGraph1)
        # self.layoutMenu.addAction(self.layoutGraph2)
        # self.layoutMenu.addAction(self.layoutFFT)
        # self.layoutGraph1.setChecked(False)
        # self.layoutGraph2.setChecked(False)
        # self.layoutFFT.setChecked(False)

        # Add submenus to the Data Management menu
        self.menuData_Management.addMenu(self.deviceSRMenu)
        self.menuData_Management.addMenu(self.windowSizeMenu)
        self.menuData_Management.addMenu(self.layoutMenu)

        # Setup Filters menu
        self.menuFilters = StickyMenu("Filters", self.menubar)
        self.menubar.addMenu(self.menuFilters)

        # Create submenus for each filter type
        self.lowPassMenu = StickyMenu("Low Pass", self.menuFilters)
        self.highPassMenu = StickyMenu("High Pass", self.menuFilters)
        self.bandPassMenu = StickyMenu("Band Pass", self.menuFilters)
        self.bandStopMenu = StickyMenu("Band Stop", self.menuFilters)
        self.selectedFilterMenu = StickyMenu("Selected Filter", self.menuFilters)

        # Add submenus to the Filters menu
        self.menuFilters.addMenu(self.lowPassMenu)
        self.menuFilters.addMenu(self.highPassMenu)
        self.menuFilters.addMenu(self.bandPassMenu)
        self.menuFilters.addMenu(self.bandStopMenu)
        self.menuFilters.addMenu(self.selectedFilterMenu)

        # Create action group for exclusive selection
        self.selectedFilterActionGroup = QtWidgets.QActionGroup(self.selectedFilterMenu)
        self.selectedFilterActionGroup.setExclusive(True)

        # Add filter options to the "Selected Filter" menu
        for filter_name in ["None", "Lowpass", "Highpass", "Bandpass", "Bandstop"]:
            action = QtWidgets.QAction(filter_name, self.selectedFilterMenu, checkable=True)
            self.selectedFilterMenu.addAction(action)
            self.selectedFilterActionGroup.addAction(action)
            if filter_name == "None":  # Default selection
                action.setChecked(True)

        # Add options for each submenu
        self.addFilterOptions(self.lowPassMenu, ["Cutoff Freq", "SF", "Order"])
        self.addFilterOptions(self.highPassMenu, ["Cutoff Freq", "SF", "Order"])
        self.addFilterOptions(self.bandPassMenu, ["Lowcut Freq", "Highcut Freq", "SF", "Order"])
        self.addFilterOptions(self.bandStopMenu, ["Lowcut Freq", "Highcut Freq", "SF", "Order"])

        # Adding Exit action to File menu
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.menuFile.addAction(self.actionExit)

        # Adding menus to the menubar
        self.menubar.addMenu(self.menuFile)
        self.menubar.addMenu(self.menuData_Management)
        self.menubar.addMenu(self.menuFilters)

        for action in self.windowSizeActionGroup.actions():
            action.triggered.connect(self.handle_window_size_change)

        self.actionExit.triggered.connect(MainWindow.close)

        self.gridLayout.addWidget(self.verticalLayoutWidget, 1, 0, 8, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_4, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_5, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_6, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_11, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_7, 0, 4, 1, 1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 5, 1, 1)
        self.gridLayout.addLayout(self.treatment_method_layout, 0, 6, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_12, 0, 7, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_10, 0, 8, 1, 1)
        self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 1, 5, 5)
        self.gridLayout.addWidget(self.verticalLayoutWidget_3, 6, 1, 3, 5)
        self.gridLayout.addWidget(self.verticalLayoutWidget_9, 1, 6, 8, 3)





        # serial port menu
        self.actionSerial_Port = StickyMenu("Serial Port", self.centralwidget)
        self.devicePortActionGroup = QtWidgets.QActionGroup(self.actionSerial_Port)
        self.devicePortActionGroup.setExclusive(True)
        for rate in ["COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9"]:
            action = QtWidgets.QAction(rate, self.actionSerial_Port, checkable=True)
            self.actionSerial_Port.addAction(action)
            self.devicePortActionGroup.addAction(action)
            if rate == "COM3":  # Default selection
                action.setChecked(True)

        # baud rate menu
        self.actionBaudrate = StickyMenu("Baudrate", self.centralwidget)
        self.deviceBaudrateActionGroup = QtWidgets.QActionGroup(self.actionBaudrate)
        for rate in ["921600", "115200", "48000", "1000000", "2000000"]:
            action = QtWidgets.QAction(rate, self.actionBaudrate, checkable=True)
            self.actionBaudrate.addAction(action)
            self.deviceBaudrateActionGroup.addAction(action)
            if rate == "921600":  # Default selection
                action.setChecked(True)


        self.menuFile.addMenu(self.actionSerial_Port)
        self.menuFile.addMenu(self.actionBaudrate)

        # Add Save Settings action in data management tab
        self.saveSettingsAction = QtWidgets.QAction("Save Settings", self.menuData_Management)
        self.menuData_Management.addAction(self.saveSettingsAction)
        self.saveSettingsAction.triggered.connect(self.save_settings)

        # Add Import Settings action in data management tab
        self.importSettingsAction = QtWidgets.QAction("Import Settings", self.menuData_Management)
        self.menuData_Management.addAction(self.importSettingsAction)
        self.importSettingsAction.triggered.connect(self.import_settings)

        # Add "hide all plot lines" in data management
        self.actionHidePlots = QtWidgets.QAction("Hide Plots", self.menuData_Management)
        self.menuData_Management.addAction(self.actionHidePlots)
        self.actionHidePlots.triggered.connect(self.hide_plot_lines)

        # Add "show all plot lines" in data management
        self.actionShowPlots = QtWidgets.QAction("Show Plots", self.menuData_Management)
        self.menuData_Management.addAction(self.actionShowPlots)
        self.actionShowPlots.triggered.connect(self.show_plot_lines)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.connection_btn.setText(_translate("MainWindow", "Connection"))
        self.recorder_btn.setText(_translate("MainWindow", "Recorder"))
        self.clear_data_btn.setText(_translate("MainWindow", "Clear Data"))
        self.label_41.setText(_translate("MainWindow", "StatusLabel"))
        self.label_40.setText(_translate("MainWindow", "StatusLabel"))
        self.pause_btn.setText(_translate("MainWindow", "Pause/Resume"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuData_Management.setTitle(_translate("MainWindow", "Data Management"))
        self.menuFilters.setTitle(_translate("MainWindow", "Filters"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.logout_button.setText(_translate("MainWindow", "Logout"))
        self.setup_time_display()

    def hide_plot_lines(self):
        for plot_curve in self.plot_curves_v1 + self.plot_curves_v2 + self.plot_curves_v3:
            plot_curve.setVisible(False)

        for plot_widget in self.plot_widgets:  # Update plot 1 after hiding plot lines
            plot_widget.update()
        self.plot_widget2.update()  # Update plot 2 after hiding plot lines
        self.plot_widget3.update()  # Update plot 3 after hiding plot lines

    def show_plot_lines(self):
        for plot_curve in self.plot_curves_v1 + self.plot_curves_v2 + self.plot_curves_v3:
            plot_curve.setVisible(True)

        for plot_widget in self.plot_widgets:  # Update plot 1 after hiding plot lines
            plot_widget.update()
        self.plot_widget2.update()  # Update plot 2 after hiding plot lines
        self.plot_widget3.update()  # Update plot 3 after hiding plot lines

    def save_settings(self):
        save_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if save_path:
            settings = {
                'device_sr': self.deviceSRActionGroup.checkedAction().text(),
                'window_size': self.windowSizeActionGroup.checkedAction().text(),
                'serial_port': self.devicePortActionGroup.checkedAction().text(),
                'baud_rate': self.deviceBaudrateActionGroup.checkedAction().text(),
                'selected_filter': self.selectedFilterActionGroup.checkedAction().text(),
                'lowpass': {
                    'cutoff_freq': self.extract_checked(self.lowPassMenu, "Cutoff Freq"),
                    'sf': self.extract_checked(self.lowPassMenu, "SF"),
                    'order': self.extract_checked(self.lowPassMenu, "Order")
                },
                'highpass': {
                    'cutoff_freq': self.extract_checked(self.highPassMenu, "Cutoff Freq"),
                    'sf': self.extract_checked(self.highPassMenu, "SF"),
                    'order': self.extract_checked(self.highPassMenu, "Order")
                },
                'bandpass': {
                    'lowcut_freq': self.extract_checked(self.bandPassMenu, "Lowcut Freq"),
                    'highcut_freq': self.extract_checked(self.bandPassMenu, "Highcut Freq"),
                    'sf': self.extract_checked(self.bandPassMenu, "SF"),
                    'order': self.extract_checked(self.bandPassMenu, "Order")
                },
                'bandstop': {
                    'lowcut_freq': self.extract_checked(self.bandStopMenu, "Lowcut Freq"),
                    'highcut_freq': self.extract_checked(self.bandStopMenu, "Highcut Freq"),
                    'sf': self.extract_checked(self.bandStopMenu, "SF"),
                    'order': self.extract_checked(self.bandStopMenu, "Order")
                }
            }
            with open(f"{save_path}/settings.json", 'w') as f:
                json.dump(settings, f, indent=4)

        if not save_path:
            return

    def extract_checked(self, menu, option):
        for submenu in menu.findChildren(StickyMenu):
            if submenu.title() == option:
                if submenu.actions():
                    for action in submenu.actions():
                        if action.isChecked():
                            return action.text()
        return None

    def import_settings(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select JSON File", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                settings = json.load(f)
                # Apply the settings using a helper function
                self.apply_action_by_text(self.deviceSRMenu, settings['device_sr'])
                self.apply_action_by_text(self.windowSizeMenu, settings['window_size'])
                self.apply_action_by_text(self.actionSerial_Port, settings['serial_port'])
                self.apply_action_by_text(self.actionBaudrate, settings['baud_rate'])
                self.apply_action_by_text(self.selectedFilterMenu, settings['selected_filter'])

                # Apply filter settings
                self.apply_specific_filter_settings(self.lowPassMenu, settings['lowpass'])
                self.apply_specific_filter_settings(self.highPassMenu, settings['highpass'])
                self.apply_specific_filter_settings(self.bandPassMenu, settings['bandpass'])
                self.apply_specific_filter_settings(self.bandStopMenu, settings['bandstop'])

        if not file_path:
            return

    def apply_action_by_text(self, menu, text):
        for action in menu.actions():
            if action.text() == text:
                action.trigger()
                break

    def apply_specific_filter_settings(self, menu, filter_settings):
        for sub_action in menu.actions():
            if sub_action.menu():  # Checks if the action opens a submenu
                submenu = sub_action.menu()
                for setting_key, setting_value in filter_settings.items():
                    if submenu.title().replace(" ", "").lower() == setting_key.replace("_", "").lower():
                        for action in submenu.actions():
                            if action.text() == str(setting_value):
                                action.trigger()
                                break

    def adjust_layout_visibility(self):
        graph1_visible = self.layoutGraph1.isChecked()
        graph2_visible = self.layoutGraph2.isChecked()
        fft_visible = self.layoutFFT.isChecked()

        # Show/hide and resize layouts based on the checked state
        if graph1_visible and not graph2_visible and not fft_visible:
            self.verticalLayoutWidget.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget, 1, 0, 10, 7)  # Expand to cover entire area
            self.verticalLayoutWidget_2.hide()
            self.verticalLayoutWidget_3.hide()
        elif graph2_visible and not graph1_visible and not fft_visible:
            self.verticalLayoutWidget.hide()
            self.verticalLayoutWidget_2.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 0, 10, 7)  # Expand to cover entire area
            self.verticalLayoutWidget_3.hide()
        elif fft_visible and not graph1_visible and not graph2_visible:
            self.verticalLayoutWidget.hide()
            self.verticalLayoutWidget_2.hide()
            self.verticalLayoutWidget_3.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_3, 1, 0, 10, 7)  # Expand to cover entire area
        elif graph1_visible and graph2_visible and not fft_visible:
            self.verticalLayoutWidget.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget, 1, 0, 10, 2)  # Adjust to take half vertical space
            self.verticalLayoutWidget_2.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 2, 10, 5)  # Adjust to take the other half
            self.verticalLayoutWidget_3.hide()
        elif graph2_visible and fft_visible and not graph1_visible:
            self.verticalLayoutWidget.hide()
            self.verticalLayoutWidget_2.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 0, 5, 7)  # Adjust to take half vertical space
            self.verticalLayoutWidget_3.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_3, 6, 0, 5, 7)  # Adjust to take the other half
        elif graph1_visible and fft_visible and not graph2_visible:
            self.verticalLayoutWidget.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget, 1, 0, 10, 2)  # Adjust to take half vertical space
            self.verticalLayoutWidget_2.hide()
            self.verticalLayoutWidget_3.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_3, 1, 2, 10, 5)  # Adjust to take the other half
        elif not graph1_visible and not graph2_visible and not fft_visible:
            self.verticalLayoutWidget.hide()
            self.verticalLayoutWidget_2.hide()
            self.verticalLayoutWidget_3.hide()
        else:
            # Default to showing all
            self.verticalLayoutWidget.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget, 1, 0, 10, 2)
            self.verticalLayoutWidget_2.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_2, 1, 2, 6, 5)
            self.verticalLayoutWidget_3.show()
            self.gridLayout.addWidget(self.verticalLayoutWidget_3, 7, 2, 4, 5)

    def addFilterOptions(self, menu, options):
        for option in options:
            subMenu = StickyMenu(option, menu)
            subMenu.setTitle(option)  # Ensure the title is set as expected
            actionGroup = QtWidgets.QActionGroup(subMenu)
            actionGroup.setExclusive(True)
            values = self.getOptionValues(option)
            for value in values:
                action = QtWidgets.QAction(str(value), subMenu, checkable=True)
                subMenu.addAction(action)
                actionGroup.addAction(action)
                if (option == "SF" and value == 1000) or (option == "Order" and value == 1):
                    action.setChecked(True)
            menu.addMenu(subMenu)

    def getOptionValues(self, option):
        if "Freq" in option:
            return [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    200]
        elif option == "SF":
            return [500, 1000, 16000]
        elif option == "Order":
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return []

    def setup_time_display(self):
        # GUI Start Time Setup
        self.start_time = QTime(0, 0, 0)  # Start at 00:00:00
        self.gui_start_timer = QTimer(self.centralwidget)
        self.gui_start_timer.timeout.connect(self.update_gui_start_time)
        self.label_41.setText(self.start_time.toString("HH:mm:ss"))
        font = self.label_41.font()
        font.setPointSize(14)  # Set font size to 14 points
        self.label_41.setFont(font)

        # Current Time Setup
        self.current_time_timer = QTimer(self.centralwidget)
        self.current_time_timer.timeout.connect(self.update_current_time)
        self.current_time_timer.start(1000)  # Update every second
        self.label_40.setText(QDateTime.currentDateTime().toString("HH:mm:ss"))
        font = self.label_40.font()
        font.setPointSize(14)  # Set font size to 14 points
        self.label_40.setFont(font)

    def update_gui_start_time(self):
        self.start_time = self.start_time.addSecs(1)
        self.label_41.setText(self.start_time.toString("HH:mm:ss"))

    def update_current_time(self):
        self.label_40.setText(QDateTime.currentDateTime().toString("HH:mm:ss"))

    def on_plot_click(self, event, plot_index):
        plot_widget = self.plot_widgets[plot_index]
        plot_item = plot_widget.getPlotItem()
        mouse_point = plot_item.vb.mapSceneToView(event.scenePos())

        if not self.legend_items[plot_index].sceneBoundingRect().contains(event.scenePos()):
            if event.button() == Qt.LeftButton:
                x_click, y_click = mouse_point.x(), mouse_point.y()
                curve = self.plot_curves_v1[plot_index]
                curve_data = curve.getData()

                # Calculate distances and find nearest point
                x_data, y_data = curve_data
                distances = np.sqrt((x_data - x_click) ** 2 + (y_data - y_click) ** 2)
                min_distance_index = np.argmin(distances)
                nearest_x, nearest_y = x_data[min_distance_index], y_data[min_distance_index]

                # Manage overlays efficiently
                if self.dots[plot_index] is not None:
                    plot_item.removeItem(self.dots[plot_index])
                if self.text_items[plot_index] is not None:
                    plot_item.removeItem(self.text_items[plot_index])

                self.dots[plot_index] = pg.ScatterPlotItem([nearest_x], [nearest_y], size=10, pen=pg.mkPen(None),
                                                           brush=pg.mkBrush('r'))
                plot_item.addItem(self.dots[plot_index])
                self.text_items[plot_index] = pg.TextItem(f'({nearest_x:.2f}, {nearest_y:.2f})', anchor=(0, 0))
                self.text_items[plot_index].setPos(nearest_x, nearest_y)
                plot_item.addItem(self.text_items[plot_index])

    def on_plot_click2(self, event, plot_index):
        plot_item2 = self.plot_widget2.getPlotItem()
        mouse_point = plot_item2.vb.mapSceneToView(event.scenePos())

        # Get the bounds of the legend item
        legend_bounds = self.legend2.sceneBoundingRect()

        # Check if the click event is outside the legend bounds
        if not legend_bounds.contains(event.scenePos()):
            if event.button() == Qt.LeftButton:
                x_click, y_click = mouse_point.x(), mouse_point.y()
                curve = self.plot_curves_v2[plot_index]
                curve_data = curve.getData()

                # Calculate distances and find nearest point
                x_data, y_data = curve_data
                distances = np.sqrt((x_data - x_click) ** 2 + (y_data - y_click) ** 2)
                min_distance_index = np.argmin(distances)
                nearest_x, nearest_y = x_data[min_distance_index], y_data[min_distance_index]

                # Manage overlays efficiently
                if self.dots2[plot_index] is not None:
                    plot_item2.removeItem(self.dots2[plot_index])
                if self.text_items2[plot_index] is not None:
                    plot_item2.removeItem(self.text_items2[plot_index])

                self.dots2[plot_index] = pg.ScatterPlotItem([nearest_x], [nearest_y], size=10, pen=pg.mkPen(None),
                                                            brush=pg.mkBrush('r'))
                plot_item2.addItem(self.dots2[plot_index])
                self.text_items2[plot_index] = pg.TextItem(f'({nearest_x:.2f}, {nearest_y:.2f})', anchor=(0, 0))
                self.text_items2[plot_index].setPos(nearest_x, nearest_y)
                plot_item2.addItem(self.text_items2[plot_index])

    def start_connection_thread(self):
        # Ensure any existing thread is stopped before starting a new one
        if hasattr(self, 'worker') and hasattr(self.worker, 'stop'):
            self.worker.stop()
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.thread.quit()
                self.thread.wait()

        self.thread = QThread()
        self.worker = Worker(self)
        self.worker.moveToThread(self.thread)
        self.worker.data_ready.connect(self.handle_data)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        print("Worker and thread started.")  # Debug statement

    def calculate_and_verify_checksum(self, data):
        def calculate_checksum_2s_complement(data_bytes):
            # Sum all byte values
            checksum = sum(data_bytes)

            # Take the 2's complement
            checksum = (-checksum) & 0xFF

            return checksum

        def calculate_combined_checksum(hex_string):
            # Convert hex string to integer bytes
            data_bytes = [int(hex_string[i:i + 2], 16) for i in range(0, len(hex_string), 2)]

            # Sum the bytes
            checksum = sum(data_bytes)

            return checksum

        # Extract the received checksum (last 2 characters)
        received_checksum = int(data[-2:], 16)

        # Calculate the checksum for the first 8 characters (count data)
        count_data = data[:8]
        count_checksum = calculate_combined_checksum(count_data)

        # Calculate the checksum for the first 48 characters in segments of 6
        segment_checksums = []
        for i in range(8, 56, 6):
            segment = data[i:i + 6]
            segment_checksum = calculate_combined_checksum(segment)
            segment_checksums.append(segment_checksum)

        # Calculate the checksum for the 8 characters (hand grip data)
        hand_grip_data = data[56:64]
        hand_grip_checksum = calculate_combined_checksum(hand_grip_data)

        # Calculate the checksum for the 8 characters (battery data)
        battery_data = data[64:72]
        battery_checksum = calculate_combined_checksum(battery_data)

        # Calculate the checksum for 'aa' and 'bb' combined
        additional_data_checksum = calculate_combined_checksum('aabb')

        # Append all calculated checksums to the segment checksums list
        segment_checksums.append(additional_data_checksum)
        segment_checksums.append(count_checksum)
        segment_checksums.append(hand_grip_checksum)
        segment_checksums.append(battery_checksum)

        # Calculate the final checksum
        final_checksum = calculate_checksum_2s_complement(segment_checksums)

        return final_checksum, received_checksum

    def butter_lowpass_filter(self, cutoff_freq, fs, order):
        """
        Applies a Butterworth low-pass filter to the data.

        Parameters:
        - data (numpy array): The input data from a single channel.
        - cutoff_freq (float): The cutoff frequency of the filter in Hz.
        - fs (float): The sampling rate of the data in Hz.
        - order (int): The order of the filter.

        Returns:
        - numpy array: The data after applying the low-pass filter.
        """
        nyquist_freq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist_freq
        self.b_lowpass, self.a_lowpass = butter(order, normal_cutoff, btype='low', analog=False)
        return self.b_lowpass, self.a_lowpass

    def butter_highpass_filter(self, cutoff_freq, fs, order):
        """
        Applies a Butterworth high-pass filter to a dataset.

        Parameters:
        - data (numpy array): The input data to be filtered.
        - cutoff_freq (float): The cutoff frequency of the high-pass filter in Hz.
        - fs (float): The sampling rate of the data in Hz.
        - order (int): The order of the filter. Higher order means a steeper filter.

        Returns:
        - numpy array: The filtered data.
        """
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff_freq / nyq
        self.b_highpass, self.a_highpass = butter(order, normal_cutoff, btype='high', analog=False)
        return self.b_highpass, self.a_highpass

    def butter_bandpass_filter(self, lowcut, highcut, fs, order):
        """
        Applies a Butterworth bandpass filter to a dataset.

        Parameters:
        - data (numpy array): The input data to be filtered.
        - lowcut (float): The lower frequency bound of the passband (Hz).
        - highcut (float): The upper frequency bound of the passband (Hz).
        - fs (float): The sampling rate of the data (Hz).
        - order (int): The order of the filter.

        Returns:
        - numpy array: The filtered data.
        """
        nyq = 0.5 * fs  # Nyquist Frequency
        low = lowcut / nyq
        high = highcut / nyq
        self.b_bandpass, self.a_bandpass = butter(order, [low, high], btype='bandpass', analog=False)
        return self.b_bandpass, self.a_bandpass

    def butter_bandstop_filter(self, lowcut, highcut, fs, order):
        """
        Applies a Butterworth bandstop filter to a dataset.

        Parameters:
        - data (numpy array): The input data to be filtered.
        - lowcut (float): The lower frequency bound of the stopband (Hz).
        - highcut (float): The upper frequency bound of the stopband (Hz).
        - fs (float): The sampling rate of the data (Hz).
        - order (int): The order of the filter.

        Returns:
        - numpy array: The filtered data.
        """
        nyq = 0.5 * fs  # Nyquist Frequency
        low = lowcut / nyq
        high = highcut / nyq
        self.b_bandstop, self.a_bandstop = butter(order, [low, high], btype='bandstop', analog=False)
        return self.b_bandstop, self.a_bandstop

    def send_command(self, ser, command):
        encoded_command = command.encode('gbk')
        if command != "+++":
            encoded_command += b'\r\n'
        ser.write(encoded_command)
        print(f"Sent command: {encoded_command.decode()}")
        # time.sleep(1)
        response = ser.read(ser.in_waiting)
        return response

    def ensure_command_response(self, ser, command):
        while True:
            response = self.send_command(ser, command)
            time.sleep(0.1)
            if "OK" in response.decode('latin1'):
                return response

    def initial_command_sequence(self, ser):
        while True:
            response = self.send_command(ser, "+++")
            time.sleep(0.1)
            if "OK" in response.decode('latin1'):
                break
            else:
                self.ensure_command_response(ser, "AT+EXIT")
        return

    def process_data(self, data_segment_aabb, data_segment_aacc, data_segment_aadd):
        if data_segment_aabb:
            self.process_aabb_data(data_segment_aabb)
        if data_segment_aacc:
            self.process_aacc_data(data_segment_aacc)
        if data_segment_aadd:
            self.process_aadd_data(data_segment_aadd)

    def process_aabb_data(self, data_segment):
        self.eight_channel_data_list = []
        if len(data_segment) == 48:
            channels_hex = [data_segment[i:i + 6] for i in range(0, len(data_segment), 6)]
            channels_int = []
            for channel_hex1 in channels_hex:
                bin_data = bin(int(channel_hex1, 16))[2:].zfill(24)
                if bin_data[0] == '1':
                    flipped_bin = ''.join('1' if b == '0' else '0' for b in bin_data)
                    int_data = -1 * (int(flipped_bin, 2) + 1)
                else:
                    int_data = int(bin_data, 2)
                channels_int.append(int_data)
            factor = (2 * 4500) / (2 ** 24)
            formatted_data = [round(x * factor, 2) for x in channels_int]
            """discard_packet = False
            threshold = 50  # burst limit
            for value in formatted_data:
                if value > threshold or value < -threshold:
                    discard_packet = True
                    break
            if not discard_packet:"""
            self.eight_channel_data_list.append(formatted_data)
        return self.eight_channel_data_list

    def process_aacc_data(self, data_segment):
        self.hand_grip_values = []
        if len(data_segment) == 8:
            high_word = int(data_segment[:4], 16)
            low_word = int(data_segment[4:], 16)
            combined_value = (high_word << 16) | low_word
            if high_word >= 0x8000:
                combined_value -= 0x100000000
            hand_grip_kg = combined_value / 100.0
            self.hand_grip_values.append(hand_grip_kg)
        return self.hand_grip_values

    def process_aadd_data(self, data_segment):
        self.battery_data = []
        if len(data_segment) == 8:
            reversed_battery_hex = data_segment[6:] + data_segment[4:6] + data_segment[2:4] + data_segment[:2]
            battery_voltage = struct.unpack('>f', bytes.fromhex(reversed_battery_hex))[0]
            battery_voltage = round(battery_voltage, 2)
            self.battery_data.append(battery_voltage)
        return self.battery_data

    def handle_data(self, formatted_data_list, hand_grip_data, battery_data):
        self.formatted_data_list = formatted_data_list
        self.hand_grip_data = hand_grip_data
        self.battery_data = battery_data
        self.update_battery_icon()

        if self.formatted_data_list:
            self.formatted_data2 = []
            for self.formatted_data2 in self.formatted_data_list:
                row_data = [self.record_id, self.formatted_data2[0]] + self.formatted_data2[1:9] + [
                    self.hand_grip_data[0] if self.hand_grip_data else None]
                if self.is_recording:
                    with open(self.csv_file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row_data)
                    self.record_id += 1


        if self.formatted_data_list:
            self.formatted_data3 = []
            for self.formatted_data3 in self.formatted_data_list:
                for i in range(8):
                    self.data_buffers_v1[i].append(self.formatted_data3[i + 1])
                    self.data_buffers_v2[i].append(self.formatted_data3[i + 1])
                    self.data_buffers_v3[i].append(self.formatted_data3[i + 1])
                    if len(self.data_buffers_v1[i]) > self.get_current_window_size():
                        self.data_buffers_v1[i].pop(0)
                    if len(self.data_buffers_v2[i]) > self.get_current_window_size():
                        self.data_buffers_v2[i].pop(0)
                    if len(self.data_buffers_v3[i]) > self.get_current_window_size():
                        self.data_buffers_v3[i].pop(0)

        if self.hand_grip_data:
            self.formatted_data4 = []
            for self.formatted_data4 in self.hand_grip_data:
                self.data_buffers_v4.append(self.formatted_data4)
                if len(self.data_buffers_v4) > self.get_current_window_size():
                    self.data_buffers_v4.pop(0)

    def low_pass_filter(self, data, cutoff_frequency, sampling_rate):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def start_gui_timer(self):
        self.gui_start_timer.start(1000)
        self.timer_started = True

    def update_plot(self):
        if not self.is_paused:
            selected_filter = self.get_checked_filter()
            sampling_rate = self.get_current_sampling_rate()
            data_received = False
            for i in range(8):
                self.plot_curves_v1[i].clear()
                if len(self.data_buffers_v1[i]) > 0:
                    data_received = True
                    filtered_data = self.apply_filter(selected_filter, self.data_buffers_v1[i])

                    # Apply the low-pass filter to reduce noise
                    #filtered_data = self.low_pass_filter(filtered_data, cutoff_frequency = 5, sampling_rate = sampling_rate)

                    x_data = np.array([index / self.get_current_sampling_rate() for index in range(len(filtered_data))])
                    y_data = np.array(filtered_data)

                    if len(x_data) > 1 and len(y_data) > 1:
                        cubic_interp = CubicSpline(x_data, y_data)
                        x_interp = np.linspace(x_data.min(), x_data.max(), num=len(x_data) * 10)
                        y_interp = cubic_interp(x_interp)
                        self.plot_curves_v1[i].setData(x_interp, y_interp)

            # Start the timer when data is received for the first time
            if data_received and not self.timer_started:
                self.start_gui_timer()

    def update_plot2(self):
        if not self.is_paused:
            selected_filter = self.get_checked_filter()
            sampling_rate = self.get_current_sampling_rate()
            for i, plot_curve2 in enumerate(self.plot_curves_v2):
                plot_curve2.clear()
                if len(self.data_buffers_v2[i]) > 0:
                    filtered_data = self.apply_filter(selected_filter, self.data_buffers_v2[i])

                    # Apply the low-pass filter to reduce noise
                    #filtered_data = self.low_pass_filter(filtered_data, cutoff_frequency=5, sampling_rate=sampling_rate)

                    x_data = np.array([index / self.get_current_sampling_rate() for index in range(len(filtered_data))])
                    y_data = np.array(filtered_data)

                    if len(x_data) > 1 and len(y_data) > 1:  # Ensure there are enough points to interpolate
                        cubic_interp = CubicSpline(x_data, y_data)
                        x_interp = np.linspace(x_data.min(), x_data.max(), num=len(x_data) * 10)
                        y_interp = cubic_interp(x_interp)
                        plot_curve2.setData(x_interp, y_interp)

    def update_plot3(self):
        if not self.is_paused:
            selected_filter = self.get_checked_filter()
            sampling_rate = self.get_current_sampling_rate()
            for i, plot_curve3 in enumerate(self.plot_curves_v3):
                plot_curve3.clear()
                if len(self.data_buffers_v3[i]) > 0:
                    filtered_data = self.apply_filter(selected_filter, self.data_buffers_v3[i])

                    # Apply the low-pass filter to reduce noise
                    #filtered_data = self.low_pass_filter(filtered_data, cutoff_frequency=5, sampling_rate=sampling_rate)

                    x_data = np.array([index / self.get_current_sampling_rate() for index in range(len(filtered_data))])
                    y_data = np.array(filtered_data)

                    if len(x_data) > 1 and len(y_data) > 1:  # Ensure there are enough points to interpolate
                        cubic_interp = CubicSpline(x_data, y_data)
                        x_interp = np.linspace(x_data.min(), x_data.max(), num=len(x_data) * 10)
                        y_interp = cubic_interp(x_interp)
                        plot_curve3.setData(x_interp, y_interp)

    def get_barrange_data(self):
        self.dialog = BarrangeCalculation()
        self.dialog.values_set.connect(lambda lower, higher: self.update_ranges(lower, higher))
        self.dialog.show()

    def update_ranges(self, lower, higher):
        self.lower_range = higher
        self.higher_range = lower

    def update_plot4(self):
        if not self.is_paused and self.data_buffers_v4:

            # Filter out values above the threshold
            threshold = 100
            filtered_values = [value for value in self.data_buffers_v4 if value <= threshold]
            if not filtered_values:
                return

            target_hand_grip_value = self.data_buffers_v4[-1] if self.data_buffers_v4 else 0

            current_display_value = getattr(self, 'current_display_value', 0)  # Get current displayed value or initialize it
            smoothing_factor = 0.1  # Control the smoothing effect (lower values are smoother but slower to respond)
            self.current_display_value = (current_display_value + smoothing_factor * (target_hand_grip_value - current_display_value))

            # Update the global maximum value
            if not hasattr(self, 'global_max_value'):
                self.global_max_value = self.current_display_value
            else:
                self.global_max_value = max(self.global_max_value, self.current_display_value)

            upper_level = self.higher_range
            lower_level = self.lower_range

            # Clear the previous bar
            self.plot_widget4.clear()

            # Determine the segment color based on the self.current_display_value
            if self.current_display_value <= lower_level:
                color = 'r'  # Below lower level
            elif lower_level < self.current_display_value <= upper_level:
                color = 'g'  # Within the normal range
            else:
                color = 'r'  # Above upper level

            # Create and add the new bar graph item directly
            bar = BarGraphItem(x=[0.5], height=[self.current_display_value], width=0.5, brush=pg.mkBrush(color))
            self.plot_widget4.addItem(bar)

            # Update the legend with the new display value and the global maximum value
            self.legend4.clear()
            self.legend_item4 = self.legend4.addItem(self.bar_item,
                                                     f'Force (kg): {self.current_display_value:.2f}  Max: {self.global_max_value:.2f}')

            # Set the font size and color for the legend item
            for sample in self.legend4.items:
                for item in sample:
                    if isinstance(item, pg.graphicsItems.LabelItem.LabelItem):
                        item.setText(
                            f'<div style="font-size: 15pt; color: black; line-height: 1.2; vertical-align: baseline;">{item.text}</div>')

            # Optionally, reset the x-range to keep it constant
            self.plot_widget4.getPlotItem().setXRange(0, 1, padding=0)

            # Schedule the next update
            if abs(self.current_display_value - target_hand_grip_value) > 0.1:  # Continue updating until the target is closely matched
                QtCore.QTimer.singleShot(10, self.update_plot4)  # Adjust this value to balance performance

    def update_battery_icon(self):
        if self.battery_data:
            battery_voltage = self.battery_data[-1]
            if battery_voltage < 3.5:
                battery_level = 0.0
            else:
                battery_level = (battery_voltage - 3.7) / 0.5  # Normalize battery voltage to range 0 to 1
            battery_level = max(0, min(battery_level, 1))  # Ensure the level is within [0, 1]
            self.battery_widget.setBatteryVoltage(battery_voltage)
            self.battery_widget.setBatteryLevel(battery_level)

    def toggle_recording(self):
        if not self.is_recording:
            if self.selected_folder:
                self.save_path = self.selected_folder
            elif not self.folder_selected:
                self.save_path = QFileDialog.getExistingDirectory(None, "Select Folder")
                if not self.save_path:
                    return
                self.folder_selected = True

            self.is_recording = True
            self.start_recording_timer()
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # Initialize the record ID counter
            self.record_id = 1

            print("text", self.treatment_method_input.text())

            if not self.selected_folder:
                user_folder_name = f"{self.user_name}-{self.gender}-{self.phone_number}"
                self.user_folder = os.path.join(self.save_path, user_folder_name)
                os.makedirs(self.user_folder, exist_ok=True)

                # Create CSV file in the user folder
                self.csv_file_path = os.path.join(self.user_folder, f"{user_folder_name}-{self.treatment_method_input.text()}-{current_time}.csv")
                with open(self.csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['ID', 'Serial', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
                                     'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8', 'Hand grip'])

                # Create image folder inside the user folder
                self.image_folder = os.path.join(self.user_folder, "images")
                os.makedirs(self.image_folder, exist_ok=True)

            else:
                self.user_folder = self.save_path
                folder_name = os.path.basename(os.path.normpath(self.save_path))

                self.csv_file_path = os.path.join(self.user_folder, f"{folder_name}-{self.treatment_method_input.text()}-{current_time}.csv")
                with open(self.csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['ID', 'Serial', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
                                     'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8', 'Hand grip'])

                # Create image folder inside the user-selected folder
                self.image_folder = os.path.join(self.user_folder, "images")
                os.makedirs(self.image_folder, exist_ok=True)

            self.recorder_btn.setIcon(QIcon('C:/Users/NICK/PycharmProjects/modi_projects/logos/recording.PNG'))
            self.recorder_btn.setStyleSheet("""
                                    QPushButton {
                                        border: 1px solid #555; /* Set border color and width */
                                        border-radius: 10px;    /* Set the roundness of the corners */
                                        padding: 5px;           /* Optional: Adjust padding */
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #fff, stop:1 #ccc); /* Gradient background */
                                        color: black;           /* Set text color */
                                    }
                                    QPushButton:hover {
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #ccc, stop:1 #fff); /* Change for hover */
                                    }
                                    QPushButton:pressed {
                                        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                          stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                                    }
                                """)

            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.capture_media)
            self.recording_timer.start(33)  # Approx 30 fps for video
        else:
            self.stop_recording()

    def start_recording_timer(self):
        self.record_start_time = QTime(0, 0, 0)
        self.recorder_btn.setText(self.record_start_time.toString("HH:mm:ss"))
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_recording_timer)
        self.timer.start(1000)  # Update every second

    def update_recording_timer(self):
        self.record_start_time = self.record_start_time.addSecs(1)
        self.recorder_btn.setText(self.record_start_time.toString("HH:mm:ss"))

    def stop_recording(self):
        self.is_recording = False
        self.recorder_btn.setStyleSheet("""
                                QPushButton {
                                    border: 1px solid #555; /* Set border color and width */
                                    border-radius: 10px;    /* Set the roundness of the corners */
                                    padding: 5px;           /* Optional: Adjust padding */
                                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                      stop:0 #fff, stop:1 #ccc); /* Gradient background */
                                    color: black;           /* Set text color */
                                }
                                QPushButton:hover {
                                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                      stop:0 #ccc, stop:1 #fff); /* Change for hover */
                                }
                                QPushButton:pressed {
                                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                                      stop:0 #aaa, stop:1 #eee); /* Change for pressed */
                                }
                            """)
        if self.recording_timer:
            self.recording_timer.stop()
        self.timer.stop()

    def capture_media(self):
        current_time = datetime.datetime.now()
        if not self.last_image_time or (current_time - self.last_image_time).total_seconds() >= 1:
            self.capture_image(current_time)
            self.last_image_time = current_time

    def capture_image(self, current_time):
        pixmap = self.centralwidget.grab()
        screenshot_path = os.path.join(self.image_folder, f"{current_time.strftime('%Y%m%d_%H%M%S')}.png")
        pixmap.save(screenshot_path, 'PNG')

    def apply_filter(self, selected_filter, data_buffer):
        if selected_filter == "Lowpass":
            cutoff_freq = self.get_checked_value(self.lowPassMenu, "Cutoff Freq")
            fs = self.get_checked_value(self.lowPassMenu, "SF")
            order = self.get_checked_value(self.lowPassMenu, "Order")

            self.b_lowpass, self.a_lowpass = self.butter_lowpass_filter(cutoff_freq, fs, order)
            self.zi_lowpass = lfilter_zi(self.b_lowpass, self.a_lowpass) * data_buffer[0]
            filtered_data, self.zi_lowpass = lfilter(self.b_lowpass, self.a_lowpass, data_buffer, zi=self.zi_lowpass)
            return filtered_data

        elif selected_filter == "Highpass":
            cutoff_freq = self.get_checked_value(self.highPassMenu, "Cutoff Freq")
            fs = self.get_checked_value(self.highPassMenu, "SF")
            order = self.get_checked_value(self.highPassMenu, "Order")

            self.b_highpass, self.a_highpass = self.butter_highpass_filter(cutoff_freq, fs, order)
            self.zi_highpass = lfilter_zi(self.b_highpass, self.a_highpass) * data_buffer[0]
            filtered_data, self.zi_highpass = lfilter(self.b_highpass, self.a_highpass, data_buffer,
                                                      zi=self.zi_highpass)
            return filtered_data

        elif selected_filter == "Bandpass":
            lowcut = self.get_checked_value(self.bandPassMenu, "Lowcut Freq")
            highcut = self.get_checked_value(self.bandPassMenu, "Highcut Freq")
            fs = self.get_checked_value(self.bandPassMenu, "SF")
            order = self.get_checked_value(self.bandPassMenu, "Order")

            self.b_bandpass, self.a_bandpass = self.butter_bandpass_filter(lowcut, highcut, fs, order)
            self.zi_bandpass = lfilter_zi(self.b_bandpass, self.a_bandpass) * data_buffer[0]
            filtered_data, self.zi_bandpass = lfilter(self.b_bandpass, self.a_bandpass, data_buffer,
                                                      zi=self.zi_bandpass)
            return filtered_data

        elif selected_filter == "Bandstop":
            lowcut = self.get_checked_value(self.bandStopMenu, "Lowcut Freq")
            highcut = self.get_checked_value(self.bandStopMenu, "Highcut Freq")
            fs = self.get_checked_value(self.bandStopMenu, "SF")
            order = self.get_checked_value(self.bandStopMenu, "Order")

            self.b_bandstop, self.a_bandstop = self.butter_bandstop_filter(lowcut, highcut, fs, order)
            self.zi_bandstop = lfilter_zi(self.b_bandstop, self.a_bandstop) * data_buffer[0]
            filtered_data, self.zi_bandstop = lfilter(self.b_bandstop, self.a_bandstop, data_buffer,
                                                      zi=self.zi_bandstop)
            return filtered_data

        elif selected_filter == "None":
            return data_buffer

    def get_checked_value(self, menu, option_name):
        """Retrieve the checked value from the submenu."""
        # print(f"Available actions in '{menu.title()}': {[action.text() for action in menu.actions()]}")
        for subMenuAction in menu.actions():
            if isinstance(subMenuAction, QtWidgets.QWidgetAction):
                if subMenuAction.defaultWidget().title() == option_name:
                    for action in subMenuAction.defaultWidget().actions():
                        if action.isChecked():
                            return float(action.text())
            elif isinstance(subMenuAction, QtWidgets.QAction) and subMenuAction.menu():
                if subMenuAction.menu().title() == option_name:
                    for action in subMenuAction.menu().actions():
                        if action.isChecked():
                            return float(action.text())
        # print(f"Submenu {option_name} not found in {menu.title()}")
        return None

    def get_checked_filter(self):
        """ Retrieve the currently checked filter from the Selected Filter menu. """
        for action in self.selectedFilterActionGroup.actions():
            if action.isChecked():
                return action.text()
        return "None"  # Default to 'None' if no action is selected

    def get_current_sampling_rate(self):
        """Retrieve the currently selected sampling rate from the Device SR menu."""
        for action in self.deviceSRActionGroup.actions():
            if action.isChecked():
                return float(action.text())

    def handle_window_size_change(self):
        new_window_size = self.get_current_window_size()
        # Ensure new_window_size is an integer
        new_window_size = int(new_window_size)
        for buffer_list in [self.data_buffers_v1, self.data_buffers_v2, self.data_buffers_v3]:
            for buffer in buffer_list:
                if len(buffer) > new_window_size:
                    # Ensure len(buffer) is an integer
                    buffer_length = int(len(buffer))
                    slice_index = max(0, buffer_length - new_window_size)
                    del buffer[:slice_index]

    def get_current_window_size(self):
        """Retrieve the currently selected window size from the Window Size menu."""
        for action in self.windowSizeActionGroup.actions():
            if action.isChecked():
                return float(action.text())
        return 1000  # Default size if none is selected

    def toggle_pause_resume(self):
        """Toggle the pause/resume state of the plots."""
        if self.is_paused:
            # Resume plotting
            self.timer1.start(50)
            self.timer2.start(50)
            self.timer3.start(50)
            self.timer4.start(10)
            self.pause_btn.setText("Pause")
            #self.pause_btn.setIcon(QIcon('C:/Users/NICK/PycharmProjects/modi_projects/logos/play.PNG'))
            self.pause_btn.setIcon(QIcon(resource_path('logos/play.PNG')))

        else:
            # Pause plotting
            self.timer1.stop()
            self.timer2.stop()
            self.timer3.stop()
            self.timer4.stop()
            self.pause_btn.setText("Resume")
            #self.pause_btn.setIcon(QIcon('C:/Users/NICK/PycharmProjects/modi_projects/logos/pause-button.PNG'))
            self.pause_btn.setIcon(QIcon(resource_path('logos/pause-button.PNG')))

        self.is_paused = not self.is_paused

    def clear_plots(self):
        # Clear data buffers
        for buffer_list in (self.data_buffers_v1, self.data_buffers_v2, self.data_buffers_v3):
            for buffer in buffer_list:
                buffer.clear()

        # Clear the hand grip data buffer
        self.data_buffers_v4.clear()

        # Clear plot curves
        for plot_curve in self.plot_curves_v1:
            plot_curve.setData([], [])
        for plot_curve in self.plot_curves_v2:
            plot_curve.setData([], [])
        for plot_curve in self.plot_curves_v3:
            plot_curve.setData([], [])

        # Clear dots and text items
        for i in range(8):
            if self.dots[i] is not None:
                self.plot_widgets[i].getPlotItem().removeItem(self.dots[i])
                self.dots[i] = None
            if self.text_items[i] is not None:
                self.plot_widgets[i].getPlotItem().removeItem(self.text_items[i])
                self.text_items[i] = None

            if self.dots2[i] is not None:
                self.plot_widget2.getPlotItem().removeItem(self.dots2[i])
                self.dots2[i] = None
            if self.text_items2[i] is not None:
                self.plot_widget2.getPlotItem().removeItem(self.text_items2[i])
                self.text_items2[i] = None

        # Clear hand grip plot and legend
        self.plot_widget4.clear()
        self.legend4.clear()
        self.bar_curve = []
        self.global_max_value = 0  # Reset the global maximum value

    def get_current_serial_port(self):
        """Retrieve the currently selected sampling rate from the Device SR menu."""
        for action in self.devicePortActionGroup.actions():
            if action.isChecked():
                return action.text()
        return "COM4"  # Default to 1000 Hz if no action is selected

    def get_current_baudrate(self):
        """Retrieve the currently selected sampling rate from the Device SR menu."""
        for action in self.deviceBaudrateActionGroup.actions():
            if action.isChecked():
                return float(action.text())
        return 921600  # Default to 1000 Hz if no action is selected

    def logout_function(self):
        self.main_window.hide()
        get_user_data(self)


class BarrangeCalculation(QtWidgets.QDialog):
    values_set = QtCore.pyqtSignal(float, float)  # Signal to emit lower and upper values

    def __init__(self, parent=None):
        super(BarrangeCalculation, self).__init__(parent, QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowTitle("Bar Range Calculation")

        layout = QtWidgets.QVBoxLayout(self)


        # Higher value
        higher_layout = QtWidgets.QHBoxLayout()
        higher_label = QtWidgets.QLabel("Higher Value:")
        self.higher_value = QtWidgets.QLabel("")
        higher_layout.addWidget(higher_label)
        higher_layout.addWidget(self.higher_value)
        layout.addLayout(higher_layout)

        # Lower value
        lower_layout = QtWidgets.QHBoxLayout()
        lower_label = QtWidgets.QLabel("Lower Value:")
        self.lower_value = QtWidgets.QLabel("")
        lower_layout.addWidget(lower_label)
        lower_layout.addWidget(self.lower_value)
        layout.addLayout(lower_layout)

        # Enter value text box
        self.enter_value_textbox = QtWidgets.QLineEdit()
        self.enter_value_textbox.setPlaceholderText("Enter float value")
        float_validator = QtGui.QDoubleValidator()
        float_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.enter_value_textbox.setValidator(float_validator)
        layout.addWidget(self.enter_value_textbox)

        # Buttons 20% and 50%
        buttons_layout = QtWidgets.QHBoxLayout()
        self.twenty_percent_button = QtWidgets.QPushButton("20%")
        self.fifty_percent_button = QtWidgets.QPushButton("50%")
        buttons_layout.addWidget(self.twenty_percent_button)
        buttons_layout.addWidget(self.fifty_percent_button)
        layout.addLayout(buttons_layout)

        # Set button
        self.set_button = QtWidgets.QPushButton("Set")
        layout.addWidget(self.set_button)

        # Close window button
        self.close_button = QtWidgets.QPushButton("Close Window")
        layout.addWidget(self.close_button)

        # Connections
        self.enter_value_textbox.textChanged.connect(self.enable_set_button)
        self.set_button.clicked.connect(self.set_values)
        self.close_button.clicked.connect(self.close)
        self.twenty_percent_button.clicked.connect(lambda: self.calculate_percent(0.2))
        self.fifty_percent_button.clicked.connect(lambda: self.calculate_percent(0.5))

        # Initially disable Set button
        self.set_button.setEnabled(False)

    def enable_set_button(self, text):
        self.set_button.setEnabled(bool(text.strip()))

    def calculate_percent(self, percent):
        entered_value = float(self.enter_value_textbox.text())
        self.higher_range = entered_value * percent * 1.1
        self.lower_range = entered_value * percent * 0.9
        self.higher_value.setText(f"{self.higher_range:.2f}")
        self.lower_value.setText(f"{self.lower_range:.2f}")


    def set_values(self):
        rounded_higher = round(self.higher_range)
        rounded_lower = round(self.lower_range)
        self.values_set.emit(rounded_higher, rounded_lower)

def get_user_data(ui_mainwindow):
    dialog = UserInputDialog(ui_mainwindow.device_id)
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        user_data = dialog.get_inputs()
        ui_mainwindow.user_name = user_data["name"]  # Store the user's name globally in the main UI class
        ui_mainwindow.gender = user_data["gender"]  # Store the user's name globally in the main UI class
        ui_mainwindow.phone_number = user_data["phone_number"]  # Store the phone number globally in the main UI class
        ui_mainwindow.selected_folder = user_data["folder"]  # Store the selected folder globally in the main UI class
        ui_mainwindow.device_id = user_data["device_id"]  # Store the device ID globally in the main UI class
        ui_mainwindow.main_window.show()

class UserInputDialog(QtWidgets.QDialog):

    def __init__(self, last_device_id="", parent=None):

        super(UserInputDialog, self).__init__(parent)
        self.setWindowTitle("User Information")

        # Define the file to store user names
        self.NAMES_FILE = os.path.join(tempfile.gettempdir(), 'user_names.json')
        self.DEVICE_IDS_FILE = os.path.join(tempfile.gettempdir(), 'device_ids.json')
        self.selected_folder = None  # Variable to store the selected folder path

        layout = QtWidgets.QVBoxLayout(self)

        # Load previously used names and device IDs
        self.names = self.load_names()
        self.device_ids = self.load_device_ids()

        # Name
        self.name_label = QtWidgets.QLabel("Name:")
        self.name_input = QtWidgets.QLineEdit()
        completer = QtWidgets.QCompleter(self.names, self.name_input)
        self.name_input.setCompleter(completer)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # Gender
        self.gender_label = QtWidgets.QLabel("Gender:")
        self.gender_input = QtWidgets.QComboBox()
        self.gender_input.addItems(["Male", "Female", "Other"])
        layout.addWidget(self.gender_label)
        layout.addWidget(self.gender_input)

        # phone_number
        self.phone_number_label = QtWidgets.QLabel("Phone number:")
        self.phone_number_input = QtWidgets.QLineEdit()
        self.phone_number_input.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r'\d{11}')))  # Allow exactly 10 digits
        layout.addWidget(self.phone_number_label)
        layout.addWidget(self.phone_number_input)

        # Device ID
        self.device_id_label = QtWidgets.QLabel("Device ID:")
        self.device_id_input = QtWidgets.QLineEdit()
        if self.device_ids:  # Check if there are any saved device IDs
            self.device_id_input.setText(self.device_ids[0])  # Set the last device ID from the list
        device_id_completer = QtWidgets.QCompleter(self.device_ids, self.device_id_input)
        self.device_id_input.setCompleter(device_id_completer)
        layout.addWidget(self.device_id_label)
        layout.addWidget(self.device_id_input)



        # Folder selection button
        self.folder_button = QtWidgets.QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_button)

        # OK Button
        self.ok_button = QtWidgets.QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        # Apply the stylesheet to the buttons
        self.ok_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #555; /* Set border color and width */
                border-radius: 10px;    /* Set the roundness of the corners */
                padding: 5px;           /* Optional: Adjust padding */
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #fff, stop:1 #ccc); /* Gradient background */
                color: black;           /* Set text color */
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #ccc, stop:1 #fff); /* Change for hover */
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #aaa, stop:1 #eee); /* Change for pressed */
            }
        """)
        self.folder_button.setStyleSheet(self.ok_button.styleSheet())

        # Remove the help button from the dialog window
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    def select_folder(self):
        self.selected_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.selected_folder:
            QtWidgets.QMessageBox.information(self, "Folder Selected", f"Selected folder: {self.selected_folder}")
            self.accept()  # Accept the dialog and proceed to the main window

    def accept(self):
        # Validate inputs
        if not self.device_id_input.text():
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter the Device ID.")
        elif not self.selected_folder and (
                not self.name_input.text() or not self.phone_number_input.text() or self.gender_input.currentText() not in [
            "Male", "Female", "Other"]):
            QtWidgets.QMessageBox.warning(self, "Input Error",
                                          "Please enter all fields: Name, Phone number, and Gender or select a folder.")
        else:
            # Save the name and device ID if they are new
            name = self.name_input.text()
            if name and name not in self.names:
                self.names.append(name)
                self.save_names(self.names)
            device_id = self.device_id_input.text()

            if device_id:
                self.save_device_ids(device_id)

            super(UserInputDialog, self).accept()

    def get_inputs(self):
        return {
            "name": self.name_input.text(),
            "gender": self.gender_input.currentText(),
            "phone_number": self.phone_number_input.text(),
            "device_id": self.device_id_input.text(),
            "folder": self.selected_folder
        }

    def load_names(self):
        if os.path.exists(self.NAMES_FILE):
            with open(self.NAMES_FILE, 'r') as file:
                return json.load(file)
        return []

    def save_names(self, names):
        with open(self.NAMES_FILE, 'w') as file:
            json.dump(names, file)

    def load_device_ids(self):
        if os.path.exists(self.DEVICE_IDS_FILE):
            with open(self.DEVICE_IDS_FILE, 'r') as file:
                return json.load(file)
        return []

    def save_device_ids(self, device_id):
        existing_ids = self.load_device_ids()
        if device_id in existing_ids:
            existing_ids.remove(device_id)  # Remove existing entry to avoid duplicates
        existing_ids.insert(0, device_id)  # Insert the new ID at the front

        # Save the updated list back to the JSON file
        with open(self.DEVICE_IDS_FILE, 'w') as file:
            json.dump(existing_ids, file)

class CustomSplashScreen(QtWidgets.QSplashScreen):

    def __init__(self, pixmap, flags):
        super().__init__(pixmap, flags)

    # Override the mousePressEvent to ignore all mouse clicks
    def mousePressEvent(self, event):
        pass

def showSplashScreen(app):
    animation_display_time = 2000  # Display seconds
    splash_pix = QtGui.QPixmap(resource_path("logos/logo-no-background.png"))
    splash_pix = splash_pix.scaled(320, 240, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    splash = CustomSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setWindowOpacity(0.9)
    splash.show()

    effect = QtWidgets.QGraphicsOpacityEffect()
    splash.setGraphicsEffect(effect)

    anim = QtCore.QPropertyAnimation(effect, b"opacity")
    anim.setDuration(animation_display_time)
    anim.setStartValue(1)
    anim.setEndValue(0)
    anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

    # Ensure the animation object is not garbage collected
    app.animation = anim

    anim.start()
    QtCore.QTimer.singleShot(animation_display_time, splash.close)
    return splash

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui_display_time = 1500

    splash = showSplashScreen(app)

    # Prepare the main window
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.main_window = MainWindow  # Store a reference to MainWindow in the UI class

    QtCore.QTimer.singleShot(gui_display_time, lambda: get_user_data(ui))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()