import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QSlider, QDoubleSpinBox, 
                             QGroupBox, QToolBar, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtGui

# Import your refactored piezo controller
from pze_control import PiezoController

from camera_control import CameraController

class QIUP_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. SETUP PATHS & WINDOW
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_path = os.path.join(self.base_dir, 'logo', 'Logo_ANT.png')
        
        self.setWindowTitle("QIUP Experimental Control")
        self.setWindowIcon(QtGui.QIcon(self.logo_path))
        self.setFixedSize(1200, 750)

        self.piezo = None
        self.init_ui()

    def init_ui(self):
        # --- TOOLBAR ---
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(self.toolbar.sizePolicy().Expanding, self.toolbar.sizePolicy().Preferred)
        self.toolbar.addWidget(spacer)

        # Connect Button
        self.btn_connect = QPushButton(" Connect Devices")
        if os.path.exists(self.logo_path):
            self.btn_connect.setIcon(QtGui.QIcon(self.logo_path))
        self.btn_connect.setFixedWidth(150)
        self.btn_connect.clicked.connect(self.attempt_connection)
        self.toolbar.addWidget(self.btn_connect)

        # Disconnect button
        self.btn_disconnect = QPushButton(" Disconnect")
        self.btn_disconnect.setFixedWidth(120)
        self.btn_disconnect.clicked.connect(self.disconnect_devices)
        self.btn_disconnect.setEnabled(False) # Disabled initially
        self.toolbar.addWidget(self.btn_disconnect)

        # --- MAIN LAYOUT ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- LEFT SIDE: CONTROLS ---
        self.controls_group = QGroupBox("Hardware Control")
        self.controls_group.setEnabled(False) 
        ctrl_layout = QVBoxLayout()

        v_label = QLabel("Piezo Voltage (Open Loop):")
        self.voltage_spin = QDoubleSpinBox()
        self.voltage_spin.setRange(0.0, 75.0)
        self.voltage_spin.setSuffix(" V")
        self.voltage_spin.setSingleStep(0.1)
        
        self.voltage_slider = QSlider(Qt.Horizontal)
        self.voltage_slider.setRange(0, 750) 
        
        self.voltage_slider.valueChanged.connect(lambda v: self.voltage_spin.setValue(v/10.0))
        self.voltage_spin.valueChanged.connect(self.update_piezo_voltage)

        ctrl_layout.addWidget(v_label)
        ctrl_layout.addWidget(self.voltage_spin)
        ctrl_layout.addWidget(self.voltage_slider)
        ctrl_layout.addStretch()
        
        self.controls_group.setLayout(ctrl_layout)
        self.main_layout.addWidget(self.controls_group, 1)

        # --- RIGHT SIDE: IMAGE DISPLAYS ---
        display_layout = QHBoxLayout()
        self.vis_container = self.create_image_placeholder("Visibility Map")
        self.contrast_container = self.create_image_placeholder("Contrast Image")
        self.phase_container = self.create_image_placeholder("Phase Reconstruction")
        
        display_layout.addWidget(self.vis_container)
        display_layout.addWidget(self.contrast_container)
        display_layout.addWidget(self.phase_container)
        
        self.main_layout.addLayout(display_layout, 4)

        self.statusBar().showMessage("System Offline - Please connect hardware.")

    def create_image_placeholder(self, title):
        container = QWidget()
        layout = QVBoxLayout(container)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        
        img_box = QLabel("No Data")
        img_box.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333; color: #555; border-radius: 4px;")
        img_box.setFixedSize(300, 300)
        img_box.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(img_box)
        container.img_display = img_box 
        return container

    def attempt_connection(self):
        self.statusBar().showMessage("Connecting...")
        self.btn_connect.setEnabled(False)
        QApplication.processEvents()
        
        self.piezo = PiezoController()
        success, message = self.piezo.connect()
        
        if success:
            self.statusBar().showMessage("Connected.")
            self.controls_group.setEnabled(True)
            self.btn_disconnect.setEnabled(True) # Enable disconnect
            self.btn_connect.setText(" Connected")
            self.btn_connect.setStyleSheet("background-color: #d4edda; color: #155724;")
            
            self.vis_container.img_display.setText("Ready for Acquisition")
            self.contrast_container.img_display.setText("Ready for Acquisition")
            self.phase_container.img_display.setText("Ready for Acquisition")
        else:
            self.statusBar().showMessage(f"Connection Failed: {message}")
            self.btn_connect.setEnabled(True)

    def disconnect_devices(self):
        """Safely shuts down hardware and resets UI state."""
        if self.piezo:
            self.piezo.disconnect()
            self.piezo = None
        
        # Reset UI
        self.controls_group.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText(" Connect Devices")
        self.btn_connect.setStyleSheet("") # Reset to default
        
        self.vis_container.img_display.setText("No Data")
        self.contrast_container.img_display.setText("No Data")
        self.phase_container.img_display.setText("No Data")
        
        self.statusBar().showMessage("Devices Disconnected.")

    def update_piezo_voltage(self):
        voltage = self.voltage_spin.value()
        if self.piezo and self.piezo.is_connected:
            self.piezo.set_voltage(voltage)
            self.voltage_slider.blockSignals(True)
            self.voltage_slider.setValue(int(voltage * 10))
            self.voltage_slider.blockSignals(False)
            self.statusBar().showMessage(f"Piezo set to {voltage:.2f} V")

if __name__ == "__main__":
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('applied.nano.qiup.v1')

    app = QApplication(sys.argv)
    window = QIUP_GUI()
    window.show()
    sys.exit(app.exec_())