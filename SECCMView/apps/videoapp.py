import sys
import os
import pandas as pd
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QLineEdit, QGroupBox, QFileDialog, QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph as pg

class ExperimentVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # State Variables
        self.df = None
        self.video_path = None
        self.cap = None
        self.time_offset = 0.0 
        
        # Data Arrays
        self.raw_times = np.array([])
        self.raw_voltage = np.array([])
        self.raw_current = np.array([])
        self.split_idx = 0
        
        # Playback State
        self.fps = 30.0
        self.total_frames = 0
        self.video_duration = 0
        self.ms_per_frame = 33.33
        self.is_playing = False
        
        # UI Setup
        self.init_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def init_ui(self):
        self.setWindowTitle("Experiment Visualizer (CV & Constant V)")
        self.resize(1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # ==================== LEFT SIDE (Controls & Video) ====================
        left_panel = QVBoxLayout()
        
        # --- 1. FILE LOADING AREA ---
        file_group = QGroupBox("1. File Selection")
        file_layout = QFormLayout_Custom() # Helper layout below
        
        self.btn_load_csv = QPushButton("Load CSV...")
        self.btn_load_csv.clicked.connect(self.load_csv_dialog)
        self.lbl_csv_name = QLabel("No CSV loaded")
        self.lbl_csv_name.setStyleSheet("color: #666;")
        
        self.btn_load_vid = QPushButton("Load Video...")
        self.btn_load_vid.clicked.connect(self.load_video_dialog)
        self.lbl_vid_name = QLabel("No Video loaded")
        self.lbl_vid_name.setStyleSheet("color: #666;")
        
        file_layout.addRow(self.btn_load_csv, self.lbl_csv_name)
        file_layout.addRow(self.btn_load_vid, self.lbl_vid_name)
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)

        # --- 2. VIDEO PLAYER ---
        self.video_label = QLabel("Please Load Video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444; color: white;")
        left_panel.addWidget(self.video_label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100) 
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.setEnabled(False) # Disabled until video loaded
        left_panel.addWidget(self.slider)
        
        controls_layout = QHBoxLayout()
        self.step_back_btn = QPushButton("<< Frame")
        self.step_back_btn.clicked.connect(lambda: self.step_frame(-1))
        controls_layout.addWidget(self.step_back_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setFixedWidth(100)
        controls_layout.addWidget(self.play_btn)

        self.step_fwd_btn = QPushButton("Frame >>")
        self.step_fwd_btn.clicked.connect(lambda: self.step_frame(1))
        controls_layout.addWidget(self.step_fwd_btn)
        left_panel.addLayout(controls_layout)
        
        self.time_label = QLabel("Video: 0.00s | Exp: 0.00s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        left_panel.addWidget(self.time_label)

        # --- 3. SYNC SETTINGS ---
        sync_group = QGroupBox("Synchronization")
        sync_layout = QHBoxLayout()
        
        self.vid_input = QLineEdit()
        self.vid_input.setPlaceholderText("0.000")
        self.vid_input.setText("0.000")
        self.vid_input.returnPressed.connect(self.apply_sync) 
        sync_layout.addWidget(QLabel("Video T (s):"))
        sync_layout.addWidget(self.vid_input)

        self.cv_input = QLineEdit()
        self.cv_input.setPlaceholderText("0.000")
        self.cv_input.setText("0.000")
        self.cv_input.returnPressed.connect(self.apply_sync)
        sync_layout.addWidget(QLabel("Exp T (s):"))
        sync_layout.addWidget(self.cv_input)
        
        self.apply_btn = QPushButton("Apply Sync")
        self.apply_btn.clicked.connect(self.apply_sync)
        sync_layout.addWidget(self.apply_btn)
        
        sync_group.setLayout(sync_layout)
        left_panel.addWidget(sync_group)
        
        main_layout.addLayout(left_panel, stretch=4)

        # ==================== RIGHT SIDE (Graph) ====================
        right_panel = QVBoxLayout()
        
        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        toolbar_layout.addWidget(QLabel("<b>Plot Mode:</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Current vs Voltage (CV)")
        self.combo_mode.addItem("Current vs Time (Constant V)")
        self.combo_mode.currentIndexChanged.connect(self.change_plot_mode)
        toolbar_layout.addWidget(self.combo_mode)
        
        toolbar_layout.addSpacing(20)
        
        self.btn_pan = QPushButton("Pan / Drag")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.clicked.connect(self.set_pan_mode)
        toolbar_layout.addWidget(self.btn_pan)

        self.btn_rect = QPushButton("Rect Zoom")
        self.btn_rect.setCheckable(True)
        self.btn_rect.clicked.connect(self.set_rect_mode)
        toolbar_layout.addWidget(self.btn_rect)

        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_plot_view)
        toolbar_layout.addWidget(self.btn_reset)
        
        right_panel.addLayout(toolbar_layout)

        # --- Plot Widget ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', False) 
        
        self.plot_widget = pg.PlotWidget(title="Experiment Data")
        self.plot_widget.setLabel('left', 'Current', units='A')
        self.plot_widget.setLabel('bottom', 'Voltage', units='V')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setClipToView(False) 
        
        # Initialize Curves (Empty for now)
        # Background traces (Faint)
        self.bg_fwd = self.plot_widget.plot(pen=pg.mkPen(color=(180, 220, 180), width=1))
        self.bg_rev = self.plot_widget.plot(pen=pg.mkPen(color=(220, 200, 180), width=1))
        
        # Active traces (Strong)
        self.active_fwd = self.plot_widget.plot(pen=pg.mkPen(color='#00AA00', width=2))
        self.active_rev = self.plot_widget.plot(pen=pg.mkPen(color='#FF8800', width=2))
        
        # Current Point
        self.current_point_marker = self.plot_widget.plot(symbol='o', symbolBrush='r', symbolSize=10)
        
        # Legend
        self.legend = self.plot_widget.addLegend()
        self.legend.addItem(self.active_fwd, "Phase 1 (Fwd)")
        self.legend.addItem(self.active_rev, "Phase 2 (Rev)")

        right_panel.addWidget(self.plot_widget)
        main_layout.addLayout(right_panel, stretch=5)

    # ==================== DATA LOADING ====================
    def load_csv_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '.', 'CSV Files (*.csv)')
        if fname:
            self.load_csv(fname)

    def load_csv(self, filepath):
        try:
            # Load and Preprocess
            data_tr = pd.read_csv(filepath)
            # Ensure columns exist (adjust indices if your format changes)
            df_cv = data_tr.iloc[:, [3, 5, 12]].copy()
            df_cv.columns = ['Voltage (V)', 'Current (A)', 'Delta Time (s)']
            df_cv['Cumulative Time (s)'] = df_cv['Delta Time (s)'].cumsum()
            
            # Clean
            cols = ['Voltage (V)', 'Current (A)', 'Cumulative Time (s)']
            for c in cols:
                df_cv[c] = pd.to_numeric(df_cv[c], errors='coerce')
            df_cv = df_cv.dropna(subset=cols)
            
            if df_cv.empty:
                raise ValueError("CSV is empty or columns not found.")

            self.df = df_cv
            self.lbl_csv_name.setText(os.path.basename(filepath))
            
            # Extract Arrays
            self.raw_times = self.df['Cumulative Time (s)'].values
            self.raw_voltage = self.df['Voltage (V)'].values
            self.raw_current = self.df['Current (A)'].values
            
            # Determine Split Point (Peak Voltage)
            self.split_idx = np.argmax(self.raw_voltage)
            
            # Refresh Plot
            self.change_plot_mode(self.combo_mode.currentIndex())
            
            # Reset Sync
            self.time_offset = 0.0
            self.cv_input.setText("0.000")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CSV", str(e))

    def load_video_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', '.', 'Video Files (*.mp4 *.avi *.mov)')
        if fname:
            self.load_video(fname)

    def load_video(self, filepath):
        self.video_path = filepath
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {filepath}")
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.ms_per_frame = (1000.0 / self.fps) if self.fps > 0 else 33.33
        self.timer_interval = int(self.ms_per_frame)
        
        self.lbl_vid_name.setText(os.path.basename(filepath))
        
        # Enable controls
        self.slider.setEnabled(True)
        self.slider.setRange(0, int(self.video_duration * 1000))
        self.slider.setValue(0)
        
        # Reset Sync
        self.vid_input.setText("0.000")
        self.update_display(0)

    # ==================== PLOT MODES ====================
    def change_plot_mode(self, index):
        if self.df is None:
            return

        # 0 = Current vs Voltage (CV)
        # 1 = Current vs Time (Constant V)
        is_cv_mode = (index == 0)

        # 1. Determine X-Axis Data
        if is_cv_mode:
            x_data = self.raw_voltage
            self.plot_widget.setLabel('bottom', 'Voltage', units='V')
            self.plot_widget.setTitle("Cyclic Voltammetry (I vs V)")
        else:
            x_data = self.raw_times
            self.plot_widget.setLabel('bottom', 'Time', units='s')
            self.plot_widget.setTitle("Current vs Time")

        y_data = self.raw_current

        # 2. Update Background Traces (Static)
        # Fwd Phase (0 to split)
        self.bg_fwd.setData(x_data[:self.split_idx], y_data[:self.split_idx])
        # Rev Phase (split to end)
        self.bg_rev.setData(x_data[self.split_idx:], y_data[self.split_idx:])

        # 3. Calculate View Bounds
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.1
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
        
        self.view_bounds_x = (x_min - x_pad, x_max + x_pad)
        self.view_bounds_y = (y_min - y_pad, y_max + y_pad)
        
        self.reset_plot_view()
        
        # 4. Trigger display update to draw Active Traces correctly
        self.update_display(self.slider.value())

    # ==================== CORE LOGIC ====================
    def update_display(self, time_ms):
        vid_time_sec = time_ms / 1000.0
        
        # --- A. Update Video ---
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(pixmap)
        
        # --- B. Update Plot ---
        if self.df is None:
            return

        target_exp_time = vid_time_sec + self.time_offset
        self.time_label.setText(f"Video: {vid_time_sec:.2f}s | Exp: {target_exp_time:.2f}s")

        if target_exp_time < 0:
            self.active_fwd.clear()
            self.active_rev.clear()
            self.current_point_marker.clear()
        else:
            # 1. Find current index based on TIME (always syncs by time)
            idx = np.searchsorted(self.raw_times, target_exp_time)
            
            # 2. Determine X-Axis Data based on current Mode
            is_cv_mode = (self.combo_mode.currentIndex() == 0)
            x_data = self.raw_voltage if is_cv_mode else self.raw_times
            y_data = self.raw_current

            # 3. Draw Curves (Split Logic)
            if idx <= self.split_idx:
                # In First Phase
                self.active_fwd.setData(x_data[:idx], y_data[:idx])
                self.active_rev.clear()
            else:
                # In Second Phase
                self.active_fwd.setData(x_data[:self.split_idx], y_data[:self.split_idx])
                self.active_rev.setData(x_data[self.split_idx:idx], y_data[self.split_idx:idx])

            # 4. Draw Red Dot Marker
            if idx < len(x_data):
                self.current_point_marker.setData([x_data[idx]], [y_data[idx]])

    # ==================== CONTROLS & EVENTS ====================
    def apply_sync(self):
        if self.df is None: return
        current_video_time_s = self.slider.value() / 1000.0
        
        try: vid_t = float(self.vid_input.text())
        except ValueError: vid_t = current_video_time_s
        
        try: exp_t = float(self.cv_input.text())
        except ValueError: exp_t = 0.0

        self.time_offset = exp_t - vid_t
        self.vid_input.setText(f"{vid_t:.3f}")
        self.cv_input.setText(f"{exp_t:.3f}")
        self.update_display(self.slider.value())
        self.vid_input.clearFocus()
        self.cv_input.clearFocus()

    def toggle_play(self):
        if self.video_path is None: return
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(self.timer_interval)
            self.play_btn.setText("Pause")
        self.is_playing = not self.is_playing

    def step_frame(self, direction):
        if self.video_path is None: return
        if self.is_playing: self.toggle_play()
        
        current_ms = self.slider.value()
        new_ms = current_ms + (direction * self.ms_per_frame)
        new_ms = max(0, min(new_ms, self.slider.maximum()))
        self.slider.setValue(int(new_ms))
        self.update_display(int(new_ms))

    def next_frame(self):
        current_slider_val = self.slider.value()
        next_val = current_slider_val + self.timer_interval
        if next_val >= self.slider.maximum():
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("Play")
        else:
            self.slider.setValue(next_val)
            self.update_display(next_val)

    def slider_moved(self, val):
        self.update_display(val)
        if not self.vid_input.hasFocus():
            self.vid_input.setText(f"{val / 1000.0:.3f}")

    def slider_pressed(self):
        if self.is_playing: self.timer.stop()

    def slider_released(self):
        if self.is_playing: self.timer.start(self.timer_interval)

    def set_pan_mode(self):
        self.btn_pan.setChecked(True)
        self.btn_rect.setChecked(False)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

    def set_rect_mode(self):
        self.btn_pan.setChecked(False)
        self.btn_rect.setChecked(True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def reset_plot_view(self):
        if hasattr(self, 'view_bounds_x'):
            self.plot_widget.setRange(xRange=self.view_bounds_x, yRange=self.view_bounds_y)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

# Helper class for sidebar layout
class QFormLayout_Custom(QVBoxLayout):
    def addRow(self, widget1, widget2):
        h = QHBoxLayout()
        h.addWidget(widget1)
        h.addWidget(widget2)
        self.addLayout(h)

# ==================== MAIN ====================
if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = ExperimentVisualizer()
    window.show()
    app.exec()