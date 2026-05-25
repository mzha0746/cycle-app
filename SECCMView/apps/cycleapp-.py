import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch, find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QFileDialog, 
                             QFormLayout, QGroupBox, QMessageBox,
                             QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# =============================================================================
#  1. DATA STRUCTURES
# =============================================================================
class ExperimentData:
    def __init__(self, df):
        self.df = df
        self.t = df['Time'].values
        self.i = df['Current'].values
        self.v = df['Voltage'].values
        self.line = df['Line'].values
        
        self.unique_lines = np.unique(self.line)
        self.unique_lines.sort()

# =============================================================================
#  2. ANALYSIS STRATEGIES
# =============================================================================
class AnalysisStrategy:
    def name(self): raise NotImplementedError
    def get_params(self): return {}
    def process(self, data, params): raise NotImplementedError
    def visualize_results(self, layout_container, result_data): raise NotImplementedError

# --- MODE 1: LINE INSPECTOR ---
class LineInspectorStrategy(AnalysisStrategy):
    def name(self): return "1. Line Number Inspector"
    
    def process(self, data, params):
        return {'data': data}

    def visualize_results(self, container, res):
        container.clear()
        plot = container.addPlot()
        plot.setTitle("Current vs Time (Colored by Line Number)")
        plot.setLabel('left', 'Current', 'A')
        plot.setLabel('bottom', 'Time', 's')
        
        data = res['data']
        for idx, line_num in enumerate(data.unique_lines):
            mask = (data.line == line_num)
            t_chunk = data.t[mask]
            i_chunk = data.i[mask]
            if len(t_chunk) == 0: continue
            
            color = pg.intColor(idx, hues=15, alpha=200) 
            plot.plot(t_chunk, i_chunk, pen=pg.mkPen(color, width=2), name=f"Line {line_num}")

# --- MODE 2: SINGLE CYCLE ---
class SingleCycleStrategy(AnalysisStrategy):
    def name(self): return "2. Single Cycle Extractor"
    
    def get_params(self):
        return {
            'Auto Detect Cycle': {'type': 'bool', 'default': True}, 
            'Target Cycle': {'type': 'int', 'default': 1},
            'Start Line Number': {'type': 'int', 'default': 5},
            'Prominence (V)': {'type': 'float', 'default': 0.1},
            'Smooth Window': {'type': 'int', 'default': 51},
        }

    def process(self, data, params):
        cycle = params['Target Cycle']
        auto_detect = params['Auto Detect Cycle']

        if not auto_detect:
            start = params['Start Line Number']
            base = start + 2 * (cycle - 1)
            lines = [base, base + 1]
            mask = np.isin(data.line, lines)
            if not np.any(mask): return {'error': f"Lines {lines} not found."}
            return {'v': data.v[mask], 'i': data.i[mask], 'cycle': cycle}
        else:
            win_len = params['Smooth Window']
            if win_len % 2 == 0: win_len += 1
            prom = params['Prominence (V)']
            if len(data.v) > win_len:
                v_smooth = savgol_filter(data.v, win_len, 3)
            else:
                v_smooth = data.v
            peaks, _ = find_peaks(v_smooth, prominence=prom)
            valleys, _ = find_peaks(-v_smooth, prominence=prom)
            turnarounds = np.sort(np.concatenate(([0], peaks, valleys, [len(v_smooth)-1])))
            start_idx_pos = 2 * (cycle - 1)
            end_idx_pos = 2 * cycle
            if end_idx_pos < len(turnarounds):
                start_idx = turnarounds[start_idx_pos]
                end_idx = turnarounds[end_idx_pos]
                return {'v': data.v[start_idx:end_idx], 'i': data.i[start_idx:end_idx], 'cycle': cycle}
            else:
                return {'error': f"Auto detect failed for Cycle {cycle}."}

    def visualize_results(self, container, res):
        container.clear()
        plot = container.addPlot()
        if 'error' in res:
            plot.setTitle(res['error'])
            return
        plot.setTitle(f"Cycle {res['cycle']}")
        plot.setLabel('left', 'Current', 'A')
        plot.setLabel('bottom', 'Voltage', 'V')
        plot.plot(res['v'], res['i'], pen=pg.mkPen('c', width=3))

# --- MODE 3: RANGE FFT (Swapped with original Mode 4) ---
class RangeCyclesFFTStrategy(AnalysisStrategy):
    def name(self): return "3. Range Cycles FFT (Smoothable)"
    
    def get_params(self):
        return {
            'Start Line Number': {'type': 'int', 'default': 5},
            'Start Cycle': {'type': 'int', 'default': 1},
            'End Cycle': {'type': 'int', 'default': 100},
            'Detrend Poly Order': {'type': 'int', 'default': 3},
            'Enable Smoothing': {'type': 'bool', 'default': False},
            'Smooth Window': {'type': 'int', 'default': 11},
            'Smooth Poly': {'type': 'int', 'default': 2},
        }

    def process(self, data, params):
        start_line = params['Start Line Number']
        c_start = params['Start Cycle']
        c_end = params['End Cycle']
        poly_order = params['Detrend Poly Order']
        do_smooth = params['Enable Smoothing']
        win_len = params['Smooth Window']
        if win_len % 2 == 0: win_len += 1
        poly_smooth = params['Smooth Poly']
        
        if c_start > c_end: return {'error': "Start > End Cycle."}
        
        results = []
        for c_idx in range(c_start, c_end + 1):
            base = start_line + 2 * (c_idx - 1)
            lines = [base, base + 1]
            mask = np.isin(data.line, lines)
            if np.any(mask):
                t_seg = data.t[mask]
                i_seg = data.i[mask]
                dt_vals = np.diff(t_seg)
                if len(dt_vals) == 0 or np.mean(dt_vals) == 0: continue
                fs = 1.0 / np.mean(dt_vals)
                if poly_order > 0:
                    t_local = t_seg - t_seg[0]
                    coeffs = np.polyfit(t_local, i_seg, poly_order)
                    signal_to_fft = i_seg - np.polyval(coeffs, t_local)
                else:
                    signal_to_fft = i_seg
                n = len(signal_to_fft)
                nperseg_val = max(256, n // 4)
                if nperseg_val > n: nperseg_val = n
                freqs, power_density = welch(signal_to_fft, fs=fs, window='hann', nperseg=nperseg_val, scaling='density')
                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                power_pos = power_density[pos_mask]
                if do_smooth and len(power_pos) > win_len:
                    try:
                        power_pos = savgol_filter(power_pos, win_len, poly_smooth)
                        power_pos = np.maximum(power_pos, 1e-20) 
                    except: pass
                results.append({'num': c_idx, 'freqs': freqs_pos, 'power': power_pos})
        if not results: return {'error': "No data found."}
        return {'cycles': results, 'min_c': c_start, 'max_c': c_end}

    def visualize_results(self, container, res):
        container.clear()
        plot = container.addPlot()
        plot.setLogMode(x=True, y=True)
        if 'error' in res:
            plot.setTitle(res['error'])
            return
        cycles = res['cycles']
        c_min, c_max = res['min_c'], res['max_c']
        total_range = c_max - c_min if c_max > c_min else 1
        plot.setTitle(f"FFT Overlay: Cycles {c_min}-{c_max}")
        plot.setLabel('left', 'Power Density (Log)', 'A^2/Hz')
        plot.setLabel('bottom', 'Frequency (Log)', 'Hz')
        plot.showGrid(x=True, y=True)
        cmap = pg.colormap.get('viridis')
        for item in cycles:
            progress = (item['num'] - c_min) / total_range
            color = cmap.mapToQColor(progress)
            color.setAlpha(150)
            plot.plot(item['freqs'], item['power'], pen=pg.mkPen(color, width=1))

# --- MODE 4: RANGE OVERLAY (Swapped with original Mode 3) ---
class RangeCyclesStrategy(AnalysisStrategy):
    def name(self): return "4. Range Cycles Overlay (CV & IT)"
    
    def get_params(self):
        return {
            'Start Line Number': {'type': 'int', 'default': 5},
            'Start Cycle': {'type': 'int', 'default': 1},
            'End Cycle': {'type': 'int', 'default': 100},
        }

    def process(self, data, params):
        start_line = params['Start Line Number']
        c_start = params['Start Cycle']
        c_end = params['End Cycle']
        if c_start > c_end: return {'error': "Start > End Cycle."}
        cycles_data = []
        for c_idx in range(c_start, c_end + 1):
            base = start_line + 2 * (c_idx - 1)
            lines = [base, base + 1]
            mask = np.isin(data.line, lines)
            if np.any(mask):
                cycles_data.append({'t': data.t[mask], 'v': data.v[mask], 'i': data.i[mask], 'num': c_idx})
        if not cycles_data: return {'error': "No data found."}
        return {'cycles': cycles_data, 'min_c': c_start, 'max_c': c_end}

    def visualize_results(self, container, res):
        container.clear()
        if 'error' in res:
            plot = container.addPlot(); plot.setTitle(res['error']); return
        cycles = res['cycles']; c_min, c_max = res['min_c'], res['max_c']
        total_range = c_max - c_min if c_max > c_min else 1
        p1 = container.addPlot(row=0, col=0, title=f"CV: Current vs Voltage (Cycles {c_min}-{c_max})")
        p1.setLabel('left', 'Current', 'A'); p1.setLabel('bottom', 'Voltage', 'V'); p1.showGrid(x=True, y=True)
        p2 = container.addPlot(row=1, col=0, title="IT: Current vs Time")
        p2.setLabel('left', 'Current', 'A'); p2.setLabel('bottom', 'Time', 's'); p2.showGrid(x=True, y=True)
        cmap = pg.colormap.get('viridis')
        for item in cycles:
            progress = (item['num'] - c_min) / total_range
            color = cmap.mapToQColor(progress); color.setAlpha(150)
            pen = pg.mkPen(color, width=1.5)
            p1.plot(item['v'], item['i'], pen=pen)
            p2.plot(item['t'], item['i'], pen=pen)

# --- MODE 5: RAW TIME-SERIES (VOLTAGE VS TIME) ---
class RawVoltageTimeSeriesStrategy(AnalysisStrategy):
    def name(self): return "5. Raw Time-Series (Control V)"

    def get_params(self):
        return {
            'Sampling Interval (s)': {'type': 'float', 'default': 0.002},
            'Start Index': {'type': 'int', 'default': 0},
            'End Index (0 for all)': {'type': 'int', 'default': 0},
        }

    def process(self, data, params):
        dt, start_idx, end_idx = params['Sampling Interval (s)'], params['Start Index'], params['End Index (0 for all)']
        if not hasattr(data, 'v'): return {'error': "Voltage data (data.v) missing."}
        total_points = len(data.v)
        if end_idx <= 0 or end_idx > total_points: end_idx = total_points
        if start_idx >= end_idx or start_idx < 0: return {'error': "Invalid indices."}
        return {'time': np.arange(start_idx, end_idx) * dt, 'voltage': data.v[start_idx:end_idx]}

    def visualize_results(self, container, res):
        container.clear()
        if 'error' in res:
            plot = container.addPlot(); plot.setTitle(res['error'], color='r'); return
        p1 = container.addPlot(title="Control Voltage vs Time (Channel 4)")
        p1.setLabel('left', 'Voltage', 'V'); p1.setLabel('bottom', 'Time', 's'); p1.showGrid(x=True, y=True)
        p1.plot(res['time'], res['voltage'], pen=pg.mkPen(color=(0, 0, 150), width=2))
        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((100, 100, 100), style=Qt.PenStyle.DashLine))
        h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((100, 100, 100), style=Qt.PenStyle.DashLine))
        p1.addItem(v_line); p1.addItem(h_line)
        label = pg.TextItem(anchor=(0, 1), color='w', fill=(0, 0, 0, 150)); p1.addItem(label)
        def mouseMoved(evt):
            pos = evt[0]
            if p1.sceneBoundingRect().contains(pos):
                mousePoint = p1.vb.mapSceneToView(pos)
                v_line.setPos(mousePoint.x()); h_line.setPos(mousePoint.y())
                label.setHtml(f"<div style='color: white;'><b>Time:</b> {mousePoint.x():.3f}s<br><b>V:</b> {mousePoint.y():.4f}V</div>")
                label.setPos(mousePoint.x(), mousePoint.y())
        self.proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

# =============================================================================
#  3. MAIN APPLICATION
# =============================================================================
class CVAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Analyzer: Cycles & Bubbles")
        self.resize(1400, 1000)
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k'); pg.setConfigOption('antialias', False) 
        self.data_obj = None
        # Swapped original 3 and 4 here
        self.strategies = [
            LineInspectorStrategy(), 
            SingleCycleStrategy(), 
            RangeCyclesFFTStrategy(), # Mode 3
            RangeCyclesStrategy(),    # Mode 4
            RawVoltageTimeSeriesStrategy()
        ]
        self.current_strategy = self.strategies[0]
        self.param_inputs = {}
        self.init_ui()

    def init_ui(self):
        w = QWidget(); self.setCentralWidget(w)
        layout = QHBoxLayout(w)
        side = QVBoxLayout(); side_widget = QWidget(); side_widget.setFixedWidth(340); side_widget.setLayout(side)
        
        grp_load = QGroupBox("1. Data Loading"); l_load = QVBoxLayout()
        self.btn_load = QPushButton("Load CSV"); self.btn_load.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No File Loaded"); self.lbl_file.setWordWrap(True)
        l_load.addWidget(self.btn_load); l_load.addWidget(self.lbl_file)
        grp_load.setLayout(l_load); side.addWidget(grp_load)
        
        grp_mode = QGroupBox("2. Analysis Mode"); l_mode = QVBoxLayout()
        self.combo_mode = QComboBox()
        for s in self.strategies: self.combo_mode.addItem(s.name())
        self.combo_mode.currentIndexChanged.connect(self.change_strategy)
        l_mode.addWidget(self.combo_mode); grp_mode.setLayout(l_mode); side.addWidget(grp_mode)
        
        self.grp_param = QGroupBox("3. Parameters"); self.form_layout = QFormLayout(); self.grp_param.setLayout(self.form_layout); side.addWidget(self.grp_param)
        self.btn_run = QPushButton("Process Data"); self.btn_run.setStyleSheet("background-color: #d4f1f4; font-weight: bold; height: 40px;"); self.btn_run.clicked.connect(self.run_analysis); self.btn_run.setEnabled(False); side.addWidget(self.btn_run)
        
        side.addStretch(); layout.addWidget(side_widget)
        self.plot_container = pg.GraphicsLayoutWidget(); layout.addWidget(self.plot_container)
        self.main_plot = self.plot_container.addPlot(title="Status: Ready"); self.main_plot.showGrid(x=True, y=True)
        self.refresh_params_ui()

    def load_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", ".", "CSV (*.csv)")
        if not fname: return
        try:
            self.lbl_file.setText("Loading..."); QApplication.processEvents() 
            df = pd.read_csv(fname)
            if df.shape[1] < 13: raise ValueError("CSV columns mismatch.")
            data_df = df.iloc[:, [9, 3, 5, 12]].copy(); data_df.columns = ['Line', 'Voltage', 'Current', 'DeltaTime']
            for c in data_df.columns: data_df[c] = pd.to_numeric(data_df[c], errors='coerce')
            data_df = data_df.dropna(); data_df['Line'] = data_df['Line'].astype(int); data_df['Time'] = data_df['DeltaTime'].cumsum()
            self.data_obj = ExperimentData(data_df); self.lbl_file.setText(f"Loaded: {len(data_df)} rows"); self.btn_run.setEnabled(True); self.run_analysis()
        except Exception as e:
            self.lbl_file.setText("Load Failed"); QMessageBox.critical(self, "Error", str(e))

    def change_strategy(self, idx):
        self.current_strategy = self.strategies[idx]; self.refresh_params_ui()
        if self.data_obj: self.run_analysis()

    def refresh_params_ui(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        self.param_inputs = {}
        for key, info in self.current_strategy.get_params().items():
            w = QCheckBox() if info['type'] == 'bool' else QLineEdit()
            if info['type'] == 'bool': w.setChecked(info['default'])
            else: w.setText(str(info['default']))
            self.param_inputs[key] = w; self.form_layout.addRow(key, w)

    def run_analysis(self):
        if not self.data_obj: return
        params = {}; defaults = self.current_strategy.get_params()
        for k, w in self.param_inputs.items():
            info = defaults[k]
            if info['type'] == 'bool': params[k] = w.isChecked()
            else:
                try:
                    val = float(w.text())
                    params[k] = int(val) if info['type'] == 'int' else val
                except: params[k] = info['default']
        try:
            res = self.current_strategy.process(self.data_obj, params)
            self.current_strategy.visualize_results(self.plot_container, res)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    w = CVAnalyzerApp(); w.show(); sys.exit(app.exec())