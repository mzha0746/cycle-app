import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QFileDialog, 
                             QFormLayout, QGroupBox, QMessageBox,
                             QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# =============================================================================
#  1. DATA STRUCTURES
# =============================================================================
class CycleSegment:
    """
    Represents a single bubble evolution cycle.
    """
    def __init__(self, raw_t, raw_i, raw_v):
        self.raw_t = raw_t
        self.raw_i = raw_i
        self.raw_v = raw_v
        
        self.avg_voltage = np.mean(raw_v) if len(raw_v) > 0 else 0
        self.duration = raw_t[-1] - raw_t[0] if len(raw_t) > 0 else 0
        
        self.norm_t = np.array([])
        self.norm_i = np.array([])
        self._normalize()

    def _normalize(self):
        if len(self.raw_t) < 2: return
        
        # Normalize Time: 0.0 start, 1.0 end
        t_start = self.raw_t[0]
        t_range = self.raw_t[-1] - t_start
        if t_range == 0: t_range = 1e-9
        self.norm_t = (self.raw_t - t_start) / t_range
        
        # Normalize Current: 0.0 min, 1.0 max
        i_min = np.min(self.raw_i)
        i_range = np.max(self.raw_i) - i_min
        if i_range == 0: i_range = 1e-9
        self.norm_i = (self.raw_i - i_min) / i_range

# =============================================================================
#  2. ANALYSIS STRATEGIES
# =============================================================================
class AnalysisStrategy:
    def name(self): raise NotImplementedError
    def get_params(self): return {}
    def process(self, t, i, v, params): raise NotImplementedError
    def visualize_results(self, plot_widget, cycles, layout_item=None): raise NotImplementedError

class BasePeakStrategy(AnalysisStrategy):
    """
    Base class that handles the common Peak Detection logic.
    Subclasses only need to implement visualize_results.
    """
    def get_params(self):
        return {
            'Invert Signal': {'type': 'bool', 'default': True, 'desc': 'Check if bubble detachment is a valley (dip)'},
            'Prominence': {'type': 'float', 'default': 1e-6, 'desc': 'Vertical threshold (Amps).'},
            'Min Distance': {'type': 'int', 'default': 50, 'desc': 'Min samples between peaks.'},
            'Width': {'type': 'int', 'default': 5, 'desc': 'Min width of peak in samples.'}
        }

    def process(self, t, i, v, params):
        sig = -i if params['Invert Signal'] else i
        
        peaks, _ = find_peaks(sig, 
                              prominence=params['Prominence'], 
                              distance=params['Min Distance'], 
                              width=params['Width'])
        
        cycles = []
        # Need at least 2 peaks to form a cycle.
        # Logic naturally discards pre-1st peak and post-last peak data.
        if len(peaks) > 1:
            for k in range(len(peaks) - 1):
                start_idx = peaks[k]
                end_idx = peaks[k+1]
                
                c_t = t[start_idx : end_idx+1]
                c_i = i[start_idx : end_idx+1]
                c_v = v[start_idx : end_idx+1]
                
                if len(c_t) > 5:
                    cycles.append(CycleSegment(c_t, c_i, c_v))
        
        return cycles, peaks

# --- Mode 1: The Original Overlay ---
class OverlayVisualization(BasePeakStrategy):
    def name(self): return "Mode 1: Normalized Cycle Overlay"
    
    def visualize_results(self, plot_widget, cycles, layout_item):
        plot_widget.clear()
        plot_widget.setTitle(f"Overlay: {len(cycles)} Cycles (Color = Voltage)")
        plot_widget.setLabel('left', 'Norm. Current (0-1)')
        plot_widget.setLabel('bottom', 'Norm. Time (0-1)')
        
        if not cycles: return

        # 1. Setup Color Map
        all_avg_v = [c.avg_voltage for c in cycles]
        min_v = min(all_avg_v)
        max_v = max(all_avg_v)
        if max_v == min_v: max_v += 1e-9
        
        # We need to access the histogram from the layout to update levels
        if layout_item:
            layout_item.setLevels(min_v, max_v)
            pg_cmap = layout_item.gradient.colorMap()
        else:
            return

        # 2. Plot Lines
        for cycle in cycles:
            norm_v_val = (cycle.avg_voltage - min_v) / (max_v - min_v)
            color = pg_cmap.mapToQColor(norm_v_val)
            color.setAlpha(150)
            plot_widget.plot(cycle.norm_t, cycle.norm_i, pen=pg.mkPen(color, width=2))

# --- Mode 2: The New Scatter Plot ---
class ScatterVisualization(BasePeakStrategy):
    def name(self): return "Mode 2: Duration vs Voltage Scatter"
    
    def visualize_results(self, plot_widget, cycles, layout_item):
        plot_widget.clear()
        plot_widget.setTitle(f"Kinetics: {len(cycles)} Cycles Detected")
        plot_widget.setLabel('bottom', 'Average Voltage', 'V')
        plot_widget.setLabel('left', 'Cycle Duration', 's')
        
        if not cycles: return

        # Extract Data
        x_data = [c.avg_voltage for c in cycles] # Voltage
        y_data = [c.duration for c in cycles]    # Time
        
        # Plot Scatter
        scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 150))
        scatter.addPoints(x=x_data, y=y_data)
        plot_widget.addItem(scatter)
        
        # Disable Histogram if present (it's not used here)
        if layout_item:
            # We can't easily hide it without removing from layout, 
            # so let's just set it to a neutral range or ignore it.
            pass

# =============================================================================
#  3. MAIN APPLICATION
# =============================================================================
class BubbleCycleAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bubble Cycle Analysis & Normalization")
        self.resize(1400, 1000)
        
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)
        
        self.df = None
        # Register Strategies
        self.strategies = [OverlayVisualization(), ScatterVisualization()]
        self.current_strategy = self.strategies[0]
        self.param_inputs = {} 
        
        self.init_ui()

    def init_ui(self):
        w = QWidget(); self.setCentralWidget(w)
        layout = QHBoxLayout(w)
        
        # --- SIDEBAR ---
        side = QVBoxLayout()
        side_widget = QWidget(); side_widget.setFixedWidth(320); side_widget.setLayout(side)
        
        # 1. Load
        grp_load = QGroupBox("1. Data Loading")
        l_load = QVBoxLayout()
        self.btn_load = QPushButton("Load CV CSV")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No File Loaded")
        self.lbl_file.setWordWrap(True)
        l_load.addWidget(self.btn_load); l_load.addWidget(self.lbl_file)
        grp_load.setLayout(l_load); side.addWidget(grp_load)
        
        # 2. Strategy
        grp_strat = QGroupBox("2. Analysis Mode")
        l_strat = QVBoxLayout()
        self.combo_strat = QComboBox()
        for s in self.strategies: self.combo_strat.addItem(s.name())
        self.combo_strat.currentIndexChanged.connect(self.change_strategy)
        l_strat.addWidget(self.combo_strat)
        grp_strat.setLayout(l_strat); side.addWidget(grp_strat)
        
        # 3. Parameters
        self.grp_param = QGroupBox("3. Peak Detection Parameters")
        self.form_param = QFormLayout()
        self.grp_param.setLayout(self.form_param)
        side.addWidget(self.grp_param)
        
        # 4. Action
        grp_act = QGroupBox("4. Process")
        l_act = QVBoxLayout()
        self.btn_process = QPushButton("Run Analysis")
        self.btn_process.setStyleSheet("background-color: #d4f1f4; font-weight: bold; padding: 10px;")
        self.btn_process.clicked.connect(self.run_analysis)
        self.btn_process.setEnabled(False)
        l_act.addWidget(self.btn_process)
        grp_act.setLayout(l_act); side.addWidget(grp_act)
        
        side.addStretch()
        layout.addWidget(side_widget)
        
        # --- PLOT AREA ---
        plot_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_layout)
        
        # Plot 1: Raw Overview
        self.plot_raw = plot_layout.addPlot(row=0, col=0, title="Raw Data (Select Region)")
        self.plot_raw.setLabel('left', 'Current', 'A')
        self.plot_raw.setLabel('bottom', 'Time', 's')
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.plot_raw.addItem(self.region)
        self.region.sigRegionChanged.connect(self.on_region_change)
        
        # Plot 2: Detailed Selection & Peaks
        self.plot_detail = plot_layout.addPlot(row=1, col=0, title="Selection & Detected Breakpoints")
        self.plot_detail.setLabel('left', 'Current', 'A')
        self.plot_detail.setLabel('bottom', 'Time', 's')
        self.plot_detail.showGrid(x=True, y=True)
        
        # Plot 3: Result (Dynamic based on strategy)
        self.plot_result = plot_layout.addPlot(row=2, col=0, title="Analysis Result")
        self.plot_result.showGrid(x=True, y=True)
        
        # Color Bar (Helper for Mode 1)
        self.cmap_histogram = pg.HistogramLUTItem()
        self.cmap_histogram.gradient.loadPreset('viridis')
        plot_layout.addItem(self.cmap_histogram, row=2, col=1)

        self.refresh_params_ui()

    # --- LOGIC ---
    def change_strategy(self, idx):
        self.current_strategy = self.strategies[idx]
        self.refresh_params_ui()
        # If we have data loaded, we can re-run automatically or wait for user
        # Let's wait for user to click Run
        self.plot_result.clear()
        self.plot_result.setTitle(f"Analysis Result ({self.current_strategy.name()})")

    def load_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", ".", "CSV (*.csv)")
        if not fname: return
        
        try:
            df = pd.read_csv(fname, dtype=str)
            df = df.iloc[:, [3, 5, 12]].copy()
            df.columns = ['Voltage', 'Current', 'DeltaTime']
            for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna()
            df['Time'] = df['DeltaTime'].cumsum()
            
            self.df = df
            self.lbl_file.setText(f"Loaded: {len(df)} rows")
            
            self.plot_raw.clear()
            self.plot_raw.addItem(self.region)
            self.plot_raw.plot(df['Time'].values, df['Current'].values, pen='k')
            
            t = df['Time'].values
            mid = (t[-1]+t[0])/2
            span = (t[-1]-t[0])*0.1
            self.region.setRegion([mid-span, mid+span])
            
            self.btn_process.setEnabled(True)
            self.update_detail_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def refresh_params_ui(self):
        while self.form_param.count():
            item = self.form_param.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.param_inputs = {}
        
        params = self.current_strategy.get_params()
        for key, info in params.items():
            if info['type'] == 'bool':
                w = QCheckBox()
                w.setChecked(info['default'])
            else:
                w = QLineEdit()
                w.setText(str(info['default']))
                if 'desc' in info: w.setToolTip(info['desc'])
            
            self.param_inputs[key] = w
            self.form_param.addRow(key, w)

    def on_region_change(self):
        self.update_detail_view()

    def update_detail_view(self):
        if self.df is None: return
        min_t, max_t = self.region.getRegion()
        mask = (self.df['Time'] >= min_t) & (self.df['Time'] <= max_t)
        sub_df = self.df.loc[mask]
        
        self.plot_detail.clear()
        if sub_df.empty: return
        self.plot_detail.plot(sub_df['Time'].values, sub_df['Current'].values, pen='k')

    def run_analysis(self):
        if self.df is None: return
        
        # 1. Get Data Selection
        min_t, max_t = self.region.getRegion()
        mask = (self.df['Time'] >= min_t) & (self.df['Time'] <= max_t)
        sub_df = self.df.loc[mask]
        if len(sub_df) < 10: 
            QMessageBox.warning(self, "Warning", "Selection too small.")
            return
        
        t = sub_df['Time'].values
        i = sub_df['Current'].values
        v = sub_df['Voltage'].values
        
        # 2. Parse Params
        params = {}
        default_params = self.current_strategy.get_params()
        
        for key, widget in self.param_inputs.items():
            def_info = default_params[key]
            if def_info['type'] == 'bool':
                params[key] = widget.isChecked()
            else:
                try:
                    val_str = widget.text()
                    if def_info['type'] == 'float': params[key] = float(val_str)
                    else: params[key] = int(float(val_str))
                except ValueError:
                    widget.setText(str(def_info['default']))
                    QMessageBox.warning(self, "Invalid Input", f"Resetting {key}")
                    return

        # 3. Execute
        try:
            cycles, peak_indices = self.current_strategy.process(t, i, v, params)
            
            # --- VISUALIZE ---
            # A. Peaks on Detail Plot
            self.update_detail_view()
            if len(peak_indices) > 0:
                self.plot_detail.plot(t[peak_indices], i[peak_indices], pen=None, symbol='o', symbolBrush='r', symbolSize=8)
            
            # B. Strategy Specific Plot (Overlay vs Scatter)
            if not cycles:
                self.plot_result.clear()
                QMessageBox.information(self, "Result", "No cycles detected.")
                return
                
            self.current_strategy.visualize_results(self.plot_result, cycles, self.cmap_histogram)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))

if __name__ == "__main__":
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    w = BubbleCycleAnalyzer()
    w.show()
    app.exec()