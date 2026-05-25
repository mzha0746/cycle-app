import sys
import os
import datetime
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QFileDialog, 
                             QFormLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                             QDialog, QDialogButtonBox, QCheckBox)
from PyQt6.QtCore import Qt, QRectF
import pyqtgraph as pg

# =============================================================================
#  0. INFRASTRUCTURE: DATA CONTEXT
# =============================================================================
class DataContext:
    def __init__(self, df, time_col, signal_col, aux_col=None, aux_label=None, signal_label="Signal", original_filename=""):
        self.df = df
        self.time_col = time_col
        self.signal_col = signal_col
        self.aux_col = aux_col       
        self.aux_label = aux_label   
        self.signal_label = signal_label
        self.original_filename = original_filename 

    def get_time(self): return self.df[self.time_col].values
    def get_signal(self): return self.df[self.signal_col].values
    def get_aux(self): return self.df[self.aux_col].values if self.aux_col else None
    
    def subset(self, t_min, t_max):
        mask = (self.df[self.time_col] >= t_min) & (self.df[self.time_col] <= t_max)
        return DataContext(self.df.loc[mask].copy(), self.time_col, self.signal_col, 
                           self.aux_col, self.aux_label, self.signal_label, self.original_filename)
    
    @property
    def has_aux(self): return self.aux_col is not None

# =============================================================================
#  1. HELPER: CUSTOM AXIS
# =============================================================================
class AuxLookupAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_data = np.array([])
        self.aux_data = np.array([])

    def set_lookup_data(self, time_arr, aux_arr):
        self.time_data = time_arr
        self.aux_data = aux_arr

    def get_val_at_time(self, t):
        if len(self.time_data) == 0: return np.nan
        return np.interp(t, self.time_data, self.aux_data, left=np.nan, right=np.nan)

    def tickStrings(self, values, scale, spacing):
        if len(self.time_data) == 0: return []
        try:
            mapped = np.interp(values, self.time_data, self.aux_data, left=np.nan, right=np.nan)
            return ["" if np.isnan(v) else f"{v:.3f}" for v in mapped]
        except: return [""] * len(values)

# =============================================================================
#  2. GENERIC CSV IMPORT DIALOG (Enhanced with Delta Time)
# =============================================================================
class GenericCSVDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.df = None
        self.setWindowTitle("Import Generic CSV")
        self.resize(450, 300)
        
        layout = QVBoxLayout(self)
        
        # Header Toggle
        self.chk_header = QCheckBox("First row contains headers")
        self.chk_header.setChecked(True)
        self.chk_header.toggled.connect(self.reload_data)
        layout.addWidget(self.chk_header)

        # Delta Time Toggle
        self.chk_delta = QCheckBox("Use Delta Time (dt) column")
        self.chk_delta.setToolTip("If checked, Time will be calculated as cumulative sum of the selected column.")
        self.chk_delta.toggled.connect(self.update_labels)
        layout.addWidget(self.chk_delta)
        
        # Column Selection
        grp = QGroupBox("Select Data Columns")
        form = QFormLayout()
        self.combo_time = QComboBox()
        self.combo_signal = QComboBox()
        
        self.lbl_time = QLabel("Time Column (X):")
        form.addRow(self.lbl_time, self.combo_time)
        form.addRow("Signal Column (Y):", self.combo_signal)
        grp.setLayout(form)
        layout.addWidget(grp)
        
        # Info Label
        self.lbl_info = QLabel("Loading preview...")
        self.lbl_info.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(self.lbl_info)
        
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
        # Initial Load
        self.reload_data()

    def update_labels(self):
        if self.chk_delta.isChecked():
            self.lbl_time.setText("Delta Time Column (dt):")
        else:
            self.lbl_time.setText("Time Column (X):")

    def reload_data(self):
        try:
            header_arg = 0 if self.chk_header.isChecked() else None
            self.df = pd.read_csv(self.filepath, header=header_arg)
            
            valid_cols = []
            for col in self.df.columns:
                temp = pd.to_numeric(self.df[col], errors='coerce')
                if temp.notna().sum() > 0:
                    valid_cols.append(col)
            
            self.combo_time.clear()
            self.combo_signal.clear()
            
            def display_name(c):
                return str(c) if self.chk_header.isChecked() else f"Column {c}"

            for col in valid_cols:
                name = display_name(col)
                self.combo_time.addItem(name, userData=col)
                self.combo_signal.addItem(name, userData=col)
            
            self.lbl_info.setText(f"Loaded {len(self.df)} rows. Found {len(valid_cols)} numeric columns.")
            
        except Exception as e:
            self.lbl_info.setText(f"Error reading file: {e}")
            self.combo_time.clear()
            self.combo_signal.clear()

    def get_data(self):
        """Returns (DataFrame, time_column_key, signal_column_key)"""
        t_col_raw = self.combo_time.currentData()
        s_col = self.combo_signal.currentData()
        
        self.df[t_col_raw] = pd.to_numeric(self.df[t_col_raw], errors='coerce')
        self.df[s_col] = pd.to_numeric(self.df[s_col], errors='coerce')
        self.df = self.df.dropna(subset=[t_col_raw, s_col])
        
        t_final_col = t_col_raw
        
        # Handle Delta Time Logic
        if self.chk_delta.isChecked():
            t_final_col = "Calculated_Time"
            self.df[t_final_col] = self.df[t_col_raw].cumsum()
        
        return self.df, t_final_col, s_col


# =============================================================================
#  3. ABSTRACT STRATEGY INTERFACE
# =============================================================================
class AnalysisStrategy:
    def name(self): raise NotImplementedError
    def description(self): raise NotImplementedError
    def load_data(self, parent_widget, filepath=None): raise NotImplementedError 
    def get_required_parameters(self): return {}
    def execute(self, context, params): raise NotImplementedError
    def plot_results(self, raw_plot_widget, result_layout_widget, results): raise NotImplementedError
    def save_results(self, results, parent_widget): raise NotImplementedError
    
    def clear_overlays(self, raw_plot_widget): pass

    def _get_save_path(self, parent_widget, suffix, ctx=None):
        folder = QFileDialog.getExistingDirectory(parent_widget, "Select Output Folder")
        if not folder: return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        name_part = ""
        if ctx and ctx.original_filename:
            base_name = os.path.splitext(os.path.basename(ctx.original_filename))[0]
            name_part = f"_{base_name}"
            
        return f"{folder}/Analysis_{ts}_{suffix}{name_part}.csv"

    def _save_raw_slice(self, parent_widget, ctx, time_lbl="Time", sig_lbl="Signal", aux_lbl="Aux"):
        path = self._get_save_path(parent_widget, "RawSlice", ctx)
        if path:
            data = {
                time_lbl: ctx.get_time(),
                sig_lbl: ctx.get_signal()
            }
            if ctx.has_aux:
                data[aux_lbl] = ctx.get_aux()
                
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)

# =============================================================================
#  4. CONCRETE STRATEGIES
# =============================================================================

# --- A. CV Specific Strategy ---
class CV_PolyFFT(AnalysisStrategy):
    def __init__(self):
        self.trend_item = None

    def name(self): return "CV: Poly Detrend + PSD"
    def description(self): return "Specific to CV data (Voltage/Current). Includes Voltage Axis."

    def load_data(self, parent, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(parent, 'Open CV CSV', '.', 'CSV Files (*.csv)')
        if not filepath: return None, None
        
        try:
            df = pd.read_csv(filepath, dtype=str)
            df = df.iloc[:, [3, 5, 12]].copy()
            df.columns = ['Voltage', 'Current', 'DeltaTime']
            for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna()
            df['Time'] = df['DeltaTime'].cumsum()
            
            ctx = DataContext(df, 'Time', 'Current', 'Voltage', 'Voltage (V)', 'Current (A)', filepath)
            return ctx, filepath
        except Exception as e:
            QMessageBox.warning(parent, "Load Error", f"CV Load Failed: {e}")
            return None, None

    def get_required_parameters(self):
        return {
            'Poly Order': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
            'Sampling Rate (Hz)': {'type': 'float', 'default': 1000.0}
        }

    def execute(self, ctx, params):
        t = ctx.get_time()
        y = ctx.get_signal()
        t_c = t - t[0]
        coeffs = np.polyfit(t_c, y, params['Poly Order'])
        trend = np.polyval(coeffs, t_c)
        resid = y - trend
        
        n = len(resid)
        win = np.hanning(n)
        fft_res = np.fft.fft(resid * win)
        freqs = np.fft.fftfreq(n, d=1/params['Sampling Rate (Hz)'])
        mask = freqs > 0
        power = (np.abs(fft_res[mask]) ** 2) / n
        
        return {
            'ctx': ctx, 'trend': trend, 'resid': resid,
            'freqs': freqs[mask], 'power': power
        }

    def clear_overlays(self, raw_plot):
        if self.trend_item and self.trend_item in raw_plot.listDataItems():
            raw_plot.removeItem(self.trend_item)
            self.trend_item = None

    def plot_results(self, raw_plot, res_layout, res):
        self.clear_overlays(raw_plot)
        self.trend_item = raw_plot.plot(res['ctx'].get_time(), res['trend'], pen=pg.mkPen('r', width=2), name="Fit")
        
        res_layout.clear()
        ctx = res['ctx']
        ax_args = {}
        if ctx.has_aux:
            ax = AuxLookupAxis(orientation='top')
            ax.set_lookup_data(ctx.get_time(), ctx.get_aux())
            ax.setLabel(ctx.aux_label)
            ax_args = {'axisItems': {'top': ax}}

        p1 = res_layout.addPlot(row=0, col=0, title="Residuals", **ax_args)
        p1.plot(ctx.get_time(), res['resid'], pen='g')
        p1.setLabel('left', 'Delta ' + ctx.signal_label)
        p1.showGrid(x=True, y=True)
        if ctx.has_aux: p1.showAxis('top')

        res_layout.nextRow()
        p2 = res_layout.addPlot(row=1, col=0, title="Power Spectral Density")
        p2.plot(res['freqs'], res['power'], pen='b')
        p2.setLogMode(x=True, y=True)
        p2.setLabel('bottom', 'Frequency', 'Hz')
        p2.showGrid(x=True, y=True)

    def save_results(self, res, parent):
        self._save_raw_slice(parent, res['ctx'], time_lbl="Time (s)", sig_lbl="Current (A)", aux_lbl="Voltage (V)")

        path_res = self._get_save_path(parent, "CV_Results", res['ctx'])
        if path_res:
            data = {
                'Time (s)': res['ctx'].get_time(), 
                'Current (A)': res['ctx'].get_signal(), 
                'Trend (A)': res['trend'], 
                'Residual (A)': res['resid']
            }
            if res['ctx'].has_aux: data['Voltage (V)'] = res['ctx'].get_aux()
            pd.DataFrame(data).to_csv(path_res, index=False)
        
        path_psd = self._get_save_path(parent, "CV_PSD", res['ctx'])
        if path_psd:
            pd.DataFrame({'Frequency (Hz)': res['freqs'], 'Power Density': res['power']}).to_csv(path_psd, index=False)

        QMessageBox.information(parent, "Saved", f"Data saved successfully.")

# --- B. CV STFT Strategy ---
class CV_STFT(CV_PolyFFT):
    def name(self): return "CV: STFT Spectrogram"
    
    def get_required_parameters(self):
        p = super().get_required_parameters()
        p.update({
            'Window Size': {'type': 'int', 'default': 256},
            'Overlap Ratio': {'type': 'float', 'default': 0.8}
        })
        return p

    def execute(self, ctx, params):
        parent_res = super().execute(ctx, params)
        resid = parent_res['resid']
        fs = params['Sampling Rate (Hz)']
        nperseg = params['Window Size']
        noverlap = int(nperseg * params['Overlap Ratio'])
        
        f, t_spec, Sxx = spectrogram(resid, fs, nperseg=nperseg, noverlap=noverlap)
        return {
            'ctx': ctx, 'trend': parent_res['trend'], 'resid': resid,
            'spec_t': t_spec + ctx.get_time()[0], 'spec_f': f, 'spec_p': Sxx
        }

    def plot_results(self, raw_plot, res_layout, res):
        self.clear_overlays(raw_plot)
        self.trend_item = raw_plot.plot(res['ctx'].get_time(), res['trend'], pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        
        res_layout.clear()
        ctx = res['ctx']
        ax_args = {}
        if ctx.has_aux:
            ax = AuxLookupAxis(orientation='top')
            ax.set_lookup_data(ctx.get_time(), ctx.get_aux())
            ax.setLabel(ctx.aux_label)
            ax_args = {'axisItems': {'top': ax}}
            
        p1 = res_layout.addPlot(row=0, col=0, title="Noise Trace", **ax_args)
        p1.plot(ctx.get_time(), res['resid'], pen='g')
        if ctx.has_aux: p1.showAxis('top')
        
        res_layout.nextRow()
        # Spectrogram Plot
        p2 = res_layout.addPlot(row=1, col=0, title="Spectrogram")
        p1.setXLink(p2)
        
        img = pg.ImageItem()
        p2.addItem(img)
        power_db = 10 * np.log10(res['spec_p'] + 1e-12)
        img.setImage(power_db.T)
        
        t, f = res['spec_t'], res['spec_f']
        img.setRect(QRectF(t[0], f[0], t[-1]-t[0], f[-1]-f[0]))
        p2.setLabel('left', 'Frequency', 'Hz')
        p2.setLabel('bottom', 'Time', 's')

        # Add Histogram / Color Bar
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        hist.gradient.loadPreset('viridis')
        res_layout.addItem(hist, row=1, col=1)

    def save_results(self, res, parent):
        self._save_raw_slice(parent, res['ctx'], time_lbl="Time (s)", sig_lbl="Current (A)", aux_lbl="Voltage (V)")
        path_res = self._get_save_path(parent, "STFT_Residuals", res['ctx'])
        if path_res:
             data = {'Time (s)': res['ctx'].get_time(), 'Trend (A)': res['trend'], 'Residual (A)': res['resid']}
             if res['ctx'].has_aux: data['Voltage (V)'] = res['ctx'].get_aux()
             pd.DataFrame(data).to_csv(path_res, index=False)
        QMessageBox.information(parent, "Saved", f"Data saved successfully.")

# --- C. GENERIC Strategy ---
class Generic_PolyFFT(AnalysisStrategy):
    def __init__(self):
        self.trend_item = None

    def name(self): return "Generic: Poly Detrend + PSD"
    def description(self): return "For any CSV. You select Time and Signal columns."

    def load_data(self, parent, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(parent, 'Open Generic CSV', '.', 'CSV Files (*.csv)')
        if not filepath: return None, None
        
        try:
            # Use Enhanced Dialog
            dlg = GenericCSVDialog(filepath, parent)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                df, t_col, y_col = dlg.get_data()
                df = df.sort_values(by=t_col)
                ctx = DataContext(df, t_col, y_col, None, None, str(y_col), filepath)
                return ctx, filepath
            else:
                return None, None
        except Exception as e:
            QMessageBox.warning(parent, "Load Error", str(e))
            return None, None

    def get_required_parameters(self):
        return {
            'Poly Order': {'type': 'int', 'default': 2},
            'Sampling Rate (Hz)': {'type': 'float', 'default': 1.0} 
        }

    def execute(self, ctx, params):
        t = ctx.get_time()
        y = ctx.get_signal()
        t_c = t - t[0]
        coeffs = np.polyfit(t_c, y, params['Poly Order'])
        trend = np.polyval(coeffs, t_c)
        resid = y - trend
        
        n = len(resid)
        win = np.hanning(n)
        fft_res = np.fft.fft(resid * win)
        freqs = np.fft.fftfreq(n, d=1/params['Sampling Rate (Hz)'])
        mask = freqs > 0
        power = (np.abs(fft_res[mask]) ** 2) / n
        
        return {
            'ctx': ctx, 'trend': trend, 'resid': resid,
            'freqs': freqs[mask], 'power': power
        }

    def clear_overlays(self, raw_plot):
        if self.trend_item and self.trend_item in raw_plot.listDataItems():
            raw_plot.removeItem(self.trend_item)
            self.trend_item = None

    def plot_results(self, raw_plot, res_layout, res):
        self.clear_overlays(raw_plot)
        self.trend_item = raw_plot.plot(res['ctx'].get_time(), res['trend'], pen=pg.mkPen('r', width=2), name="Fit")
        
        res_layout.clear()
        p1 = res_layout.addPlot(row=0, col=0, title="Residuals")
        p1.plot(res['ctx'].get_time(), res['resid'], pen='g')
        p1.showGrid(x=True, y=True)
        
        res_layout.nextRow()
        p2 = res_layout.addPlot(row=1, col=0, title="Power Spectral Density")
        p2.plot(res['freqs'], res['power'], pen='b')
        p2.setLogMode(x=True, y=True)
        p2.showGrid(x=True, y=True)

    def save_results(self, res, parent):
        self._save_raw_slice(parent, res['ctx'], time_lbl="Time", sig_lbl="Signal", aux_lbl="Aux")
        path = self._get_save_path(parent, "Generic_Results", res['ctx'])
        if path:
            pd.DataFrame({'Time': res['ctx'].get_time(), 'Signal': res['ctx'].get_signal(), 'Trend': res['trend'], 'Residual': res['resid']}).to_csv(path, index=False)
        QMessageBox.information(parent, "Saved", f"Data saved successfully.")


# =============================================================================
#  5. MAIN APP
# =============================================================================
class UniversalAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal Time-Series Noise Analyzer")
        self.resize(1400, 950)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.strategies = [CV_PolyFFT(), CV_STFT(), Generic_PolyFFT()]
        self.curr_strat = self.strategies[0]
        self.data_ctx = None
        self.results = None
        self.last_loaded_file = None 
        
        self.init_ui()

    def init_ui(self):
        w = QWidget(); self.setCentralWidget(w)
        layout = QHBoxLayout(w)
        
        side = QVBoxLayout()
        side_widget = QWidget(); side_widget.setFixedWidth(300); side_widget.setLayout(side)
        
        grp_strat = QGroupBox("1. Analysis Mode")
        l_strat = QVBoxLayout()
        self.combo_strat = QComboBox()
        for s in self.strategies: self.combo_strat.addItem(s.name())
        self.combo_strat.currentIndexChanged.connect(self.on_strat_change)
        self.lbl_desc = QLabel(self.curr_strat.description())
        self.lbl_desc.setWordWrap(True)
        l_strat.addWidget(self.combo_strat)
        l_strat.addWidget(self.lbl_desc)
        grp_strat.setLayout(l_strat); side.addWidget(grp_strat)
        
        grp_load = QGroupBox("2. Data")
        l_load = QVBoxLayout()
        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(lambda: self.load_data_flow(None))
        self.lbl_file = QLabel("No Data")
        l_load.addWidget(self.btn_load); l_load.addWidget(self.lbl_file)
        grp_load.setLayout(l_load); side.addWidget(grp_load)
        
        grp_param = QGroupBox("3. Parameters")
        self.l_param = QFormLayout()
        grp_param.setLayout(self.l_param)
        side.addWidget(grp_param)
        self.param_inputs = {}
        
        grp_act = QGroupBox("4. Execute")
        l_act = QVBoxLayout()
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_save = QPushButton("Save Results")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        l_act.addWidget(self.btn_run); l_act.addWidget(self.btn_save)
        grp_act.setLayout(l_act); side.addWidget(grp_act)
        
        side.addStretch()
        layout.addWidget(side_widget)
        
        center = QVBoxLayout()
        self.lbl_hover = QLabel("Hover for Info")
        self.lbl_hover.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.lbl_hover)
        
        self.ax_top = AuxLookupAxis(orientation='top')
        self.plot_raw = pg.PlotWidget(axisItems={'top': self.ax_top}, title="Raw Data (Select Region)")
        self.plot_raw.showGrid(x=True, y=True)
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.plot_raw.scene().sigMouseMoved.connect(self.on_mouse_move)
        
        center.addWidget(self.plot_raw, 1)
        self.plot_layout = pg.GraphicsLayoutWidget()
        center.addWidget(self.plot_layout, 2)
        layout.addLayout(center)
        self.refresh_params()

    def on_strat_change(self, idx):
        self.curr_strat.clear_overlays(self.plot_raw)
        self.curr_strat = self.strategies[idx]
        self.lbl_desc.setText(self.curr_strat.description())
        self.refresh_params()
        self.plot_layout.clear()
        
        if self.last_loaded_file:
            self.lbl_file.setText("Attempting reload...")
            success = self.load_data_flow(self.last_loaded_file)
            if not success:
                 self.data_ctx = None
                 self.plot_raw.clear()
                 self.lbl_file.setText("Reload Failed. Pick Data.")

    def refresh_params(self):
        while self.l_param.count():
            item = self.l_param.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.param_inputs = {}
        
        for k, v in self.curr_strat.get_required_parameters().items():
            if v['type'] == 'int':
                w = QSpinBox(); w.setRange(v.get('min', 1), v.get('max', 99999)); w.setValue(v['default'])
            elif v['type'] == 'float':
                w = QDoubleSpinBox(); w.setRange(v.get('min', 0.0), v.get('max', 1e9)); w.setValue(v['default']); w.setSingleStep(0.1)
            self.param_inputs[k] = w
            self.l_param.addRow(k, w)

    def load_data_flow(self, filepath=None):
        ctx, loaded_path = self.curr_strat.load_data(self, filepath)
        if ctx:
            self.data_ctx = ctx
            self.last_loaded_file = loaded_path
            self.lbl_file.setText(f"Loaded: {len(ctx.df)} rows")
            
            self.plot_raw.clear()
            self.plot_raw.addItem(self.region)
            self.plot_raw.plot(ctx.get_time(), ctx.get_signal(), pen='k')
            self.plot_raw.setLabel('bottom', "Time")
            self.plot_raw.setLabel('left', ctx.signal_label)
            
            if ctx.has_aux:
                self.ax_top.set_lookup_data(ctx.get_time(), ctx.get_aux())
                self.ax_top.setLabel(ctx.aux_label)
                self.plot_raw.showAxis('top')
            else:
                self.plot_raw.hideAxis('top')
                self.ax_top.set_lookup_data([], [])

            t = ctx.get_time()
            if len(t) > 1:
                mid = (t[-1] + t[0]) / 2
                span = (t[-1] - t[0]) * 0.2
                self.region.setRegion([mid - span, mid + span])
                
                dt_vals = np.diff(t)
                dt_avg = np.mean(dt_vals)
                if dt_avg > 0:
                    fs_est = 1.0 / dt_avg
                    if 'Sampling Rate (Hz)' in self.param_inputs:
                        self.param_inputs['Sampling Rate (Hz)'].setValue(fs_est)

            return True
        return False

    def run_analysis(self):
        if not self.data_ctx: return
        min_t, max_t = self.region.getRegion()
        sub_ctx = self.data_ctx.subset(min_t, max_t)
        if len(sub_ctx.get_time()) < 10:
            QMessageBox.warning(self, "Error", "Selection too small")
            return
            
        p = {k: w.value() for k, w in self.param_inputs.items()}
        try:
            self.results = self.curr_strat.execute(sub_ctx, p)
            self.curr_strat.plot_results(self.plot_raw, self.plot_layout, self.results)
            self.btn_save.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))

    def save_results(self):
        if self.results: self.curr_strat.save_results(self.results, self)

    def on_mouse_move(self, pos):
        if not self.data_ctx: return
        vb = self.plot_raw.plotItem.vb
        if self.plot_raw.sceneBoundingRect().contains(pos):
            pt = vb.mapSceneToView(pos)
            t = pt.x()
            msg = f"Time: {t:.3f} s"
            if self.data_ctx.has_aux:
                val = self.ax_top.get_val_at_time(t)
                if not np.isnan(val): msg += f" | {self.data_ctx.aux_label}: {val:.3f}"
            self.lbl_hover.setText(msg)

if __name__ == "__main__":
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    w = UniversalAnalyzer()
    w.show()
    app.exec()