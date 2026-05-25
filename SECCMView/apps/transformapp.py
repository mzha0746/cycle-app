import sys
import os
import csv
import numpy as np
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QGroupBox, QFileDialog, QRadioButton, QProgressBar, 
                             QTextEdit, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class ProcessingWorker(QThread):
    """
    Background worker to handle file IO and processing without freezing the UI.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str) # Success, Message

    def __init__(self, tsv_path, output_path, mode, voltage_col_idx=None, split_dir=None):
        super().__init__()
        self.tsv_path = tsv_path
        self.output_path = output_path
        self.mode = mode # "TRANSPOSE_ONLY" or "SPLIT_CYCLES"
        self.voltage_col_idx = voltage_col_idx
        self.split_dir = split_dir

    def run(self):
        try:
            self.log_signal.emit("Reading TSV file into memory...")
            self.progress_signal.emit(10)

            # 1. FAST READ & TRANSPOSE
            # We use Python's native csv module. Pandas is slow here because
            # creating a DataFrame with millions of columns (before transpose) is expensive.
            with open(self.tsv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f, delimiter='\t')
                # list(reader) reads the whole file into RAM. 
                # zip(*data) transposes rows to columns efficiently.
                rows = list(reader)
            
            if not rows:
                raise ValueError("Input file is empty.")

            self.log_signal.emit(f"File read. Dimensions: {len(rows)} variables x {len(rows[0])} points.")
            self.log_signal.emit("Transposing data...")
            
            # The Magic Transpose Line
            transposed_data = list(zip(*rows))
            
            self.progress_signal.emit(40)

            # 2. BRANCH BASED ON MODE
            if self.mode == "TRANSPOSE_ONLY":
                self.save_transpose_only(transposed_data)
            elif self.mode == "SPLIT_CYCLES":
                self.process_and_split(transposed_data)

        except Exception as e:
            import traceback
            self.log_signal.emit(f"ERROR: {str(e)}")
            print(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

    def save_transpose_only(self, data):
        self.log_signal.emit(f"Saving transposed CSV to: {self.output_path}")
        try:
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            self.progress_signal.emit(100)
            self.finished_signal.emit(True, "Transpose Complete!")
        except Exception as e:
            raise IOError(f"Failed to write CSV: {e}")

    def process_and_split(self, data):
        self.log_signal.emit("Analyzing cycles based on Voltage column...")
        
        # Convert data to numpy for analysis (handling headers)
        # Assuming Row 0 is header
        headers = data[0]
        values = data[1:]
        
        if self.voltage_col_idx >= len(headers):
            raise IndexError(f"Column Index {self.voltage_col_idx} out of bounds (Max {len(headers)-1})")

        # Extract Voltage Column for Signal Processing
        # We try to convert to float, filling errors with 0 or NaN
        try:
            # Transpose the values part back to column-major for numpy extraction? 
            # No, 'values' is currently [Row1(t1), Row2(t2)...]. 
            # We need to extract the specific element from each row.
            
            # Optimization: Just iterate and extract
            voltage_data = []
            for row in values:
                try:
                    val = float(row[self.voltage_col_idx])
                except (ValueError, IndexError):
                    val = 0.0
                voltage_data.append(val)
            
            voltage_arr = np.array(voltage_data)
        except Exception as e:
            raise ValueError(f"Could not parse voltage column data: {e}")

        # Cycle Detection Algorithm
        # CV typically goes Min -> Max -> Min. 
        # We find the "Troughs" (Minima) to define cycle boundaries.
        # We use find_peaks on inverted data to find minima.
        
        # Invert voltage
        inv_voltage = -voltage_arr
        
        # Heuristic: min distance between peaks to avoid noise. 
        # Assuming a cycle takes at least 50 data points.
        peaks, _ = find_peaks(inv_voltage, distance=50)

        # If the scan starts at the minimum, the first point is index 0.
        # We ensure 0 is included if it's not detected as a peak
        split_indices = list(peaks)
        if len(split_indices) == 0 or split_indices[0] > 100:
             # If no peaks found or first peak is far away, assume start is 0
             split_indices.insert(0, 0)
        
        # Ensure the last chunk is included if meaningful
        if split_indices[-1] < len(values) - 10:
            split_indices.append(len(values))

        self.log_signal.emit(f"Detected {len(split_indices)-1} cycles.")
        
        base_name = os.path.splitext(os.path.basename(self.tsv_path))[0]
        
        # Write Files
        total_cycles = len(split_indices) - 1
        for i in range(total_cycles):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Slice
            cycle_rows = [headers] + values[start_idx:end_idx]
            
            # Define filename
            out_name = f"{base_name}_Cycle_{i+1:03d}.csv"
            out_full_path = os.path.join(self.split_dir, out_name)
            
            with open(out_full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(cycle_rows)
            
            # Update Progress
            prog = 40 + int(60 * (i / total_cycles))
            self.progress_signal.emit(prog)
            
        self.progress_signal.emit(100)
        self.finished_signal.emit(True, f"Successfully split into {total_cycles} files.")


class CVTransposerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High-Performance CV Processor")
        self.resize(600, 500)
        self.worker = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- 1. File Selection ---
        grp_files = QGroupBox("1. File Selection")
        layout_files = QVBoxLayout()
        
        # Input
        hbox_in = QHBoxLayout()
        self.line_input = QLineEdit()
        self.line_input.setPlaceholderText("Select source .tsv file...")
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)
        hbox_in.addWidget(self.line_input)
        hbox_in.addWidget(btn_browse_in)
        layout_files.addLayout(hbox_in)
        
        grp_files.setLayout(layout_files)
        layout.addWidget(grp_files)

        # --- 2. Processing Mode ---
        grp_mode = QGroupBox("2. Processing Mode")
        layout_mode = QVBoxLayout()
        
        self.radio_simple = QRadioButton("Simple Transpose (TSV -> CSV)")
        self.radio_simple.setChecked(True)
        self.radio_simple.toggled.connect(self.toggle_mode_ui)
        
        self.radio_split = QRadioButton("Transpose & Split by CV Cycles")
        self.radio_split.toggled.connect(self.toggle_mode_ui)
        
        layout_mode.addWidget(self.radio_simple)
        layout_mode.addWidget(self.radio_split)
        
        # --- Split Settings (Hidden by default) ---
        self.widget_split_settings = QWidget()
        layout_split = QVBoxLayout(self.widget_split_settings)
        layout_split.setContentsMargins(20, 0, 0, 0) # Indent
        
        # Voltage Column
        hbox_col = QHBoxLayout()
        hbox_col.addWidget(QLabel("Voltage Column Index (0-based):"))
        self.spin_col_idx = QSpinBox()
        self.spin_col_idx.setRange(0, 1000)
        self.spin_col_idx.setValue(3) # Default to 0
        hbox_col.addWidget(self.spin_col_idx)
        hbox_col.addStretch()
        layout_split.addLayout(hbox_col)
        
        # Output Directory
        hbox_dir = QHBoxLayout()
        self.line_out_dir = QLineEdit()
        self.line_out_dir.setPlaceholderText("Select output folder for cycles...")
        btn_browse_dir = QPushButton("Select Folder")
        btn_browse_dir.clicked.connect(self.browse_output_dir)
        hbox_dir.addWidget(self.line_out_dir)
        hbox_dir.addWidget(btn_browse_dir)
        layout_split.addLayout(hbox_dir)
        
        layout_mode.addWidget(self.widget_split_settings)
        self.widget_split_settings.setVisible(False)
        
        grp_mode.setLayout(layout_mode)
        layout.addWidget(grp_mode)

        # --- 3. Output File (For Simple Mode) ---
        self.grp_simple_out = QGroupBox("3. Output File")
        layout_simple_out = QHBoxLayout()
        self.line_out_csv = QLineEdit()
        self.line_out_csv.setPlaceholderText("Save as...")
        btn_save_csv = QPushButton("Browse Save")
        btn_save_csv.clicked.connect(self.browse_save_csv)
        layout_simple_out.addWidget(self.line_out_csv)
        layout_simple_out.addWidget(btn_save_csv)
        self.grp_simple_out.setLayout(layout_simple_out)
        layout.addWidget(self.grp_simple_out)

        # --- 4. Action Area ---
        layout.addSpacing(10)
        self.btn_process = QPushButton("Run Processing")
        self.btn_process.setMinimumHeight(40)
        self.btn_process.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_process.clicked.connect(self.run_processing)
        layout.addWidget(self.btn_process)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(100)
        layout.addWidget(self.log_view)

    # --- UI Logic ---
    def toggle_mode_ui(self):
        is_split = self.radio_split.isChecked()
        self.widget_split_settings.setVisible(is_split)
        self.grp_simple_out.setVisible(not is_split)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select TSV", "", "TSV Files (*.tsv *.txt);;All Files (*)")
        if path:
            self.line_input.setText(path)
            # Auto-suggest output
            if not self.line_out_csv.text():
                self.line_out_csv.setText(path.replace(".tsv", ".csv"))
            if not self.line_out_dir.text():
                self.line_out_dir.setText(os.path.dirname(path))

    def browse_save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", self.line_out_csv.text(), "CSV Files (*.csv)")
        if path:
            self.line_out_csv.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.line_out_dir.setText(path)

    def log(self, msg):
        self.log_view.append(msg)

    # --- Processing Logic ---
    def run_processing(self):
        tsv_path = self.line_input.text()
        if not os.path.exists(tsv_path):
            QMessageBox.critical(self, "Error", "Input file does not exist.")
            return

        # Prepare parameters
        mode = "SPLIT_CYCLES" if self.radio_split.isChecked() else "TRANSPOSE_ONLY"
        out_csv = self.line_out_csv.text()
        split_dir = self.line_out_dir.text()
        vol_idx = self.spin_col_idx.value()

        if mode == "SPLIT_CYCLES" and not split_dir:
            QMessageBox.critical(self, "Error", "Please select an output directory.")
            return

        # UI State
        self.btn_process.setEnabled(False)
        self.progress.setValue(0)
        self.log_view.clear()
        self.log("Starting process...")

        # Start Thread
        self.worker = ProcessingWorker(tsv_path, out_csv, mode, vol_idx, split_dir)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, msg):
        self.btn_process.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Failed", msg)

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = CVTransposerApp()
    window.show()
    app.exec()