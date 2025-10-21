import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
                            QLabel, QSpinBox, QFileDialog, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ResultsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FL Results Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Set up matplotlib with publication-ready settings
        plt.style.use('default')  # Reset to default style
        self.figure, self.ax = plt.subplots(figsize=(3.5, 2.5))  # Standard figure size for two-column papers
        self.figure.set_dpi(300)  # High DPI for quality

        # Configure plot style manually
        self.ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_axisbelow(True)  # Place grid below data points

        # Set font sizes and styles for publication
        plt.rcParams.update({
            'font.size': 8,  # Base font size
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'mathtext.fontset': 'stix',
        })
        # Create top panel for controls
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)

        # Create controls widget
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Add folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Data folder:")
        self.folder_button = QPushButton("Browse")
        self.folder_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_button)
        controls_layout.addLayout(folder_layout)

        self.label_inputs = []
        for i in range(4):  # Support for 4 timestamps
            label_layout = QHBoxLayout()
            label_layout.addWidget(QLabel(f"Label for timestamp {i + 1}:"))
            label_input = QLineEdit()
            self.label_inputs.append(label_input)
            label_layout.addWidget(label_input)
            controls_layout.addLayout(label_layout)

        # Add checkboxes
        checkbox_layout = QHBoxLayout()
        self.show_average_checkbox = QCheckBox("Show Average Only")
        self.show_average_checkbox.stateChanged.connect(self.plot_results)
        self.show_rounds_only_checkbox = QCheckBox("Show Rounds Only")  # New checkbox
        self.show_rounds_only_checkbox.stateChanged.connect(self.plot_results)
        checkbox_layout.addWidget(self.show_average_checkbox)
        checkbox_layout.addWidget(self.show_rounds_only_checkbox)
        controls_layout.addLayout(checkbox_layout)

        # Add plot button, save button, and status
        button_status_layout = QHBoxLayout()
        self.plot_button = QPushButton("Plot Results")
        self.plot_button.clicked.connect(self.plot_results)
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.status_label = QLabel("")
        button_status_layout.addWidget(self.plot_button)
        button_status_layout.addWidget(self.save_button)
        button_status_layout.addWidget(self.status_label)
        controls_layout.addLayout(button_status_layout)

        top_layout.addWidget(controls_widget)

        # Add timestamp list with search
        timestamp_widget = QWidget()
        timestamp_layout = QVBoxLayout(timestamp_widget)
        timestamp_layout.addWidget(QLabel("Timestamps:"))

        # Add search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search timestamps...")
        self.search_bar.textChanged.connect(self.filter_timestamps)
        timestamp_layout.addWidget(self.search_bar)

        self.timestamp_list = QListWidget()
        self.timestamp_list.setMaximumHeight(200)
        self.timestamp_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.timestamp_list.itemSelectionChanged.connect(self.handle_timestamp_selection)
        timestamp_layout.addWidget(self.timestamp_list)
        top_layout.addWidget(timestamp_widget)

        layout.addWidget(top_panel)

        # Create plot panel
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_panel)

        layout.setStretchFactor(top_panel, 1)
        layout.setStretchFactor(plot_panel, 4)

        self.base_path = None
        self.selected_timestamps = []
        self.colors = ['#0077BB', '#EE7733', '#009988', '#CC3311']
        self.all_timestamps = []

        self.plot_button.setEnabled(False)
        self.save_button.setEnabled(False)

    # [Previous methods remain unchanged until plot_results]

    def plot_results(self):
        if not self.selected_timestamps:
            return

        self.ax.clear()
        all_data = []

        for timestamp_idx, timestamp_folder in enumerate(self.selected_timestamps):
            print(f"\nProcessing timestamp: {timestamp_folder}")

            fs_results = self.load_fs_results(timestamp_folder)
            print(f"FS results type: {type(fs_results)}")
            print(f"FS results value: {fs_results}")

            client_rounds = self.load_client_data(timestamp_folder)

            if self.show_average_checkbox.isChecked():
                if client_rounds:
                    max_rounds = max(len(rounds) for rounds in client_rounds)
                    avg_rounds = []
                    for round_idx in range(max_rounds):
                        round_scores = [float(client[round_idx]) for client in client_rounds if round_idx < len(client)]
                        avg_rounds.append(float(np.mean(round_scores)))

                    data = []
                    if not self.show_rounds_only_checkbox.isChecked():
                        if fs_results is not None and len(fs_results) > 1:
                            # Full FS case - use all values
                            data.extend(list(map(float, fs_results)))
                        else:
                            # No FS case - use initial value at Fed FS position
                            initial_value = float(fs_results[0]) if fs_results else None
                            data.extend([None, None, initial_value])
                    data.extend(avg_rounds)
                    all_data.append((timestamp_folder, data))
            else:
                for client_idx, client_scores in enumerate(client_rounds):
                    data = []
                    if not self.show_rounds_only_checkbox.isChecked():
                        if fs_results is not None and len(fs_results) > 1:
                            data.extend(list(map(float, fs_results)))
                        else:
                            initial_value = float(fs_results[0]) if fs_results else None
                            data.extend([None, None, initial_value])
                    data.extend(client_scores)
                    all_data.append((f"{timestamp_folder} (Client {client_idx + 1})", data))

        # Create x-axis labels
        if all_data:
            max_points = max(len(data) for _, data in all_data)
            x_positions = list(range(max_points))

            # Generate x-axis labels
            if self.show_rounds_only_checkbox.isChecked():
                labels = [str(i) for i in range(max_points)]
            else:
                labels = ['Initial', 'FS', 'FE']
                num_rounds = max_points - len(labels)
                labels.extend([str(i) for i in range(num_rounds)])

            # Plot data with improved visual settings
            for idx, (label, data) in enumerate(all_data):
                label_text = ""
                if idx < len(self.label_inputs):
                    label_text = self.label_inputs[idx].text().strip()
                display_label = label_text or label

                timestamp_idx = idx
                if not self.show_average_checkbox.isChecked():
                    timestamp_idx = idx // max(1, len(client_rounds))

                color = self.colors[timestamp_idx % len(self.colors)]

                # Filter out None values and their corresponding positions
                valid_data = [(i, val) for i, val in enumerate(data) if val is not None]
                if valid_data:
                    x_vals, y_vals = zip(*valid_data)
                    self.ax.plot(x_vals, y_vals, 'o-',
                                label=display_label,
                                color=color,
                                linewidth=1.5,
                                markersize=4,
                                alpha=0.9)

            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha='right')
            self.ax.set_ylabel('F1 Score')

            # Improved grid settings
            self.ax.grid(True, linestyle='--', alpha=0.3, which='major')

            # Create semi-transparent white background for legend
            legend = self.ax.legend(bbox_to_anchor=(0.98, 0.02),
                                  loc='lower right',
                                  framealpha=0.6,  # Reduced opacity of legend background
                                  edgecolor='none',
                                  ncol=1,
                                  bbox_transform=self.ax.transAxes,  # Use axes coordinates
                                  facecolor='white')  # White background

            # Make legend entries more transparent
            for text in legend.get_texts():
                text.set_alpha(0.9)  # Slightly transparent text

            # Tight layout with specific margins for publication
            self.figure.tight_layout(pad=0.2)
            self.canvas.draw()

    # [Rest of the methods remain unchanged]
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.base_path = folder
            self.folder_label.setText(f"Data folder: {folder}")
            self.update_timestamp_list()

    def update_timestamp_list(self):
        self.timestamp_list.clear()
        if not self.base_path:
            return

        self.all_timestamps = [d for d in os.listdir(self.base_path)
                               if os.path.isdir(os.path.join(self.base_path, d))]
        self.all_timestamps.sort(reverse=True)

        self.timestamp_list.addItems(self.all_timestamps)

        if self.all_timestamps:
            self.plot_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.status_label.setText(f"Found {len(self.all_timestamps)} folders")
        else:
            self.status_label.setText("No timestamp folders found")

    def filter_timestamps(self, search_text):
        from PyQt6.QtWidgets import QListWidgetItem  # Add this import
        from PyQt6.QtGui import QFont, QColor, QBrush

        search_text = search_text.lower()

        # Get currently selected items before modifying the list
        selected_timestamps = set(item.text() for item in self.timestamp_list.selectedItems())

        # Store current connection state and disconnect if connected
        try:
            self.timestamp_list.itemSelectionChanged.disconnect(self.handle_timestamp_selection)
        except:
            pass  # If not connected, ignore the error

        # Clear and repopulate with all items
        self.timestamp_list.clear()

        normal_font = QFont()
        bold_font = QFont()
        bold_font.setBold(True)

        gray_color = QColor(128, 128, 128)  # Color for non-matching items

        for timestamp in self.all_timestamps:
            item = QListWidgetItem(timestamp)
            if search_text:
                if search_text in timestamp.lower():
                    item.setFont(bold_font)  # Make matching items bold
                    item.setForeground(QBrush(QColor(0, 0, 0)))  # Black text for matches
                else:
                    item.setFont(normal_font)
                    item.setForeground(QBrush(gray_color))  # Gray out non-matching items
            else:
                item.setFont(normal_font)
                item.setForeground(QBrush(QColor(0, 0, 0)))  # Reset to black text

            self.timestamp_list.addItem(item)

            # Restore selection if this item was previously selected
            if timestamp in selected_timestamps:
                item.setSelected(True)

        # Reconnect selection signal
        self.timestamp_list.itemSelectionChanged.connect(self.handle_timestamp_selection)

    def load_fs_results(self, timestamp_folder):
        fs_path = os.path.join(self.base_path, timestamp_folder, 'feature_election_results.csv')
        if os.path.exists(fs_path):
            try:
                df = pd.read_csv(fs_path)
                results = []

                # Convert DataFrame to native Python types immediately
                df_dict = df.to_dict('records')[0]  # Get first row as dict

                # Add each stage in order
                stages = ['Initial F1', 'Local FS', 'Feature Election']
                for stage in stages:
                    if stage in df_dict:
                        results.append(float(df_dict[stage]))

                print(f"FS Results for {timestamp_folder}: {results}")
                return results if results else None
            except Exception as e:
                print(f"Error reading FS results: {str(e)}")
                return None
        return None

    def load_client_data(self, timestamp_folder):
        csv_folder = os.path.join(self.base_path, timestamp_folder, 'CSV')
        if not os.path.exists(csv_folder):
            return []

        client_results = []
        for file in os.listdir(csv_folder):
            if 'client' in file and not 'train' in file:
                file_path = os.path.join(csv_folder, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'Round' in df.columns and 'F1 Score' in df.columns:
                        scores = df.sort_values('Round')['F1 Score'].tolist()
                        scores = [float(score) for score in scores]  # Convert to native Python floats
                        client_results.append(scores)
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")

        return client_results

    def handle_timestamp_selection(self):
        selected_items = self.timestamp_list.selectedItems()
        if not selected_items:
            return

        selected_timestamps = [item.text() for item in selected_items]

        # Limit to 4 most recent selections
        if len(selected_timestamps) > 4:
            self.timestamp_list.itemSelectionChanged.disconnect(self.handle_timestamp_selection)
            self.timestamp_list.clearSelection()
            for item_text in selected_timestamps[-4:]:
                items = self.timestamp_list.findItems(item_text, Qt.MatchFlag.MatchExactly)
                if items:
                    items[0].setSelected(True)
            self.timestamp_list.itemSelectionChanged.connect(self.handle_timestamp_selection)
            selected_timestamps = selected_timestamps[-4:]

        self.selected_timestamps = selected_timestamps
        self.plot_results()

    def save_plot(self):
        if not self.selected_timestamps:
            self.status_label.setText("Please select at least one timestamp")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "",
            "PDF Files (*.pdf);;EPS Files (*.eps);;PNG Files (*.png);;All Files (*)"
        )

        if file_name:
            # Set figure size for two-column publication format
            self.figure.set_size_inches(3.5, 2.5)  # Standard width for two-column papers

            # Save with appropriate settings based on file format
            if file_name.lower().endswith('.pdf'):
                self.figure.savefig(file_name,
                                    format='pdf',
                                    dpi=300,
                                    bbox_inches='tight',
                                    pad_inches=0.02,
                                    transparent=True)
            elif file_name.lower().endswith('.eps'):
                self.figure.savefig(file_name,
                                    format='eps',
                                    dpi=300,
                                    bbox_inches='tight',
                                    pad_inches=0.02)
            else:  # PNG or other formats
                self.figure.savefig(file_name,
                                    dpi=600,  # Higher DPI for raster formats
                                    bbox_inches='tight',
                                    pad_inches=0.02)

            self.status_label.setText(f"Saved figure to {file_name}")


def main():
    app = QApplication(sys.argv)
    viewer = ResultsViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()