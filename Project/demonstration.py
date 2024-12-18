import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QFrame, QSpacerItem, QSizePolicy, QPushButton
)
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QCategoryAxis, QScatterSeries
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt

import csv

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
timestamplist = [""]

class StockPredictorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        charts_layout = QHBoxLayout()

        # Chart section
        peakChart_layout = QVBoxLayout()
        chart_label = QLabel("Stock Prediction Model")
        chart_label.setAlignment(Qt.AlignCenter)
        font = QFont('Segoe UI', 20, QFont.Bold)
        font.setUnderline(True)
        chart_label.setFont(font)
        peakChart_layout.addWidget(chart_label)

        # Create a line chart
        peakChart = QChart()
        peakChart.setTitle("Stock: Apple")
        line_series = QLineSeries()
        scatter_series = QScatterSeries()
        axis = QCategoryAxis()
        vertical_line = QLineSeries()
        print(timestamp)
        for x, y in enumerate(close_values):
            line_series.append(x, y)
            xlabel = timestamplist[x][0:10]
            axis.append(xlabel, x)
            if (timestamplist[x].strip() == timestamp.strip()):
                print("Added!")
                print(y)
                vertical_line.append(x, y)
                vertical_line.append(x+1, y+1)
                scatter_series.append(x, y)

        line_series.setColor(QColor("blue"))
        vertical_line.setColor(QColor("red"))
        
        peakChart.addSeries(vertical_line)
        peakChart.addSeries(line_series)
        
        peakChart.setAxisX(axis,line_series)
        peakChart.setAxisX(axis, vertical_line)
        peakChart_view = QChartView(peakChart)
        peakChart_view.setRenderHint(QPainter.Antialiasing)
        peakChart_view.setFixedSize(800, 400)
        peakChart_layout.addWidget(peakChart_view)
        
        peakChart_layout.setAlignment(Qt.AlignCenter)
        charts_layout.addLayout(peakChart_layout)

        main_layout.addLayout(charts_layout)
        # Prediction data
        prediction_layout = QHBoxLayout()
        peak_label = QLabel("Prediction Point: ")
        peak_label.setFont(QFont('Segoe UI', 16, QFont.Bold))
        prediction_layout.addWidget(peak_label)
        prediction_layout.setAlignment(Qt.AlignCenter)
        peak_value = QLabel(f"{timestamp}")
        font = QFont('Segoe UI', 16)
        font.setUnderline(True)
        peak_value.setFont(font)
        prediction_layout.addWidget(peak_value)
        
        main_layout.addLayout(prediction_layout)
        
        result_layout= QVBoxLayout()
        layer1_layout = QHBoxLayout()
        layer2_layout = QHBoxLayout()
        
        class_pred = QLabel(f"{prediction}")
        class_pred_label = QLabel("Predicted Label: ")
        true_class_label = QLabel("True Label: ")
        true_class = QLabel(f"{true_label}")
        font = QFont('Segoe UI', 12)
        font.setUnderline(True)
        
        class_pred_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
        true_class_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
        class_pred.setFont(font)
        font.setUnderline(False)
        true_class.setFont(font)
        
        layer1_layout.setAlignment(Qt.AlignCenter)
        layer2_layout.setAlignment(Qt.AlignCenter)
        layer1_layout.addWidget(true_class_label)
        layer1_layout.addWidget(true_class)
        layer2_layout.addWidget(class_pred_label)
        layer2_layout.addWidget(class_pred)
        result_layout.addLayout(layer1_layout)
        result_layout.addLayout(layer2_layout)
        result_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(result_layout)
        
        spacer = QSpacerItem(50,50, QSizePolicy.Minimum, QSizePolicy.Fixed)
        spacerbox = QVBoxLayout()
        spacerbox.addItem(spacer)

        toggle_button = QPushButton("Static Data")
        toggle_button.clicked.connect(self.toggle_label)
        static_string = ""
        index = 7
        for key, value in static_data.items():
            string_append = f"  {key}: {value:.2f}\n"
            static_string += string_append
            index += -1
            if index == 0:
                break
        
        toggle_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        toggle_button.setFixedSize(200, 50)
        
        button_layout = QVBoxLayout()

        self.expanding_label = QLabel(static_string)
        self.expanding_label.setAlignment(Qt.AlignLeft)
        self.expanding_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.expanding_label.setVisible(False)
        
        button_layout.addWidget(toggle_button)
        button_layout.addWidget(self.expanding_label)
        button_layout.setAlignment(Qt.AlignCenter)
        
        main_layout.addLayout(button_layout)
        main_layout.addLayout(spacerbox)
#----------------------------------------------------------------------------------------------------------------
        # News and sentiment section
        news_layout = QVBoxLayout()
        news_label = QLabel("Today's News (Average Sentiment Value: +89%)")
        news_label.setAlignment(Qt.AlignCenter)
        news_layout.addWidget(news_label)

        news_table = QTableWidget(2, 2)
        news_table.setHorizontalHeaderLabels(["Article Title", "Sentiment Score"])
        news_table.setItem(0, 0, QTableWidgetItem("Steve Jobs hires 100,000 new employees"))
        news_table.setItem(0, 1, QTableWidgetItem("+93%"))
        news_table.setItem(1, 0, QTableWidgetItem("Customer outrage at iPhone 19 not having 3D cameras"))
        news_table.setItem(1, 1, QTableWidgetItem("-5%"))
        news_layout.addWidget(news_table)

        main_layout.addLayout(news_layout)
        
        news_table.setColumnWidth(1000, 400)
        news_table.setFixedWidth(1500)

        # Finalize layout
        self.setLayout(main_layout)
        
    def toggle_label(self):
        current_visibility = self.expanding_label.isVisible()
        self.expanding_label.setVisible(not current_visibility)
        

def get_random_prediction_full(model_path='my_model.keras',
                               seq_val_file='sequential_features_normalized_window12_val.csv',
                               static_val_file='static_features_normalized_window12_val.csv',
                               label_val_file='labels_interval_5_window_12_val.csv',
                               data_5_path='data_5.csv',
                               sequence_length=390):
    """
    Loads the trained model and necessary data, selects a random sequence from the validation set,
    makes a prediction, and returns the corresponding Timestamp, predicted class as string,
    the true label as string, the non-normalized static features, and the list of non-normalized 'Close' values for the sequence.

    Parameters:
    - model_path (str): Path to the saved Keras model file (e.g., 'my_model.h5').
    - seq_val_file (str): Path to the normalized sequential features validation CSV.
    - static_val_file (str): Path to the normalized static features validation CSV.
    - label_val_file (str): Path to the labels validation CSV.
    - data_5_path (str): Path to the non-normalized data CSV ('data_5.csv').
    - sequence_length (int): Number of time steps in each sequence used during training.

    Returns:
    - chosen_timestamp (str): The timestamp corresponding to the randomly selected validation sequence.
    - predicted_class_str (str): The class predicted by the model ('Neither', 'Peak', 'Valley').
    - true_label_str (str): The true class label ('Neither', 'Peak', 'Valley').
    - static_data (dict): Dictionary containing non-normalized static feature values at the chosen timestamp.
    - close_values (list of float): List of non-normalized 'Close' values for the sequence.
    """
    
    # Define label mapping
    def map_labels(labels):
        label_mapping = {-1: 'Valley', 0: 'Neither', 1: 'Peak'}
        return np.vectorize(label_mapping.get)(labels)
    
    # Check if all files exist
    required_files = [model_path, seq_val_file, static_val_file, label_val_file, data_5_path]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: The file '{file}' does not exist.")
            return None, None, None, None, None
    
    # Load validation data
    try:
        seq_val_df = pd.read_csv(seq_val_file, parse_dates=['Timestamp'])
        static_val_df = pd.read_csv(static_val_file, parse_dates=['Timestamp'])
        label_val_df = pd.read_csv(label_val_file, parse_dates=['Timestamp'])
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None, None, None, None, None
    
    # Verify alignment of Timestamps across dataframes
    def verify_alignment(*dfs):
        for i in range(1, len(dfs)):
            if not dfs[0]['Timestamp'].equals(dfs[i]['Timestamp']):
                raise ValueError(f"Timestamps are not aligned across DataFrames at position {i}.")
    
    try:
        verify_alignment(seq_val_df, static_val_df, label_val_df)
    except ValueError as ve:
        print(f"Data alignment error: {ve}")
        return None, None, None, None, None
    
    # Define feature columns
    seq_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
    static_features = [
        'Inflation', 'Inflation_RateChange', 'InterestRate', 'InterestRate_RateChange',
        'Sentiment_Avg', 'Sentiment_SD', 'HourOfDay',
        'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday',
        'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfMonth'
    ]
    
    # Convert dataframes columns to numpy arrays
    try:
        seq_val_data = seq_val_df[seq_features].values
        static_val_data = static_val_df[static_features].values.astype(np.float32)
        Y_val_original = label_val_df['Class'].values
        Y_val_mapped = map_labels(Y_val_original)
    except Exception as e:
        print(f"Error processing validation data: {e}")
        return None, None, None, None, None
    
    # Create sequences for validation
    def create_sequences(seq_data, static_data, labels, sequence_length):
        """
        Generates sequences of data for LSTM input.
        
        Parameters:
        - seq_data (np.array): Sequential features.
        - static_data (np.array): Static features.
        - labels (np.array): Labels corresponding to each interval.
        - sequence_length (int): Number of past intervals to include in each sequence.
        
        Returns:
        - X_seq (np.array): Sequences of sequential features.
        - X_static (np.array): Corresponding static features.
        - Y (np.array): Labels for each sequence.
        - timestamps (list): Timestamps corresponding to each label.
        """
        X_seq = []
        X_static = []
        Y = []
        timestamps = []
        num_samples = len(seq_data)
        print(seq_val_df['Timestamp'].iloc[sequence_length])
        for i in range(sequence_length, num_samples):
            X_seq.append(seq_data[i-sequence_length:i, :])
            X_static.append(static_data[i])
            Y.append(labels[i])
            timestamps.append(seq_val_df['Timestamp'].iloc[i])
            timestamplist.append(seq_val_df['Timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'))

        return np.array(X_seq), np.array(X_static), np.array(Y), timestamps

    X_seq_val, X_static_val, Y_val, timestamps_val = create_sequences(seq_val_data, static_val_data, Y_val_mapped, sequence_length)
    
    # Load non-normalized data
    try:
        data_5_df = pd.read_csv(data_5_path, parse_dates=['Timestamp'])
        data_5_df = data_5_df.sort_values('Timestamp').reset_index(drop=True)  # Ensure sorted
    except Exception as e:
        print(f"Error loading non-normalized data: {e}")
        return None, None, None, None, None
    
    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None
    
    # Check if there are validation sequences
    val_size = X_seq_val.shape[0]
    if val_size <= 0:
        print("No validation sequences available.")
        return None, None, None, None, None
    
    # Select a random index from the validation set
    # Define the desired date range
    start_date = pd.Timestamp("2024-01-24")
    end_date = pd.Timestamp("2024-01-31")

    # Filter timestamps within the desired range
    filtered_indices = [i for i, ts in enumerate(timestamps_val) if start_date <= ts <= end_date]

    # Ensure there are valid indices within the range
    if not filtered_indices:
        print("No validation sequences within the specified date range.")
        return None, None, None, None, None

    # Select a random index from the filtered indices
    rand_idx = random.choice(filtered_indices)

    
    # Get the chosen timestamp from the validation timestamps
    chosen_timestamp = timestamps_val[rand_idx]
    
    # Ensure the chosen timestamp exists in the non-normalized data
    matching_rows = data_5_df[data_5_df['Timestamp'] == chosen_timestamp]
    if matching_rows.empty:
        print("Chosen timestamp not found in non-normalized data.")
        return None, None, None, None, None
    
    # Find the index of the chosen timestamp in data_5_df
    chosen_index = data_5_df[data_5_df['Timestamp'] == chosen_timestamp].index[0]
    
    # Extract non-normalized 'Close' values for the sequence
    start_index = chosen_index - sequence_length
    end_index = chosen_index  # Exclusive
    if start_index < 0:
        print("Not enough data to extract 'Close' values for the sequence.")
        return None, None, None, None, None
    
    # Extract the 'Close' values for the sequence
    close_values = data_5_df.loc[start_index:end_index-1, 'Close'].tolist()
    # Note: 'Close' values for the sequence are from start_index to end_index-1 inclusive
    
    # Extract non-normalized static features at the chosen timestamp
    static_features_non_normalized = [
        'Inflation', 'Inflation_RateChange', 'InterestRate', 'InterestRate_RateChange',
        'Sentiment_Avg', 'Sentiment_SD', 'HourOfDay',
        'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday',
        'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfMonth'
    ]
    
    static_data_row = data_5_df.iloc[chosen_index][static_features_non_normalized]
    static_data = static_data_row.to_dict()
    
    # Prepare the input sample for prediction
    X_seq_sample = X_seq_val[rand_idx:rand_idx+1]       # Shape: (1, sequence_length, num_seq_features)
    X_static_sample = X_static_val[rand_idx:rand_idx+1] # Shape: (1, num_static_features)
    
    # Make a prediction using the model
    try:
        prediction_probs = model.predict([X_seq_sample, X_static_sample], verbose=0)  # Shape: (1, 3)
        predicted_class_num = np.argmax(prediction_probs, axis=1)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None, None, None
    
    # Map the predicted class to its string representation
    class_mapping = {0: 'Neither', 1: 'Peak', 2: 'Valley'}
    predicted_class_str = class_mapping.get(predicted_class_num, "Unknown")
    
    # Get the true label
    true_label_str = Y_val[rand_idx]  # Already mapped to strings
    
    
    # Return the timestamp, predicted class string, true label string, static data, and 'Close' values
    return chosen_timestamp.strftime('%Y-%m-%d %H:%M:%S'), predicted_class_str, true_label_str, static_data, close_values



def extract_second_column(file_path):
    """
    Reads the second column of a CSV file and maps its values to indices.

    :param file_path: Path to the CSV file.
    :return: List of tuples where each tuple is (index, value).
    """
    data_points = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row

            for index, row in enumerate(reader, start=1):
                # Extract the second column and map it to the index
                data_points.append((index, float(row[1])))

        return data_points

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError:
        print("Error: Could not convert data to floats. Check the file's content.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []




    
if __name__ == "__main__":
    """
    Example usage of the get_random_prediction_full function.
    """
    # Define file paths (modify these paths if your files are located elsewhere)
    MODEL_PATH = 'my_model.keras'
    SEQ_VAL_FILE = 'sequential_features_normalized_window12_val.csv'
    STATIC_VAL_FILE = 'static_features_normalized_window12_val.csv'
    LABEL_VAL_FILE = 'labels_interval_5_window_12_val.csv'
    DATA_5_PATH = 'data_5.csv'
    SEQUENCE_LENGTH = 390  # Must match the sequence length used during training


    # Call the function to get a random prediction with full data
    timestamp, prediction, true_label, static_data, close_values = get_random_prediction_full(
        model_path=MODEL_PATH,
        seq_val_file=SEQ_VAL_FILE,
        static_val_file=STATIC_VAL_FILE,
        label_val_file=LABEL_VAL_FILE,
        data_5_path=DATA_5_PATH,
        sequence_length=SEQUENCE_LENGTH
    )
    
    
    
    if (timestamp is not None and prediction is not None and 
        true_label is not None and static_data is not None and 
        close_values is not None):
        print("Timestamp:", timestamp)
        print("Predicted Class:", prediction)
        print("True Class:", true_label)
        print("\nNon-normalized Static Data:")
        for key, value in static_data.items():
            print(f"  {key}: {value}")
        #print("\nNon-normalized 'Close' values for the sequence:")
        #print(close_values)
    else:
        print("Prediction could not be made due to an error.")
        
    app = QApplication(sys.argv)
    window = StockPredictorUI()
    window.show()
    sys.exit(app.exec_())