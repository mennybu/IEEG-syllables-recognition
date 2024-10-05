import os
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Define filter parameters
nyquist_frequency = 1000  # Hz
cutoff_frequency = 600  # Hz, adjust this value based on your needs
order = 2  # Filter order

# Define filter parameters for high-pass filter
high_cutoff_frequency = 0.1  # Hz
high_order = 3  # Filter order

# Normalize the cutoff frequency
normalized_cutoff = cutoff_frequency / nyquist_frequency

# Normalize the high cutoff frequency
normalized_high_cutoff = high_cutoff_frequency / nyquist_frequency

# Design the low-pass filter
b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

# Design the high-pass filter
b_high, a_high = signal.butter(high_order, normalized_high_cutoff, btype='high', analog=False)

# Function to apply the filters to a single row
def apply_filters(row):
    row = signal.filtfilt(b, a, row)
    #row = signal.filtfilt(b_high, a_high, row)
    return row

# Function to process a single file
def process_file(input_file, output_file):
    data = pd.read_csv(input_file, header=None)
    filtered_data = data.apply(apply_filters, axis=1, result_type="expand")
    # Round the values to the nearest integer
    rounded_data = filtered_data.round().astype(int)
    rounded_data.to_csv(output_file, header=False, index=False)
    print(f"Processed {input_file} -> {output_file}")

# Function to process files in a directory structure
def process_files(input_root, output_root):
    for subdir, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, input_root)
                output_subdir = os.path.join(output_root, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Creating output file name
                file_name, file_extension = os.path.splitext(file)
                output_file_path = os.path.join(output_subdir, f"{file_name}_filtered.csv")
                
                # Process the file
                process_file(input_file_path, output_file_path)

# usage
input_root = '/home/tauproj6/EEG_proj/patient_1_output_1.5s_122ch'
output_root = '/home/tauproj6/EEG_proj/patient_1_output_1.5s_filtered_lp'
process_files(input_root, output_root)

