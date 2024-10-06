# import pandas as pd
# import h5py
#
# # Load the MATLAB file using h5py
# file_path = "C:/Users/amhec/Downloads/For_Meni_and_Ezra/CSC1.mat"
# with h5py.File(file_path, 'r') as mat_data:
#     # Print keys (variables) in the MATLAB file
#     print("Keys in the MATLAB file:")
#     print(mat_data.keys())
#     # Convert the MATLAB data to a DataFrame
#     data_df = pd.DataFrame({key: mat_data[key][()].squeeze() for key in mat_data.keys()})
#
# # Create a CSV file
# csv_file_path = "C:/Users/amhec/Downloads/CSC1_data.csv"
# data_df.to_csv(csv_file_path, index=False)
#
# print("CSV file has been created successfully.")

import scipy.io
import numpy as np
import os
import csv
import h5py

# Function to load EEG data from .mat file
def load_eeg_data(file_path):
    with h5py.File(file_path, 'r') as f:
        eeg_signals = f['data'][:]  # Assuming the EEG signals are stored under the key 'data'
    return eeg_signals

# Function to extract relevant EEG signals for each event
def extract_eeg_signals(eeg_data, start_time, end_time):
    sampling_rate = 2000  # Assuming a sampling rate of 2000 Hz
    start_index = int(start_time * (sampling_rate / 1000000))  # Convert microseconds to index
    end_index = int(end_time * (sampling_rate / 1000000))
    eeg_signals = eeg_data[start_index:end_index]
    return eeg_signals

# Function to load EEG data from a file for all electrodes
def load_all_eeg_data():
    eeg_data = {}
    for electrode_num in range(1, 127):
        if electrode_num in [2, 4, 66, 68]:
            continue
        #eeg_file_path = rf"/Users/mennymac/Desktop/University/EE final project /patient_1_meas/For_Meni_and_Ezra/CSC{electrode_num}.mat"
        
        eeg_file_path = rf"/home/tauproj6/EEG_proj/For_Meni_and_Ezra/CSC{electrode_num}.mat"
        eeg_data[electrode_num] = load_eeg_data(eeg_file_path)
    return eeg_data

# Function to save EEG signals to a CSV file
def save_to_csv(eeg_signals, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in eeg_signals:
            # Remove brackets from each number in the row
            cleaned_row = [str(entry).strip('[]') for entry in row]
            writer.writerow(cleaned_row)

# Main function
def main():
    # Load EEG data for all electrodes
    all_eeg_data = load_all_eeg_data()
    print("loaded")

    # Iterate over each event
    e = 1
    for event in event_data:
        start_time = event["start_time"]
        end_time = event["end_time"]
        label = event["label"]

        # Load EEG data for all electrodes
        eeg_grid = []
        for electrode_num in range(1, 127):
            if electrode_num in [2, 4, 66, 68]:
                continue
            eeg_signals = extract_eeg_signals(all_eeg_data[electrode_num], start_time, end_time)
            eeg_grid.append(eeg_signals)
        # Convert eeg_grid to numpy array
        eeg_grid = np.array(eeg_grid)

        # Create directory for the event label if it doesn't exist
        #output_dir = f'/Users/mennymac/Desktop/University/EE final project /patient_1_output/{label_to_info[label][1]}\{label_to_info[label][0]}'
        output_dir = rf"/home/tauproj6/EEG_proj/patient_1_output_1.5s_122ch/{label_to_info[label][1]}/{label_to_info[label][0]}"
        os.makedirs(output_dir, exist_ok=True)


        # Save EEG signals to a CSV file
        output_file_path = os.path.join(output_dir, f'#{e}_{label_to_info[label][1]}_{label_to_info[label][0]}_eeg_event_{start_time}_{end_time}.csv')
        e+=1
        save_to_csv(eeg_grid, output_file_path)

if __name__ == "__main__":
    # Open the text file
    #with open(r"/Users/mennymac/Desktop/University/EE final project /audio_times - training only_with_changes.log", "r") as file:
    with open(fr"/home/tauproj6/EEG_proj/audio_times - training only_with_changes.log") as file:
        # Initialize a list to store event data
        event_data = []

        # Read the first line from the file
        line = file.readline()
        while "skip" in line:
            line = file.readline()

        # Continue reading lines until reaching the end of the file
        while line:
            # Check if the line contains "BEEP" or "IMAGERY"
            if "BEEP" in line or "IMAGERY" in line:
                # Process the event and extract relevant data
                # For example, get the starting and ending times
                event_time_start = int(line.split()[1])
                label = line.split()[2]

                # Read the next line
                next_line = file.readline()

                while next_line and "BEEP" not in next_line and "IMAGERY" not in next_line:
                    # Process the middle line if it exists
                    # For example, extract the label
                    if next_line.strip():  # Check if the line is not empty
                        label = next_line.split()[2]
                    # Read the next line
                    next_line = file.readline()

                if next_line and (int(next_line.split()[1]) - event_time_start)/1e6 < 1.5:
                    event_time_end = int(next_line.split()[1])
                else:
                    event_time_end = event_time_start + 1.5*1e6
                # event_time_end = int(next_line.split()[1]) if next_line else None

                # Append the extracted data to the list
                event_data.append({
                    "start_time": event_time_start,
                    "end_time": event_time_end,
                    "label": label if 'label' in locals() else None
                })
                line = next_line
                while "skip" in line:
                    line = file.readline()

            else:
                # Read the next line
                line = file.readline()

    event_data.pop()  # Remove the last event from the list

    # Print the extracted event data
    for event in event_data:
        print("Event Start Time:", event["start_time"])
        print("Event End Time:", event["end_time"])
        print("Event Label:", event["label"])

    label_to_info = {
        "ANNOTATE_A_OTHER": ["a","other"],
        "ANNOTATE_A": ["a","annotate"],
        "ANNOTATE_A_IMAGERY": ["a","imagery"],
        "ANNOTATE_E_OTHER": ["e","other"],
        "ANNOTATE_E": ["e","annotate"],
        "ANNOTATE_E_IMAGERY": ["e","imagery"],
        "ANNOTATE_I_OTHER": ["i","other"],
        "ANNOTATE_I": ["i","annotate"],
        "ANNOTATE_I_IMAGERY": ["i","imagery"],
        "ANNOTATE_O_OTHER": ["o","other"],
        "ANNOTATE_O": ["o","annotate"],
        "ANNOTATE_O_IMAGERY": ["o","imagery"],
        "ANNOTATE_U_OTHER": ["u","other"],
        "ANNOTATE_U": ["u","annotate"],
        "ANNOTATE_U_IMAGERY": ["u","imagery"],
    }

    main()
