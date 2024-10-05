import scipy.io
import numpy as np
import os
import csv
import h5py
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert


def sliding_window_zscore(data, window_size=60000):
    # Pad the data to handle the edges
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    
    # Calculate cumulative sum for efficient mean and std computation
    cumsum = np.cumsum(padded_data)
    cumsum_sq = np.cumsum(padded_data ** 2)
    
    # Calculate moving mean and std
    moving_mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Add a small epsilon to avoid negative values due to floating-point precision
    epsilon = 1e-8
    var = (cumsum_sq[window_size:] - cumsum_sq[:-window_size]) / window_size - moving_mean ** 2
    moving_std = np.sqrt(np.maximum(var, epsilon))
    
    # Avoid division by zero
    moving_std = np.maximum(moving_std, epsilon)
    
    # Calculate z-score
    zscore = (data - moving_mean) / moving_std
    return zscore

def preprocess_and_filter_data(data, sampling_rate=2000):
    nyquist_frequency = sampling_rate / 2

    # Step 1: Common Average Referencing (CAR)
    car_data = data - np.mean(data, axis=0)

    # Step 2: Notch filters at 50Hz and 100Hz
    notch_freq = [50, 100]
    for freq in notch_freq:
        b_notch, a_notch = signal.iirnotch(freq, Q=50, fs=sampling_rate)
        car_data = signal.filtfilt(b_notch, a_notch, car_data, axis=1)

    # # Step 3: 100 Hz low-pass filter
    # lowpass_100hz = 100 / nyquist_frequency
    # b_lowpass, a_lowpass = signal.butter(4, lowpass_100hz, btype='low')
    # filtered_100hz = signal.filtfilt(b_lowpass, a_lowpass, car_data, axis=1)

    # Step 4: Define filter parameters for specific bands
    lowpass_high = 100 / nyquist_frequency
    highpass_low = 70 / nyquist_frequency
    highpass_high = 150 / nyquist_frequency

    # Step 5: Design and apply the specific band filters
    b_low, a_low = signal.butter(8, lowpass_high, btype='low')
    b_high, a_high = signal.butter(8, [highpass_low, highpass_high], btype='band')

    low_filtered = signal.filtfilt(b_low, a_low, data, axis=1)
    high_filtered = signal.filtfilt(b_high, a_high, data, axis=1)

    # Step 6: Convert high-frequency data to amplitude envelope
    analytic_signal = hilbert(high_filtered, axis=1)
    amplitude_envelope = np.abs(analytic_signal)

    # Step 7: Apply sliding-window z-score normalization
    low_freq_zscore = np.apply_along_axis(sliding_window_zscore, 1, low_filtered)
    high_freq_zscore = np.apply_along_axis(sliding_window_zscore, 1,amplitude_envelope)

    # Step 8: Create 2D array with z-scored low-frequency data and z-scored amplitude envelope of high-frequency data
    filtered_array = np.vstack((low_freq_zscore, high_freq_zscore))

    return filtered_array


# Example usage:
# Assuming 'data' is your pandas DataFrame
# result = preprocess_and_filter_data(data.values)



def filter_and_save_plot(row, sampling_rate=2000, output_dir=r'', file_name=rf'filter_plot.png'):
    # Define filter parameters
    nyquist_frequency = sampling_rate / 2  # Hz
    cutoff_frequency = 500  # Hz
    order = 2  # Filter order
    high_cutoff_frequency = 0.2  # Hz
    high_order = 3  # Filter order

    # Normalize the cutoff frequencies
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    normalized_high_cutoff = high_cutoff_frequency / nyquist_frequency

    # Design the filters
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    b_high, a_high = signal.butter(high_order, normalized_high_cutoff, btype='high', analog=False)

    # Apply the filters
    row_low_pass = signal.filtfilt(b, a, row)
    row_filtered = signal.filtfilt(b_high, a_high, row_low_pass)

    return row_filtered


    # #procces the files
    # filtered_data = data.apply(apply_filters, axis=1, result_type="expand")
    # # Round the values to the nearest integer
    # return filtered_data

# Function to load EEG data from .mat file
def load_eeg_data(file_path):
    with h5py.File(file_path, 'r') as f:
        eeg_signals = f['data'][:]  # Assuming the EEG signals are stored under the key 'data'
        eeg_signals_filtered = preprocess_and_filter_data(eeg_signals.T)
        print(f'proccesed {file_path}')
    return eeg_signals_filtered.T.astype(np.float16)



# Function to extract relevant EEG signals for each event
def extract_eeg_signals(eeg_data, start_time, end_time):
    sampling_rate = 2000  # Assuming a sampling rate of 2000 Hz
    start_index = int(start_time * (sampling_rate / 1000000))  # Convert microseconds to index
    end_index = int(end_time * (sampling_rate / 1000000))
    eeg_signals_low = eeg_data.T[0][start_index:end_index]
    eeg_signals_high = eeg_data.T[1][start_index:end_index]
    return [eeg_signals_low, eeg_signals_high]

# Function to load EEG data from a file for all electrodes
def load_all_eeg_data():
    eeg_data = {}
    for electrode_num in range(1, 7):
        if electrode_num in [2, 4, 65, 66, 68]:
            continue
        # eeg_file_path = rf"/Users/mennymac/Desktop/University/EE final project /patient_1_meas/For_Meni_and_Ezra/CSC{electrode_num}.mat"

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
        for electrode_num in range(1, 7):
            if electrode_num in [2, 4, 65, 66, 68]:
                continue

            # eeg_after_filter = filter_and_save_plot(all_eeg_data[electrode_num].T)

            eeg_signals = extract_eeg_signals(all_eeg_data[electrode_num], start_time, end_time)
            eeg_grid.extend(eeg_signals)
        # Convert eeg_grid to numpy array
        eeg_grid = np.array(eeg_grid)

        # Create directory for the event label if it doesn't exist
        # output_dir = f'/Users/mennymac/Desktop/University/EE final project /patient_1_output/{label_to_info[label][1]}\{label_to_info[label][0]}'
        output_dir = rf"/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_test_with_env/{label_to_info[label][1]}/{label_to_info[label][0]}"
        os.makedirs(output_dir, exist_ok=True)

        # Save EEG signals to a CSV file
        output_file_path = os.path.join(output_dir,
                                        f'#{e}_{label_to_info[label][1]}_{label_to_info[label][0]}_eeg_event_{start_time}_{end_time}.csv')
        e += 1
        save_to_csv(eeg_grid, output_file_path)


if __name__ == "__main__":
    # Open the text file
    # with open(r"/Users/mennymac/Desktop/University/EE final project /audio_times - training only_with_changes.log", "r") as file:
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

                if next_line and (int(next_line.split()[1]) - event_time_start) / 1e6 < 1:
                    event_time_end = int(next_line.split()[1])
                else:
                    event_time_end = event_time_start + 1 * 1e6
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
        "ANNOTATE_A_OTHER": ["a", "other"],
        "ANNOTATE_A": ["a", "annotate"],
        "ANNOTATE_A_IMAGERY": ["a", "imagery"],
        "ANNOTATE_E_OTHER": ["e", "other"],
        "ANNOTATE_E": ["e", "annotate"],
        "ANNOTATE_E_IMAGERY": ["e", "imagery"],
        "ANNOTATE_I_OTHER": ["i", "other"],
        "ANNOTATE_I": ["i", "annotate"],
        "ANNOTATE_I_IMAGERY": ["i", "imagery"],
        "ANNOTATE_O_OTHER": ["o", "other"],
        "ANNOTATE_O": ["o", "annotate"],
        "ANNOTATE_O_IMAGERY": ["o", "imagery"],
        "ANNOTATE_U_OTHER": ["u", "other"],
        "ANNOTATE_U": ["u", "annotate"],
        "ANNOTATE_U_IMAGERY": ["u", "imagery"],
    }

    main()
