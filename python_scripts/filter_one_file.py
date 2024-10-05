import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv(r'/home/tauproj6/EEG_proj/patient_1_output_1.5s_122ch/annotate/a/#51_annotate_a_eeg_event_1650755660_1652255660.0.csv', header=None )

# Define filter parameters
nyquist_frequency = 1000  # Hz
cutoff_frequency = 200  # Hz, adjust this value based on your needs
order = 2  # Filter order

# Define filter parameters for high-pass filter
high_cutoff_frequency = 0.2  # Hz
high_order = 3  # Filter order

# Normalize the cutoff frequency
normalized_cutoff = cutoff_frequency / nyquist_frequency

# Normalize the high cutoff frequency
normalized_high_cutoff = high_cutoff_frequency / nyquist_frequency

# Design the low-pass filter
b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

# Design the high-pass filter
b_high, a_high = signal.butter(high_order, normalized_high_cutoff, btype='high', analog=False)


# Function to apply the filter to a single row
def apply_filter(row):
    return signal.filtfilt(b, a, row)

# Function to apply the high-pass filter to a single row
def apply_high_pass_filter(row):
    return signal.filtfilt(b_high, a_high, row)



# Apply the filter to each row
filtered_data_lp = data.apply(apply_filter, axis=1, result_type="expand")
filtered_data = filtered_data_lp.apply(apply_high_pass_filter, axis=1, result_type="expand")

# Function to compute the FFT and plot the frequency spectrum
def plot_frequency_spectrum(data_row, title):
    # Compute the FFT
    fft_result = np.fft.fft(data_row)
    # Compute the corresponding frequencies
    frequencies = np.fft.fftfreq(len(data_row), d=0.5/nyquist_frequency)
    # Plot the magnitude spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.yscale('log')
    plt.grid()
    plt.show()

# Plot original and filtered data for the first row as an example
plt.figure(figsize=(12, 6))
plt.plot(data.iloc[1], label='Original')
plt.plot(filtered_data.iloc[1], label='Filtered')
plt.legend()
plt.title('Original vs Filtered Data (First Row)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# Plot the frequency spectrum for the original and filtered data for the first row
print(data.shape)
plot_frequency_spectrum(data.iloc[1], 'Frequency Spectrum of Original Data (First Row)')
plot_frequency_spectrum(filtered_data.iloc[1], 'Frequency Spectrum of Filtered Data (First Row)')

# Save the filtered data to a new file
filtered_data.to_csv('/home/tauproj6/EEG_proj/#51_annotate_a_eeg_event_1650755660_1652255660.0.csv_filtered_data.txt', header=False, index=False)

print("Filtering complete. The new matrix with filtered data has been saved as 'filtered_data.txt'.")
