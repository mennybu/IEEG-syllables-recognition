import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Function to recursively find all CSV files in the directory and subdirectories
def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Function to perform PCA on data and save the projected data
def perform_pca_and_save(input_dir, output_dir):
    # Get the list of all CSV files
    file_list = find_csv_files(input_dir)
   
    # Load CSV files into a list of DataFrames
    data_frames = [pd.read_csv(file, header=None) for file in file_list]

    # Ensure all DataFrames have the same shape (e.g., 122 rows)
    for df in data_frames:
        assert df.shape[0] == 242, "All CSV files must have 122 rows (electrodes)."

    # Convert DataFrames to numpy arrays and stack them
    data1 = np.stack([df.values for df in data_frames//2], axis=0)  # Shape: (n_samples, 121, n_time_points)
    data2 = np.stack([df.values for df in data_frames//2], axis=0)  # Shape: (n_samples, 121, n_time_points)

    # Convert to torch tensor
    data1_tensor = torch.tensor(data, dtype=torch.float32)
    data2_tensor = torch.tensor(data, dtype=torch.float32)

    # Reshape data to (n_samples * n_time_points, 122)
    n_samples, n_electrodes//2, n_time_points = data1_tensor.shape
    reshaped_data1 = data_tensor.permute(0, 2, 1).reshape(-1, n_electrodes)
    reshaped_data2 = data_tensor.permute(0, 2, 1).reshape(-1, n_electrodes)


    # Normalize the data for PCA
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(reshaped_data1)
    normalized_data = scaler.fit_transform(reshaped_data2)

    # Convert normalized data back to torch tensor
    normalized_data_tensor1 = torch.tensor(normalized_data1, dtype=torch.float32)
    normalized_data_tensor2 = torch.tensor(normalized_data2, dtype=torch.float32)

    # Perform PCA, retaining 99% of the variance
    U, S, V = torch.pca_lowrank(normalized_data_tensor, q=n_electrodes)
    explained_variance_ratio = (S ** 2) / torch.sum(S ** 2)
    cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
    n_components = torch.sum(cumulative_variance_ratio < 0.99).item() + 1

    # Project original data onto the first n_components principal components
    # Reshape original data for projection
    reshaped_data_for_projection = data_tensor.permute(0, 2, 1).reshape(-1, n_electrodes)
    projected_data = torch.matmul(reshaped_data_for_projection, V[:, :n_components])
    # Reshape projected data back to original form but with reduced dimensions
    reshaped_projected_data = projected_data.reshape(n_samples, n_time_points, n_components)

    # Process each file and save the projected data
    for file_path in file_list:
        # Get relative path for maintaining directory structure
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Define the output file name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{file_name}_PCA.csv"
        output_file_path = os.path.join(output_subdir, output_file_name)
       
        # Find the corresponding sample index
        file_index = file_list.index(file_path)
       
        # Transpose the data (swap axes 1 and 2)
        transposed_projected_data = reshaped_projected_data[file_index].transpose(1, 0)
       
        # Save the transposed projected data to CSV
        df_projected = pd.DataFrame(transposed_projected_data.numpy())
        df_projected.to_csv(output_file_path, header=False, index=False)
        print(f"Saved PCA projected data for {file_path} to {output_file_path}")

    print("Number of components to retain 99% variance:", n_components)
    print("Principal Components shape:", V[:, :n_components].shape)
    print("Projected Data shape:", reshaped_projected_data.shape)
    print(f"Projected data saved to directory: {output_dir}")

# List of (input_dir, output_dir) pairs
directory_pairs = [
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/annotate', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca/annotate'),
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/other', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca/other'),
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/imagery', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca/imagery')
]

# Process each directory pair
for input_dir, output_dir in directory_pairs:
    perform_pca_and_save(input_dir, output_dir)
