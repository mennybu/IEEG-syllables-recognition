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



def perform_split_pca_and_save(input_dir, output_dir):
    file_list = find_csv_files(input_dir)
    data_frames = [pd.read_csv(file, header=None) for file in file_list]

    for df in data_frames:
        assert df.shape[0] == 242, "All CSV files must have 242 rows (electrodes)."

    data = np.stack([df.values for df in data_frames], axis=0)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    n_samples, n_electrodes, n_time_points = data_tensor.shape

    # Split the data into odd and even rows
    data_odd = data_tensor[:, 0::2, :]
    data_even = data_tensor[:, 1::2, :]

    def process_data(data_part, name):
        reshaped_data = data_part.permute(0, 2, 1).reshape(-1, data_part.shape[1])
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(reshaped_data)
        normalized_data_tensor = torch.tensor(normalized_data, dtype=torch.float32)

        U, S, V = torch.pca_lowrank(normalized_data_tensor, q=data_part.shape[1])
        explained_variance_ratio = (S ** 2) / torch.sum(S ** 2)
        cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
        n_components = torch.sum(cumulative_variance_ratio < 0.99).item() + 1

        print(f"Number of components to retain 99% variance for {name} rows:", n_components)
        print(f"Principal Components shape for {name} rows:", V[:, :n_components].shape)

        return n_components, V

    n_components_odd, V_odd = process_data(data_odd, "odd")
    n_components_even, V_even = process_data(data_even, "even")

    # Use the maximum number of components for both
    max_components = max(n_components_odd, n_components_even)

    def project_data(data_part, V, n_components):
        reshaped_data = data_part.permute(0, 2, 1).reshape(-1, data_part.shape[1])
        projected_data = torch.matmul(reshaped_data, V[:, :n_components])
        return projected_data.reshape(n_samples, n_time_points, n_components)

    projected_odd = project_data(data_odd, V_odd, max_components)
    projected_even = project_data(data_even, V_even, max_components)

    # Stack the projected data vertically
    stacked_projected_data = torch.cat((projected_odd, projected_even), dim=2)

    print("Stacked Projected Data shape:", stacked_projected_data.shape)

    for file_path in file_list:
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{file_name}_OddEven_PCA.csv"
        output_file_path = os.path.join(output_subdir, output_file_name)

        file_index = file_list.index(file_path)
        # Reshape to [pca_channels, t]
        reshaped_projected_data = stacked_projected_data[file_index].transpose(1, 0)

        df_projected = pd.DataFrame(reshaped_projected_data.numpy())
        df_projected.to_csv(output_file_path, header=False, index=False)
        print(f"Saved Odd-Even PCA projected data for {file_path} to {output_file_path}")

    print(f"Projected data saved to directory: {output_dir}")

# List of (input_dir, output_dir) pairs
directory_pairs = [
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/annotate', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca_sep/annotate'),
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/other', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca_sep/other'),
    ('/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/imagery', '/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca_sep/imagery')
]

# Process each directory pair
for input_dir, output_dir in directory_pairs:
    perform_split_pca_and_save(input_dir, output_dir)
