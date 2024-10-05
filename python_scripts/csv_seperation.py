import csv
import numpy as np
import os
import pandas as pd
def split_to_five_columns(input_file, output_file_first, output_file_second, output_file_third, output_file_fourth, output_file_fifth):
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    # Convert string values to float, replacing commas with dots
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i].replace(',', '.'))
            except ValueError:
                pass  # Keep non-numeric values as they are
    
    df = pd.DataFrame(data)
    
    df_part1 = df.iloc[:, 0::5]
    #df_part2 = df.iloc[:, 1::5]
    #df_part3 = df.iloc[:, 2::5]
    #df_part4 = df.iloc[:, 3::5]
    #df_part5 = df.iloc[:, 4::5]
    
    # Write the resulting dataframes to new CSV files
    df_part1.to_csv(output_file_first, index=False, float_format='%.6f', header=False)
    #df_part2.to_csv(output_file_second, index=False, float_format='%.6f', header=False)
    #df_part3.to_csv(output_file_third, index=False, float_format='%.6f', header=False)
    #df_part4.to_csv(output_file_fourth, index=False, float_format='%.6f', header=False)
    #df_part5.to_csv(output_file_fifth, index=False, float_format='%.6f', header=False)

def process_files(input_root, output_root):
    for subdir, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, input_root)
                output_subdir = os.path.join(output_root, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Creating output file names
                file_name, file_extension = os.path.splitext(file)
                output_file_first = os.path.join(output_subdir, f"{file_name}_first{file_extension}")
                output_file_second = os.path.join(output_subdir, f"{file_name}_second{file_extension}")
                output_file_third = os.path.join(output_subdir, f"{file_name}_third{file_extension}")
                output_file_fourth = os.path.join(output_subdir, f"{file_name}_fourth{file_extension}")
                output_file_fifth = os.path.join(output_subdir, f"{file_name}_fifth{file_extension}")                
                # Processing the file
                try:
                    split_to_five_columns(input_file_path, output_file_first, output_file_second, output_file_third, output_file_fourth, output_file_fifth)
                    print(f"Processed {input_file_path} -> {output_file_first}, {output_file_second}, {output_file_third}, {output_file_fourth}, {output_file_fifth}")
                except Exception as e:
                    print(f"Error processing {input_file_path}: {str(e)}")

# Example usage
input_csv_path = r'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_copy/'
output_root = r'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_downsampled/'
process_files(input_csv_path, output_root)
