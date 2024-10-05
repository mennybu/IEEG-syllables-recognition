import os
import csv

def average_every_two_cells(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]

    averaged_data = []
    for row in data:
        averaged_row = []
        for i in range(0, len(row), 2):
            if i + 1 < len(row):
                avg = (float(row[i]) + float(row[i+1])) / 2
            else:
                avg = float(row[i])  # Handle case where row has an odd number of elements
            averaged_row.append(avg)
        averaged_data.append(averaged_row)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(averaged_data)

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
                output_file_name = f"{file_name}_averaged{file_extension}"
                output_file_path = os.path.join(output_subdir, output_file_name)
               
                # Processing the file
                average_every_two_cells(input_file_path, output_file_path)
                print(f"Processed {input_file_path} -> {output_file_path}")

# Example usage
input_root = '/home/tauproj6/EEG_proj/patient_1_output_3s_test'
output_root = '/home/tauproj6/EEG_proj/patient_1_output_3s_test_avg'
process_files(input_root, output_root)
