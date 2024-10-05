import os
import pandas as pd
import random

# Base directory
#path to file where 'annotate','other', and 'imagery' files are:
#base_dir = "C:\output - Copy"#r"C:\EEG_project_2.0"
base_dir = r"/Users/mennymac/Desktop/University/EE final project /patient_1_output"  #r"C:\EEG_project_2.0"
# Source directories
source_dirs = ["ANNOTATE", "IMAGERY", "OTHER"]

# List of vowels
vowels = ['a', 'e', 'i', 'o', 'u']


# Function to get all CSV files from a given directory
def get_csv_files(dir_path):
    csv_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


# Create a dataframe to keep track of batches and their use in training/testing
columns = ["group", "vowel_a", "vowel_e", "vowel_i", "vowel_o", "vowel_u", "used", "epoch"]
data = []

for source_dir in source_dirs:
    print(f"Processing group: {source_dir}")
    # Get files for each vowel in the current source directory
    group_files = {vowel: get_csv_files(os.path.join(base_dir, source_dir, vowel)) for vowel in vowels}

    # Debugging: print out the number of files found for each vowel
    for vowel, files in group_files.items():
        print(f"  Vowel '{vowel}' has {len(files)} files")

    # Continue as long as there is at least one file for each vowel
    while all(len(files) > 0 for files in group_files.values()):
        batch = [source_dir]
        for vowel in vowels:
            selected_file = random.choice(group_files[vowel])
            group_files[vowel].remove(selected_file)
            batch.append(selected_file)
        batch.append("unused")  # Indicator for whether this batch has been used
        data.append(batch)
        # Debugging: print out the batch created
        print(f"  Created batch: {batch}")

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(data, columns=columns)
df.to_csv(rf"{base_dir}/file_usage_batches.csv", index=False)
print("File usage batches CSV created.")
