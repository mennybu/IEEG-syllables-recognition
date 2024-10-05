import os
import shutil
import random


def create_directory_structure(base_dir, subdirs):
    """Create base directory with specified subdirectories and their respective subfolders."""
    for subdir in subdirs:
        for label in ['a', 'e', 'i', 'o', 'u']:
            os.makedirs(os.path.join(base_dir, f"{subdir}_validation", label), exist_ok=True)


def move_files_to_validation(original_base_dir, validation_base_dir, subdirs, validation_split=0.1):
    """Move a percentage of files from the original directories to the validation directories."""
    for subdir in subdirs:
        for label in ['a', 'e', 'i', 'o', 'u']:
            original_dir = os.path.join(original_base_dir, subdir, label)
            validation_dir = os.path.join(validation_base_dir, f"{subdir}_validation", label)

            # Get list of files in the original directory
            files = os.listdir(original_dir)
            files = [file for file in files if file.endswith('.csv')]

            # Calculate the number of files to move
            num_files_to_move = max(1, int(round(len(files) * validation_split)))

            # Randomly select files to move
            files_to_move = random.sample(files, num_files_to_move)

            # Move the files
            for file in files_to_move:
                shutil.move(os.path.join(original_dir, file), os.path.join(validation_dir, file))


if __name__ == "__main__":
    # Define the original and validation base directories
    original_base_dir = r'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca_sep/'
    validation_base_dir = r'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper_pca_sep/'
    # Define the subdirectories
    subdirs = ['other', 'annotate', 'imagery']

    # Create the validation directory structure
    create_directory_structure(validation_base_dir, subdirs)

    # Move 10% of the files to the validation directories
    move_files_to_validation(original_base_dir, validation_base_dir, subdirs, validation_split=0.1)

    print("Files moved to validation directories successfully.")
