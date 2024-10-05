import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import time
import main_code_post_cnn_15s_for_filtered
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # Define data directory
    val_data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_1.5s_filtered/annotate_validation'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    # Create validation dataset instance
    val_dataset = main_code_post_cnn_15s_for_filtered.EEGDataset(val_data_dir, label_to_idx)

    # Define model settings
    model_settings = {
        'rnn_dim': 40,
        'KS': 6,
        'num_layers': 3,
        'dropout': 0.7,
        'n_classes': 5,
        'bidirectional': False, 
        'in_channels': 122,
        'keeptime': True,
        'token_input': False
    }

    # Initialize the model
    model = main_code_post_cnn_15s_for_filtered.CnnRnnClassifier(**model_settings)

    # Define batch size
    batch_size = 300


    # Create validation data loader
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(labels)
        return inputs, labels


    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Move the model to the appropriate device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    models_dir = r"/home/tauproj6/EEG_proj/patient_1_output_1.5s_filtered/annotate_model_26.07_filtered_40cnn/"

    #create a list for accuracy and validation loss
    val_data = []


    # Load the pre-trained models
    for epoch in range(10, 4010, 10):
        model_path = rf'{models_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth'
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)

            # Validation phase
            val_running_loss = 0.0
            val_correct_predictions = 0

            # Initialize dictionaries to store the count of predictions and correct predictions per syllable
            predictions_per_syllable = {key: 0 for key in label_to_idx.keys()}
            correct_predictions_per_syllable = {key: 0 for key in label_to_idx.keys()}


            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct_predictions += (predicted == labels).sum().item()

                     # Count the predictions and correct predictions for each syllable
                    for label, prediction in zip(labels, predicted):
                        label_syllable = idx_to_label[label.item()]
                        prediction_syllable = idx_to_label[prediction.item()]

                        predictions_per_syllable[prediction_syllable] += 1
                        if label == prediction:
                            correct_predictions_per_syllable[label_syllable] += 1

            val_epoch_loss = val_running_loss / len(val_dataset)
            val_epoch_accuracy = val_correct_predictions / len(val_dataset)

            print(f'\nValidation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f} for epoch model num {epoch}')
            print("Predictions per syllable:", predictions_per_syllable)
            print("Correct predictions per syllable:", correct_predictions_per_syllable)

            val_data.append((epoch, val_epoch_loss, val_epoch_accuracy))

    #create a csv
    data = pd.DataFrame(val_data)
    data.to_csv(fr'{models_dir}/csv_of_validation.csv', index=False)
    
    data = np.array(data).T
    # Unpack the data
    epochs, val_loss, val_accuracy = data[0], data[1], data[2]

    # Calculate averages
    avg_loss = np.mean(val_loss)
    avg_accuracy = np.mean(val_accuracy)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot validation loss
    ax1.plot(epochs, val_loss, label='Validation Loss', color='red')
    ax1.axhline(y=avg_loss, color='darkred', linestyle='--', label=f'Average Loss: {avg_loss:.4f}')
    ax1.set_title('Validation Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(epochs, val_accuracy, label='Validation Accuracy', color='blue')
    ax2.axhline(y=avg_accuracy, color='darkblue', linestyle='--', label=f'Average Accuracy: {avg_accuracy:.4f}')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

