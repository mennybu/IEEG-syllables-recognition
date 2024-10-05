import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import time
import main_code
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # Define data directory
    val_data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_3s_v2/annotate'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    # Create validation dataset instance
    val_dataset = main_code.EEGDataset(val_data_dir, label_to_idx)

    # Define model settings
    model_settings = {
        'rnn_dim': 60,
        'KS': 6,
        'num_layers': 3,
        'dropout': 0.7,
        'n_classes': 5,
        'bidirectional': True,
        'in_channels': 126,
        'keeptime': False,
        'token_input': False
    }

    # Initialize the model
    model = main_code.CnnRnnClassifier(**model_settings)

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
    models_dir = r"/home/tauproj6/EEG_proj/patient_1_output_3s_v2/annotate_model_3layer_0.7do_01_07_lr0.0001_60cnn/"

    #create a list for accuracy and validation loss
    val_data = []


    # Load the pre-trained models
    for epoch in range(10,4010,50):
        model_path = rf'{models_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth'
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
            model.eval()


            model.to(device)

            # Validation phase
            val_running_loss = 0.0
            val_correct_predictions = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct_predictions += (predicted == labels).sum().item()

            val_epoch_loss = val_running_loss / len(val_dataset)
            val_epoch_accuracy = val_correct_predictions / len(val_dataset)

            print(f'Training Loss: {val_epoch_loss:.4f}, Training Accuracy: {val_epoch_accuracy:.4f}\n for epoch model num {epoch}')

            val_data.append((epoch, val_epoch_loss, val_epoch_accuracy))
    

    #create a csv
    data = pd.DataFrame(val_data)
    data.to_csv(fr'{models_dir}/csv_of_training.csv', index=False)
    

    # Unpack the data
    epochs, val_loss, val_accuracy = np.array(data[0]), np.array(data[1]), np.array(data[2])

    # Calculate averages
    avg_loss = np.mean(val_loss)
    avg_accuracy = np.mean(val_accuracy)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot validation loss
    ax1.plot(epochs, val_loss, label='Training Loss', color='red')
    ax1.axhline(y=avg_loss, color='darkred', linestyle='--', label=f'Average Loss: {avg_loss:.4f}')
    ax1.set_title('Training Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(epochs, val_accuracy, label='Training Accuracy', color='blue')
    ax2.axhline(y=avg_accuracy, color='darkblue', linestyle='--', label=f'Average Accuracy: {avg_accuracy:.4f}')
    ax2.set_title('Training Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
