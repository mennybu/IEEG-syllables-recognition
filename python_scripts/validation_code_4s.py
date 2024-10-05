import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import time
import main_code



if __name__ == "__main__":
    # Define data directory
    val_data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_4s/annotate_validation'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    # Create validation dataset instance
    val_dataset = main_code.EEGDataset(val_data_dir, label_to_idx)

    # Define model settings
    model_settings = {
        'rnn_dim': 120,
        'KS': 6,
        'num_layers': 2,
        'dropout': 0.5,
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

    # Load the pre-trained models
    for epoch in range(10,910,10):
        model.load_state_dict(torch.load(rf'/home/tauproj6/EEG_proj/patient_1_output_4s/annotate_model_2layer/trained_model_ANNOTATE_epoch_{epoch}_1layer.pth'))
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

        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}\n for epoch model num {epoch}')
