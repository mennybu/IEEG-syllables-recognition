import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import time

def save_parameters_to_file(params,  tw, filename):
    with open(filename, 'w') as file:
        file.write('Parameters:\n')
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
        file.write('\nAdditional Parameters:\n')
        file.write(f'learning_rate: {lr}\n')
        #file.write(f'time_window: {tw}\n')
        file.write(f'Model description: 1Dcnn for each chanel seperately, with keep_time == true \n')


class EEGDataset(Dataset):
    def __init__(self, data_dir, label_to_idx):
        self.data_dir = data_dir
        self.label_to_idx = label_to_idx
        self.samples = []

        # # Iterate over all files in the data directory
        # for root, _, files in os.walk(data_dir):
        #     for file in files:
        #         if file.endswith('.csv'):
        #             file_path = os.path.join(root, file)
        #             label = os.path.basename(root)
        #             self.samples.append((file_path, label))
        # Iterate over all files in the data directory
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label = os.path.basename(root)  # Extract label from second last directory
                   # label = os.path.basename(os.path.dirname(os.path.dirname(root)))  # Extract label from third last directory
                   #  print(label)
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        eeg_data = self.load_eeg_data(file_path)
        label_idx = self.label_to_idx[label]  # Convert label to index (e.g., 'a' -> 0, 'e' -> 1, ...)
        return eeg_data, label_idx

    def load_eeg_data(self, file_path):
        # Load EEG data from CSV file and convert it to a tensor
        df = pd.read_csv(file_path, header=None)  # Specify header=None to include all rows
        eeg_data = torch.tensor(df.values, dtype=torch.float32)
        # Transpose the tensor to have the format (num_samples, num_features)
        eeg_data = eeg_data.t()
        return eeg_data
class RnnCnnClassifier(torch.nn.Module):
    def __init__(self, rnn_dim, cnn_out_channels, KS, num_layers, dropout, n_classes, bidirectional, in_channels, dense_samples, keeptime=True, token_input=None ):
        super().__init__()

        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.keeptime = keeptime
        self.in_channels = in_channels
        self.dense_samples = dense_samples
        # RNN layer (GRU)
        self.BiGRU = nn.GRU(input_size=in_channels,
                            hidden_size=rnn_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        # Determine the multiplier for bidirectional RNN
        self.mult = 2 if bidirectional else 1

        # Linear layer to match RNN output dimension with input dimension
        #self.match_dim = nn.Linear(rnn_dim * self.mult, in_channels)

        # CNN layer
        self.cnn = nn.Conv1d(in_channels=in_channels,
                             out_channels=cnn_out_channels,
                             kernel_size=KS,
                             stride=6)

        self.dropout = nn.Dropout(dropout)

        dense_samples = 100

        # Adjust the input size of the dense layer based on the CNN output
        self.dense = nn.Linear(cnn_out_channels * self.dense_samples, n_classes)

    def forward(self, x):
        # x shape: [batch_size, time_steps, in_channels]
        original_x = x
       
        # Permute for RNN input: [time_steps, batch_size, in_channels]
        x = x.permute(1, 0, 2)
       
        # Apply RNN
        rnn_out, _ = self.BiGRU(x)
       
        # Match RNN output dimension with input dimension
        #rnn_out = self.match_dim(rnn_out)
       
        # Multiply RNN output with original input (element-wise)
        x = rnn_out * original_x.permute(1, 0, 2)
       
        # Permute for CNN input: [batch_size, channels, time_steps]
        x = x.permute(1, 2, 0)
       
        print(x.size())
        # Apply CNN
        x = F.softmax(self.cnn(x), dim=2)
       
        # Adaptive pooling to fix the output size
        x = F.adaptive_avg_pool1d(x, self.dense_samples)
       
        # Flatten
        x = x.view(x.size(0), -1)
       
        # Apply dropout
        x = self.dropout(x)
       
        # Fully connected layer
        x = self.dense(x)
       
        # Apply softmax
        out = F.softmax(x, dim=1)
       
        return out


if __name__ == "__main__":

    # Define data directory
    data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_1.5s/annotate'

    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    # Define the directory where you want to save the model
    model_dir = f'/home/tauproj6/EEG_proj/patient_1_output_1.5s/annotate_model_rnn_than_cnn/'

    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create dataset instance
    dataset = EEGDataset(data_dir, label_to_idx)

    # Define model settings
    model_settings = {
        'rnn_dim': 126,
        'cnn_out_channels': 80,
        'KS': 6,
        'num_layers': 2,
        'dropout': 0.7,
        'dense_samples': 100,
        'n_classes': 5,
        'bidirectional': False,
        'in_channels': 126,
        'keeptime': True,
        'token_input': False
    }

    # Initialize the model
    model = RnnCnnClassifier(**model_settings)

    # Define batch size
    batch_size = 30

    # Create data loader
    def collate_fn(batch):
        # Extract inputs and labels from the batch
        inputs, labels = zip(*batch)
        # Pad sequences
        global batch_size
        # for i in range(batch_size):
        #     print(inputs[1].shape)
        inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
        # Convert labels to Tensor
        labels = torch.tensor(labels)
        return inputs, labels

    # Create data loader with custom collate function
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    # Define the loss function (e.g., CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (e.g., Adam optimizer)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    #save model parameters

    save_parameters_to_file(model_settings, lr, fr'{model_dir}/model_parameters.txt')

    # Define the number of epochs
    num_epochs = 2001

    # Move the model to the appropriate device (e.g., GPU if available)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    
    #check number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    # Training loop
    for epoch in range(num_epochs):
        t_start_epoch = time.time()
        # Set the model to training mode
        model.train()

        # Initialize variables to keep track of the loss and number of correct predictions
        running_loss = 0.0
        correct_predictions = 0

        # Iterate over the data loader
        for inputs, labels in data_loader:
            # Move the inputs and labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            t1 = time.time()
            # Compute the loss
            loss = criterion(outputs, labels)

            t2 = time.time()
            print(f"forward {t2-t1}")
            # Backward pass
            loss.backward()
            t3= time.time()
            print("back", t3-t2)
            # Update the parameters
            optimizer.step()

            # Update the running loss
            running_loss += loss.item() * inputs.size(0)

            # Compute the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            print("Correct predictions:", correct_predictions)
            


        # Compute the average loss and accuracy for the epoch
        epoch_loss = running_loss / len(dataset)
        epoch_accuracy = correct_predictions / len(dataset)

        # Print the average loss and accuracy for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        t_end_epoch = time.time()
        print("Running time for this epoch is:", t_end_epoch-t_start_epoch, "[s]")
        if epoch % 10 == 0 and epoch != 0:
            # Save the trained model
            torch.save(model.state_dict(), fr'{model_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth')
        
