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
class CnnRnnClassifier(torch.nn.Module):

    def __init__(self, rnn_dim, KS, num_layers, dropout, n_classes, bidirectional, in_channels, keeptime=False,
                 token_input=None):
        super().__init__()

        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                            out_channels=rnn_dim,
                                            kernel_size=KS,
                                            stride=KS)

        ###########

        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        ###########

        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.keeptime = keeptime

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else:
            mult = 1
        self.mult = mult

        ###########

        if keeptime:
            self.postprocessing_conv = nn.ConvTranspose1d(in_channels=rnn_dim * mult,
                                                          out_channels=rnn_dim * mult,
                                                          kernel_size=KS,
                                                          stride=KS)

        ###########

        self.dense = nn.Linear(rnn_dim * mult, n_classes)

        ###########

    def forward(self, x):
        # x comes in bs = batch size (23?), t = length of vectors, c = number of channals (126 electrodes in our case)
        x = x.contiguous().permute(0, 2, 1)  # rearrange input
        # now bs, c, t
        x = self.preprocessing_conv(x)  # applying conv1D
        #         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        t1 = time.time()
        output, x = self.BiGRU(x)  # output: t,bs,d*c - containing the output features from the last layer of the GRU, for each t
        t2= time.time()
        # x: d*nl,bs,c - vector of shape d*num_layers x bs(=1 in our case) x c (number of output channals)
        if not self.keeptime:
            x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)  # reshaping
            # -1 is a placeholder that tells PyTorch to infer the size of this dimension based on the other dimensions and the
            # total number of elements in the tensor.
            # from git: (2, bs, rnn_dim)
            # or i think Shape: (num_layers = nl, mult = d, bs, rnn_dim)
            x = x[-1]  # Only care about the output at the final layer. This line selects only the last element along the first
            # dimension of the tensor x. Since the tensor x has been reshaped in the previous line such that the
            # first dimension represents the layers of the GRU, selecting x[-1] effectively selects the
            # output from the last layer of the GRU.
            # from git: (2, bs, rnn_dim)
            # or i think Shape: (mult = d, bs, rnn_dim)
            x = x.contiguous().permute(1, 0, 2)  # swapping first 2 dimensions - x: (bs, mult = d, rnn_dim)
            x = x.contiguous().view(x.shape[0], -1)  # now x:(bs, mult * rnn_dim)
        else:
            x = output.contiguous().permute(1, 2, 0)  # bs,d*c,t
            x = self.postprocessing_conv(x)
            x = x.permute(0, 2, 1)  # bs,t,d*c
        # following the keeptime = false option for our case, we now have:
        # x: (bs=1, mult * rnn_dim)
        x = self.dropout(x)
        out = self.dense(x)
        # Add a softmax layer at the end
        # out = F.softmax(out, dim=1)
        # now for our case x:(1,n_classes)
        return out


if __name__ == "__main__":

    # Define data directory
    data_dir = r'/home/tauproj6/EEG_proj/patient_1_output/annotate'

    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    # Define the directory where you want to save the model
    model_dir = f'/home/tauproj6/EEG_proj/patient_1_output/annotate_model_more_dropout/'

    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create dataset instance
    dataset = EEGDataset(data_dir, label_to_idx)

    # Define model settings
    model_settings = {
        'rnn_dim': 65,
        'KS': 6,
        'num_layers': 1,
        'dropout': 0.50,
        'n_classes': 5,
        'bidirectional': True,
        'in_channels': 126,
        'keeptime': False,
        'token_input': False
    }

    # Initialize the model
    model = CnnRnnClassifier(**model_settings)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the number of epochs
    num_epochs = 301

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
            torch.save(model.state_dict(), fr'{model_dir}/trained_model_ANNOTATE_epoch_{epoch}_1layer.pth')
        
