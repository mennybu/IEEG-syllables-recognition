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

class EEGDataset(Dataset):
    def __init__(self, data_dir, label_to_idx):
        self.data_dir = data_dir
        self.label_to_idx = label_to_idx
        self.samples = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        eeg_data = self.load_eeg_data(file_path)
        label_idx = self.label_to_idx[label]
        return eeg_data, label_idx

    def load_eeg_data(self, file_path):
        df = pd.read_csv(file_path, header=None)
        eeg_data = torch.tensor(df.values, dtype=torch.float32)
        eeg_data = eeg_data.t()
        return eeg_data

class SeparateConv1DClassifier(nn.Module):
    def __init__(self, conv_out_channels1, conv_out_channels2, conv_out_channels3, kernel_sizes, num_classes):
        super(SeparateConv1DClassifier, self).__init__()
        self.num_classes = num_classes
        self.convs1 = nn.ModuleList([nn.Conv1d(1, conv_out_channels1, kernel_sizes[0],stride=kernel_sizes[0]) for _ in range(75)])
        self.convs2 = nn.ModuleList([nn.Conv1d(1, conv_out_channels2, kernel_sizes[1], stride=kernel_sizes[1]) for _ in range(75)])
        self.convs3 = nn.ModuleList([nn.Conv1d(1, conv_out_channels3, kernel_sizes[2], stride=kernel_sizes[2]) for _ in range(75)])
        self.fc = nn.Linear(7200, self.num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
       

         # x shape: [30, 3000, 126]
        x = x.permute(0, 2, 1)  # New shape: [30, 126, 3000]
        
        # First set of convolutions
        x = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.convs1)]
        x = torch.stack(x, dim=1)
        x = F.softmax(x, dim=1)
        
        # print("Shape after first convolution and stack:", x.shape)
        
        

        # Second set of convolutions
        x = x.view(x.size(0) , x.size(1) * x.size(2), x.size(3))
        #now the tensor is [BS, channels*out_channels1, new_length]
        # print("Shape before second convolution:", x.shape)
        x = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.convs2)]
        x = torch.stack(x, dim=1)
        x = F.softmax(x, dim=1)
        # print("Shape after second convolution and stack:", x.shape)
        
        # Third set of convolutions
        x = x.view(x.size(0) , x.size(1) * x.size(2), x.size(3))
        # print("Shape before third convolution:", x.shape)
        x = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.convs3)]
        x = torch.stack(x, dim=1)
        x = F.softmax(x, dim=1)
        # print("Shape after third convolution and stack:", x.shape)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        #print("Shape after flattening:", x.size())
        
        x = self.fc(x)
        x = self.dropout(x)
        out = F.softmax(x, dim=1)  # Apply softmax along the class dimension
        return out
    

if __name__ == "__main__":
    data_dir = r'/home/tauproj6/final_runs_EEG_proj/patient_1_output_1.5s_pca_v2/annotate'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    model_dir = f'/home/tauproj6/EEG_proj/patient_1_output_1.5s_pca_v2/annotate_model_06.10_seperate_conv1D/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = EEGDataset(data_dir, label_to_idx)

    # model = SeparateConv1DClassifier(
    #     conv_out_channels1=64,
    #     conv_out_channels2=128,
    #     conv_out_channels3=256,
    #     kernel_sizes=[3, 5, 7],
    #     num_classes=5
    # )


    model_settings = {
       'conv_out_channels1': 128,
       'conv_out_channels2': 32,
       'conv_out_channels3': 16,
        'kernel_sizes' : [4, 10, 12],
       'num_classes': 5,
    }

    model = SeparateConv1DClassifier(**model_settings)

    


    batch_size = 30

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(labels)
        return inputs, labels

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #save model parameters

    save_parameters_to_file(model_settings, lr, fr'{model_dir}/model_parameters.txt')

    num_epochs = 4001

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)
    model.to(device)
    
    # Find the most recent checkpoint
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('trained_model_ANNOTATE_epoch_')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_checkpoint = checkpoint_files[-1]
        start_epoch = int(last_checkpoint.split('_')[-1].split('.')[0]) + 1
        model.load_state_dict(torch.load(os.path.join(model_dir, last_checkpoint)))
        print(f"Loaded model from epoch {start_epoch - 1}")
    else:
        start_epoch = 0

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    for epoch in range(start_epoch, num_epochs):
        t_start_epoch = time.time()
        model.train()

        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            t1 = time.time()
            loss = criterion(outputs, labels)

            t2 = time.time()
            print(f"forward {t2-t1}")
            loss.backward()
            t3= time.time()
            print("back", t3-t2)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            print("Correct predictions:", correct_predictions)

        epoch_loss = running_loss / len(dataset)
        epoch_accuracy = correct_predictions / len(dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        t_end_epoch = time.time()
        print("Running time for this epoch is:", t_end_epoch-t_start_epoch, "[s]")

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), fr'{model_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth')
            print(f"Saved model checkpoint at epoch {epoch}")
