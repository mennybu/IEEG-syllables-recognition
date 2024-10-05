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

t_samples = 10

def save_parameters_to_file(params,  tw, filename):
    with open(filename, 'w') as file:
        file.write('Parameters:\n')
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
        file.write('\nAdditional Parameters:\n')
        file.write(f'learning_rate: {lr}\n')
        file.write('\nModel name : CnnRnnmodel with post cnn\n')
        file.write(f'\n t_samples : {t_samples}')
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

class CnnRnnClassifier(torch.nn.Module):
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_classes, bidirectional, in_channels, keeptime=False, token_input=None):
        super().__init__()

        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                            out_channels=rnn_dim,
                                            kernel_size=KS,
                                            stride=KS)

        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.keeptime = keeptime

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else:
            mult = 1
        self.mult = mult

        if keeptime:
            self.postprocessing_conv = nn.ConvTranspose1d(in_channels=rnn_dim * mult,
                                                          out_channels=rnn_dim * mult,
                                                          kernel_size=KS,
                                                          stride=KS)

        global t_samples
        self.dense = nn.Linear(rnn_dim * mult * t_samples, n_classes)

    def forward(self, x):
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        output, x = self.BiGRU(x)
        if not self.keeptime:
            x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
            x = x[-1]
            x = x.contiguous().permute(1, 0, 2)
        else:
            x = output.contiguous().permute(1, 2, 0)  # bs,d*c,t 
            x = self.postprocessing_conv(x)
            x = x.permute(0, 2, 1)  # bs, t, d*c

            # Generate 10 equally spaced indices
            indices = torch.linspace(0, len(x[0]) - 1, t_samples).long()
            # Select the elements at these indices
            x = x[:, indices, :]
            # Flatten the tensor to shape [batch_size, t_samples * d * c]
            x = x.contiguous().view(x.shape[0], -1)  # bs, t_samples * d * c

        x = self.dropout(x)
        x = self.dense(x)
        out = F.softmax(x, dim=1)
        return out

if __name__ == "__main__":
    data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_1.5s_pca_filtered/annotate'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    model_dir = f'/home/tauproj6/EEG_proj/patient_1_output_1.5s_pca_filtered/annotate_model_28.07_CnnRnn_post_cnn_35cnn/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = EEGDataset(data_dir, label_to_idx)

    model_settings = {
        'rnn_dim': 35,
        'KS': 6,
        'num_layers': 3,
        'dropout': 0.7,
        'n_classes': 5,
        'bidirectional': False,
        'in_channels': 74,
        'keeptime': True,
        'token_input': False
    }

    model = CnnRnnClassifier(**model_settings)

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
