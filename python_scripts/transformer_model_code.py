
##this is the all transformer:
# MHA_block
# this is the attention block ( the core of the transformer):
# self.MHA = nn.MultiheadAttention(self.d_core, heads, dropout=self.dropout, batch_first=True)
# this is like the most basic transformer, see if it runs like this, if not tell me
# and the input should be (batch, length, dimension) if i remember correctly

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
from einops import rearrange, reduce, repeat
import math


def save_parameters_to_file(params,  tw, filename):
    with open(filename, 'w') as file:
        file.write('Parameters:\n')
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
        file.write('\nAdditional Parameters:\n')
        file.write('\nTransformer model\n')
        # file.write(f'learning_rate: {lr}\n')
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




class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = rearrange(x, "b l d -> l b d")
        x = x + self.pe[:x.size(0)]
        return rearrange(self.dropout(x), "l b d -> b l d")

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.dropout(F.gelu(self.lin1(x)))
        return self.lin2(x)

class MHA_block(nn.Module):
    def __init__(self, d_core,seq_length = 2000 , d_embed=None, dropout=0, heads=4):
        super(MHA_block, self).__init__()
        self.dropout = dropout
        self.d_core = d_core
        self.d_embed = d_core if d_embed is None else d_embed
        self.pos_encoder = PositionalEncoding(d_core, dropout, seq_length)
        self.QQ = nn.Linear(d_core, self.d_embed)
        self.KK = nn.Linear(d_core, self.d_embed)
        self.VV = nn.Linear(d_core, self.d_embed)
        self.MHA = nn.MultiheadAttention(self.d_core, heads, dropout=self.dropout) #, batch_first=True)
        self.ff1 = FeedForward(self.d_embed, dropout=dropout)
        self.act = nn.GELU()
        self.ff2 = FeedForward(self.d_embed, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_core)
        self.norm2 = nn.LayerNorm(d_core)
        self.expand = 1
        return

    def forward(self, u, o=None):
        # (B, L, D)
        u = self.pos_encoder(u)
        if o is None:
            o = u
        k = self.KK(o)
        v = self.VV(o)
        q = self.QQ(u)
        out, _ = self.MHA(q, k, v, need_weights=False)
        out = self.norm1(u + out)
        out = self.ff1(out)
        out = self.act(out)
        out = self.ff2(out)
        out = self.norm2(u + out)
        return out
    

class SepChannels1DConv(nn.Module):
    def __init__(self, conv_in_channels,conv_out_channels, kernel_size, cnn_dropout=0.1):
        super().__init__()
        self.conv1D = nn.ModuleList([nn.Conv1d(1, conv_out_channels, kernel_size,stride=kernel_size) for _ in range(conv_in_channels)])
        self.dropout = nn.Dropout(cnn_dropout)
        
    def forward(self, x):
        

        # x shape: [30, 3000, 126]
        x = x.permute(0, 2, 1)  # New shape: [30, 126, 3000]
        # First set of convolutions
       # x = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.conv1D)]
        x = torch.stack(x, dim=1)
        x = F.softmax(x, dim=1)
        x = x.view(x.size(0) , x.size(1) * x.size(2), x.size(3))
        return x
        
    
class TransformerWithSepChannels1DConv(nn.Module):
    def __init__(self,d_core, n_classes, seq_length, kernel_size, d_embed=None, dropout=0, heads=4, conv_in_channels=126, conv_out_channels=5):
        super().__init__()
        self.d_embed = d_core if d_embed is None else d_embed
        self.MHA = MHA_block( d_core, seq_length=seq_length, d_embed=d_embed ,dropout=dropout, heads=heads)
        self.conv1D =  SepChannels1DConv(kernel_size=kernel_size, conv_in_channels=conv_in_channels, conv_out_channels=conv_out_channels)
        l_out = int(seq_length / kernel_size)
        self.fully_connected = nn.Linear(self.d_embed * conv_out_channels *  l_out, n_classes)

    def forward(self, x):
        x = self.MHA(x)
        x = self.conv1D(x)
        x =  x.view(x.size(0) , x.size(1) * x.size(2))
        out = self.fully_connected(x)        
        return out
        


if __name__ == "__main__":
    data_dir = r'/home/tauproj6/EEG_proj/patient_1_output_1.5s/annotate'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    model_dir = f'/home/tauproj6/EEG_proj/patient_1_output_1.5s/annotate_model_25.07_transformer_with_sep_conv1D/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = EEGDataset(data_dir, label_to_idx)

    # pe_settings = {
    #     'd_model': 126,
    #     'dropout': 0.2,
    #     'max_len': 3000
    # }


    model_settings = {
        'd_core' : 126,
        'd_embed' : None,
        'dropout' : 0.6,
        'heads' : 1,
        'seq_length' : 3000,
        'kernel_size' : 50,
        'conv_in_channels': 126,
        'conv_out_channels' : 5,
        'n_classes' : 5
    }
    


    # positional_encoding = PositionalEncoding(**pe_settings)
    model = TransformerWithSepChannels1DConv(**model_settings)
    


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

        if epoch % 1 == 0 and epoch != 0:
            torch.save(model.state_dict(), fr'{model_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth')
            print(f"Saved model checkpoint at epoch {epoch}")
