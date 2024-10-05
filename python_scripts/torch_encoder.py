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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import numpy as np
import math
from einops import rearrange, reduce, repeat

t_samples = 10

def save_parameters_to_file(params, lr, filename):
    with open(filename, 'w') as file:
        file.write('Parameters:\n')
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
        file.write('\nAdditional Parameters:\n')
        file.write(f'initial_learning_rate: {lr}\n')
        file.write('Learning rate scheduler: ReduceLROnPlateau\n')
        file.write('Scheduler parameters:\n')
        file.write('  mode: min\n')
        file.write('  factor: 0.1\n')
        file.write('  patience: 10\n')
        file.write('\nModel name: Transformer with mean_pool\n')
        file.write(f'\nt_samples: {t_samples}\n')



def save_metrics_to_csv(epoch, loss, accuracy, file_path):
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Loss', 'Accuracy'])
        writer.writerow([epoch, loss, accuracy])


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
    
        # Check for non-numeric data
        if not df.applymap(np.isreal).all().all():
            print(f"Non-numeric data found in {file_path}")
            print(df.dtypes)
            # Handle non-numeric data (e.g., convert to numeric, or drop)
            df = df.apply(pd.to_numeric, errors='coerce')
    
        # Handle NaN values
        if df.isnull().values.any():
            print(f"NaN values found in {file_path}")
            df = df.fillna(0)  # or use df.dropna()
    
        # Convert to float
        df = df.astype(float)
    
        # Create tensor
        try:
            eeg_data = torch.tensor(df.values, dtype=torch.float32)
        except TypeError as e:
            print(f"Error creating tensor from {file_path}: {e}")
            print(df.dtypes)
            raise
    
        eeg_data = eeg_data.t()
        return eeg_data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
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
        self.lin1 = nn.Linear(d_model, d_model//2)
        self.dropout = nn.Dropout(p=dropout)
        #self.lin2 = nn.Linear(d_model//2, d_model)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.dropout(F.gelu(self.lin1(x)))
        #return self.lin2(x)
        return x

class MHA_block(nn.Module):
    def __init__(self, d_core,seq_length  , heads, d_embed=None, dropout=0):
        super(MHA_block, self).__init__()
        self.dropout = dropout
        self.d_core = d_core
        self.d_embed = d_core if d_embed is None else d_embed
        self.pos_encoder = PositionalEncoding(d_core, dropout, max_len = seq_length)
        #self.QQ = nn.Linear(d_core, self.d_embed)
        #self.KK = nn.Linear(d_core, self.d_embed)
        #self.VV = nn.Linear(d_core, self.d_embed)
        ####shared weights:
        #self.QKV = nn.Linear(d_core, self.d_embed * 3)
        self.MHA = nn.MultiheadAttention(self.d_core, heads, dropout=self.dropout) #, batch_first=True)
        self.ff1 = FeedForward(self.d_embed, dropout=dropout)
        self.act = nn.GELU()
        #####can reduce ff2
        #self.ff2 = FeedForward(self.d_embed, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_core)
        ######can reduce norm2
        #self.norm2 = nn.LayerNorm(d_core)
        self.expand = 1
        

    def forward(self, u, o=None):
        # (B, L, D)


        u = self.pos_encoder(u)

        # Transpose from (B, L, D) to (L, B, D)
        u = u.transpose(0, 1)
        print(u.size())
        if o is None:
            o = u
        #k = self.KK(o)
        #v = self.VV(o)
        #q = self.QQ(u)
        #out, _ = self.MHA(q, k, v)#, need_weights=False)
        #######shared weights approach:
        #qkv = self.QKV(u)
        #q, k, v = qkv.chunk(3, dim=-1)
        #out, _ = self.MHA(q, k, v)

        ##### no pre-weights approach:
        out, _ = self.MHA(u, u, u, need_weights=False)
        #Transpose back from (L, B, D) to (B, L, D)

        out = out.transpose(0, 1)
        u = u.transpose(0, 1)
        out = self.norm1(u + out)
        out = self.ff1(out)
        #out = self.act(out)
        #out = self.ff2(out)
        #out = self.norm2(u + out)
        return out
    

        
    
class TransformerWithFF(nn.Module):
    def __init__(self,d_core, n_classes, seq_length, heads, d_embed=None, dropout=0.1):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoderLayer(d_core, )
        self.dense = nn.Linear(self.d_embed  *  t_samples//2, n_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.MHA(x)
        print(x.size())
        #[bs,seq_l, d_core]

        # Divide the sequence into 10 segments and apply mean pooling to each
        batch_size, seq_len, d_core = x.size()
        x = self.dropout(x)
        segment_size = seq_len // t_samples
        
        if seq_len % t_samples != 0:
        # If not, we'll pad the sequence
            padding_size = t_samples - (seq_len % t_samples)
            x = F.pad(x, (0, 0, 0, padding_size))
            seq_len += padding_size

        segment_size = seq_len // t_samples

        # Reshape to [bs, 10, segment_size, d_core]
        x = x.reshape(batch_size, t_samples, segment_size, d_core)
        
        # Apply mean pooling to each segment
        x = torch.mean(x, dim=2)
        #print("After mean pooling:", x.size())
        # Now x has shape [bs, 10, d_core]

        # Flatten the tensor to shape [batch_size, 10 * d_core]
        x = x.reshape(batch_size, -1)
        #print("After flattening:", x.size())
        x = self.dense(x)
        #print("After fully connected:", x.size())
        out = F.softmax(x, dim=1)
        return out

if __name__ == "__main__":
    data_dir = r'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper/annotate'
    label_to_idx = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

    model_dir = f'/home/tauproj6/EEG_proj/patient_1_1s_preproccesed_steeper/annotate_model_03.08_Transformer_80_embed/'
    csv_file_path = os.path.join(model_dir, 'training_metrics.csv') 

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = EEGDataset(data_dir, label_to_idx)

    model_settings = {
        'd_core' : 242,
        'd_embed' : 80,
        'dropout' : 0.7,
        'heads' : 1,
        'seq_length' : 2000,
        'n_classes' : 5
    }

    model = TransformerWithFF(**model_settings)

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=0.5e-4)

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

            assert torch.isfinite(inputs).all(), "Inputs contain NaN or infinity"

            optimizer.zero_grad()

            outputs = model(inputs)

            t1 = time.time()
            loss = criterion(outputs, labels)

            t2 = time.time()
            print(f"forward {t2-t1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    
        # Update the learning rate
        scheduler.step(epoch_loss)
    
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')

        t_end_epoch = time.time()
        print("Running time for this epoch is:", t_end_epoch-t_start_epoch, "[s]")

        if epoch % 1 == 0 and epoch != 0:
            torch.save(model.state_dict(), fr'{model_dir}/trained_model_ANNOTATE_epoch_{epoch}.pth')
            print(f"Saved model checkpoint at epoch {epoch}")
            save_metrics_to_csv(epoch + 1, epoch_loss, epoch_accuracy, csv_file_path)
