import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os

import pandas as pd
import pickle as pkl

import numpy as np
from tqdm import tqdm

import argparse

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x[:,:1]


def main(args):

    input_size = 290  # The input feature size
    hidden_size = 50  # The size of the hidden layer
    lr = args.learning_rate  # The learning rate
    num_epochs = args.num_epochs  # The number of training epochs



    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)


    train_cps = pd.read_csv(f'{args.fp_data_dir}/train_comp_fps.csv').values[:,1:]
    val_cps = pd.read_csv(f'{args.fp_data_dir}/val_comp_fps.csv').values[:,1:]
    mean_cps = train_cps.mean(axis=0)
    std_cps = train_cps.std(axis=0)

    train_cps_norm = (train_cps - mean_cps) / (std_cps + 1e-5)
    val_cps_norm = (val_cps - mean_cps) / (std_cps + 1e-5)

    with open(os.path.join(save_dir, 'scalar.p'), 'wb') as f:
        pkl.dump((mean_cps, std_cps), f)


    with open(f'{args.fp_data_dir}/ml_train.pkl','rb') as f:
        train_pairs, train_label, val_pairs, val_label = pkl.load(f)


    train_data = torch.Tensor(np.abs(train_cps_norm[train_pairs[:,0]] - train_cps_norm[train_pairs[:,1]]))
    train_targets = torch.Tensor(train_label)
    val_data = torch.Tensor(np.abs(val_cps_norm[val_pairs[:,0]] - val_cps_norm[val_pairs[:,1]]))
    val_targets = torch.Tensor(val_label)


    # Create data loaders for the training and testing data
    train_dataset = TensorDataset(train_data, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataset = TensorDataset(val_data, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # Create the model and optimizer
    model = MLP(input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)



    model = model.cuda()

    best_acc = 0.0
    best_epoch = -1
    ckpt_list = []
    # Train the model
    for epoch in range(num_epochs):

        losses = []
        model.train()
        for data, targets in tqdm(train_loader):
            data = data.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = nn.BCELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.array(losses).mean():.4f}")

        # Compute the testing accuracy for this epoch
        if epoch % 5 == 4:
            model.eval()
            pred = []
            for data, targets in tqdm(val_loader):
                data = data.cuda()
                targets = targets.cuda()
                val_outputs = model(data)
                pred.append(val_outputs.detach().cpu().numpy())
            pred = np.concatenate(pred,axis=0)
            val_acc = ((pred > 0.5) == (val_label > 0.5)).astype(np.float32).mean()
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                state = {
                    "model": model.state_dict(),
                    'cur_epoch': epoch,
                    'best_acc': best_acc,
                }
                checkpoint = os.path.join(save_dir,f'checkpoint-{epoch}.ckpt')
                if len(ckpt_list) >= 5:
                    try:
                        os.remove(ckpt_list[0])
                    except:
                        print('Remove checkpoint failed for', ckpt_list[0])
                    ckpt_list = ckpt_list[1:]
                    ckpt_list.append(checkpoint)
                else:
                    ckpt_list.append(checkpoint)
                torch.save(state, checkpoint)

            print(f"Epoch {epoch+1}/{num_epochs}, Val Acc: {val_acc:.4f}, Best Acc: {best_acc:.4f}, Best Epoch: {best_epoch}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_data_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    args = parser.parse_args()