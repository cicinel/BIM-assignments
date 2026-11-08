#!/usr/bin/env python3


import argparse
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from train_utils import load_data

# for data curation
REST_TOLERANCE = 0.01

# for data curation - I am filtering out some data rows that I determined are bad
MAX_JOINT_1 = 1.6


def make_features(states):
  
    states = np.asarray(states, dtype=np.float64)
    x, y, z, j1, j3 = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]

    distance = np.sqrt(x ** 2 + y ** 2)   
    azimuth = np.arctan2(y, x)           
    elevation = np.arctan2(z, distance)   

    return np.stack([x, y, z, j1, j3, azimuth, elevation, distance], axis=1)


def compute_normalization_parameters(data):
  
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-8] = 1.0   
    return mean.reshape(1, -1), std.reshape(1, -1)


def normalize(data, mean, std):
   
    return (data - mean) / std


class PolicyModel(nn.Module):
  

    def __init__(self, num_inputs=8):
        super(PolicyModel, self).__init__()
        self.hidden1 = nn.Linear(num_inputs, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 2)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        return self.output(x)


def load_at_rest_data(data_files):
    
    all_states = []
    all_actions = []
    for data_file in data_files:
        states, actions = load_data(data_file)
        at_rest = np.zeros(len(states), dtype=bool)
        diff = np.abs(states[1:, 3:5] - actions[:-1, :])
        at_rest[1:] = np.all(diff < REST_TOLERANCE, axis=1)

        all_states.append(states[at_rest])
        all_actions.append(actions[at_rest])

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

  
    finite = np.all(np.isfinite(states), axis=1) & np.all(np.isfinite(actions), axis=1)
    states = states[finite]
    actions = actions[finite]


    in_range = (np.abs(states[:, 3]) < MAX_JOINT_1) & (np.abs(actions[:, 0]) < MAX_JOINT_1)
    return states[in_range], actions[in_range]


def train_model(model, x_train, y_train, x_val, y_val, mean, std, output_dir,
                epochs=200, learning_rate=1e-3, batch_size=64):


    x_train = torch.as_tensor(normalize(x_train, mean, std), dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_val = torch.as_tensor(normalize(x_val, mean, std), dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val),
                                             batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion = nn.SmoothL1Loss(beta=0.01)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)
    writer.add_graph(model, x_train[:1].to(device))

    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'model.pth')

    for epoch in range(epochs):
  
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

  
        model.eval()
        val_loss = 0.0
        val_mae_j1 = 0.0
        val_mae_j3 = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_mae_j1 += torch.mean(torch.abs(outputs[:, 0] - targets[:, 0])).item()
                val_mae_j3 += torch.mean(torch.abs(outputs[:, 1] - targets[:, 1])).item()


        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_mae_j1 = np.rad2deg(val_mae_j1 / len(val_loader))
        val_mae_j3 = np.rad2deg(val_mae_j3 / len(val_loader))

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE_deg/Val_j1', val_mae_j1, epoch)
        writer.add_scalar('MAE_deg/Val_j3', val_mae_j3, epoch)

  
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

        if epoch % 10 == 0:
            print("Epoch {}/{}, Train Loss: {:.6f}, Val Loss: {:.6f}, Val MAE j1: {:.4f} deg, "
                  "Val MAE j3: {:.4f} deg".format(epoch, epochs, train_loss, val_loss, val_mae_j1, val_mae_j3))

    writer.close()


    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model


def evaluate_model(model, x_val, y_val, mean, std):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = torch.as_tensor(normalize(x_val, mean, std), dtype=torch.float32).to(device)
        predicted = model(inputs).cpu().numpy()

    errors = np.abs(predicted - y_val)  

    print("\nEVALUATION ON VALIDATION SET:")
    print("MAE joint_1: {:.4f} deg".format(np.rad2deg(np.mean(errors[:, 0]))))
    print("MAE joint_3: {:.4f} deg".format(np.rad2deg(np.mean(errors[:, 1]))))
    print("Max error: {:.4f} deg".format(np.rad2deg(np.max(errors))))
    print("avg(|d_j1| + |d_j3|): {:.6f} rad".format(np.mean(errors[:, 0] + errors[:, 1])))


def main():
    parser = argparse.ArgumentParser(description='Learn an imitation policy from expert data')
    parser.add_argument('data', nargs='+', help='path(s) to the training data file(s)')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--output_dir', default='.', help='directory for model.pth and norm_params.json')
    args = parser.parse_args()


    np.random.seed(0)
    torch.manual_seed(0)

    states, actions = load_at_rest_data(args.data)
    features = make_features(states)

 
    indices = np.random.permutation(len(features))
    num_train = int(0.8 * len(features))
    x_train, y_train = features[indices[:num_train]], actions[indices[:num_train]]
    x_val, y_val = features[indices[num_train:]], actions[indices[num_train:]]


    mean, std = compute_normalization_parameters(x_train)
    os.makedirs(args.output_dir, exist_ok=True)
    norm_params_path = os.path.join(args.output_dir, 'norm_params.json')
    with open(norm_params_path, 'w') as fid:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, fid)

    model = PolicyModel(num_inputs=x_train.shape[1])

    model = train_model(model, x_train, y_train, x_val, y_val, mean, std, args.output_dir,
                        epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)

    evaluate_model(model, x_val, y_val, mean, std)


if __name__ == '__main__':
    main()
