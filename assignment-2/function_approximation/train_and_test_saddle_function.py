#!/usr/bin/env python3
# Script to train and test a neural network with PyTorch

import os
import sys
import argparse
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import saddle_function_utils as sfu


def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev


def normalize_data_per_row(data, mean, stdev):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :param mean: mean for normalization
    :param stdev: standard deviation for normalization
    :return: whitened data, (data - mean) / stdev
    """

    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    centered = data - np.tile(mean, (data.shape[0], 1))
    normalized_data = np.divide(centered, np.tile(stdev, (data.shape[0],1)))

    return normalized_data


class LinearModel(nn.Module):
    """
    Linear neural network model with PyTorch
    """
    def __init__(self, num_inputs):
        super(LinearModel, self).__init__()
        self.hidden1 = nn.Linear(num_inputs, 64, bias=True)
        self.hidden2 = nn.Linear(64, 128, bias=True)
        self.output = nn.Linear(128, 1, bias=True)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x


def build_linear_model(num_inputs):
    """
    Build NN model with PyTorch
    :param num_inputs: number of input features for the model
    :return: PyTorch model
    """
    return LinearModel(num_inputs)


class NonLinearModel(nn.Module):

    def __init__(self, num_inputs):
        super(NonLinearModel, self).__init__()
        self.hidden1 = nn.Linear(num_inputs, 64, bias=True)
        self.hidden2 = nn.Linear(64, 128, bias=True)
        self.output = nn.Linear(128, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x


def build_nonlinear_model(num_inputs):
  
    return NonLinearModel(num_inputs)



def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: PyTorch model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # normalize
    norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # Convert to PyTorch tensors
    train_input_tensor = torch.as_tensor(norm_train_input, dtype=torch.float32)
    train_target_tensor = torch.as_tensor(train_target, dtype=torch.float32)
    val_input_tensor = torch.as_tensor(norm_val_input, dtype=torch.float32)
    val_target_tensor = torch.as_tensor(val_target, dtype=torch.float32)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_input_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_input_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer, loss, and device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)
    dummy_in = torch.as_tensor(norm_train_input[:1], dtype=torch.float32).to(device)
    writer.add_graph(model, dummy_in)

    best_val_loss = float('inf')
    best_model_path = os.path.join(logs_dir, 'best_monkey_weights.pth')

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_input, batch_target in train_loader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch_target)).item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_target)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - batch_target)).item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
    
        writer.add_scalar('Loss-Train', avg_train_loss, epoch)
        writer.add_scalar('Loss-Val', avg_val_loss, epoch)
        writer.add_scalar('MAE-Train', avg_train_mae, epoch)
        writer.add_scalar('MAE-Val', avg_val_mae, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}')

   
    writer.close()


    model.load_state_dict(torch.load(best_model_path, weights_only=True))


def test_model(model, test_input, input_mean, input_stdev, batch_size=60):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # normalize
    norm_test_input = normalize_data_per_row(test_input, input_mean, input_stdev)

    # Convert to PyTorch tensor
    test_input_tensor = torch.as_tensor(norm_test_input, dtype=torch.float32)
    
    # Set model to evaluation mode and get device
    model.eval()
    device = next(model.parameters()).device
    
    predicted_targets = []
    
    # Evaluate in batches
    with torch.no_grad():
        for i in range(0, len(test_input_tensor), batch_size):
            batch_input = test_input_tensor[i:i+batch_size].to(device)
            batch_predictions = model(batch_input)
            predicted_targets.append(batch_predictions.cpu().numpy())
    
    # Concatenate all predictions
    predicted_targets = np.concatenate(predicted_targets, axis=0)
    
    return predicted_targets


def compute_average_L2_error(test_target, predicted_targets):
    """
    Compute the average L2 error for the predictions
    :param test_target: matrix with ground truth targets [N x 1]
    :param predicted_targets: matrix with predicted targets [N x 1]
    :return: average L2 error
    """
    diff = predicted_targets - test_target
    l2_err = np.sqrt(np.sum(np.power(diff, 2), axis=1))
    assert l2_err.shape[0] == predicted_targets.shape[0], \
        "Invalid dim {} vs {}".format(l2_err.shape, predicted_targets.shape)
    average_l2_err = np.mean(l2_err)

    return average_l2_err


def main(num_examples, epochs, lr, visualize_training_data, build_fn=build_linear_model, batch_size=16):
    """
    Main function
    :param num_training: Number of examples to generate for the problem (including training, testing, and val)
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param visualize_training_data: visualize the training data?
    """

    np.random.seed(0) # make the generated values deterministic. do not change!
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0) # For multi-GPU
    random.seed(0)
    
    # generate data
    monkey_function = lambda x: np.power(x[0], 3) - 3*x[0]*np.power(x[1],2)
    input, target = sfu.generate_data_for_2D_function(monkey_function, N=num_examples)

    # split data into training (70%) and testing (30%)
    all_train_input, all_train_target, test_input, test_target = sfu.split_data(input, target, 0.6)

    # visualize all training/testing (uncomment if you want to visualize the whole dataset)
    # plot_train_and_test(all_train_input, all_train_target, test_input, test_target, "train", "test", title="Train/Test Data")

    # split training data into actual training and validation
    train_input, train_target, val_input, val_target = sfu.split_data(all_train_input, all_train_target, 0.8)

    # visualize training/validation (uncomment if you want to visualize the training/validation data)
    if visualize_training_data:
        sfu.plot_train_and_test(train_input, train_target, val_input, val_target, "train", "validation", title="Train/Val Data")

    # normalize input data and save normalization parameters to file
    mean, stdev = compute_normalization_parameters(train_input)

    # build the model
    model = build_fn(train_input.shape[1])

    # train the model
    print("\n\nTRAINING...")
    train_model(model, train_input, train_target, val_input, val_target, mean, stdev,
                epochs=epochs, learning_rate=lr, batch_size=batch_size)

    # test the model
    print("\n\nTESTING...")
    predicted_targets = test_model(model, test_input, mean, stdev)

    # Report average L2 error
    l2_err = compute_average_L2_error(test_target, predicted_targets)
    print("L2 Error on Testing Set: {}".format(l2_err))

    # visualize the result
    sfu.plot_test_predictions(test_input, test_target, predicted_targets, title="Predictions")


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="total number of examples (including training, testing, and validation)",
                        type=int, default=600)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=16)
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=50)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=1e-2)
    parser.add_argument("--visualize_training_data", help="visualize training data",
                        action="store_true")
    parser.add_argument("--build_fn", help="model to train (e.g., 'linear')",
                        type=str, default="linear")
    parser.add_argument("--load_model", help="path to the model",
                        type=str, default="")
    args = parser.parse_args()

    # define the model function that we will use to assemble the Neural Network
    if args.build_fn == "linear":
        build_fn = build_linear_model # function that builds linear model
    elif args.build_fn == "nonlinear":
        build_fn = build_nonlinear_model 
    else:
        print("Invalid build function name {}".format(args.build_fn))
        sys.exit(1)

 
    if len(args.load_model) > 0:
        original_build_fn = build_fn

        def load_pytorch_model(num_inputs):
         
            model = original_build_fn(num_inputs)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(args.load_model, map_location=device, weights_only=True))
            return model

        build_fn = load_pytorch_model


    # run the main function
    main(args.n, args.epochs, args.lr, args.visualize_training_data, build_fn=build_fn, batch_size=args.batch_size)
    sys.exit(0)