import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing

# Load dataset
inputs, targets = fetch_california_housing(return_X_y=True)
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
train_ds = TensorDataset(inputs, targets)
batch_size = 100
train_dl = DataLoader(train_ds, batch_size, shuffle=True) # data loader

# define model
inputSize, outputSize = (8, 1)
model = nn.Linear(inputSize, outputSize) # model
loss_fn = F.mse_loss # loss function
opt = torch.optim.SGD(model.parameters(), lr=1e-5) # optimizer

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        # Train with batches of data
        for xb,yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            # 3. Compute gradients
            loss.backward()
            # 4. Update parameters using gradients
            opt.step()
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt, train_dl)