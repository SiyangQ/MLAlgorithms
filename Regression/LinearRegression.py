import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

'''
Model definition
'''
class Linear_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
    def training_step(self, batch):
        x, y = batch 
        out = self(x)                  # Generate predictions
        loss = nn.MSELoss(out, y) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        x, y = batch 
        out = self(x)                    # Generate predictions
        loss = nn.MSELoss(out, y)   # Calculate loss
        acc = accuracy(out, y)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

# Utility function to train the model
def fit(epochs, lr, model, trainLoader, valLoader, opt_func=torch.optim.SGD, plot=False, leaveProgressBar=False):
    optimizer = opt_func(model.parameters(), lr)
    history = []

    for epoch in trange(epochs):
        # Train with batches of data
        for batch in tqdm(trainLoader):
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, valLoader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, dataLoader):
    with torch.no_grad(): 
        outputs = [model.validation_step(batch) for batch in dataLoader]
    return model.validation_epoch_end(outputs)

# Load dataset
x_numpy, y_numpy = datasets.fetch_california_housing(data_home='./Regression/datasets/', download_if_missing=True, return_X_y=True)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(x_numpy.shape, y_numpy.shape)
dataset = TensorDataset(x, y)
trainSet, testSet = random_split(dataset, [0.6*len(dataset), 0.4*len(dataset)])

batchSize = 100
trainLoader = DataLoader(trainSet, batchSize, shuffle=True) # data loader
testLoader = DataLoader(testSet, batchSize, shuffle=False) # data loader

# Train model
inputSize, outputSize = (8, 1)
model = nn.Linear(inputSize, outputSize) # model
loss_fn = nn.MSELoss() # loss function
opt = torch.optim.SGD(model.parameters(), lr=1e-5) # optimizer

fit(100, model, loss_fn, opt, trainLoader)

# Test model
result = evaluate(model, testLoader)
print(f"Test accuracy={result['val_acc']}")