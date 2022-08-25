#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from tqdm.notebook import tqdm, trange

'''
Helper function to display an image
Usage (without prediction): 
    image, label = mnist_train[3]
    displayImage(image, label)
'''
def displayImage(image, label, prediction = None):
    image = image.reshape([28, 28])
    title = f"Label: {label}"
    if not prediction == None: title += f" Prediction: {prediction}"
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()

'''
Model definition
'''
class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.linear(x)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, plot=False, leaveProgressBar=False):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in trange(epochs, leave=leaveProgressBar):
        
        # Training Phase 
        for batch in tqdm(train_loader, leave=leaveProgressBar):
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def evaluate(model, val_loader):
    with torch.no_grad(): 
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
