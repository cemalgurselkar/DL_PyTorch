"""
FashionMNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np


def get_data(bacth_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.FashionMNIST(root="./data",train=True, transform=transform,download=True)
    test_dataset = datasets.FashionMNIST(root="./data",train=False,transform=transform,download=True)
    train_loader = DataLoader(train_dataset, batch_size=bacth_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=bacth_size, shuffle=False)
    return train_loader, test_loader

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), #28*28 ---> 784
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#callback: early stopping

class EarlyStopping:

    def __init__(self,patience=5,min_delta=0.001):
        self.patiance = patience#kaç epoch gelişme olmazsa dursun
        self.min_delta = min_delta#kayıptaki minimum iyileşme miktarı
        self.best_loss = None#en iyi kayıp değeri
        self.counter = 0# sabit kalan epoch  sayac

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else: #gelişme yok
            self.counter += 1
        
        if self.counter >= self.patiance:
            return True # trainingi durdur.
        
        return False

#hyperparams
epochs = 50
lr = 1e-3

model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
early_stop = EarlyStopping(patience=5, min_delta=0.001)

train_loader, test_loader = get_data(bacth_size=32)

def train_model(model, train_loaders, optimizer, criterion, early_stopping, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input, _ in train_loaders:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loaders)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

        if early_stopping(avg_loss):
            print(f"Early Stopping at epoch {epoch+1}")
            break

train_model(model, train_loader,optimizer,criterion, early_stop, epochs=20)