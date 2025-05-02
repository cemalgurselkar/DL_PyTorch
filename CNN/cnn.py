import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
"""
1. load datasets
2. visual datasets
3. build CNN Model
4. define loss funcs and optimizer
5. training
6. test
"""

def get_data(bathc_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(((0.5,0.5,0.5)), (0.5,0.5,0.5)) #rgb kanallarını normalize et.
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data",train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, bathc_size,shuffle=True)
    test_loader = DataLoader(test_set, bathc_size, shuffle=False)

    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        # image 3x32x32 -> conv(32) -> relu(32) -> pool(16)
        # conv(16) -> relu(16) -> pool(8)

        #Fully Connected
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, out_channel)
    
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x))) # birinci conv katmanı
        x = self.pool(self.relu(self.conv2(x))) # ikinci conv katmanı
        x = x.view(-1, 64*8*8)
        x = self.fc3(self.fc2(self.fc1(x))) # fully connection
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(3,10).to(device=device)

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
)
criterion, optimizer = define_loss_and_optimizer(model)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predicted = model(images)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss: .3f}")
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("Model_Train_Performs.png")
    print("Model Sonuçları Kaydedildi.")
    plt.close()

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # gradyan hesaplama işlemlerini kapatmak için
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # doğru tahminleri say
    
    print(f"Test Accuracy: {100*correct/total: .3f}")

if __name__ == "__main__":
    train_loader, test_loader = get_data()
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    test_model(model, test_loader)