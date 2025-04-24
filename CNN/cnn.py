import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(((0.5,0.5,0.5)), (0.5,0.5,0.5)) #rgb kanallarını normalize et.
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data",train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data",train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.2) # dropout %20 oranında çalıştır
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*8*8)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

define_loss_optimiezr = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("Train_loss_Performs.png")
    plt.close()

def test_model(model, test_loader, dataset_type):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"For {dataset_type}, accuracy: {100* correct/total}%")

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader()

    model = ConvolutionNeuralNetwork().to(device)
    criterion, optimizer = define_loss_optimiezr(model)
    train_model(model, train_loader,criterion, optimizer, epochs=10)
    test_model(model, test_loader, dataset_type="test")
    test_model(model, train_loader, dataset_type="training")