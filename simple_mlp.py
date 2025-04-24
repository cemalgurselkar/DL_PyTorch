import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    train_data = datasets.MNIST(root=".data/MNIST",train=True,download=True, transform=transform)
    test_data = datasets.MNIST(root=".data/MNIST",train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def model_train(model, dataloader,loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for x,y in dataloader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred,y).item() # to retrivial a pure float value.
            correct += (pred.argmax(1) == y).sum().item()
    
    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def plot_training(train_losses, test_accuracies):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.legend()

    plt.show()

def main():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(10):
        train_loss = model_train(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_accuracy = test_model(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1} => Train Loss: {train_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    plot_training(train_losses, test_accuracies)

if __name__ == "__main__":
    main()