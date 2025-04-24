"""
Problem Type: Classification with MNİST datasets using Artifical Neural Network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(batch_size=16):
    transforms = transform.Compose([
        transform.ToTensor(), # görüntüyü tensore çevirir ve 0-255 -> 0-1 ölçeklendirir.
        transform.Normalize((0.5,),(0.5,)) #piksel değerlerini -1 ile 1 arasında ölçeklendirir.
    ])

    train_set = torchvision.datasets.MNIST(root="./data",train=True, download=True, transform=transforms)
    test_set = torchvision.datasets.MNIST(root="./data",train=False, download=True, transform=transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #elimizde bulunan görüntüleri (2D) vektör haline çevirelim (1D)
        self.flatten = nn.Flatten()
        # ilk tam bağlı katmanı oluştur
        self.fc1 = nn.Linear(28*28, 128)
        # aktivasyon fonks oluştur
        self.relu = nn.ReLU()
        # ikinci tam bağlı katman
        self.fc2 = nn.Linear(128, 64)
        #çıktı katmanı oluştur
        self.fc3 = nn.Linear(64,10) # burada son kısım, sınıf sayısı olmak zorunda. 
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)
#loss function and define the optimization algorithm
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr=0.001)
)

criterion, optimizer = define_loss_and_optimizer(model)
# Train
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("Model_Train_Performs.png")
    print("Model Sonuçları Kaydedildi.")
    plt.close()

train_model(model, train_loader, criterion, optimizer, epochs=10)

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

test_model(model, test_loader)