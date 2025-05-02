import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataloader


def generate_data(seq_length=50, num_samples=7000):
    x = np.linspace(0,200,num_samples)
    y = np.sin(x)
    sequences = []
    targets = []

    for i in range(len(x) - seq_length):
        sequences.append(y[i:i+seq_length])
        targets.append(y[i+seq_length])

    """plt.figure(figsize=(8,4))
    plt.plot(x,y)
    plt.title("Sinus Dalga Grafiği")
    plt.xlabel("Zaman")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()"""

    return np.array(sequences), np.array(targets)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        """
        RNN -> Linear
        Returns
        """
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out, _ = self.rnn(x)
        out = self.fc1(out[:,-1,:]) #son zaman adimindaki çiktiyi al ve fc layera bağla.
        return out

#hyperparams
seq_length = 50
input_size = 1
hidden_size = 16
output_size = 1
num_layers = 1
epochs = 10
batch_size = 32
learning_rate = 0.001

#set the data
X,y = generate_data(seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # unsqueeze(-1) = boş kısımları kendin doldur.
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(X,y) #pytorch datasets oluşturma
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_losses = []
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predicted = model(batch_x)
        loss = criterion(predicted, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item(): .4f}")

x_test = np.linspace(100, 150, seq_length).reshape(1,-1)
y_test = np.sin(x_test)

x_test2 = np.linspace(125, 190, seq_length).reshape(1,-1)
y_test2 = np.sin(x_test2)

X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1) # unsqueeze(-1) = boş kısımları kendin doldur.
X_test2 = torch.tensor(x_test2, dtype=torch.float32).unsqueeze(-1) # unsqueeze(-1) = boş kısımları kendin doldur.

model.eval()
prediction = model(X_test).detach().numpy()
prediction2 = model(X_test2).detach().numpy()

plt.figure()
plt.plot(np.linspace(0,100, len(y)), y, marker="o", label="Training dataset")
plt.plot(X_test.numpy().flatten(), marker="o",label="Test 1")
plt.plot(X_test2.numpy().flatten(), marker="o",label="Test 2")
plt.plot(np.arange(seq_length, seq_length+1), prediction.flatten(), "ro", label="Prediction1")
plt.plot(np.arange(seq_length, seq_length+1), prediction2.flatten(), "ro", label="Prediction2")
plt.legend()
plt.show()