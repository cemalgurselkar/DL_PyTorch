import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_data(seq_length = 50, num_sample=1000):
    x = np.linspace(0,100,num_sample)
    y = np.sin(x)
    sequence = []
    targets = []

    for i in range(len(x) - seq_length):
        sequence.append(y[i:i+seq_length])
        targets.append(y[i+seq_length])
    
    plt.figure(figsize=(8,4))
    plt.plot(x,y,label="sin(t)", color="b",linewidth=2)
    plt.title("Sinüs Dalga Grafiği")
    plt.xlabel("Zaman (radyan)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.savefig("Sample_data.png")
    plt.close()
    
    return np.array(sequence), np.array(targets)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1):
        super().__init__()
        # input_size = giriş boyutu
        # hidden_size = rnn gizli katman cell sayisi
        # num_layers = rnn layer sayisi
        self.rnn = nn.RNN(input_size, hidden_size, num_layer, batch_first= True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self,x):
        out,_ = self.rnn(x)
        out = self.fc(out[:,-1,:]) # son zaman adimindaki çiktiyi al ve fc layera bağla
        return out

seq_length = 50
input_size = 1
hidden_size = 16
output_size = 1
num_layer = 1
epochs = 15
batch_size = 32
learning_rate = 0.001

X,y = generate_data()
X = torch.tensor(X, dtype = torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(X,y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, output_size, num_layer)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

X_test = np.linspace(100, 110, seq_length).reshape(1,-1)
y_test = np.sin(X_test) 

X_test2 = np.linspace(120,130,seq_length).reshape(1,-1)
y_test2 = np.sin(X_test2)

X_test = torch.tensor(y_test, dtype = torch.float32).unsqueeze(-1)
X_test2 = torch.tensor(y_test2, dtype = torch.float32).unsqueeze(-1)

model.eval()
prediction1 = model(X_test).detach().numpy()
prediction2 = model(X_test2).detach().numpy()

plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker = "o", label = "Training dataset")
plt.plot(X_test.numpy().flatten(), marker = "o", label = "Test 1")
plt.plot(X_test2.numpy().flatten(), marker = "o", label = "Test 2")

plt.plot(np.arange(seq_length, seq_length + 1), prediction1.flatten(), "ro", label = "Prediction 1")
plt.plot(np.arange(seq_length, seq_length + 1), prediction2.flatten(), "ro", label = "Prediction 2")
plt.legend()
plt.savefig("Predict-True_Result.png")
plt.close()