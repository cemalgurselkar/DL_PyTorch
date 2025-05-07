"""
Radial Basis Function:
It is a classical ANN but it has radian kernel.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#veri seti oluşturma.
def to_tensor(data,target):
    return torch.tensor(data,dtype=torch.float32), torch.tensor(target, dtype=torch.long)

def get_data():
    df = pd.read_csv("iris.data", header=None)

    X = df.iloc[:,:-1].values
    y,_ = pd.factorize(df.iloc[:,-1])

    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    x_train,y_train = to_tensor(x_train,y_train)
    x_test, y_test = to_tensor(x_test,y_test)

    return x_train,x_test,y_train,y_test

x_train,x_test,y_train_,y_test = get_data()

def rbf_kernel(X,centers,beta):
    return torch.exp(-beta * torch.cdist(X, centers)**2)

class RBFN(nn.Module):
    def __init__(self,num_centers, input_dim, output_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.beta = nn.Parameter(torch.ones(1) * 2.0)
        self.fc = nn.Linear(num_centers, output_dim)

    def forward(self,x):
        #rbf kernel
        phi = rbf_kernel(x,self.centers,self.beta)
        return self.fc(phi)

num_centers = 10
model = RBFN(input_dim=4, num_centers=num_centers, output_dim=3)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_func(output, y_train_)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(x_test)
    accuracy = (torch.argmax(y_pred, axis=1) == y_test).float().mean().item()#type:ignore
    print(f"accuracy: {accuracy}")