"""
image generation with MNIST datasets
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
img_size = 28*28

def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    datasets = MNIST(root="./data",train=True, transform=transform,download=True)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    return dataloader

dataloader = get_data(batch_size)

class Generator(nn.Module):
    def __init__(self, z_dim,img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,img_size),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x).view(-1,1,28,28) # kendin belirle demek -1. çıktıyı 28x28 çevirir. 


class Discriminator(nn.Module):
    def __init__(self,img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x.view(-1, img_size))

# GAN Training.
#hyperparams
lr = 0.0002
z_dim = 100 #rastgele gürültü
epochs = 10

#1. model başlatma
generator = Generator(z_dim,img_size).to(device)
discriminator = Discriminator(img_size).to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))

d_losses = []
g_losses = []

#3. eğitim döngüsü
for epoch in range(epochs):
    for i,(real_imgs,_) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        z = torch.randn(batch_size, z_dim).to(device) #rastgele gürültü oluştur
        fake_img = generator(z)
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_img.detach()), fake_labels)
        d_loss = real_loss + fake_loss #toplam discriminator kaybı

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.append(d_loss.item())

        #generator
        g_loss = criterion(discriminator(fake_img), real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_losses.append(g_loss.item())
    print(f"Epoch: {epoch+1}/{epochs}, d_loss: {d_loss.item():.3f}, g_loss: {g_loss.item():.3f}")

plt.figure(figsize=(10,5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.show()

#Model testing

with torch.no_grad():
    z = torch.randn(16,z_dim).to(device)
    sample_imgs = generator(z).to(device)
    grid = np.transpose(utils.make_grid(sample_imgs, nrow=4, normalize=True), (1,2,0))
    plt.imshow(grid)
    plt.show()