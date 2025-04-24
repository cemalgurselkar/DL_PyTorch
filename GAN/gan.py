import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
image_size = 28*28

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

dataset = datasets.MNIST(root="./data",train=True, transform=transform, download=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,image_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x).view(-1,1,28,28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,img):
        return self.model(img.view(-1, image_size))

learning_rate = 0.0002
z_dim = 100
epochs = 20

generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5,0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999))

for epoch in range(epochs):
    for i, (real_img, _) in enumerate(dataloader):
        real_img = real_img.to(device)
        batch_size = real_img.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = generator(z)
        real_loss = criterion(discriminator(real_img), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()),fake_labels)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} d_loss: {d_loss.item():.3f}, g_loss: {g_loss.item():.3f}")

with torch.no_grad():
    z = torch.randn(16, z_dim).to(device)
    sample_imgs = generator(z).cpu()
    grid = np.transpose(utils.make_grid(sample_imgs, nrow=4, normalize=True),(1,2,0))
    plt.imshow(grid)
    plt.show()