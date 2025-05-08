import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare_data(batch_size):
    transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transfrom)
    testset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transfrom)
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(testset,batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = prepare_data(batch_size=32)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channel, stride=1, downsampling=None):
        """
        conv2d -> batchNorm -> conv2d -> batchNorm -> downsampling
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=3,stride=stride, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.batchNorm2 = nn.BatchNorm2d(out_channel)

        #downsampling
        self.downsampling = downsampling

    def forward(self,x):
        identity = x 
        
        if self.downsampling is not None:
            identity = self.downsampling(x)
        
        out = self.conv1(x)
        out = self.batchNorm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchNorm2(out)
        out = out + identity #skip connection
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,num_classes):
        """
            conv -> batchNorm -> relu -> maxpool -> 4xlayer -> avgpool -> fc
        """
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bc1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        
        self.layer1 = self._make_layer(64,64,2,2)
        self.layer2 = self._make_layer(64,128,2,2)
        self.layer3 = self._make_layer(128,256,2,2)
        self.layer4 = self._make_layer(256,512,2,2)

        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)        
    
    def _make_layer(self,in_channels,out_channels,blocks,stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResidualBlock(in_channels,out_channels,stride,downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bc1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
use_custom_model = True #eğer true ise custom çalışsın

if use_custom_model:
    model = ResNet(10).to(device)

else:
    model = models.resnet18(pretrain=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential( # type: ignore
        nn.Linear(num_ftrs,256),
        nn.ReLU(),
        nn.Linear(256,10))
    model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epoch = 7
for epoch in tqdm(range(num_epoch)):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {running_loss:.5f}")
print()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images,labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100 * correct / total}%s")