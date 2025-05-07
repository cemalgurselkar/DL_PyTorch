import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(batch_size):
    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(), #görüntüleri yatay çevirerek veri arttırma.
    transforms.RandomRotation(degrees=10), #görüntüleri rastgele 10 derece döndürür
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    train_dataset = dataset.Flowers102(root="./data", split="train", transform=transform_train,download=True)
    test_dataset = dataset.Flowers102(root="./data",split="test", transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = prepare_data(batch_size=16)

model = models.mobilenet_v2(pretrained = True)

#sınıflandırıcı katmani ekleme
num_feautres = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_feautres, 102) # type: ignore

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #step lr

epochs = 5
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    schedular.step()
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss:.5f}")

torch.save(model.state_dict(), "custom_mobileNet.pth")