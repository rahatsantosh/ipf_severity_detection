import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from shallow_autoenc import Autoencoder
from autoencoder_dataset import Dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name())
cudnn.benchmark = True

tf = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(45),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.Resize((1024, 1024)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x:torch.reshape(x, (-1, x.shape[0], x.shape[1], x.shape[2])))
])

root = "../../data/external/chest_xray/img"

training_set = torchvision.datasets.ImageFolder(root, transform=tf)
training_generator = DataLoader(training_set, batch_size = 32)

model = Autoencoder()
if use_cuda:
	model.to(device)

def train_model(model,train_loader,optimizer,n_epochs=10,gpu=True):
    loss_list=[]
    for epoch in range(n_epochs):
        for x, _ in train_loader:
            if gpu:
                # Transfer to GPU
                x = x.to(device)

            model.train()
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)
        print('Epoch : ',epoch,'    Loss : ',loss.data)

    return loss_list

criterion = nn.CosineSimilarity()
optimizer = torch.optim.Adam(model.parameters())

loss_list = train_model(
    model=model,
    train_loader=training_generator,
    optimizer=optimizer,
    n_epochs=100,
    gpu=True
)
print("-------------Done--------------")

plt.plot(np.arange(len(loss_list)),loss_list)
plt.savefig('../../reports/figures/xray_autoenc_loss.png')

model_path = "../../models/autoenc.pt"
torch.save(model, model_path)
