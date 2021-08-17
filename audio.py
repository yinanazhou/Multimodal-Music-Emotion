import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


class LoadData:
    def __init__(self, dataset_path, transform=None, idx=None):
        self.dataset = json.load(open(dataset_path, 'r'))
        self.data = np.array(self.dataset['melspectrogram'])
        self.targets = np.array(self.dataset['labels'])
        if idx is not None:
            self.targets = self.targets[idx]
            self.data = self.data[idx]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index].squeeze(), int(self.targets[index])
        img = Image.fromarray((img * 255).astype('uint8'), mode='L')
        # img = Image.fromarray(img.astype('uint8'), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 8), padding=(0, 3))
        # self.conv2 = nn.Conv2d(32, 16, kernel_size=(1, 8), padding=(0, 3))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 8))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(1, 8))
        # self.fc1 = nn.Linear(16*32*322, 64)
        self.fc1 = nn.Linear(16 * 8 * 78, 64)
        self.fc2 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.5)

        # self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # self.fc1 = nn.Linear(16 * 323 * 3, 64)
        # self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        # 1*1292*13
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=4, stride=4)  # 1*13*1292 => 32*13*1291 => 32*6*645
        x = self.dropout(x)
        # print(x.shape)
        # 32*1291*13 => 32*645*6
        # 32*1292*13 => 32*646*6
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=4, stride=4)  # 16*6*644 => 16*3*322
        x = self.dropout(x)
        # print(x.shape)
        # 16*646*6 => 16*323*3
        x = x.view(-1, 16 * 8 * 78)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# hyperparameters
BATCH_SIZE = 5  # feel free to change it
EPOCH = 5
DATASET_PATH = '../dataset/json/melspectrogram.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
torch.cuda.manual_seed(0)

""" Load data"""

transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),
    # transforms.Resize(IMG_SIZE)  
])


""" Split train and test"""


dataset = LoadData(DATASET_PATH, transform=transformer, idx=None)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

VAL_SPLIT = 0.20
shuffle = True

# Creating indices for train and val split:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(VAL_SPLIT * dataset_size))
if shuffle:
    # set random seed so that we get the same split everytime
    np.random.seed(0)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_dataset = LoadData(DATASET_PATH, transform=transformer, idx=train_indices)
test_dataset = LoadData(DATASET_PATH, transform=transformer, idx=test_indices)

# separate loaders for train and val data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""Model"""

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# optimizer = optim.RMSprop(model.parameters(), lr=1e-5)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(3)]
test_accu = []


def train(epoch, model, loader):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        # print(output.shape)
        # target = torch.argmax(target, dim=1) # convert from 1-hot to 1D
        loss = F.cross_entropy(output, target, reduction='sum')  # negative log likelihood loss
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(loader.dataset),
              100. * batch_idx / len(loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
              (batch_idx*64) + ((epoch-1)*len(loader.dataset)))
            torch.save(model.state_dict(), 'model\model15.pth')
            torch.save(optimizer.state_dict(), 'model\optimizer15.pth')


def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    test_losses.append(test_loss)
    test_accu.append(correct/len(loader.dataset))
    print('Epoch: {}, Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      epoch, test_loss, correct, len(loader.dataset),
      100. * correct / len(loader.dataset)))


for epoch in range(1, EPOCH+1):
    train(epoch, model, train_loader)
    test(model, test_loader)

# convert val_accus (tensor) to list, for plotting
accu_list = []
for accu in test_accu:
    accu_list.append(accu.item())

accu = pd.DataFrame(accu_list)
accu.to_csv('results/audio_train_15.csv')
print('Results Saved!')

