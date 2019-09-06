import glob
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class BallDataset(Dataset):

    def __init__(self,root_dir,numFrames,numVel,transform=None):
        self.transform = transform
        self.numFrames = numFrames
        self.numVel = numVel
        self.root_dir = root_dir
        self.filenames = sorted(glob.glob(self.root_dir+"/*.npz"))
        loaded = np.load(self.filenames[0])
        self.numSamplesPerFile = (loaded['velArray'].shape[1] - (numFrames+numVel-1))
        self.numFiles = len(glob.glob(self.root_dir+"/*.npz"))

    def __len__(self):
        return self.numFiles * self.numSamplesPerFile

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fileIndex = idx // self.numSamplesPerFile
        sequenceNum = idx % self.numSamplesPerFile
        sequenceIndex = sequenceNum + self.numFrames - 1
        loaded = np.load(self.filenames[fileIndex])
        imageSequence = loaded['ballImgArray'][:,:,:,sequenceIndex-self.numFrames+1:sequenceIndex+1]
        imageSequence = imageSequence.reshape((128,128,3*self.numFrames))
        if self.transform:
            imageSequence = self.transform(imageSequence)
        predictSequence = loaded['velArray'][:,sequenceIndex:sequenceIndex+self.numVel]
        predictSequence = torch.tensor(predictSequence,dtype=torch.float32)
        sample = (imageSequence, predictSequence)
        return sample

class Net(nn.Module):
    def __init__(self,batch_size,numFrames,numVel):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.numVel = numVel
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3*numFrames, 96, kernel_size=11, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.Sequential(
            nn.Linear(128, 1000),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(1000,1000,2)
        self.decoder = nn.Sequential(
            nn.Linear(1000,1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,2*numVel),
        )

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(2, self.batch_size, 1000),
                torch.zeros(2, self.batch_size, 1000))

    def forward(self,input):
        input = self.cnn1(input)
        input = torch.squeeze(input)
        input = self.encoder(input)
        lstm_out, self.hidden = self.lstm(input.view(-1, self.batch_size,1000))
        predict = self.decoder(lstm_out[-1].view(self.batch_size, -1))
        return predict.view(self.batch_size,2,self.numVel)

class Weighted_MSE_Loss(torch.nn.Module):
    def __init__(self,weightVector):
        super(Weighted_MSE_Loss,self).__init__()
        self.weightVector = weightVector

    def forward(self,x,y):
        sqError = torch.sum((x - y) ** 2,dim=1)
        weightedError = sqError * self.weightVector.expand_as(sqError)
        return(torch.sum(weightedError))

numFrames = 4
numVel = 20
batch_size = 50
learning_rate = 0.001
max_iters = 10

net = Net(batch_size,numFrames,numVel)
train_loader = DataLoader(
    BallDataset(
        "./",numFrames,numVel,
        transform=transforms.Compose([
            transforms.ToTensor()])
    ),
    batch_size=batch_size
    )

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
weightVectorNp = np.exp(-1*np.power(np.arange(numVel),1.0/4.0))
weightVector = Variable(torch.tensor(weightVectorNp,dtype=torch.float32), requires_grad=True)
criterion = Weighted_MSE_Loss(weightVector)

for epoch in range(max_iters):
    total_loss = 0.0
    total_acc = 0.0
    start = time.time()
    print("Starting Training For epoch {:02d}".format(epoch))
    for data in train_loader:
        inputs, preditVel = data
        optimizer.zero_grad()
        net.hidden = net.init_hidden()
        outputs = net(inputs)
        loss = criterion(outputs, preditVel)
        loss.backward()
        optimizer.step()
    end = time.time()

