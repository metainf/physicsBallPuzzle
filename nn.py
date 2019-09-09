import glob
import time
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class BallDataset(Dataset):

    def __init__(self,root_dir,numFrames,numVel,transform=None):
        self.transform = transform
        self.numFrames = numFrames
        self.numVel = numVel
        self.root_dir = root_dir
        self.filenames = sorted_nicely(glob.glob(self.root_dir+"/*N.npz"))
        loaded = np.load(self.filenames[0])
        self.numSamplesPerFile = (loaded['velArray'].shape[1] - (numFrames+numVel-1))
        self.numFiles = len(self.filenames)

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
        self.features = nn.Sequential(
            nn.Conv2d(3*numFrames, 64, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True),

            nn.Conv2d(64, 192, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=0),
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

    def init_hidden(self,device):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(2, self.batch_size, 1000).to(device),
                torch.zeros(2, self.batch_size, 1000).to(device))

    def forward(self,imageSeq):
        features = self.features(imageSeq)
        features = torch.squeeze(features)
        encoded = self.encoder(features)
        lstm_out, self.hidden = self.lstm(encoded.view(-1, self.batch_size,1000))
        predict = self.decoder(lstm_out[-1].view(self.batch_size, -1))
        return predict.view(self.batch_size,2,self.numVel)

class Weighted_MSE_Loss(torch.nn.Module):
    def __init__(self,weightVector):
        super(Weighted_MSE_Loss,self).__init__()
        self.weightVector = weightVector

    def forward(self,x,y):
        L2Error = torch.sqrt(torch.sum((x - y) ** 2,dim=1))
        weightedError = L2Error * self.weightVector.expand_as(L2Error)
        return(torch.sum(weightedError))

if __name__ == '__main__':
    numFrames = 4
    numVel = 20
    batch_size = 50
    learning_rate = 0.0001
    max_iters = 10
    alexnet = models.alexnet(pretrained=True)
    net = Net(batch_size,numFrames,numVel)

    alexnetDict = alexnet.state_dict()
    del alexnetDict['features.0.weight']
    del alexnetDict['features.0.bias']
    del alexnetDict['features.8.weight']
    del alexnetDict['features.8.bias']

    net.load_state_dict(alexnetDict,strict=False)

    train_loader = DataLoader(
        BallDataset(
            "./",numFrames,numVel,
            transform=transforms.Compose([
                transforms.ToTensor()])
        ),
        batch_size=batch_size,num_workers=4
        )

    device = torch.device("cuda:0")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    weightVectorNp = np.exp(-1*np.power(np.arange(numVel),1.0/4.0))
    weightVector = Variable(torch.tensor(weightVectorNp,dtype=torch.float32), requires_grad=True)
    criterion = Weighted_MSE_Loss(weightVector.to(device))


    for epoch in range(max_iters):
        start = time.time()
        print("Starting Training For epoch {:02d}".format(epoch))
        for data in train_loader:
            net.zero_grad()
            optimizer.zero_grad()

            inputs, preditVel = data
            inputs, preditVel = inputs.to(device),preditVel.to(device)

            net.hidden = net.init_hidden(device)
            outputs = net(inputs)
            loss = criterion(outputs, preditVel)
            loss.backward()
            optimizer.step()
            print(loss)
        end = time.time()
        print("Finished Training For epoch {:02d}, took {:.2f} seconds".format(epoch,end-start))
        torch.save(net.state_dict(),"lstmNN.pt")





