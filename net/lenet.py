import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Defining the convolutional neural network
class LeNet5Encoder(nn.Module):
    def __init__(self,n_fc=84):
        super(LeNet5Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(320, 256)
        self.bn = nn.LayerNorm([256])
        self.fc1 = nn.Linear(256, n_fc)
        self.bn1 = nn.LayerNorm([n_fc])

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.bn(self.fc(out)))
        out = F.relu(self.bn1(self.fc1(out)))
        return out


class MultiLeNetR(nn.Module):
    def __init__(self, d):
        super(MultiLeNetR, self).__init__()
        self._conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self._conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self._conv2_drop = nn.Dropout2d()
        self._fc = nn.Linear(320, d)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(
                torch.bernoulli(torch.ones(1, channel_size, 1, 1) *
                                0.5).to(x.device))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask):
        x = F.relu(F.max_pool2d(self._conv1(x), 2))
        x = self._conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training:
            x = x * mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self._fc(x))
        return x, mask

class MLP(nn.Module):
    def __init__(self, d=84):
        super(MLP, self).__init__()
        self._fc1 = nn.Linear(d, d)
        self._fc2 = nn.Linear(d, d)
        self._fc3 = nn.Linear(d, 10)
        return

    def forward(self, x):
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return F.log_softmax(x, dim=1)