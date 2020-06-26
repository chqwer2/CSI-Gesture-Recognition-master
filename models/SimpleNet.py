import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleNet(nn.Module):
    def __init__(self,normalized = True):
        super(SimpleNet, self).__init__()
        
        self.normalized = normalized

        self.conv_3_6_5x1 = nn.Conv2d(3,6,(5,1),1)
        self.conv_6_16_5x1 = nn.Conv2d(6,16,(5,1),1)
        self.conv_16_32_5x1 = nn.Conv2d(16,32,(5,1),1)
        self.conv_32_64_5x1 = nn.Conv2d(32,64,(5,1),1)
        self.conv_64_64_2x2 = nn.Conv2d(64,64,2,1,(1,0))
        self.conv_64_128_3x3 = nn.Conv2d(64,128,3,1)
        self.conv_128_64_1x1 = nn.Conv2d(128,64,1)
        self.conv_64_16_1x1 = nn.Conv2d(64,16,1)
        
        self.pool_2x2_2_1 = nn.MaxPool2d(2,(2,1),(0,1))
        self.pool_2x2_1_2 = nn.MaxPool2d(2,(1,2),(1,0))
        self.pool_2x2_1_1 = nn.MaxPool2d(2,1)
        self.pool_2x2_2_2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x): #shape of x is 3x200x30
        x = self.pool_2x2_2_1(F.relu(self.conv_3_6_5x1(x)))  # 6x98x31
        x = self.pool_2x2_2_1(F.relu(self.conv_6_16_5x1(x)))  # 16x47x32
        x = self.pool_2x2_1_1(F.relu(self.conv_16_32_5x1(x)))  # 32x42x31
        x = self.pool_2x2_2_1(F.relu(self.conv_32_64_5x1(x)))  # 64x19x32
        x = self.pool_2x2_2_1(F.relu(self.conv_64_64_2x2(x)))  # 64x10x32
        x = self.pool_2x2_2_2(F.relu(self.conv_64_128_3x3(x)))  # 128x4x15
        x = self.pool_2x2_1_2(F.relu(self.conv_128_64_1x1(x)))  # 64x5x7   test
        x = F.relu(self.conv_64_16_1x1(x))  #16x5x7
        x = x.view(-1, 16 * 5 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.normalized:
            norm = x.norm(dim=1,p=2,keepdim=True)   #sqrt((x1)^2 + (x2)^2+   )
            x = x.div(norm.expand_as(x))
        return x


def Simple_Net( normalized, pretrained = False, model_path = None , ):

    model = SimpleNet(normalized)
    return model

def main():
    model = Simple_Net(normalized = True)
    images = Variable(torch.ones(8, 3, 200, 30))
    out_ = model(images)
    print(out_.data.shape)

if __name__ == '__main__':
    main()
