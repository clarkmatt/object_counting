import torch
from torch import nn

class Column(nn.Module):
    def __init__(self, out_size, filter_size):
        super(Column, self).__init__()
        self.out_size = out_size
        self.filter_size = filter_size

        self.conv1 = nn.Conv2d( 1, 2*out_size, filter_size, padding=int((filter_size-1)/2))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2*out_size, 4*out_size, filter_size-2, padding=int((filter_size-1)/2))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(4*out_size, 2*out_size, filter_size-2, padding=int((filter_size-1)/2))
        self.conv4 = nn.Conv2d(2*out_size,  out_size, filter_size-2, padding=int((filter_size-1)/2))

    def forward(self, x):
        x = self.conv1(x) 
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        self.column1 = Column(8, 9)
        self.column2 = Column(10, 7)
        self.column3 = Column(12, 5)

        out_size = self.column1.out_size + self.column2.out_size + self.column3.out_size
        self.merge_conv = nn.Conv2d(out_size, 1, 1)
        print(out_size)


    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)

        x = self.merge_conv(torch.cat((x1, x2, x3), 1))

        return x


