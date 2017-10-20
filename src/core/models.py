import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)


class LossUtil:

    @staticmethod
    def label_loss(gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        loss = -(gt_label * torch.log(pred_label + 1e-10) + (1 - gt_label) * torch.log(1 - pred_label + 1e-10))
        return torch.mean(loss)


    @staticmethod
    def bbox_loss(bbox_target,bbox_pred):
        square_error=torch.sum(torch.pow(bbox_pred-bbox_target,2))
        return torch.mean(square_error)



class PNet(nn.Module):
    ''' PNet '''

    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        self.is_train = is_train


        # backend
        self.feature = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.feature(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)

        if self.is_train is True:
            # label_loss = LossUtil.label_loss(self.gt_label,torch.squeeze(label))
            # bbox_loss = LossUtil.bbox_loss(self.gt_bbox,torch.squeeze(offset))
            return label,offset
        #landmark = self.conv4_3(x)
        return label, offset





class RNet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False):
        super(RNet, self).__init__()
        # backend
        self.feature = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()  # prelu3

        )
        self.conv4 = nn.Linear(64, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 2)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.feature(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        if self.is_train is True:
            return det, box
        #landmard = self.conv5_3(x)
        return det, box




class ONet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False):
        super(RNet, self).__init__()
        # backend
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.feature(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        if self.is_train is True:
            return det, box
        #landmard = self.conv5_3(x)
        return det, box




if __name__ == '__main__':


    net = PNet(is_train=True)
    x = Variable(torch.rand(5, 3, 12, 12))
    y1, y2 = net(x)

    print(torch.squeeze(y1))
    print(torch.squeeze(y2))


    #print(Variable(torch.FloatTensor([[0.5, -1, 2.2, 0.6],[0.5, -1, 2.2, 0.6],[0.5, -1, 2.2, 0.6],[0.5, -1, 2.2, 0.6],[0.5, -1, 2.2, 0.6]  ])))