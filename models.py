import torch.nn as nn
import torchvision.models as tvmodels

class Net(nn.Module):

    def __init__(self, args):

        super(Net, self).__init__()
        self.resnet18 = tvmodels.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children()) [:-2])

        #print(self.resnet18)

        ''' declare layers used in this network'''
        # first block
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 64x64 -> 64x64
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        # second block
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # third block
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 16x16
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # fourth block
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16 -> 16x16
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        # five block
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 16x16 -> 16x16
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True )

    def forward(self, img):

        x = self.resnet18(img)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return x

