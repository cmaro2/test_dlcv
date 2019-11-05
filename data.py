import os
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, args, mode='train', type = 0):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.type = type
        self.dir = 'hw2_data/'+mode+'/img'
        self.data_dir = os.listdir(self.dir)
        self.img_dir = [self.dir + '/' + photo for photo in self.data_dir]

        self.dir_seg = 'hw2_data/' + mode + '/seg'
        self.data_dir_seg = os.listdir(self.dir_seg)
        self.img_dir_seg = [self.dir_seg + '/' + photo for photo in self.data_dir_seg]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])



    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img_dir[idx]
        seg_path = self.img_dir_seg[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path)

        seg = np.array(seg)
        seg = torch.from_numpy(seg)
        seg = seg.long()
        seg = torch.squeeze(seg)

        if self.type == 1:
            return self.transform(img), seg, os.path.basename(img_path)
        else:
            return self.transform(img), seg

