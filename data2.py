import os
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

class DATA2(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.dir = args.dir_img
        self.data_dir = os.listdir(self.dir)
        self.img_dir = [self.dir + '/' + photo for photo in self.data_dir]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img_dir[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), os.path.basename(img_path)

