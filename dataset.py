import os

import jittor as jt 

from jittor import transform

from jittor.dataset import Dataset
import numpy as np
from PIL import Image


class TsinghuaDog(Dataset):
    def __init__(self, root_dir, batch_size, part='train', train=False, shuffle=False, transform=None, num_workers=1):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size 
        self.train = train
        self.num_classes = 130 
        self.part = part
        self.shuffle = shuffle
        self.image_list = []
        self.id_list = []
        self.transform = transform
        if part == 'train':
            list_path = os.path.join(self.root_dir, 'TrainAndValList/train.lst')
        elif part == 'val':
            list_path = os.path.join(self.root_dir, 'TrainAndValList/validation.lst')

        self.root_dir = os.path.join(self.root_dir, 'low-resolution/')

        with open(list_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                img_name = line.split('/')[-2] + '/' + line.split('/')[-1]
                cls_name = line.split('/')[-2]
                label = int(cls_name.split('-')[1][-3:]) - 1
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()

        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.id_list),
            shuffle=self.shuffle
        )

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label = self.id_list[idx]
        image_path = self.root_dir + image_name 
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image) 
        img = np.asarray(image)
        return img, label



def test_dataset():
    root = '/home/gmh/dataset/TsinghuaDog'
    part = 'train'
    # from torchvision import transforms
    rgb_mean = [0.5,0.5,0.5]
    rgb_std = [0.5,0.5,0.5]
    
    transform_val = transform.Compose([
        transform.Resize((299,299)),
        transform.ToTensor(),
        transform.ImageNormalize(rgb_mean, rgb_std),
    ])

    dataloader = TsinghuaDog(root, batch_size=16, train=False, part=part, shuffle=True, transform=transform_val)
    # def __init__(self, root_dir, batch_size, part='train', train=True, shuffle=False, transform=None, num_workers=1):
    
    for images,labels in dataloader:
        # print(images.size(),labels.size(),labels)
        pass 


if __name__=='__main__':
    test_dataset()