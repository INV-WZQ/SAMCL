import os
import json
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor
class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )
    
    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

class RandomGenerator(object):
    def __init__(self, output_size, low_res=64, bbox_shift=20, get_point=1, SAM2=False):
        self.output_size = output_size
        self.low_res = low_res
        self.bbox_shift = bbox_shift
        self.get_point = get_point
        if SAM2:
            self.transform = SAM2Transforms(1024, 0, 0, 0)
        else:
            self.transform = None

    def __call__(self, sample):
        # image, label= sample['image'], sample['label']
        # image = np.array(image)
        # image = torch.from_numpy(image.astype(np.float32))
        # label = (np.array(label)/255).astype(np.int32)
        # image = image.permute(2,0,1)
        # label = torch.from_numpy(label.astype(np.float32))
        # label_h, label_w = label.shape
        # low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        # low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        # sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label}
        if self.transform is not None:
            image, label= sample['image'], sample['label']
            image = self.transform(image)
        else:
            image, label= sample['image'], sample['label']
            image = np.array(image)
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2,0,1)
            
        label = (np.array(label)/255).astype(np.int32)
        label = torch.from_numpy(label.astype(np.float32))
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label}
        return sample

class ImageFolder(Dataset):
    def __init__(self, path, size=1024, repeat=1, cache='none', mask=False):
        self.cache = cache
        self.path = path
        self.mask = mask
        self.filenames = sorted([f for f in os.listdir(path) if f != 'points.json'])
        self.files = []

        for filename in self.filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

        if mask==True:
            with open(path + '/points.json', 'r', encoding='utf-8') as f:
                self.dict_points = json.load(f)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            x = self.img_process(x)
        
        if self.mask==True:
            points = self.dict_points[self.filenames[idx % len(self.filenames)]]
            return x, points
        else:
            return x
        
    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

class SAM_dataset(Dataset):
    def __init__(self, dataset_location, transform=None, inp_size=1024, type='train'):
        self.dataset = PairedImageFolders(dataset_location+"/images", dataset_location+"/masks")
        self.inp_size = inp_size
        self.type = type
        self.transform = transform
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor()
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        mask, points = mask
        x, y = points[0], points[1]
        points[0], points[1] = torch.as_tensor(x, dtype=torch.float), torch.as_tensor(y, dtype=torch.float)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
        sample = {'image': img, 'label': mask}

        if sample!=None:
            sample = self.transform(sample)
        sample['point'] = points

        return sample