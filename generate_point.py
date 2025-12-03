import torch
import json
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def draw_image(image, points=None, alpha=0.4):
    image = torch.tensor(np.array(image))
    image = image.unsqueeze(0)
    print(image.size())
    if points is not None:
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.uint8)
        print(points)
        for point in range(points[0].size(0)):
            now = points[0][point].numpy()  
            print(now)
            x, y = int(now[0]), int(now[1])  
            image = np.ascontiguousarray(image)
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        return image


def img_process(file):
    return Image.open(file).convert('L')

def process(label):
    label = (np.array(label)/255).astype(np.int32)
    assert (np.max(label) == 1 and np.min(label) == 0.0) or (np.sum(label)==0.), f"{label} {len(label)} {np.max(label)} \
        {np.min(label)} ground truth should be 0, 1"
    label = torch.from_numpy(label.astype(np.float32))
    points = init_point_sampling(label)
    return points

def init_point_sampling(mask, get_point=3):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if len(mask.size())>2:
        mask = mask[0]
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point #// 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels


def main(data):
    for name in data:
        dict_point = dict()
        print(name, data[name])
        path = data[name] + '/masks'
        filenames = sorted(os.listdir(path))
        for idx in range(len(filenames)):
            if 'points.json' in filenames[idx]:continue
            path_idx = path + '/' + filenames[idx]
            image = img_process(path_idx)
            image = transforms.Resize((1024, 1024), interpolation=InterpolationMode.NEAREST)(image)
            points = process(image)
            x, y = points[0].tolist(), points[1].tolist()
            dict_point[filenames[idx]] = (x, y)

            # image_output = draw_image(image, points)
            # cv2.imwrite('test.png', image_output)
        with open(path+'/points.json', 'w') as f:
            json.dump(dict_point, f)

train_data, test_data = None, None
with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
with open('datasets/datasets_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

main(train_data)
main(test_data)
