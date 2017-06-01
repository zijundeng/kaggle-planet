import os
import os.path
from PIL import Image

import numpy as np
import pandas as pd
import torch.utils.data as data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    img_to_class_csvs = [d for d in os.listdir(dir) if d.endswith('.csv')]
    classes = []
    for csv in img_to_class_csvs:
        class_content = pd.read_csv(os.path.join(dir, csv)).iloc[:, 1].values
        for e in class_content:
            classes.extend(e.split(' '))
    classes = np.unique(classes).tolist()
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_train_dataset(dir, classes, class_to_idx):
    num_classes = len(classes)
    img_to_class_csvs = [d for d in os.listdir(dir) if d.endswith('.csv')]
    img_to_classes = {}
    for csv in img_to_class_csvs:
        csv_content = pd.read_csv(os.path.join(dir, csv)).values
        for c in csv_content:
            classes_one_hot = np.zeros(num_classes)
            for class_name in c[1].split(' '):
                classes_one_hot[class_to_idx[class_name]] = 1
            img_to_classes[c[0]] = classes_one_hot

    images = []
    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            fname = os.path.splitext(fname)[0]
            if fname in img_to_classes:
                item = (path, img_to_classes[fname])
                images.append(item)
            else:
                print "Cannot find class information of image " + fname + ", skip"

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class MultipleClassImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_train_dataset(root, classes, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folder: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
