import os
import os.path
from PIL import Image

import torch.utils.data as data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderEval(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return os.path.split(path)[1], img

    def __len__(self):
        return len(self.imgs)
