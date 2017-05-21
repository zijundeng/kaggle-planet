from PIL import Image

import numpy as np

from constant import *

val_percentage = 0.05
img_list = os.listdir(train_dir)

img_list = np.random.permutation(img_list)

val_data_num = int(len(img_list) * val_percentage)
train_data = img_list[val_data_num:]
val_data = img_list[:val_data_num]

for i, t in enumerate(train_data):
    img_rgb = Image.open(os.path.join(train_dir, t)).convert('RGB')
    img_rgb.save(os.path.join(split_train_dir, t))
    print 'processed %d train images' % i

for i, v in enumerate(val_data):
    img_rgb = Image.open(os.path.join(train_dir, v)).convert('RGB')
    img_rgb.save(os.path.join(split_val_dir, v))
    print 'processed %d val images' % i
