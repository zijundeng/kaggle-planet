import os

ckpt_path = './ckpt'

ori_data_dir = '/home/b3-542/KaggleAmazon'

train_dir = os.path.join(ori_data_dir, 'train-jpg')
split_train_dir = os.path.join(ori_data_dir, 'train')
split_val_dir = os.path.join(ori_data_dir, 'val')
test_dir = os.path.join(ori_data_dir, 'test-jpg')

num_classes = 17

