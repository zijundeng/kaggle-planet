import os

ckpt_path = './ckpt'
pytorch_pretrained_root = '/media/b3-542/LIBRARY/ZijunDeng/PyTorch Pretrained'
pretrained_res152_path = os.path.join(pytorch_pretrained_root, 'ResNet', 'resnet152-b121ed2d.pth')
pretrained_inception_v3_path = os.path.join(pytorch_pretrained_root, 'Inception', 'inception_v3_google-1a9a5a14.pth')

ori_data_dir = '/home/b3-542/KaggleAmazon'

train_dir = os.path.join(ori_data_dir, 'train-jpg')
split_train_dir = os.path.join(ori_data_dir, 'train')
split_val_dir = os.path.join(ori_data_dir, 'val')
test_dir = os.path.join(ori_data_dir, 'test-jpg')

gan_ckpt = './gan_ckpt'

num_classes = 17