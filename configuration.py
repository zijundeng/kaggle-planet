import os

ckpt_path = './ckpt'

ori_data_dir = '/home/b3-542/KaggleAmazon'

train_dir = os.path.join(ori_data_dir, 'train-jpg')
split_train_dir = os.path.join(ori_data_dir, 'train')
split_val_dir = os.path.join(ori_data_dir, 'val')
test_dir = os.path.join(ori_data_dir, 'test')

num_classes = 17

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy',
           'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
           'selective_logging', 'slash_burn', 'water']

# official pretrained model
pytorch_pretrained_root = '/media/b3-542/LIBRARY/ZijunDeng/PyTorch Pretrained'
pretrained_res152_path = os.path.join(pytorch_pretrained_root, 'ResNet', 'resnet152-b121ed2d.pth')
pretrained_inception_v3_path = os.path.join(pytorch_pretrained_root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
pretrained_vgg19_path = os.path.join(pytorch_pretrained_root, 'VggNet', 'vgg19-dcbb9e9d.pth')
pretrained_dense201_path = os.path.join(pytorch_pretrained_root, 'DenseNet', 'densenet201-4c113574.pth')
