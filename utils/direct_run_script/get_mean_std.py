from PIL import Image

import numpy as np

from configuration import *

img_list = [img for img in os.listdir(train_dir) if img[-4:] != '.csv']

img_num = len(img_list)
r_sum = g_sum = b_sum = 0
for idx, img_name in enumerate(img_list, 0):
    img_data = np.array(Image.open(os.path.join(train_dir, img_name)).convert('RGB'))
    r_sum += np.sum(img_data[:, :, 0])
    g_sum += np.sum(img_data[:, :, 1])
    b_sum += np.sum(img_data[:, :, 2])
    print 'calculating mean: %d / %d' % (idx + 1, img_num)

r_mean = r_sum / (256 * 256 * img_num)
g_mean = g_sum / (256 * 256 * img_num)
b_mean = b_sum / (256 * 256 * img_num)

r_var_sum = g_var_sum = b_var_sum = 0
for idx, img_name in enumerate(img_list, 0):
    img_data = np.array(Image.open(os.path.join(train_dir, img_name)).convert('RGB'))
    r_var_sum += np.sum(np.square(img_data[:, :, 0] - r_mean))
    g_var_sum += np.sum(np.square(img_data[:, :, 1] - g_mean))
    b_var_sum += np.sum(np.square(img_data[:, :, 2] - b_mean))
    print 'calculating std: %d / %d' % (idx + 1, img_num)

r_std = np.sqrt(r_var_sum / (256 * 256 * img_num))
g_std = np.sqrt(g_var_sum / (256 * 256 * img_num))
b_std = np.sqrt(b_var_sum / (256 * 256 * img_num))

print 'r_mean: %.3f, g_mean: %.3f, b_mean: %.3f' % (r_mean / 255, g_mean / 255, b_mean / 255)  # [0.311, 0.340, 0.299]
print 'r_std: %.3f, g_std: %.3f, b_std: %.3f' % (r_std / 255, g_std / 255, b_std / 255)  # [0.167, 0.144, 0.138]

