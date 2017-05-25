import numpy as np
import pandas as pd
import scipy.io as sio
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from folder import ImageFolder

net = get_res152(snapshot_path='xxx').cuda()
net.eval()

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy',
           'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
           'selective_logging', 'slash_burn', 'water']

transform = transforms.Compose([
    transforms.Scale(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# thretholds = [-1.68, -2.06, -1.78, -2.1, -3.09, -0.98, -1.35, -2.26, -1.49, -1.88, -1.33, -2.44, -1.64, -1.71, -1.63, 0., -1.55]  # res
# thretholds = [-1.43, -1.45, -1.12, -2.54,  0., -1.09, -1.06, -2.13, -1.6, -1.84, -1.29, -2.16, -1.86, -1.57, -1.89, -2.04, -1.27]  # inception
# thretholds = [-1.39, -0.59, -1.45, -1.17, -1.95, -1.91, -1.08, -1.21, -1.38, -1.22, -1.29, -1.18, -1.68, -1.19, -1.23, -1.25, -1.24]  # vgg
thretholds = [-1.68, -1.63, -1.53, -2.07, -3.02, -1.52, -1.33, -2.79, -1.55, -1.83, -1.3, -2.09, -1.8, -1.5, -1.87, -1.59, -1.65]  # dense

predictions = []
predictions_label = []
test_img_names = os.listdir(test_dir)
test_img_names_without_ext = [os.path.splitext(i)[0] for i in test_img_names]
test_img_num = len(test_img_names)

data_set = ImageFolder(split_val_dir, transform)
data_loader = DataLoader(data_set, batch_size=256, num_workers=8)
for i, data in enumerate(data_loader, 0):
    img_names, imgs, _ = data
    imgs = Variable(imgs, volatile=True).cuda()

    outputs = net(imgs)

    outputs = outputs.squeeze(0).cpu().data.numpy()
    prediction = ' '.join([classes[c] for c, v in enumerate(outputs) if v >= thretholds[c]])
    prediction_label = np.zeros_like(outputs)
    prediction_label[outputs >= thretholds] = 1

    predictions.append(prediction)
    predictions_label.append(prediction_label)

    print 'predict %d / %d images' % (i + 1, test_img_num)

sio.savemat('./dense.mat', {'prediction': np.array(predictions_label)})

res_df = pd.DataFrame(predictions, columns=['tags'])
res_df.insert(0, column='image_name', value=pd.Series(test_img_names_without_ext))

res_df.to_csv('./dense.csv', index=False)
