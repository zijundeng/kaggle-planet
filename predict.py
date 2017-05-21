from PIL import Image

import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms

from constant import *
from planet import Planet

net = Planet(use_sigmoid=True).cuda()
net.load_state_dict(
    torch.load(ckpt_path + '/epoch_1_validation_loss_0.0794413983822_iter_xx_training_loss_0.169523105025.pth'))
net.eval()

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy',
           'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
           'selective_logging', 'slash_burn', 'water']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

thretholds = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

predictions = []
test_img_names = os.listdir(test_dir)
test_img_names_without_ext = [os.path.splitext(i)[0] for i in test_img_names]
test_img_num = len(test_img_names)
for i, img_name in enumerate(test_img_names):
    img = Image.open(os.path.join(test_dir, img_name)).convert('RGB')
    img = transform(img).unsqueeze(0)
    inputs = Variable(img, volatile=True).cuda()

    _, out = net(inputs)

    out = out.squeeze(0).cpu().data.numpy()
    prediction = ' '.join([classes[c] for c, v in enumerate(out) if v >= thretholds[c]])
    predictions.append(prediction)

    print 'predict %d / %d images' % (i, test_img_num)

res_df = pd.DataFrame(predictions, columns=['tags'])
res_df.insert(0, column='image_name', value=pd.Series(test_img_names_without_ext))

res_df.to_csv('./res.csv', index=False)
