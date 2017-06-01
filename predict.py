import numpy as np
import pandas as pd
import scipy.io as sio
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.folder_eval import ImageFolderEval
from utils.models import *

net = get_res152(num_classes=num_classes,
                   snapshot_path=os.path.join(ckpt_path, 'epoch_15_validation_loss_0.0772_iter_1000.pth')).cuda()
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
])

thretholds = [-1.650, -1.210, -1.556, -1.384, -0.981, -1.203, -1.260, -2.154, -1.548, -1.831, -1.241, -1.938, -1.832,
              -1.300, -1.691, -1.782, -1.431]

predictions = []
outputs_all = []
predictions_label = []
test_img_names = []

batch_size = 352
test_img_num = len(os.listdir(os.path.join(test_dir, 'test-jpg')))

data_set = ImageFolderEval(test_dir, transform)
data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=16)
for i, data in enumerate(data_loader, 0):
    img_names, imgs = data
    test_img_names.extend(img_names)
    imgs = Variable(imgs, volatile=True).cuda()

    outputs = net(imgs)

    outputs = outputs.cpu().data.numpy()
    outputs_all.extend(outputs)
    for o in outputs:
        prediction = ' '.join([classes[c] for c, v in enumerate(o) if v >= thretholds[c]])
        predictions.append(prediction)
    prediction_label = np.zeros_like(outputs)
    prediction_label[outputs >= thretholds] = 1
    predictions_label.extend(prediction_label)

    print 'predict %d / %d images' % ((i + 1) * batch_size, test_img_num)

test_img_names_without_ext = [os.path.splitext(i)[0] for i in test_img_names]

res_df = pd.DataFrame(predictions, columns=['tags'])
res_df.insert(0, column='image_name', value=pd.Series(test_img_names_without_ext))
res_df.to_csv('./res_results.csv', index=False)

# save these for later model fusion
sio.savemat('./res_outputs.mat', {'outputs': np.array(outputs_all), 'imgs': np.array(test_img_names_without_ext)})
sio.savemat('./res_predictions.mat',
            {'predictions': np.array(predictions_label), 'imgs': np.array(test_img_names_without_ext)})

