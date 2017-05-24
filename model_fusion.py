import numpy as np
import pandas as pd
import scipy.io as sio

from constant import *

test_img_names_without_ext = [os.path.splitext(i)[0] for i in os.listdir(test_dir)]
res = sio.loadmat('./res.mat')['prediction']
inception = sio.loadmat('./inception.mat')['prediction']
vgg = sio.loadmat('./vgg.mat')['prediction']
dense = sio.loadmat('./dense.mat')['prediction']

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy',
           'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
           'selective_logging', 'slash_burn', 'water']

total = res + inception + vgg + dense

result = np.zeros_like(total)
result[total >= 2] = 1

predictions = []
for out in result:
    prediction = ' '.join([classes[c] for c, v in enumerate(out) if v == 1])
    predictions.append(prediction)

res_df = pd.DataFrame(predictions, columns=['tags'])
res_df.insert(0, column='image_name', value=pd.Series(test_img_names_without_ext))

res_df.to_csv('./fusion_2.csv', index=False)
