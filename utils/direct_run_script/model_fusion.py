import numpy as np
import pandas as pd
import scipy.io as sio

from configuration import *

test_img_names_without_ext = [os.path.splitext(i)[0] for i in os.listdir(test_dir)]

# load the saved prediction
res = sio.loadmat('./res_predictions.mat')['predictions']
inception = sio.loadmat('./inception_predictions.mat')['predictions']
vgg = sio.loadmat('./vgg_predictions.mat')['predictions']
dense = sio.loadmat('./dense_predictions.mat')['predictions']

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
