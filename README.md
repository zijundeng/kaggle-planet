# Kaggle Planet competition code implemented in PyTorch

1. This is for the [competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) and implemented
in PyTorch

2. The model is based on ResNet152 and using a pretrained one.

3. Thanks for the nice [lr scheduler](https://github.com/pytorch/pytorch/pull/1370) provided by Jiaming Liu.


# Training tricks:
1. Use lr=1e-2 to warm up the training. At this time the log freq should be high for better monitoring.

2. Firstly crop the image. When the training is nearly finished, cancel the crop setting for possible improvement of
the model.

3. The weight decay is used periodically

4. Use SGD + Momentum + Nesterov. Don't use Adam or others.

5. When the training is nearly at the end, use very high log frequency to find best model.

6. Initially use small batch size for fast convergence. Later on increase the batch size for more steady training.

