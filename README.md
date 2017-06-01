# Kaggle Planet competition

1. This project is for the [competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) and is
implemented in PyTorch.

2. I use resnet-152 and get my best score 0.92563 on Kaggle. I have tried some model fusion tricks but it doesn't work
better.


## Training tricks

1. Firstly randomly crop the image. When the val loss doesn't decrease any more, cancel the crop setting for
possible improvement of the model.

2. The weight decay is used periodically.

3. Use SGD + Momentum + Nesterov.

4. When the validation loss decreases to very low, validate the model more frequently to find the best one.

5. Initially use small batch size for fast convergence. Later on increase the batch size for more steady training.

