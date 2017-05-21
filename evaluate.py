import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from constant import *
from multi_classes_folder import MultipleClassImageFolder
from planet import Planet


def main():
    training_batch_size = 256
    validation_batch_size = 256

    net = Planet(use_sigmoid=True).cuda()
    net.load_state_dict(
        torch.load(ckpt_path + '/epoch_1_validation_loss_0.0794413983822_iter_xx_training_loss_0.169523105025.pth'))
    net.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = MultipleClassImageFolder(split_train_dir, transform)
    train_loader = DataLoader(train_set, batch_size=training_batch_size)
    val_set = MultipleClassImageFolder(split_val_dir, transform)
    val_loader = DataLoader(val_set, batch_size=validation_batch_size)
    criterion = nn.MultiLabelSoftMarginLoss().cuda()

    batch_outputs, batch_labels = predict(net, train_loader)
    loss = criterion(batch_outputs, batch_labels)
    print 'training loss %.4f' % loss.data.numpy()[0]
    batch_outputs = batch_outputs.cpu().data.numpy()
    batch_labels = batch_labels.cpu().data.numpy()
    thretholds = find_best_threthold(batch_outputs, batch_labels)

    batch_outputs, batch_labels = predict(net, val_loader)
    loss = criterion(batch_outputs, batch_labels)
    print 'validation loss %.4f' % loss.data.numpy()[0]
    batch_outputs = batch_outputs.cpu().data.numpy()
    batch_labels = batch_labels.cpu().data.numpy()
    prediction = get_one_hot_prediction(batch_outputs, thretholds)
    evaluation = evaluate(prediction, batch_labels)
    print 'validation evaluation: accuracy %.4f, precision %.4f, recall %.4f, f2 %.4f' % (
        evaluation[0], evaluation[1], evaluation[2], evaluation[3])


def predict(net, loader):
    batch_outputs = []
    batch_labels = []
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels.float(), volatile=True).cuda()

        _, out = net(inputs)

        batch_outputs.append(out)
        batch_labels.append(labels)
        print '%d image predicted' % i

    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    return batch_outputs, batch_labels


def evaluate(one_hot_prediction, one_hot_label):
    accuracy = np.mean(one_hot_prediction == one_hot_label)
    tp = np.sum(one_hot_prediction * one_hot_label)
    fp = np.sum(one_hot_prediction) - tp
    fn = np.sum((1 - one_hot_prediction) * one_hot_label)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta = 2
    f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    return accuracy, precision, recall, f2


def get_one_hot_prediction(soft_output, thretholds):
    prediction = np.zeros_like(soft_output)
    prediction[soft_output >= thretholds] = 1
    return prediction


def find_best_threthold(soft_output, one_hot_label):
    accuracy = precision = recall = best_f2 = 1e9
    thretholds = np.zeros((soft_output.shape[1]))
    for i in xrange(thretholds.shape[0]):
        for t in np.arange(0, 1, 0.01):
            thretholds_tmp = thretholds.copy()
            thretholds_tmp[i] = t
            prediction = get_one_hot_prediction(soft_output, thretholds_tmp)
            evaluation = evaluate(prediction, one_hot_label)
            if evaluation[3] > best_f2:
                accuracy, precision, recall, best_f2 = evaluation
                thretholds[i] = t
    print 'best evaluation: accuracy %.4f, precision %.4f, recall %.4f, f2 %.4f' % (
    accuracy, precision, recall, best_f2)
    return thretholds


if __name__ == '__main__':
    main()
