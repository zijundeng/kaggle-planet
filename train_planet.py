from torch import optim, nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from utils import *
from utils import models
from configuration import *

cudnn.benchmark = True


def main():
    training_batch_size = 32
    validation_batch_size = 32
    epoch_num = 100
    iter_freq_print_training_log = 100
    iter_freq_validate = 500
    lr = 1e-2
    weight_decay = 1e-4

    net = models.get_res152(num_classes=num_classes)
    # net = get_res152(num_classes=num_classes, snapshot_path=os.path.join(ckpt_path, 'xxx.pth')).cuda()
    net.train()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
    ])

    train_set = MultipleClassImageFolder(split_train_dir, transform)
    train_loader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True, num_workers=16)
    val_set = MultipleClassImageFolder(split_val_dir, transform)
    val_loader = DataLoader(val_set, batch_size=validation_batch_size, shuffle=True, num_workers=16)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'weight_decay': weight_decay}
    ], lr=lr, momentum=0.9, nesterov=True)

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    info = [1e9, 0, 0]  # [best val loss, epoch, iter]

    for epoch in range(0, epoch_num):
        if epoch % 2 == 1:
            optimizer.param_groups[1]['weight_decay'] = 0
            print 'weight_decay is set to 0'
        else:
            optimizer.param_groups[1]['weight_decay'] = weight_decay
            print 'weight_decay is set to %.4f' % weight_decay
        train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log, iter_freq_validate,
              val_loader, info)


def train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log, iter_freq_validate, val_loader,
          info):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels.float()).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        main_loss = criterion(outputs[-1], labels)
        aux_loss = None
        for aux in outputs[:-1]:
            if aux_loss is None:
                aux_loss = criterion(aux, labels)
            else:
                aux_loss += criterion(aux, labels)
        loss = aux_loss + main_loss
        loss.backward()
        optimizer.step()

        if (i + 1) % iter_freq_print_training_log == 0:
            print '[epoch %d], [iter %d], [training_batch_total_loss %.4f], [training_batch_main_loss %.4f], ' \
                  '[training_batch_aux_loss %.4f]' % (
                      epoch + 1, i + 1, loss.data[0], main_loss.data[0], aux_loss.data[0])

        if (i + 1) % iter_freq_validate == 0:
            val_loss = validate(val_loader, net, criterion)
            if info[0] > val_loss:
                info[0] = val_loss
                info[1] = epoch + 1
                info[2] = i + 1
                torch.save(net.state_dict(),
                           ckpt_path + '/epoch_%d_validation_loss_%.4f_iter_%d.pth' % (epoch + 1, val_loss, i + 1))
            print '[best_val_loss %.4f], [best_epoch %d], [best_iter %d]' % (info[0], info[1], info[2])
            print '--------------------------------------------------------'


def validate(val_loader, net, criterion):
    net.eval()
    batch_outputs = []
    batch_labels = []
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels.float(), volatile=True).cuda()

        outputs = net(inputs)

        batch_outputs.append(outputs)
        batch_labels.append(labels)

    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    val_loss = criterion(batch_outputs, batch_labels)
    val_loss = val_loss.data[0]

    print '--------------------------------------------------------'
    print '[val_loss %.4f]' % val_loss
    net.train()
    return val_loss


if __name__ == '__main__':
    main()
