from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from constant import *
from models import *
from multi_classes_folder import MultipleClassImageFolder
from transforms import RandomVerticalFlip

cudnn.benchmark = True


def main():
    training_batch_size = 16
    validation_batch_size = 8
    epoch_num = 100
    iter_freq_print_training_log = 10
    iter_freq_validate = 30
    lr = 1e-2
    weight_decay = 1e-4

    net = get_inception_v3(num_classes=num_classes, pretrained=True).cuda()
    net.train()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        RandomVerticalFlip(),
        transforms.RandomCrop(224),  # initially crop, later on cancel it
        transforms.Scale(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = MultipleClassImageFolder(split_train_dir, transform)
    train_loader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True, num_workers=8)
    val_set = MultipleClassImageFolder(split_val_dir, transform)
    val_loader = DataLoader(val_set, batch_size=validation_batch_size, shuffle=True, num_workers=8)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    info = [1e9, 0, 0]

    for epoch in range(0, epoch_num):
        train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log, iter_freq_validate,
              val_loader, info)
        if epoch % 2 == 0:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0
        else:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = weight_decay


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

        batch_outputs.append(outputs[-1])
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
