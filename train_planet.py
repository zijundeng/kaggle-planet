from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from variation import MultipleClassImageFolder
from variation import ReduceLROnPlateau

cudnn.benchmark = True


def main():
    training_batch_size = 8
    validation_batch_size = 8
    epoch_num = 500
    iter_freq_print_training_log = 800

    net = get_res152(snapshot_path=ckpt_path+'/epoch_21_validation_loss_0.0922.pth').cuda()
    net.train()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = MultipleClassImageFolder(split_train_dir, transform)
    train_loader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True, num_workers=8)
    val_set = MultipleClassImageFolder(split_val_dir, transform)
    val_loader = DataLoader(val_set, batch_size=validation_batch_size, shuffle=True, num_workers=8)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=8, verbose=True)

    best_val_loss = 1e9
    best_epoch = 0

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    for epoch in range(21, epoch_num):
        train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log)
        val_loss = validate(val_loader, net, criterion)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        print '[best_val_loss %.4f], [best_epoch %d]' % (best_val_loss, best_epoch)
        print '--------------------------------------------------------'
        torch.save(net.state_dict(), ckpt_path + '/epoch_%d_validation_loss_%.4f.pth' % (epoch + 1, val_loss))

        scheduler.step(val_loss)


def train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels.float()).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % iter_freq_print_training_log == 0:
            print '[epoch %d], [iter %d], [training_batch_loss %.4f]' % (
                epoch + 1, i + 1, loss.data[0])


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
