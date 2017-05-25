import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from constant import *
from variation.multi_classes_folder import MultipleClassImageFolder

cudnn.benchmark = True


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    epoch_num = 300
    batch_size = 32
    num_classes = 17
    nz = 100
    ngf = 64
    ndf = 64
    nc = 3

    if not os.path.exists(gan_ckpt):
        os.mkdir(gan_ckpt)

    train_set = MultipleClassImageFolder(split_train_dir, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    netG = _NetG().cuda()
    netG.apply(weights_init)

    netD = _NetD().cuda()
    netD.apply(weights_init)

    criterion = nn.BCELoss().cuda()

    input = Variable(torch.FloatTensor(batch_size, 3, 256, 256)).cuda()
    noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)).cuda()
    d_label = Variable(torch.FloatTensor(batch_size)).cuda()

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=1e-1)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-1)

    for epoch in range(epoch_num):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            d_label.data.resize_(batch_size).fill_(real_label)

            output = netD(input, Variable(label).cuda())
            errD_real = criterion(output, d_label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            label.resize_(batch_size, num_classes, 1, 1)
            label = Variable(label).cuda()
            fake = netG(torch.cat((label, noise), 1))
            d_label.data.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, d_label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            d_label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, d_label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epoch_num, i, len(train_loader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % gan_ckpt,
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d.png' % (gan_ckpt, epoch),
                                  normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (gan_ckpt, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (gan_ckpt, epoch))


def train_d():
    pass


def train_g():
    pass


# custom weights initialization called on netG and netD
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _NetG(nn.Module):
    def __init__(self, num_classes=17):
        super(_NetG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_classes + nz, ngf * 8, 16, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class _NetD(nn.Module):
    def __init__(self, num_classes=17):
        super(_NetD, self).__init__()
        self.conv = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (1) x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 + num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        out = self.main(input).view(input.size(0), -1)
        out = self.classifier(torch.cat((label, out), 1))
        return out
