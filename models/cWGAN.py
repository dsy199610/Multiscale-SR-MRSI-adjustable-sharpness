import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def train_D(netD, real_data, fake_data, T1, flair, met_LR, upscale_factor, metname):
    """Calculate GAN loss for the discriminator"""
    x = F.interpolate(met_LR.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    input_data = torch.cat((T1, flair, x), dim=1)

    fake_data = F.interpolate(fake_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    fake = torch.cat((input_data, fake_data), 1)
    pred_fake = netD(fake.detach())
    loss_D_fake = pred_fake.mean()

    real_data = F.interpolate(real_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    real = torch.cat((input_data, real_data), 1)
    pred_real = netD(real)
    loss_D_real = pred_real.mean()

    gradient_penalty = calc_gradient_penalty(netD, real, fake)

    loss_D = loss_D_fake - loss_D_real + 10 * gradient_penalty
    W_dist = loss_D_fake - loss_D_real
    return loss_D, W_dist


def train_G(netD, real_data, fake_data, T1, flair, met_LR, upscale_factor, metname):
    """Calculate GAN loss for the generator"""
    x = F.interpolate(met_LR.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    input_data = torch.cat((T1, flair, x), dim=1)

    fake_data = F.interpolate(fake_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    fake = torch.cat((input_data, fake_data), 1)
    pred_fake = netD(fake)
    loss_G = -pred_fake.mean()
    return loss_G


def calc_gradient_penalty(netD, real_data, fake_data, device='cuda'):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    Returns the gradient penalty loss
    """
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
    alpha = alpha.to(device)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)

    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1.0) ** 2).mean()  # added eps
    return gradient_penalty


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=hidden_size, bias=False, kernel_size=(4, 4, 3), stride=(2, 2, 3), padding=(1, 1, 0))
        self.bn = nn.InstanceNorm3d(hidden_size)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(in_channels=hidden_size, out_channels=hidden_size * 2, bias=False, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.bn2 = nn.InstanceNorm3d(hidden_size * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv3d(in_channels=hidden_size * 2, out_channels=hidden_size * 4, bias=False, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.bn3 = nn.InstanceNorm3d(hidden_size * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv3d(in_channels=hidden_size * 4, out_channels=hidden_size * 8, bias=False, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.bn4 = nn.InstanceNorm3d(hidden_size * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.output1 = nn.Linear(hidden_size * 8 * 12 * 12, 16)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.output2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.conv3d(x, self.conv1.weight, bias=None, padding=(1,1,0), stride=(2,2,3))
        x = self.bn(x)
        x = self.relu1(x)

        x = F.conv3d(x, self.conv2.weight, bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn2(x)
        x = self.relu2(x)

        x = F.conv3d(x, self.conv3.weight, bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn3(x)
        x = self.relu3(x)

        x = F.conv3d(x, self.conv4.weight, bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn4(x)
        x = self.relu4(x)
        return self.output2(self.act(self.output1(x.view(x.size(0), -1))))