import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def metname_to_metcode(metname):
    met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']
    metname = list(metname)
    metcode = np.zeros((1, len(met_list)))
    idx = met_list.index(metname[0])
    metcode[0, idx] = 1.0
    return metcode


def train_D(netD, real_data, fake_data, T1, flair, met_LR, upscale_factor, metname):
    upscale_factor = torch.from_numpy(np.array([upscale_factor])).float().cuda()
    metcode = metname_to_metcode(metname)
    metcode = torch.from_numpy(metcode).float().cuda()
    #print(metname, metcode)
    """Calculate GAN loss for the discriminator"""
    x = F.interpolate(met_LR.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    input_data = torch.cat((T1, flair, x), dim=1)

    fake_data = F.interpolate(fake_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    fake = torch.cat((input_data, fake_data), 1)
    pred_fake = netD(fake.detach(), upscale_factor, metcode)
    loss_D_fake = pred_fake.mean()

    real_data = F.interpolate(real_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    real = torch.cat((input_data, real_data), 1)
    pred_real = netD(real, upscale_factor, metcode)
    loss_D_real = pred_real.mean()

    gradient_penalty = calc_gradient_penalty(netD, real, fake, upscale_factor, metcode)

    loss_D = loss_D_fake - loss_D_real + 10 * gradient_penalty
    W_dist = loss_D_fake - loss_D_real
    return loss_D, W_dist


def train_G(netD, real_data, fake_data, T1, flair, met_LR, upscale_factor, metname):
    upscale_factor = torch.from_numpy(np.array([upscale_factor])).float().cuda()
    metcode = metname_to_metcode(metname)
    metcode = torch.from_numpy(metcode).float().cuda()
    """Calculate GAN loss for the generator"""
    x = F.interpolate(met_LR.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    input_data = torch.cat((T1, flair, x), dim=1)

    fake_data = F.interpolate(fake_data.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)
    fake = torch.cat((input_data, fake_data), 1)
    pred_fake = netD(fake, upscale_factor, metcode)
    loss_G = -pred_fake.mean()
    return loss_G


def calc_gradient_penalty(netD, real_data, fake_data, upscale_factor, metcode, device='cuda'):
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
    upscale_factor.requires_grad_(True)
    metcode.requires_grad_(True)
    disc_interpolates = netD(interpolates, upscale_factor, metcode)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=(interpolates, upscale_factor, metcode),
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

    def forward(self, x, upscale_factor, metcode):
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


class Discriminator_uf(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=hidden_size, bias=False, kernel_size=(4,4,3), stride=(2,2,3), padding=(1,1,0))
        self.fs = FilterScaling(in_channels, hidden_size, latent_dim)
        self.bn = nn.InstanceNorm3d(hidden_size)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(in_channels=hidden_size, out_channels=hidden_size * 2, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs2 = FilterScaling(hidden_size, hidden_size * 2, latent_dim)
        self.bn2 = nn.InstanceNorm3d(hidden_size * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv3d(in_channels=hidden_size * 2, out_channels=hidden_size * 4, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs3 = FilterScaling(hidden_size * 2, hidden_size * 4, latent_dim)
        self.bn3 = nn.InstanceNorm3d(hidden_size * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv3d(in_channels=hidden_size * 4, out_channels=hidden_size * 8, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs4 = FilterScaling(hidden_size * 4, hidden_size * 8, latent_dim)
        self.bn4 = nn.InstanceNorm3d(hidden_size * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.output1 = nn.Linear(hidden_size * 8 * 12 * 12, 16)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.output2 = nn.Linear(16, 1)

    def forward(self, x, upscale_factor, metcode):
        x = F.conv3d(x, self.conv1.weight * self.fs(upscale_factor), bias=None, padding=(1,1,0), stride=(2,2,3))
        x = self.bn(x)
        x = self.relu1(x)

        x = F.conv3d(x, self.conv2.weight * self.fs2(upscale_factor), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn2(x)
        x = self.relu2(x)

        x = F.conv3d(x, self.conv3.weight * self.fs3(upscale_factor), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn3(x)
        x = self.relu3(x)

        x = F.conv3d(x, self.conv4.weight * self.fs4(upscale_factor), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn4(x)
        x = self.relu4(x)

        return self.output2(self.act(self.output1(x.view(x.size(0), -1))))


class Discriminator_uf_met(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=hidden_size, bias=False, kernel_size=(4,4,3), stride=(2,2,3), padding=(1,1,0))
        self.fs = FilterScalingMet(in_channels, hidden_size, latent_dim)
        self.bn = nn.InstanceNorm3d(hidden_size)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(in_channels=hidden_size, out_channels=hidden_size * 2, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs2 = FilterScalingMet(hidden_size, hidden_size * 2, latent_dim)
        self.bn2 = nn.InstanceNorm3d(hidden_size * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv3d(in_channels=hidden_size * 2, out_channels=hidden_size * 4, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs3 = FilterScalingMet(hidden_size * 2, hidden_size * 4, latent_dim)
        self.bn3 = nn.InstanceNorm3d(hidden_size * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv3d(in_channels=hidden_size * 4, out_channels=hidden_size * 8, bias=False, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        self.fs4 = FilterScalingMet(hidden_size * 4, hidden_size * 8, latent_dim)
        self.bn4 = nn.InstanceNorm3d(hidden_size * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.output1 = nn.Linear(hidden_size * 8 * 12 * 12, 16)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.output2 = nn.Linear(16, 1)

    def forward(self, x, upscale_factor, metcode):
        x = F.conv3d(x, self.conv1.weight * self.fs(upscale_factor, metcode), bias=None, padding=(1,1,0), stride=(2,2,3))
        x = self.bn(x)
        x = self.relu1(x)

        x = F.conv3d(x, self.conv2.weight * self.fs2(upscale_factor, metcode), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn2(x)
        x = self.relu2(x)

        x = F.conv3d(x, self.conv3.weight * self.fs3(upscale_factor, metcode), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn3(x)
        x = self.relu3(x)

        x = F.conv3d(x, self.conv4.weight * self.fs4(upscale_factor, metcode), bias=None, padding=(1,1,0), stride=(2,2,1))
        x = self.bn4(x)
        x = self.relu4(x)

        return self.output2(self.act(self.output1(x.view(x.size(0), -1))))


class FilterScaling(nn.Module):
    def __init__(self, in_channel, out_channel, latent_dim):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mapping = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.style = nn.Linear(latent_dim, self.out_channel * self.in_channel)

    def forward(self, hyperparameter):
        latent_fea = self.mapping(hyperparameter)
        out = self.style(latent_fea).squeeze().reshape(self.out_channel, self.in_channel).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return out


class FilterScalingMet(nn.Module):
    def __init__(self, in_channel, out_channel, latent_dim):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mapping1 = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.mapping2 = nn.Sequential(
            nn.Linear(7, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.style1 = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.style_relu = nn.LeakyReLU(0.2)
        self.style2 = nn.Linear(latent_dim * 2, self.out_channel * self.in_channel)

    def forward(self, hyperparameter, metcode):
        latent_fea1 = self.mapping1(hyperparameter).unsqueeze(0)
        latent_fea2 = self.mapping2(metcode)
        out = self.style2(self.style_relu(self.style1(torch.cat((latent_fea1, latent_fea2), dim=1))))
        out = out.squeeze().reshape(self.out_channel, self.in_channel).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return out