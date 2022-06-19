import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
import numpy as np


def data_consistency(met_LR, output, lowRes):
    met_LR_cplx = torch.cat((met_LR.unsqueeze(-1), torch.zeros_like(met_LR.unsqueeze(-1))), -1)
    subkspace = utils.fft2(met_LR_cplx)
    output_cplx = torch.cat((output.unsqueeze(-1), torch.zeros_like(output.unsqueeze(-1))), -1)
    output_kspace = utils.fft2(output_cplx)
    d = output.shape[-1] // 2
    div = lowRes
    output_kspace[:, :, d - div:d + div, d - div:d + div, :] = subkspace[:, :, d - div:d + div, d - div:d + div, :]
    output_cplx = utils.ifft2(output_kspace)
    output = torch.sqrt(output_cplx[:, :, :, :, 0] ** 2 + output_cplx[:, :, :, :, 1] ** 2)
    return output


class MUNet_AMLayer(nn.Module):

    def __init__(self, in_channels, init_features, latent_dim, out_channels=1):
        super(MUNet_AMLayer, self).__init__()
        features = init_features
        self.encoder1 = Encoder(in_channels=in_channels, init_features=features, latent_dim=latent_dim)
        self.encoder2 = Encoder(in_channels=in_channels, init_features=features, latent_dim=latent_dim)

        self.conv4 = nn.Conv3d(in_channels=features * 16 * 2, out_channels=features * 8, kernel_size=(3,3,1), stride=1, padding=(1,1,0))
        self.decoder4 = BasicConv3d(features * 8 * 3, features * 8, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.conv3 = nn.Conv3d(in_channels=features * 8, out_channels=features * 4, kernel_size=(3,3,1), stride=1, padding=(1,1,0))
        self.decoder3 = BasicConv3d(features * 4 * 3, features * 4, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.conv2 = nn.Conv3d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3,3,1), stride=1, padding=(1,1,0))
        self.decoder2 = BasicConv3d(features * 2 * 3, features * 2, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.conv1 = nn.Conv3d(in_channels=features * 2, out_channels=features, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.decoder1 = BasicConv3d(features * 3, features, latent_dim, kernel_size=(3,3,3), padding=(1,1,1))

        self.conv_output = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.global_pool = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=0)

    def forward(self, T1, flair, met_LR, lowRes, lowRes_input, met_code, adv_weight):
        lowRes_ = lowRes
        lowRes = torch.from_numpy(np.array([lowRes_input])).float().cuda()

        x = F.interpolate(met_LR.unsqueeze(-1), size=T1.shape[2:], mode='trilinear', align_corners=True)

        enc1_1, enc1_2, enc1_3, enc1_4, bottleneck1 = self.encoder1(torch.cat((x, T1), dim=1), lowRes)
        enc2_1, enc2_2, enc2_3, enc2_4, bottleneck2 = self.encoder2(torch.cat((x, flair), dim=1), lowRes)

        bottleneck = torch.cat((bottleneck1, bottleneck2), dim=1)
        dec4 = self.conv4(F.interpolate(bottleneck, scale_factor=(2,2,1), mode='trilinear', align_corners=True))
        enc4 = torch.cat((enc1_4, enc2_4), dim=1)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4, lowRes)
        dec3 = self.conv3(F.interpolate(dec4, scale_factor=(2,2,1), mode='trilinear', align_corners=True))
        enc3 = torch.cat((enc1_3, enc2_3), dim=1)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3, lowRes)
        dec2 = self.conv2(F.interpolate(dec3, scale_factor=(2,2,1), mode='trilinear', align_corners=True))
        enc2 = torch.cat((enc1_2, enc2_2), dim=1)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2, lowRes)
        dec1 = self.conv1(F.interpolate(dec2, scale_factor=(2,2,3), mode='trilinear', align_corners=True))
        enc1 = torch.cat((enc1_1, enc2_1), dim=1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1, lowRes)
        output = self.conv_output(dec1)
        output = torch.sigmoid(output)
        output = self.global_pool(output).squeeze(-1)
        output = data_consistency(met_LR, output, lowRes_)
        return output


class Encoder(nn.Module):

    def __init__(self, in_channels, init_features, latent_dim):
        super(Encoder, self).__init__()

        features = init_features
        self.encoder1 = BasicConv3d(in_channels, features, latent_dim, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,3), stride=(2,2,3))
        self.encoder2 = BasicConv3d(features, features * 2, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.encoder3 = BasicConv3d(features * 2, features * 4, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.encoder4 = BasicConv3d(features * 4, features * 8, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))

        self.bottleneck = BasicConv3d(features * 8, features * 16, latent_dim, kernel_size=(3,3,1), padding=(1,1,0))

    def forward(self, x, lowRes):
        enc1 = self.encoder1(x, lowRes)
        enc1_pool = self.pool1(enc1)
        enc2 = self.encoder2(enc1_pool, lowRes)
        enc2_pool = self.pool2(enc2)
        enc3 = self.encoder3(enc2_pool, lowRes)
        enc3_pool = self.pool3(enc3)
        enc4 = self.encoder4(enc3_pool, lowRes)
        enc4_pool = self.pool4(enc4)
        bottleneck = self.bottleneck(enc4_pool, lowRes)
        return enc1, enc2, enc3, enc4, bottleneck


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2))
        self.style = nn.Linear(latent_dim, in_channel)

        self.mapping2 = nn.Sequential(
            nn.Linear(in_channel * 2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, in_channel))

        self.mapping3 = nn.Sequential(
            nn.Linear(in_channel * 2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, in_channel))

        self.norm = nn.InstanceNorm3d(in_channel)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, hyperparameter):
        latent_fea = self.mapping(hyperparameter)

        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = self.style(latent_fea)
        mean, std = torch.mean(input, dim=(2, 3, 4)), torch.std(input, dim=(2, 3, 4))
        beta = self.mapping2(torch.cat((style, mean), dim=1))
        gamma = self.mapping3(torch.cat((style, std), dim=1))
        gamma = gamma.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        beta = beta.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        out = self.norm(input)
        out = (1. + gamma) * out + beta

        return out


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, latent_dim, kernel_size, padding):
        super(BasicConv3d, self).__init__()
        self.padding = padding

        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn = ConditionalInstanceNorm(out_channels, latent_dim)
        self.relu = nn.PReLU(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn2 = ConditionalInstanceNorm(out_channels, latent_dim)
        self.relu2 = nn.PReLU(out_channels)

    def forward(self, x, upscale_factor):
        x = F.conv3d(x, self.conv.weight, bias=None, padding=self.padding)
        x = self.bn(x, upscale_factor.expand(x.shape[0], 1))
        x = self.relu(x)

        x = F.conv3d(x, self.conv2.weight, bias=None, padding=self.padding)
        x = self.bn2(x, upscale_factor.expand(x.shape[0], 1))
        x = self.relu2(x)

        return x