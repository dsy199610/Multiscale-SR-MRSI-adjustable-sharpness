import numpy as np
import random
import torch
import argparse
import pathlib
from utils import logs
import logging
import shutil
import sys
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import torchvision

from models.MUNet import MUNet
from models.MUNet_AMLayer import MUNet_AMLayer
from models.MUNet_Hypernetworks import MUNet_Hypernetworks
from models.MUNet_FilterScaling import MUNet_FilterScaling
from models.MUNet_FilterScaling_Met import MUNet_FilterScaling_Met
from models.MUNet_FilterScaling_Met_adv import MUNet_FilterScaling_Met_adv
from models.cWGAN import train_D, train_G, Discriminator
from loader.dataloader import twoD_Data, RandomDownscale_function, RandomFlip_function, RandomShift_function, Met_Sampler


def ssimloss(output, target):
    output_ = F.interpolate(output, mode='bicubic', size=(192, 192), align_corners=True)
    target_ = F.interpolate(target, mode='bicubic', size=(192, 192), align_corners=True)
    ssim_loss = 1 - ms_ssim(output_, target_, data_range=torch.max(target_) - torch.min(target_), size_average=True)
    return ssim_loss


def create_data_loaders(args):
    train_Dataset = twoD_Data(patients=args.train_patients, transform=None, mode='train')
    met_sampler = Met_Sampler(batch_size=args.batch_size, length=train_Dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_sampler=met_sampler, num_workers=args.num_workers)
    valid_Dataset = twoD_Data(patients=args.valid_patients, transform=None, mode='valid')
    valid_loader = torch.utils.data.DataLoader(valid_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_Dataset = twoD_Data(patients=args.test_patients, transform=None, mode='test')
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    display_Dataset = [train_Dataset[i] for i in range(0, len(train_Dataset), len(train_Dataset) // 100)]
    display_loader = torch.utils.data.DataLoader(display_Dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader, display_loader


def save_python_script(args):
    ## copy training
    shutil.copyfile(sys.argv[0], str(args.exp_dir) + '/' + sys.argv[0])
    ## copy model
    shutil.copyfile('models/' + args.model + '.py', str(args.exp_dir) + '/' + args.model + '.py')
    ## copy loader
    shutil.copyfile('loader/dataloader.py', str(args.exp_dir) + '/dataloader.py')
    if args.adv_weight_train != 0.0:
        shutil.copyfile('models/cWGAN.py', str(args.exp_dir) + '/cWGAN.py')


def save_model(args, exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss):
    logging.info('Saving trained model')

    ## create a models folder if not exists
    if not (args.exp_dir / 'models').exists():
        (args.exp_dir / 'models').mkdir(parents=True, exist_ok=True)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/models/best_model.pt')
        logging.info('Done saving best model')

    if epoch == args.num_epochs - 1:
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/models/last_model.pt')
        logging.info('Done saving last model')
    return best_valid_loss


def build_model(args):
    if args.model == 'MUNet':
        model = MUNet(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()
    elif args.model == 'MUNet_AMLayer':
        model = MUNet_AMLayer(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()
    elif args.model == 'MUNet_Hypernetworks':
        model = MUNet_Hypernetworks(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()
    elif args.model == 'MUNet_FilterScaling':
        model = MUNet_FilterScaling(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()
    elif args.model == 'MUNet_FilterScaling_Met':
        model = MUNet_FilterScaling_Met(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()
    elif args.model == 'MUNet_FilterScaling_Met_adv':
        model = MUNet_FilterScaling_Met_adv(in_channels=args.in_channels, init_features=args.init_channels, latent_dim=args.latent_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    model_D = Discriminator(in_channels=args.in_channels_D, hidden_size=args.init_channels_D).cuda()
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    return model, optimizer, model_D, optimizer_D


def train_epoch(args, epoch, model, data_loader, optimizer, writer, model_D, optimizer_D):
    model.train()
    model_D.train()
    running_loss_L1, running_loss_ssim, running_loss_adv, running_Wdist = 0, 0, 0, 0
    total_data = len(data_loader)
    for iter, data in enumerate(data_loader):
        T1, flair, met_HR, data_max, Patient, sli, metname = data
        T1 = T1.float().cuda()
        flair = flair.float().cuda()
        met_HR = met_HR.float().cuda()

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 = {T1.shape}')
            logging.info(f'flair = {flair.shape}')
            logging.info(f'met_HR = {met_HR.shape} ')
            logging.info('--+' * 10)

        T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
        T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
        met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution_train)

        adv_weight = 0.0
        if args.adv_weight_train != 0.0:
            #adv_weight = args.adv_weight_train[0] * 10 ** (torch.rand(1) * np.log10(args.adv_weight_train[1] / args.adv_weight_train[0]))
            adv_weight = torch.rand(1) * args.adv_weight_train[1] + args.adv_weight_train[0]
            adv_weight = adv_weight.numpy()[0]
            if torch.rand(1) < 0.1:
                adv_weight = 0.0

        outputs = model(T1, flair, met_LR, lowRes, lowRes, metname, adv_weight)
        nonzero_mask = (met_HR != 0).float()
        outputs = torch.mul(outputs, nonzero_mask)

        ## GAN loss
        loss_adv, Wdist = torch.tensor(0.0), torch.tensor(0.0)
        if args.adv_weight_train != 0.0:
            for p in model_D.parameters():
                p.requires_grad = True  # unfix discriminator parameters

            for _ in range(5):  # train discriminator, 5x more update than generator
                optimizer_D.zero_grad()
                loss_D, Wdist = train_D(model_D, met_HR, outputs, T1, flair, met_LR, lowRes, metname)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            for p in model_D.parameters():
                p.requires_grad = False  # fix discriminator parameters

            loss_adv = train_G(model_D, met_HR, outputs, T1, flair, met_LR, lowRes, metname)

        loss_L1 = F.l1_loss(outputs, met_HR)
        loss_ssim = ssimloss(outputs, met_HR)
        loss = args.L1_weight * loss_L1 + args.SSIM_weight * loss_ssim + adv_weight * loss_adv
        #print(args.L1_weight, args.SSIM_weight, adv_weight, lowRes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss_L1 += loss_L1.item()
        running_loss_ssim += loss_ssim.item()
        running_loss_adv += loss_adv.item()
        running_Wdist += Wdist.item()

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{total_data:4d}] '
                f'L1 Loss = {loss_L1.item():.5g} '
                f'SSIM Loss = {loss_ssim.item():.5g} '
                f'Adv Loss = {loss_adv.item():.5g} '
                f'Wdist = {Wdist.item():.5g} '
            )

    loss_L1 = running_loss_L1 / total_data
    loss_ssim = running_loss_ssim / total_data
    loss_adv = running_loss_adv / total_data
    Wdist = running_Wdist / total_data
    if writer is not None:
        writer.add_scalar('Train/Loss_L1', loss_L1, epoch)
        writer.add_scalar('Train/Loss_SSIM', loss_ssim, epoch)
        writer.add_scalar('Train/Loss_adv', loss_adv, epoch)
        writer.add_scalar('Train/Wdist', Wdist, epoch)
    return loss


def valid(args, epoch, model, data_loader, writer):
    model.eval()
    running_L1, running_ssim, running_total = 0, 0, 0
    total_data = len(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution_train)

            adv_weight = 0.0
            if args.adv_weight_train != 0.0:
                #adv_weight = args.adv_weight_train[0] * 10 ** (torch.rand(1) * np.log10(args.adv_weight_train[1] / args.adv_weight_train[0]))
                adv_weight = torch.rand(1) * args.adv_weight_train[1] + args.adv_weight_train[0]
                adv_weight = adv_weight.numpy()[0]
                if torch.rand(1) < 0.1:
                    adv_weight = 0.0

            outputs = model(T1, flair, met_LR, lowRes, lowRes, metname, adv_weight)
            nonzero_mask = (met_HR != 0).float()
            outputs = torch.mul(outputs, nonzero_mask)

            loss_L1 = F.l1_loss(outputs, met_HR)
            loss_ssim = ssimloss(outputs, met_HR)
            loss = args.L1_weight * loss_L1 + args.SSIM_weight * loss_ssim

            running_L1 += loss_L1
            running_ssim += loss_ssim
            running_total += loss
        loss_L1 = running_L1 / total_data
        loss_ssim = running_ssim / total_data
        loss = running_total / total_data
    if writer is not None:
        writer.add_scalar('Dev/Loss_L1', loss_L1, epoch)
        writer.add_scalar('Dev/Loss_SSIM', loss_ssim, epoch)
        writer.add_scalar('Dev/Loss_Total', loss, epoch)
    return loss


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        image /= image.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
            T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
            met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution_train)

            adv_weight = 0.0
            if args.adv_weight_train != 0.0:
                #adv_weight = args.adv_weight_train[0] * 10 ** (torch.rand(1) * np.log10(args.adv_weight_train[1] / args.adv_weight_train[0]))
                adv_weight = torch.rand(1) * args.adv_weight_train[1] + args.adv_weight_train[0]
                adv_weight = adv_weight.numpy()[0]
                if torch.rand(1) < 0.1:
                    adv_weight = 0.0

            outputs = model(T1, flair, met_LR, lowRes, lowRes, metname, adv_weight)
            nonzero_mask = (met_HR != 0).float()
            outputs = torch.mul(outputs, nonzero_mask)

            outputs = outputs
            save_image(met_HR, 'Target')
            save_image(outputs, 'Output')
            save_image(met_LR, 'Input_Met')
            save_image(T1[:, :, :, :, 1], 'Input_T1')
            save_image(flair[:, :, :, :, 1], 'Input_flair')
            break


def main_train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    logs.set_logger(str(args.exp_dir / 'train.log'))
    logging.info('--' * 10)
    logging.info(
        '%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'train.log')))

    save_python_script(args)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    model, optimizer, model_D, optimizer_D = build_model(args)

    logging.info('--' * 10)
    logging.info(args)
    logging.info('--' * 10)
    logging.info(model)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model.parameters()))
    logging.info('--' * 10)
    logging.info(model_D)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model_D.parameters()))

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, args.lr_step_size, args.lr_gamma)

    logging.info('--' * 10)
    start_training = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start training at %s' % str(start_training))

    best_valid_loss = 1e9
    for epoch in range(0, args.num_epochs):
        logging.info('Current LR %s' % (scheduler.get_lr()[0]))
        logging.info('Current LR of D %s' % (scheduler_D.get_lr()[0]))
        torch.manual_seed(args.seed + epoch)
        train_loss = train_epoch(args, epoch, model, train_loader, optimizer, writer, model_D, optimizer_D)

        valid_loss = valid(args, epoch, model, valid_loader, writer)
        best_valid_loss = save_model(args, args.exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss)
        visualize(args, epoch, model, display_loader, writer)

        scheduler.step(epoch)
        logging.info('Epoch: %s Reduce LR to: %s' % (epoch, scheduler.get_lr()[0]))
        scheduler_D.step(epoch)
        logging.info('Epoch: %s Reduce LR of D to: %s' % (epoch, scheduler_D.get_lr()[0]))

    writer.close()


def main_evaluate(args):
    (args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))).mkdir(parents=True, exist_ok=True)
    logs.set_logger(str(args.exp_dir / ('eval_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test) + '.log')))
    logging.info('--' * 10)
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0),
                                            str(args.exp_dir / ('eval_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test) + '.log'))))

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    del train_loader, valid_loader, display_loader

    logging.info('--' * 10)
    start_eval = datetime.datetime.now().replace(microsecond=0)
    logging.info('Loading model %s' % str(args.checkpoint))
    logging.info('Start Evaluation at %s' % str(start_eval))
    logging.info('adv weight = %s' % str(args.adv_weight_test))
    logging.info('low resolution = %s' % str(args.low_resolution_test))
    logging.info('low resolution input = %s' % str(args.low_resolution_test))

    model, optimizer, model_D, optimizer_D = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()
    running_psnr, running_ssim = [], []
    total_data = len(test_loader)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution_test)

            #start = time.time()
            output = model(T1, flair, met_LR, lowRes, lowRes, metname, args.adv_weight_test)
            nonzero_mask = (met_HR != 0).float()
            output = torch.mul(output, nonzero_mask)
            #end = time.time()
            #print(end - start)

            data_max = data_max.numpy()
            output = output.squeeze().cpu().numpy() * data_max
            met_HR = met_HR.squeeze().cpu().numpy() * data_max
            met_LR = met_LR.squeeze().cpu().numpy() * data_max
            T1 = T1.squeeze().cpu().numpy()
            flair = flair.squeeze().cpu().numpy()

            PSNR = peak_signal_noise_ratio(met_HR, output, data_range=met_HR.max() - met_HR.min())
            SSIM = structural_similarity(met_HR, output, data_range=met_HR.max() - met_HR.min())

            running_psnr.append(PSNR.item())
            running_ssim.append(SSIM.item())

            sli = sli.numpy()
            logging.info(f'{Patient[0]} slice={sli[0]} met={metname[0]} PSNR={PSNR} SSIM={SSIM}')

            if metname[0] == 'Gln':
                output_file = '%s/%s_slice%s_MRI.npz' % (
                str(args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))), Patient[0], sli[0])
                np.savez(output_file, T1=T1, flair=flair)
            output_file = '%s/%s_slice%s_%s.npz' % (
            str(args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))), Patient[0], sli[0], metname[0])
            np.savez(output_file, met_LR=met_LR, output=output, met_HR=met_HR)

            met_LR_plot = met_LR.repeat(3, axis=0).repeat(3, axis=1)
            met_HR_plot = met_HR.repeat(3, axis=0).repeat(3, axis=1)
            outputs_plot = output.repeat(3, axis=0).repeat(3, axis=1)
            output_file = '%s/%s_slice%s_T1.png' % (
            str(args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((T1[:, :, 0], T1[:, :, 1], T1[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_flair.png' % (
            str(args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((flair[:, :, 0], flair[:, :, 1], flair[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_%s.png' % (
            str(args.exp_dir / ('results_adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test))), Patient[0], sli[0], metname[0])
            plt.imsave(output_file, np.concatenate((met_LR_plot, met_HR_plot, outputs_plot), axis=1), cmap='jet',
                       vmin=met_HR_plot.min(), vmax=met_HR_plot.max())

    running_psnr = np.asarray(running_psnr)
    running_ssim = np.asarray(running_ssim)
    logging.info('PSNR = %5g +- %5g' % (running_psnr.mean(), running_psnr.std()))
    logging.info('SSIM = %5g +- %5g' % (running_ssim.mean(), running_ssim.std()))
    np.savez(str(args.exp_dir) + '/adv' + str(args.adv_weight_test) + '_lr' + str(args.low_resolution_test) + '_metrics.npz', psnr=running_psnr, ssim=running_ssim)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    ## train
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40, help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--report-interval', type=int, default=10, help='Period of printing loss')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--num-workers', default=0, type=int, help='number of works')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--L1-weight', type=float, default=0.16, help='weight of L1 loss')
    parser.add_argument('--SSIM-weight', type=float, default=0.84, help='weight of SSIM loss')
    parser.add_argument('--adv-weight-train', nargs='+', type=float, default=0.0, help='range of weight of adv loss during training')
    parser.add_argument('--adv-weight-test', type=float, default=0.0, help='weight of adv loss during testing')

    ## methods
    parser.add_argument('--low-resolution-train', nargs='+', type=int, default=[8, 16], help='half of the low resolution matrix size during training')
    parser.add_argument('--low-resolution-test', type=int, default=8, help='half of the low resolution matrix size during testing')
    parser.add_argument('--model', type=str, default='MUNet_multiresHN', choices=['MUNet', 'MUNet_AMLayer'], help='model')
    parser.add_argument('--modelD', type=str, default='Discriminator', choices=['Discriminator'], help='Discriminator model')
    parser.add_argument('--in-channels', type=int, default=2, help='input channels')
    parser.add_argument('--init-channels', type=int, default=8, help='initial channels')
    parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension for conditional networks')
    parser.add_argument('--in-channels-D', type=int, default=4, help='input channels for discriminator')
    parser.add_argument('--init-channels-D', type=int, default=32, help='initial channels for discriminator')
    parser.add_argument('--test-patients', nargs='+', type=int, default=[1, 2, 4], help='Patient # for testing')
    parser.add_argument('--valid-patients', nargs='+', type=int, default=[6, 8, 9], help='Patient # for validation')
    parser.add_argument('--train-patients', nargs='+', type=int, default=[12, 14, 15, 16, 17, 18, 19, 20, 22], help='Patient # for training')
    # select from [1, 2, 4, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20, 22]
    # test-valid-train combinations: [1, 2, 4], [6, 8, 9], [12, 14, 15, 16, 17, 18, 19, 20, 22]
    #                                [6, 8, 9], [12, 14, 15], [1, 2, 4, 16, 17, 18, 19, 20, 22]
    #                                [12, 14, 15], [16, 17, 18], [1, 2, 4, 6, 8, 9, 19, 20, 22]
    #                                [16, 17, 18], [19, 20, 22], [1, 2, 4, 6, 8, 9, 12, 14, 15]
    #                                [19, 20, 22], [1, 2, 4], [6, 8, 9, 12, 14, 15, 16, 17, 18]
    parser.add_argument('--exp-dir', type=str, default='checkpoints', help='Path to save models and results')

    ## evaluation
    parser.add_argument('--evaluate', action='store_true', help='If set, test mode')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='path where the model was saved')
    args = parser.parse_args()

    args.exp_dir = args.exp_dir + '/' + args.model
    if isinstance(args.low_resolution_train, list):
        args.exp_dir = args.exp_dir + '_lr' + str(args.low_resolution_train[0]) + '-' + str(args.low_resolution_train[1]) + '_memory'
    elif isinstance(args.low_resolution_train, int):
        args.exp_dir = args.exp_dir + '_lr' + str(args.low_resolution_train)

    if isinstance(args.adv_weight_train, list) :
        args.exp_dir = args.exp_dir + '_cWGAN' + str(args.adv_weight_train[0]) + '-' + str(args.adv_weight_train[1]) + '_' + str(args.modelD) + '_0.1zeros'

    args.exp_dir = args.exp_dir + '/testpatients_' + str(args.test_patients[0]) + '_'+ str(args.test_patients[1]) + '_'+ str(args.test_patients[2])
    args.checkpoint = args.exp_dir + '/models/' + args.checkpoint
    args.exp_dir = pathlib.Path(args.exp_dir)

    return args


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        print('evaluate')
        print('--' * 10)
        main_evaluate(args)
    else:
        print('train')
        print('--' * 10)
        main_train(args)


# tensorboard --logdir='summary'