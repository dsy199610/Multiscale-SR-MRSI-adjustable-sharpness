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
from torch.utils.tensorboard import SummaryWriter
from models.UNet import UNet
from models.UNet2D import UNet2D
from models.MUNet import MUNet
from models.ResNet import ResNet
from torchvision import transforms
from loader.twoD_loader import twoD_Data, RandomFlip, RandomCrop, ToTensor, RandomFlip_function, RandomCrop_function, ToTensor_function
from torch.nn import functional as F
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import torchvision


def ssimloss(output, target):
    output_ = F.interpolate(output, mode='bicubic', size=(192, 192), align_corners=True)
    target_ = F.interpolate(target, mode='bicubic', size=(192, 192), align_corners=True)
    ssim_loss = 1 - ms_ssim(output_, target_, data_range=torch.max(target_) - torch.min(target_), size_average=True)
    return ssim_loss


def create_data_loaders(args):
    train_Dataset = twoD_Data(patients=args.train_patients, transform=None, mode='train', noiseSD=args.noiseSD, upscale_factor=args.upscale_factor)
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_Dataset = twoD_Data(patients=args.valid_patients, transform=None, mode='valid', noiseSD=args.noiseSD, upscale_factor=args.upscale_factor)
    valid_loader = torch.utils.data.DataLoader(valid_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_Dataset = twoD_Data(patients=args.test_patients, transform=None, mode='test', noiseSD=args.noiseSD, upscale_factor=args.upscale_factor)
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    display_Dataset = [train_Dataset[i] for i in range(0, len(train_Dataset), len(train_Dataset) // 100)]
    display_loader = torch.utils.data.DataLoader(display_Dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader, display_loader


def save_python_script(args):
    ## copy training
    shutil.copyfile(sys.argv[0], str(args.exp_dir) + '/' + sys.argv[0])
    ## copy model
    shutil.copyfile('models/'+args.model+'.py', str(args.exp_dir) + '/'+args.model+'.py')
    ## copy loader
    shutil.copyfile('loader/twoD_loader.py', str(args.exp_dir) + '/twoD_loader.py')


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
    if args.model == 'cWGAN':
        model = UNet(inputs=args.inputs, in_channels=args.in_channels, init_features=args.init_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    return model, optimizer


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    running_loss_L1, running_loss_ssim = 0, 0
    total_data = len(data_loader)
    for iter, data in enumerate(data_loader):
        T1, flair, met_LR, met_HR, data_max, Patient, sli, metname = data
        T1 = T1.float().cuda()
        flair = flair.float().cuda()
        met_LR = met_LR.float().cuda()
        met_HR = met_HR.float().cuda()

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 = {T1.shape}')
            logging.info(f'flair = {flair.shape}')
            logging.info(f'met_LR = {met_LR.shape} ')
            logging.info(f'met_HR = {met_HR.shape} ')
            logging.info('--+' * 10)

        T1, flair, met_LR, met_HR = RandomCrop_function(args.crop_size, T1, flair, met_LR, met_HR, args.upscale_factor)
        T1, flair, met_LR, met_HR = RandomFlip_function(T1, flair, met_LR, met_HR)
        met_LR = met_LR + args.noiseSD * torch.randn_like(met_LR)
        outputs = model(T1, flair, met_LR)
        nonzero_mask = (met_HR != 0).float()
        outputs = torch.mul(outputs, nonzero_mask)
        loss_L1 = F.l1_loss(outputs, met_HR)
        loss_ssim = ssimloss(outputs, met_HR)
        loss = args.L1_weight * loss_L1 + args.SSIM_weight * loss_ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss_L1 += loss_L1.item()
        running_loss_ssim += loss_ssim.item()

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{total_data:4d}] '
                f'L1 Loss = {loss_L1.item():.5g} '
                f'SSIM Loss = {loss_ssim.item():.5g} '
            )

    loss_L1 = running_loss_L1 / total_data
    loss_ssim = running_loss_ssim / total_data
    if writer is not None:
        writer.add_scalar('Train/Loss_L1', loss_L1, epoch)
        writer.add_scalar('Train/Loss_SSIM', loss_ssim, epoch)
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
            T1, flair, met_LR, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_LR = met_LR.float().cuda()
            met_HR = met_HR.float().cuda()

            T1, flair, met_LR, met_HR = RandomCrop_function(args.crop_size, T1, flair, met_LR, met_HR, args.upscale_factor)
            T1, flair, met_LR, met_HR = RandomFlip_function(T1, flair, met_LR, met_HR)
            met_LR = met_LR + args.noiseSD * torch.randn_like(met_LR)

            outputs = model(T1, flair, met_LR)
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
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'train.log')))

    save_python_script(args)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    model, optimizer = build_model(args)

    logging.info('--' * 10)
    logging.info(args)
    logging.info('--' * 10)
    logging.info(model)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model.parameters()))

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    logging.info('--' * 10)
    start_training = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start training at %s' % str(start_training))

    best_valid_loss = 1e9
    for epoch in range(0, args.num_epochs):
        logging.info('Current LR %s' % (scheduler.get_lr()[0]))
        torch.manual_seed(args.seed+epoch)
        train_loss = train_epoch(args, epoch, model, train_loader, optimizer, writer)

        valid_loss = valid(args, epoch, model, valid_loader, writer)
        best_valid_loss = save_model(args, args.exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss)
        visualize(args, epoch, model, display_loader, writer)

        scheduler.step(epoch)
        logging.info('Epoch: %s Reduce LR to: %s' % (epoch, scheduler.get_lr()[0]))

    writer.close()


def main_evaluate(args):
    (args.exp_dir / 'results').mkdir(parents=True, exist_ok=True)
    logs.set_logger(str(args.exp_dir / 'eval.log'))
    logging.info('--' * 10)
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'eval.log')))

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    del train_loader, valid_loader, display_loader

    logging.info('--' * 10)
    start_eval = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start Evaluation at %s' % str(start_eval))
    logging.info('Noise SD = %s' % str(args.noiseSD))

    model, optimizer = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()
    running_psnr, running_ssim = 0, 0
    total_data = len(test_loader)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            T1, flair, met_LR, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_LR = met_LR.float().cuda()
            met_HR = met_HR.float().cuda()

            met_LR = met_LR + args.noiseSD * torch.randn_like(met_LR)

            output = model(T1, flair, met_LR)
            nonzero_mask = (met_HR != 0).float()
            output = torch.mul(output, nonzero_mask)

            data_max = data_max.numpy()
            output = output.squeeze().cpu().numpy() * data_max
            met_HR = met_HR.squeeze().cpu().numpy() * data_max
            met_LR = met_LR.squeeze().cpu().numpy() * data_max
            T1 = T1.squeeze().cpu().numpy()
            flair = flair.squeeze().cpu().numpy()

            PSNR = peak_signal_noise_ratio(output, met_HR, data_range=met_HR.max()-met_HR.min())
            SSIM = structural_similarity(output, met_HR, data_range=met_HR.max()-met_HR.min())

            running_psnr += PSNR.item()
            running_ssim += SSIM.item()

            sli = sli.numpy()
            logging.info(f'{Patient[0]} slice={sli[0]} met={metname[0]} PSNR={PSNR} SSIM={SSIM} ')

            if metname[0] == 'Gln':
                output_file = '%s/%s_slice%s_MRI.npz' % (str(args.exp_dir / 'results'), Patient[0], sli[0])
                np.savez(output_file, T1=T1, flair=flair)
            output_file = '%s/%s_slice%s_%s.npz' % (str(args.exp_dir / 'results'), Patient[0], sli[0], metname[0])
            np.savez(output_file, met_LR=met_LR, output=output, met_HR=met_HR)

            met_LR_plot = met_LR.repeat(args.upscale_factor*3, axis=0).repeat(args.upscale_factor*3, axis=1)
            met_HR_plot = met_HR.repeat(3, axis=0).repeat(3, axis=1)
            outputs_plot = output.repeat(3, axis=0).repeat(3, axis=1)
            output_file = '%s/%s_slice%s_T1.png' % (str(args.exp_dir / 'results'), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((T1[:, :, 0], T1[:, :, 1], T1[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_flair.png' % (str(args.exp_dir / 'results'), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((flair[:, :, 0], flair[:, :, 1], flair[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_%s.png' % (str(args.exp_dir / 'results'), Patient[0], sli[0], metname[0])
            plt.imsave(output_file, np.concatenate((met_LR_plot, met_HR_plot, outputs_plot), axis=1), cmap='jet')

    logging.info(f'Average PSNR={running_psnr/total_data} ')
    logging.info(f'Average SSIM={running_ssim/total_data} ')


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

    ## method
    parser.add_argument('--condition', type=str, default='T1_flair', help='input images')
    parser.add_argument('--model', type=str, default='cWGAN', choices=['cWGAN'], help='model')
    parser.add_argument('--in-channels', type=int, default=2, help='input channels')
    parser.add_argument('--init-channels', type=int, default=32, help='initial channels')
    parser.add_argument('--test-patients', nargs='+', type=int, default=[1, 2, 4], help='Patient # for testing')
    parser.add_argument('--train-patients', nargs='+', type=int, default=[6, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22], help='Patient # for training')
    parser.add_argument('--exp-dir', type=str, default='checkpoints/CS', help='Path to save models and results')

    ## evaluation
    parser.add_argument('--evaluate-iterations', type=int, default=1000, help='number of iterations to optimize z')
    parser.add_argument('--upscale-factor', type=int, default=2, help='upscaling factor')
    parser.add_argument('--evaluate', action='store_true', help='If set, test mode')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='path where the model was saved')

    args = parser.parse_args()

    args.exp_dir_train = args.exp_dir + '/' + args.inputs + '_' + args.model + '_filtered_by_Gilbert'
    args.exp_dir_evaluate = args.exp_dir_train + '/' + '_uf' + str(args.upscale_factor) + '/iterations' + str(args.evaluate_iterations)
    args.checkpoint = args.exp_dir_train + '/models/' + args.checkpoint
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