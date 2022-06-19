from torch.utils.data import Dataset
import pathlib
import numpy as np
import logging
import torch
from utils import utils
from torch.utils.data.sampler import Sampler
import random
import math


class twoD_Data(Dataset):
    def __init__(self, patients, transform, mode, noiseSD=0.0, upscale_factor_range=0.0):
        self.patients = patients
        self.transform = transform
        self.mode = mode
        self.noiseSD = noiseSD
        self.upscale_factor_range = upscale_factor_range
        self.resolution = 64

        self.patient_list = ['Patient' + str(self.patients[i]) for i in range(0, len(self.patients))]
        self.met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']
        self.patient_folders = list(pathlib.Path('data_processed').iterdir())
        self.examples = []
        self.slices = []
        self.fnames = []
        self.fname2nslice = {}
        for patient in sorted(self.patient_folders):
            patientname = str(patient.name)
            if patientname not in self.patient_list:
                continue
            met = np.load(str(patient)+'/Met_filtered/Gln.npy')
            num_slices = met.shape[0]

            self.examples += [(patient, slice, met) for slice in range(num_slices) for met in self.met_list]
            self.slices += [(str(patient), slice) for slice in range(num_slices)]
            self.fnames += [patientname]
            self.fname2nslice[patientname] = num_slices

        logging.info(' ' * 10)
        logging.info('--+' * 10)
        logging.info('loading patients: %s ' % self.fname2nslice)
        logging.info('total slices: %s' % len(self.slices))
        logging.info('total mets: %s' % len(self.examples))
        logging.info('--+' * 10)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        patient, slice, metname = self.examples[idx]
        # metabolite
        #met_HR = np.load(str(patient)+'/Met_sliced_filtered_by_Gilbert/'+metname+'.npy')[slice]
        met_HR = np.load(str(patient) + '/Met_filtered/' + metname + '.npy')[slice]
        met_HR = torch.from_numpy(met_HR)
        met_max = met_HR.max()
        met_HR = met_HR / met_max

        # MRI
        T1 = np.load(str(patient)+'/MRI_sliced/T1_sliced.npy')[slice*3:(slice+1)*3, :, :]
        flair = np.load(str(patient) + '/MRI_sliced/flair_sliced.npy')[slice*3:(slice+1)*3, :, :]
        T1 = np.transpose(T1, (1, 2, 0)) / T1.max()
        flair = np.transpose(flair, (1, 2, 0)) / flair.max()

        T1, flair, met_HR = T1[None, :, :], flair[None, :, :], met_HR[None, :, :]
        sample = {'T1': T1, 'flair': flair, 'met_HR': met_HR}
        if self.transform:
            sample = self.transform(sample)
        return sample['T1'], sample['flair'], sample['met_HR'], met_max, str(patient.name), slice, metname


def RandomDownscale_function(met_HR, lowRes):
    met_HR_cplx = torch.cat((met_HR.unsqueeze(-1), torch.zeros_like(met_HR.unsqueeze(-1))), -1)
    kspace = utils.fft2(met_HR_cplx)
    kspace_trunc = torch.zeros_like(kspace)
    if isinstance(lowRes, list):
        lowRes = torch.randint(lowRes[0], lowRes[1]+1, (1, ))[0]
    div = lowRes
    kspace_trunc[:, :, 32 - div:32 + div, 32 - div:32 + div, :] = kspace[:, :, 32 - div:32 + div, 32 - div:32 + div, :]
    met_LR_cplx = utils.ifft2(kspace_trunc)
    met_LR = torch.sqrt(met_LR_cplx[:, :, :, :, 0] ** 2 + met_LR_cplx[:, :, :, :, 1] ** 2)
    return met_LR, lowRes


def RandomFlip_function(T1, flair, met_HR):
    angle = torch.randint(0, 4, (1, ))[0]
    flip = torch.rand(1)
    if flip <= 1 / 2:
        T1 = torch.flip(torch.rot90(T1, angle, [2, 3]), [2])
        flair = torch.flip(torch.rot90(flair, angle, [2, 3]), [2])
        met_HR = torch.flip(torch.rot90(met_HR, angle, [2, 3]), [2])
    else:
        T1 = torch.rot90(T1, angle, [2, 3])
        flair = torch.rot90(flair, angle, [2, 3])
        met_HR = torch.rot90(met_HR, angle, [2, 3])

    flip_MRI = torch.rand(1)
    if flip_MRI <= 1 / 2:
        T1 = torch.flip(T1, [4])
        flair = torch.flip(flair, [4])
    return T1, flair, met_HR


def RandomShift_function(T1, flair, met_HR):
    shiftx, shifty = torch.randint(-3, 4, (1, ))[0], torch.randint(-3, 4, (1, ))[0]
    met_HR = torch.roll(met_HR, (shiftx, shifty), dims=(2, 3))
    T1 = torch.roll(T1, (shiftx * 3, shifty * 3), dims=(2, 3))
    flair = torch.roll(flair, (shiftx * 3, shifty * 3), dims=(2, 3))
    return T1, flair, met_HR


class Met_Sampler(Sampler):
    def __init__(self, batch_size, length):
        self.length = length
        self.met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']
        self.num_met = len(self.met_list)
        self.indices = np.array(range(length))
        self.batch_size = batch_size

    def __iter__(self):
        batches = []
        for met_idx in range(7):
            met_indices = self.indices[0::7] + met_idx
            random.shuffle(met_indices)
            for i in range(0, math.floor(met_indices.shape[0] / self.batch_size)):
                batches.append(met_indices[i*self.batch_size:(i+1)*self.batch_size])
        random.shuffle(batches)
        #print(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.floor(self.length / 7 / self.batch_size) * 7
