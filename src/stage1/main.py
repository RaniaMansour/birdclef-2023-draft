import argparse
import datetime

# import glob
import os
import random
import warnings

# from copy import deepcopy
from functools import partial

import colorednoise as cn
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy as sp
import soundfile as sf
import timm
import torch
import torch.optim as optim
from pytorch_lightning import LightningDataModule, callbacks

# from pytorch_lightning.utilities import rank_zero_info
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

warnings.simplefilter("ignore")


train_df = pd.read_csv("../../input/birdclef-2023/train_metadata_2023.csv")
target_columns = ['abethr1',
 'abhori1',
 'abythr1',
 'afbfly1',
 'afdfly1',
 'afecuc1',
 'affeag1',
 'afgfly1',
 'afghor1',
 'afmdov1',
 'afpfly1',
 'afpkin1',
 'afpwag1',
 'afrgos1',
 'afrgrp1',
 'afrjac1',
 'afrthr1',
 'amesun2',
 'augbuz1',
 'bagwea1',
 'barswa',
 'bawhor2',
 'bawman1',
 'bcbeat1',
 'beasun2',
 'bkctch1',
 'bkfruw1',
 'blacra1',
 'blacuc1',
 'blakit1',
 'blaplo1',
 'blbpuf2',
 'blcapa2',
 'blfbus1',
 'blhgon1',
 'blhher1',
 'blksaw1',
 'blnmou1',
 'blnwea1',
 'bltapa1',
 'bltbar1',
 'bltori1',
 'blwlap1',
 'brcale1',
 'brcsta1',
 'brctch1',
 'brcwea1',
 'brican1',
 'brobab1',
 'broman1',
 'brosun1',
 'brrwhe3',
 'brtcha1',
 'brubru1',
 'brwwar1',
 'bswdov1',
 'btweye2',
 'bubwar2',
 'butapa1',
 'cabgre1',
 'carcha1',
 'carwoo1',
 'categr',
 'ccbeat1',
 'chespa1',
 'chewea1',
 'chibat1',
 'chtapa3',
 'chucis1',
 'cibwar1',
 'cohmar1',
 'colsun2',
 'combul2',
 'combuz1',
 'comsan',
 'crefra2',
 'crheag1',
 'crohor1',
 'darbar1',
 'darter3',
 'didcuc1',
 'dotbar1',
 'dutdov1',
 'easmog1',
 'eaywag1',
 'edcsun3',
 'egygoo',
 'equaka1',
 'eswdov1',
 'eubeat1',
 'fatrav1',
 'fatwid1',
 'fislov1',
 'fotdro5',
 'gabgos2',
 'gargan',
 'gbesta1',
 'gnbcam2',
 'gnhsun1',
 'gobbun1',
 'gobsta5',
 'gobwea1',
 'golher1',
 'grbcam1',
 'grccra1',
 'grecor',
 'greegr',
 'grewoo2',
 'grwpyt1',
 'gryapa1',
 'grywrw1',
 'gybfis1',
 'gycwar3',
 'gyhbus1',
 'gyhkin1',
 'gyhneg1',
 'gyhspa1',
 'gytbar1',
 'hadibi1',
 'hamerk1',
 'hartur1',
 'helgui',
 'hipbab1',
 'hoopoe',
 'huncis1',
 'hunsun2',
 'joygre1',
 'kerspa2',
 'klacuc1',
 'kvbsun1',
 'laudov1',
 'lawgol',
 'lesmaw1',
 'lessts1',
 'libeat1',
 'litegr',
 'litswi1',
 'litwea1',
 'loceag1',
 'lotcor1',
 'lotlap1',
 'luebus1',
 'mabeat1',
 'macshr1',
 'malkin1',
 'marsto1',
 'marsun2',
 'mcptit1',
 'meypar1',
 'moccha1',
 'mouwag1',
 'ndcsun2',
 'nobfly1',
 'norbro1',
 'norcro1',
 'norfis1',
 'norpuf1',
 'nubwoo1',
 'pabspa1',
 'palfly2',
 'palpri1',
 'piecro1',
 'piekin1',
 'pitwhy',
 'purgre2',
 'pygbat1',
 'quailf1',
 'ratcis1',
 'raybar1',
 'rbsrob1',
 'rebfir2',
 'rebhor1',
 'reboxp1',
 'reccor',
 'reccuc1',
 'reedov1',
 'refbar2',
 'refcro1',
 'reftin1',
 'refwar2',
 'rehblu1',
 'rehwea1',
 'reisee2',
 'rerswa1',
 'rewsta1',
 'rindov',
 'rocmar2',
 'rostur1',
 'ruegls1',
 'rufcha2',
 'sacibi2',
 'sccsun2',
 'scrcha1',
 'scthon1',
 'shesta1',
 'sichor1',
 'sincis1',
 'slbgre1',
 'slcbou1',
 'sltnig1',
 'sobfly1',
 'somgre1',
 'somtit4',
 'soucit1',
 'soufis1',
 'spemou2',
 'spepig1',
 'spewea1',
 'spfbar1',
 'spfwea1',
 'spmthr1',
 'spwlap1',
 'squher1',
 'strher',
 'strsee1',
 'stusta1',
 'subbus1',
 'supsta1',
 'tacsun1',
 'tafpri1',
 'tamdov1',
 'thrnig1',
 'trobou1',
 'varsun2',
 'vibsta2',
 'vilwea1',
 'vimwea1',
 'walsta1',
 'wbgbir1',
 'wbrcha2',
 'wbswea1',
 'wfbeat1',
 'whbcan1',
 'whbcou1',
 'whbcro2',
 'whbtit5',
 'whbwea1',
 'whbwhe3',
 'whcpri2',
 'whctur2',
 'wheslf1',
 'whhsaw1',
 'whihel1',
 'whrshr1',
 'witswa1',
 'wlwwar',
 'wookin1',
 'woosan',
 'wtbeat1',
 'yebapa1',
 'yebbar1',
 'yebduc1',
 'yebere1',
 'yebgre1',
 'yebsto1',
 'yeccan1',
 'yefcan',
 'yelbis1',
 'yenspu1',
 'yertin1',
 'yesbar1',
 'yespet1',
 'yetgre1',
 'yewgre1']

bird2id = {b: i for i, b in enumerate(target_columns)}
id2bird = {i: b for i, b in enumerate(target_columns)}


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin : begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


def load_wave_and_crop(filename, period, start=None):
    waveform_orig, sample_rate = sf.read(filename)
    wave_len = len(waveform_orig)
    waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])
    while len(waveform) < (period * sample_rate * 3):
        waveform = np.concatenate([waveform, waveform_orig])
    if start is not None:
        start = start - (period - 5) / 2 * sample_rate
        while start < 0:
            start += wave_len
        start = int(start)
        # start = int(start * sample_rate) + wave_len
    else:
        start = np.random.randint(wave_len)
    waveform_seg = waveform[start : start + int(period * sample_rate)]
    return waveform_orig, waveform_seg, sample_rate, start


class BirdClef2023Dataset(Dataset):
    def __init__(
        self,
        data_path: str = "../../input/birdclef-2023/train_audio",
        period: float = 15.0,
        secondary_coef: float = 1.0,
        smooth_label: float = 0.0,
        df: pd.DataFrame = train_df,
        train: bool = True,
    ):

        self.df = df
        self.data_path = data_path
        self.filenames = df["filename"]
        self.primary_label = df["primary_label"]

        self.secondary_labels = (
            df["secondary_labels"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("'", "")
                .split(" ")
            )
            .values
        )
        self.secondary_coef = secondary_coef
        self.type = df["type"]
        self.period = period
        self.smooth_label = smooth_label + 1e-6
        if train:
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                        ],
                        p=0.2,
                    ),
                    RandomVolume(p=0.2, limit=4),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose(
                [
                    Normalize(p=1),
                ]
            )
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_path, self.primary_label[idx], self.filenames[idx]
        )
        if self.train:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period
            )
        else:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period, 0
            )

        waveform_seg = self.wave_transforms(waveform_seg, sr=sample_rate)

        waveform_seg = torch.Tensor(np.nan_to_num(waveform_seg))

        target = np.zeros(397, dtype=np.float32)
        primary_label = bird2id[self.primary_label[idx]]
        target[primary_label] = 1.0

        for s in self.secondary_labels[idx]:
            if s == "rocpig1":
                s = "rocpig"
            if s != "" and s in bird2id.keys():
                target[bird2id[s]] = self.secondary_coef

        target = torch.Tensor(target)
        return {
            "wave": waveform_seg,
            "target": (target > 0.01).float(),
            "loss_target": target * (1 - self.smooth_label)
            + self.smooth_label / target.size(-1),
        }


class BirdClef2023DataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 8,
        period: float = 15.0,
        secondary_coef: float = 1.0,
        train_df: pd.DataFrame = train_df,
        valid_df: pd.DataFrame = train_df,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.period = period
        self.secondary_coef = secondary_coef
        self.train_df = train_df
        self.valid_df = valid_df

    def create_dataset(self, train=True):
        return (
            BirdClef2023Dataset(
                period=self.period,
                secondary_coef=self.secondary_coef,
                train=True,
                df=self.train_df,
            )
            if train
            else BirdClef2023Dataset(
                period=self.period,
                secondary_coef=self.secondary_coef,
                train=False,
                df=self.valid_df,
            )
        )

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        dataset = self.create_dataset(train)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=train,
            drop_last=train,
            worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x),
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BirdClef2023DataModule")
        parser.add_argument(
            "--num_workers",
            default=0,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        parser.add_argument(
            "--period",
            default=15.0,
            type=float,
            metavar="P",
            help="period for training",
            dest="period",
        )
        parser.add_argument(
            "--secondary_coef",
            default=1.0,
            type=float,
            metavar="SC",
            help="secondary coef",
            dest="secondary_coef",
        )
        return parent_parser


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class AttHead(nn.Module):
    def __init__(
        self, in_chans, p=0.5, num_class=264, train_period=15.0, infer_period=5.0
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)

        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)
        time_att = torch.tanh(self.attention(feat))
        assert self.train_period >= self.infer_period
        if self.training or self.train_period == self.infer_period:

            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
        else:
            feat_time = feat.size(-1)
            start = (
                feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            )
            end = start + feat_time * (self.infer_period / self.train_period)
            start = int(start)
            end = int(end)
            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att,
                dim=-1,
            )
            logits = torch.sum(
                self.fix_scale(feat) * att,
                dim=-1,
            )
            time_att = time_att[:, :, start:end]
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat).permute(0, 2, 1),
            time_att.permute(0, 2, 1),
        )


class AttModel(nn.Module):
    def __init__(
        self,
        backbone="EfficientNetB1",
        p=0.5,
        n_mels=224,
        num_class=264,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                32000,
                n_mels=n_mels,
                f_min=20,
                n_fft=2048,
                hop_length=512,
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=True, in_chans=in_chans
        )
        encoder_channels = self.backbone.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.head = AttHead(
            dense_input,
            p=p,
            num_class=num_class,
            train_period=train_period,
            infer_period=infer_period,
        )

    def forward(self, input):
        feats = self.backbone(input)
        return self.head(feats[-1])


def row_wise_f1_score_micro(y_true, y_pred, threshold=0.5):
    def event_thresholder(x, threshold):
        return x > threshold

    return f1_score(
        y_true=y_true, y_pred=event_thresholder(y_pred, threshold), average="samples"
    )

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score


class ThresholdOptimizer:
    def __init__(self, loss_fn):
        self.coef_ = {}
        self.loss_fn = loss_fn
        self.coef_["x"] = [0.5]

    def _loss(self, coef, X, y):
        ll = self.loss_fn(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def coefficients(self):
        return self.coef_["x"]

    def calc_score(self, X, y, coef):
        return self.loss_fn(y, X, coef)


class Mixup(object):
    def __init__(self, p=0.5, alpha=5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0


class BirdClef2023Model(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "EfficientNetB1",
        n_mels: int = 224,
        batch_size: int = 32,
        lr: float = 1e-3,
        backbone_lr: float = None,
        num_workers: int = 6,
        period=15.0,
        infer_period=15.0,
        mixup_p=0.0,
        mixup_alpha=0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_mels = n_mels
        # self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.num_workers = num_workers
        self.period = period
        self.infer_period = infer_period
        self.thresholder = ThresholdOptimizer(row_wise_f1_score_micro)
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)

        self.decay = 0.99

        self.__build_model()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        self.model = AttModel(
            self.backbone,
            p=0.5,
            n_mels=self.n_mels,
            num_class=264,
            train_period=self.period,
            infer_period=self.infer_period,
        )
        self.criterions = {
            "classification_clip": nn.BCEWithLogitsLoss(),
            "classification_frame": nn.BCEWithLogitsLoss(),
        }

    def forward(self, image):
        """Forward pass. Returns logits."""
        outputs = {}
        (
            outputs["logits"],
            outputs["output_clip"],
            outputs["output_frame"],
            outputs["output_attention"],
        ) = self.model(image)
        return outputs

    def loss(self, outputs, batch):
        losses = {}
        losses["loss_clip"] = self.criterions["classification_clip"](
            torch.logit(outputs["output_clip"]), batch["loss_target"]
        )
        losses["loss_frame"] = self.criterions["classification_frame"](
            outputs["output_frame"].max(1)[0], batch["loss_target"]
        )
        losses["loss"] = losses["loss_clip"] + losses["loss_frame"] * 0.5
        return losses

    def training_step(self, batch, batch_idx):
        self.mixupper.init_lambda()
        step_output = {}
        image = self.model.logmelspec_extractor(batch["wave"])[:, None]
        image = self.mixupper.lam * image + (1 - self.mixupper.lam) * image.flip(0)
        outputs = self.forward(image)
        batch["loss_target"] = self.mixupper.lam * batch["loss_target"] + (
            1 - self.mixupper.lam
        ) * batch["loss_target"].flip(0)
        batch["target"] = self.mixupper.lam * batch["target"] + (
            1 - self.mixupper.lam
        ) * batch["target"].flip(0)

        train_loss = self.loss(outputs, batch)

        step_output.update(train_loss)
        step_output.update({"output_clip": outputs["output_clip"]})
        step_output["target"] = batch["target"]
        self.log_dict(
            dict(
                train_loss=train_loss["loss"],
                train_loss_frame=train_loss["loss_frame"],
                train_loss_clip=train_loss["loss_clip"],
            )
        )
        return step_output

    def training_epoch_end(self, training_step_outputs):
        y_true = []
        y_pred = []
        for tso in training_step_outputs:
            y_true.append(tso["target"])
            y_pred.append(tso["output_clip"])
        y_true = torch.cat(y_true).cpu().numpy().astype("int")
        y_pred = torch.cat(y_pred).cpu().detach().numpy()
        self.thresholder.fit(y_pred, y_true)
        coef = self.thresholder.coefficients()

        # print(output_val.shape)
        val_df = pd.DataFrame(y_true, columns = birds)
        pred_df = pd.DataFrame(y_pred, columns = birds)
        
        avg_score = padded_cmap(val_df, pred_df, padding_factor = 5)
        avg_score2 = padded_cmap(val_df, pred_df, padding_factor = 3)
        avg_score3 = sklearn.metrics.label_ranking_average_precision_score(y_true,y_pred)
        
        f1_score = self.thresholder.calc_score(y_pred, y_true, coef)
        f1_score_05 = self.thresholder.calc_score(y_pred, y_true, [0.5])
        f1_score_03 = self.thresholder.calc_score(y_pred, y_true, [0.3])
        self.log_dict(
            dict(
                train_coef=coef,
                train_avg_score=avg_score,
                train_avg_score2=avg_score2,
                train_avg_score3=avg_score3,
            )
        )
        

    def validation_step(self, batch, batch_idx):
        step_output = {}
        image = self.model.logmelspec_extractor(batch["wave"])[:, None]
        outputs = self.forward(image)
        valid_loss = self.loss(outputs, batch)
        step_output.update({"output_clip": outputs["output_clip"]})
        step_output["target"] = batch["target"]
        self.log_dict(
            dict(
                val_loss=valid_loss["loss"],
                val_loss_frame=valid_loss["loss_frame"],
                val_loss_clip=valid_loss["loss_clip"],
            )
        )
        return step_output

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = []
        y_true = []
        for vso in validation_step_outputs:
            y_true.append(vso["target"])
            y_pred.append(vso["output_clip"])
        y_true = torch.cat(y_true).cpu().numpy().astype("int")
        y_pred = torch.cat(y_pred).cpu().detach().numpy()
        self.thresholder.fit(y_pred, y_true)
        coef = self.thresholder.coefficients()
        
        val_df = pd.DataFrame(y_true, columns = birds)
        pred_df = pd.DataFrame(y_pred, columns = birds)
        
        avg_score = padded_cmap(val_df, pred_df, padding_factor = 5)
        avg_score2 = padded_cmap(val_df, pred_df, padding_factor = 3)
        avg_score3 = sklearn.metrics.label_ranking_average_precision_score(y_true,y_pred)
        
        f1_score = self.thresholder.calc_score(y_pred, y_true, coef)
        f1_score_05 = self.thresholder.calc_score(y_pred, y_true, [0.5])
        f1_score_03 = self.thresholder.calc_score(y_pred, y_true, [0.3])
        self.log_dict(
            dict(
                val_coef=coef,
                val_avg_score=avg_score,
                val_avg_score2=avg_score2,
                val_avg_score3=avg_score3,
            )
        )
#         competition_metrics(output_val,target_val)
        print(f'epoch {self.current_epoch} validation loss {avg_loss}')
        print(f'epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
        print(f'epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}')
        print(f'epoch {self.current_epoch} validation AP score {avg_score3}')
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):

        optimizer = optim.Adam(
            [
                {"params": self.model.head.parameters(), "lr": self.lr},
                {"params": self.model.backbone.parameters(), "lr": self.backbone_lr},
            ],
            lr=self.lr,
            weight_decay=0.0001,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1.0e-6,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TransferLearningModel")
        parser.add_argument(
            "--backbone",
            default="EfficientNetB1",
            type=str,
            metavar="BK",
            help="Name (as in ``timm``) of the feature extractor",
        )
        parser.add_argument(
            "--n_mels", default=224, type=int, metavar="NM", help="nmels", dest="n_mels"
        )
        parser.add_argument(
            "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            metavar="B",
            help="batch size",
            dest="batch_size",
        )
        parser.add_argument("--gpus", type=int, default=0, help="number of gpus to use")
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate for backbone network",
            dest="backbone_lr",
        )
        parser.add_argument(
            "--mixup_p",
            default=0,
            type=float,
            metavar="MP",
            help="mixup proba",
            dest="mixup_p",
        )
        parser.add_argument(
            "--mixup_alpha",
            default=0.8,
            type=float,
            metavar="ML",
            help="mixup alpha",
            dest="mixup_alpha",
        )
        parser.add_argument(
            "--period",
            default=15.0,
            type=float,
            metavar="P",
            help="period for training",
            dest="period",
        )
        parser.add_argument(
            "--infer_period",
            default=15.0,
            type=float,
            metavar="P",
            help="period for inference",
            dest="infer_period",
        )
        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2023,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser = BirdClef2023Model.add_model_specific_args(parent_parser)
    parser = BirdClef2023DataModule.add_argparse_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    assert args.fold < 4
    for i in range(4):
        if args.fold != i:
            continue
        train_df_fold = train_df[train_df.fold != i].reset_index(drop=True)
        valid_df_fold = train_df[train_df.fold == i].reset_index(drop=True)

        datamodule = BirdClef2023DataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            period=args.period,
            secondary_coef=args.secondary_coef,
            train_df=train_df_fold,
            valid_df=valid_df_fold,
        )
        model = BirdClef2023Model(**vars(args))
        rootdir = f"../../logs/stage1/{args.logdir}/fold{i}"
        print(f"logdir = {rootdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )

        f1_checkpoint = callbacks.ModelCheckpoint(
            filename="best_f1",
            monitor="val_f1_score",
            save_top_k=1,
            mode="max",
        )
        
        Cmap_checkpoint = callbacks.ModelCheckpoint(
            filename="best_Cmap",
            monitor="val_avg_score",
            save_top_k=1,
            mode="max",
        )

        trainer = pl.Trainer(
            default_root_dir=rootdir,
            progress_bar_refresh_rate=1,
            sync_batchnorm=True,
            # precision=16,
            gpus=args.gpus,
            max_epochs=args.epochs,
            callbacks=[
                loss_checkpoint,
                f1_checkpoint,
                Cmap_checkpoint,
                lr_monitor,
            ],
            accelerator="ddp",
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
