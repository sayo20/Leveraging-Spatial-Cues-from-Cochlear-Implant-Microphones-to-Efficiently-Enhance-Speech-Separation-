import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from collections import OrderedDict
import pandas as pd
import numpy as np
from asteroid.losses import PITLossWrapper, PairwiseNegSDR, pairwise_neg_sisdr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from AudioFeatures import InputFeature
from Datasets import Datasets

class Lightning(LightningModule):
    def __init__(self, config, SuDORMRF, out_channels=256, in_channels=512, num_blocks=16, upsampling_depth=5, enc_kernel_size=21, enc_num_basis=512, num_sources=2, in_audio_channels=1, lr=1e-3, scheduler_mode='min', scheduler_factor=0.5, patience=2, sr=8000, batch_size=4):
        super(Lightning, self).__init__()
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = config["workers"]
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        self.config = config
        self.sudormf = SuDORMRF(out_channels, in_channels, num_blocks, upsampling_depth, enc_kernel_size, enc_num_basis, num_sources)

    def forward(self, x, spatial):
        return self.sudormf(x.float(), spatial.float())

    def training_step(self, batch, batch_idx):
        df = batch['df']
        refs = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()
        brir_no = df['BrirListner']

        new_targ, new_noise, mix, out_signals, refs_dry, ipds = InputFeature.OnlineMix_samp_CutFirst_reverb_IPD(refs, noise, snr_t, brir_no, az_t, az_n, self.config['Experiment'])

        new_targ = new_targ.cuda()
        new_noise = new_noise.cuda()
        mix = mix.cuda()
        ipds = ipds.cuda()
        ests = self.forward(mix, ipds)

        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        loss = loss_func(ests, refs_dry.cuda())
        loss = torch.clamp(loss, min=-30., max=+30.)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        df = batch['df']
        brir_no = df['BrirListner']
        refs = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()

        new_targ, new_noise, mix, out_signals, refs_dry, ipds = InputFeature.OnlineMix_samp_CutFirst_reverb_IPDVal(refs, noise, snr_t, brir_no, az_t, az_n, self.config['Experiment'])

        new_targ = new_targ.cuda()
        new_noise = new_noise.cuda()
        mix = mix.cuda()
        ipds = ipds.cuda()
        ests = self.forward(mix, ipds)

        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        loss = loss_func(ests, refs_dry.cuda())
        loss = torch.clamp(loss, min=-30., max=+30.)

        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        df = batch['df']
        refs_ = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()
        brir_no = df['BrirListner']
        start_time = df["Start time"].tolist()
        pad_choice = df["Pad choice"].tolist()

        new_targ, new_noise, mix, ref_signals, ipds = InputFeature.OnlineMix_samp_CutFirst_reverb_IPD_Test(refs_, noise, snr_t, brir_no, az_t, az_n, self.config["Experiment"], start_time, pad_choice)

        mix = mix.cuda()
        ests = self.forward(mix, ipds.cuda())

        df_test = pd.DataFrame(columns=["Mixed audio", "Ref source1 audio", "Ref source2 audio", "Est source1 audio", "Est source2 audio", "Start time", "Target path", "Interferer path", "Brir no", "Pad choice", "Az t", "Az n", "snr", "Mix type", "si-sdr", "sdr", "stoi", "pesq", "Avg UN_si-sdr", "Avg UN_stoi", "Avg UN_sdr", "Avg UN_pesq"])
        sr = 8000

        loss_sisdr_ = PITLossWrapper(PairwiseNegSDR('sisdr'), pit_from='pw_mtx')
        loss_sdr_ = PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')

        for i in range(len(ests)):
            loss_sisdr, reordered_sources = loss_sisdr_(ests[i].reshape(1, ests[i].shape[0], ests[i].shape[1]), ref_signals[i].reshape(1, ref_signals[i].shape[0], ref_signals[i].shape[1]).cuda(), return_est=True)
            loss_sdr = loss_sdr_(ests[i].reshape(1, ests[i].shape[0], ests[i].shape[1]), ref_signals[i].reshape(1, ref_signals[i].shape[0], ref_signals[i].shape[1]).cuda())

            est_source1 = reordered_sources[:, 0, :]
            est_source2 = reordered_sources[:, 1, :]

            stoi_s1 = short_time_objective_intelligibility(est_source1[0], ref_signals[i][0, :].cuda(), sr)
            stoi_s2 = short_time_objective_intelligibility(est_source2[0], ref_signals[i][1, :].cuda(), sr)
            avg_stoi = np.mean([stoi_s1.item(), stoi_s2.item()])

            stoi_s1_up = short_time_objective_intelligibility(mix[i][0], ref_signals[i][0, :].cuda(), sr)
            stoi_s2_up = short_time_objective_intelligibility(mix[i][0], ref_signals[i][1, :].cuda(), sr)
            avg_stoi_up = np.mean([stoi_s1_up.item(), stoi_s2_up.item()])

            pesq_s1 = perceptual_evaluation_speech_quality(est_source1[0], ref_signals[i][0, :].cuda(), sr, 'nb')
            pesq_s2 = perceptual_evaluation_speech_quality(est_source2[0], ref_signals[i][1, :].cuda(), sr, 'nb')
            avg_pesq = np.mean([pesq_s1.item(), pesq_s2.item()])

            pesq_s1_up = perceptual_evaluation_speech_quality(mix[i][0], ref_signals[i][0, :].cuda(), sr, 'nb')
            pesq_s2_up = perceptual_evaluation_speech_quality(mix[i][0], ref_signals[i][1, :].cuda(), sr, 'nb')
            avg_pesq_up = np.mean([pesq_s1_up.item(), pesq_s2_up.item()])

            source1_sidr_unp = scale_invariant_signal_distortion_ratio(mix[i][0], ref_signals[i][0, :].cuda())
            source2_sidr_unp = scale_invariant_signal_distortion_ratio(mix[i][0], ref_signals[i][1, :].cuda())
            avg_sidr_unp = np.mean([source1_sidr_unp.item(), source2_sidr_unp.item()])

            source1_sdr_unp = signal_noise_ratio(mix[i][0], ref_signals[i][0, :].cuda())
            source2_sdr_unp = signal_noise_ratio(mix[i][0], ref_signals[i][1, :].cuda())
            avg_sdr_unp = np.mean([source1_sdr_unp.item(), source2_sdr_unp.item()])

            s1_sdr = signal_noise_ratio(est_source1[0], ref_signals[i][0, :].cuda())
            s2_sdr = signal_noise_ratio(est_source2[0], ref_signals[i][1, :].cuda())
            avg_sdr = np.mean([s1_sdr.item(), s2_sdr.item()])

            df_test.loc[i, "Mixed audio"] = mix[i].cpu()
            df_test.loc[i, "Ref source1 audio"] = new_targ[i].cpu()
            df_test.loc[i, "Ref source2 audio"] = new_noise[i].cpu()
            df_test.loc[i, "Est source1 audio"] = est_source1.cpu()
            df_test.loc[i, "Est source2 audio"] = est_source2.cpu()
            df_test.loc[i, "Target path"] = refs_[i]
            df_test.loc[i, "Interferer path"] = noise[i]
            df_test.loc[i, "Brir no"] = brir_no[i]
            df_test.loc[i, "Start time"] = start_time[i]
            df_test.loc[i, "Pad choice"] = pad_choice[i]
            df_test.loc[i, "Az t"] = az_t[i]
            df_test.loc[i, "Az n"] = az_n[i]
            df_test.loc[i, "Mix type"] = "f-m"
            df_test.loc[i, "si-sdr"] = InputFeature.de_neg(loss_sisdr).cpu()
            df_test.loc[i, "Avg UN_si-sdr"] = avg_sidr_unp
            df_test.loc[i, "stoi"] = avg_stoi
            df_test.loc[i, "Avg UN_stoi"] = avg_stoi_up
            df_test.loc[i, "sdr"] = InputFeature.de_neg(loss_sdr).cpu()
            df_test.loc[i, "Avg UN_sdr"] = avg_sdr_unp
            df_test.loc[i, "pesq"] = avg_pesq
            df_test.loc[i, "Avg UN_pesq"] = avg_pesq_up

        df_test.reset_index(inplace=True)
        df_test_dict = df_test.to_dict(into=OrderedDict)
        torch.save(df_test_dict, self.config["save_dir_inf"] + self.config["save_name_inf"] + str(batch_idx) + ".pt")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        df_train = pd.read_csv("../Data/wsj0/" + self.config["train_file"])
        train_ds = Datasets(df_train, self.config["Experiment"])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        df_val = pd.read_csv("../Data/wsj0/" + self.config["val_file"])
        val_ds = Datasets(df_val, self.config["Experiment"])
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        df_test = pd.read_csv("../Data/wsj0/" + self.config["test_file"])
        test_ds = Datasets(df_test, self.config["Experiment"])
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
        return test_dl
