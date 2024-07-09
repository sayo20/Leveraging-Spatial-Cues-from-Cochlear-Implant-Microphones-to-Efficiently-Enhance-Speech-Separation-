import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sklearn
import math
import random
import torch
import torchaudio
from torchaudio import transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import os
import torchaudio.transforms as T
from scipy import signal
import cmath
import scipy
from scipy.io.wavfile import write
from torch.autograd import Variable
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)

class InputFeature:
    
    @staticmethod
    def load_audio(audio_file):
        """
        audio_file: path to the wav file
        """
        signal, sr = torchaudio.load(audio_file)
        aud_length = signal.shape[-1] // sr
        diff = (aud_length - 4)
        if diff < 0:
            dur = abs(diff) * sr
            signal = torch.cat((signal, torch.zeros(signal.shape[0], dur)), dim=1)
        four_sec = signal[:, :sr * 4]
        return four_sec, sr

    @staticmethod
    def cal_adjusted_rms(clean_rms, snr):
        a = float(snr) / 20
        noise_rms = clean_rms / (10 ** a)
        return noise_rms

    @staticmethod
    def cal_rms(amp):
        return torch.sqrt(torch.mean(torch.square(amp), axis=-1))

    @staticmethod
    def mix_audio_snr(clean_amp, noise_amp, snr):
        clean_rms = InputFeature.cal_rms(clean_amp)
        start = random.randint(0, len(noise_amp) - len(clean_amp))
        divided_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = InputFeature.cal_rms(divided_noise_amp)
        adjusted_noise_rms = InputFeature.cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
        return clean_amp, adjusted_noise_amp

    @staticmethod
    def set_snr(signal, noise, target_snr):
        signal_power = torch.sum(signal ** 2)
        noise_power = torch.sum(noise ** 2)
        scaling_factor = torch.sqrt(signal_power / (10 ** (target_snr / 10) * noise_power))
        scaled_noise = noise * scaling_factor
        return signal, scaled_noise

    @staticmethod
    def calculate_snr(signal, noise):
        signal_power = torch.mean(torch.square(signal))
        noise_power = torch.mean(torch.square(noise))
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def convolve_brir(brir_path, sound, max_s):
        brir, sr = torchaudio.load(brir_path)
        brir_size = brir.size(1)
        if brir_size > sr:
            brir = brir[:, :48000]
        else:
            padding = sr - brir_size
            brir = F.pad(brir, (0, padding))
        resample_rate = 8000
        resampler = T.Resample(sr, resample_rate)
        brir_0 = resampler(brir[0, :])
        brir_1 = resampler(brir[1, :])
        conv_left = torch.tensor(signal.convolve(sound, brir_0))
        conv_right = torch.tensor(signal.convolve(sound, brir_1))
        conv = torch.vstack((conv_left, conv_right)).T
        return conv

    @staticmethod
    def convolve_hrtf(hrtf_path, sound):
        hr = torch.tensor(scipy.io.loadmat(hrtf_path)['IR']).to(torch.float32).T
        sr = 44100
        resample_rate = 8000
        resampler = T.Resample(sr, resample_rate)
        hr_0 = resampler(hr[0, :])
        hr_1 = resampler(hr[1, :])
        conv_left = torch.tensor(signal.convolve(sound, hr_0))
        conv_right = torch.tensor(signal.convolve(sound, hr_1))
        conv = torch.vstack((conv_left, conv_right)).T
        return conv

    @staticmethod
    def spatializer(brir_base_path, norm_target, norm_noise, az_t, az_n, s1_dry, s2_dry):
        brir_target_tmic = f"../Data/BRIR_tmic_Easy/Room{brir_base_path}_{az_t}.flac"
        brir_noise_tmic = f"../Data/BRIR_tmic_Easy/Room{brir_base_path}_{az_n}.flac"
        hrtf_spk1, hrtf_spk2 = InputFeature.spatializer_hrtf(norm_target, norm_noise, az_t, az_n)
        brir_spk1, _ = torchaudio.load(brir_target_tmic)
        brir_spk2, _ = torchaudio.load(brir_noise_tmic)
        t_size = max(brir_spk1.shape[1], brir_spk2.shape[1])
        spk1_tmic = InputFeature.convolve_brir(brir_target_tmic, norm_target, t_size)
        spk2_tmic = InputFeature.convolve_brir(brir_noise_tmic, norm_noise, t_size)
        spk1_tmic_size = spk1_tmic.size(0)
        hrtf_spk1_size = hrtf_spk1.size(0)
        dif = abs(spk1_tmic_size - hrtf_spk1_size)
        hrtf_spk1 = F.pad(hrtf_spk1.T, (0, dif))
        hrtf_spk2 = F.pad(hrtf_spk2.T, (0, dif))
        s1_dry_size = s1_dry.size(1)
        dif_dry = abs(spk1_tmic_size - s1_dry_size)
        s1_dry = F.pad(s1_dry, (0, dif_dry))
        s2_dry = F.pad(s2_dry, (0, dif_dry))
        return spk1_tmic, spk2_tmic, hrtf_spk1, hrtf_spk2, s1_dry, s2_dry

    @staticmethod
    def spatializer_hrtf(norm_target, norm_noise, az_t, az_n):
        hrtf_spk1 = f"../Data/CI-HRTF-Large/Tmic{az_t}.mat"
        hrtf_spk2 = f"../Data/CI-HRTF-Large/Tmic{az_n}.mat"
        ref_spk1 = InputFeature.convolve_hrtf(hrtf_spk1, norm_target)
        ref_spk2 = InputFeature.convolve_hrtf(hrtf_spk2, norm_noise)
        return ref_spk1, ref_spk2
    @staticmethod
    def spatializer_bi(brir_basePath, norm_target, norm_noise, az_t, az_n, s1_dry, s2_dry):
        brir_target_back = f"../Data/BRIR_back_Easy/Room{brir_basePath}_{az_t}.wav"
        brir_target_tmic = f"../Data/BRIR_tmic_Easy/Room{brir_basePath}_{az_t}.flac"
        brir_noise_back = f"../Data/BRIR_back_Easy/Room{brir_basePath}_{az_n}.wav"
        brir_noise_tmic = f"../Data/BRIR_tmic_Easy/Room{brir_basePath}_{az_n}.flac"

        hrtf_spk1, hrtf_spk2 = InputFeature.spatializer_hrtf(norm_target, norm_noise, az_t, az_n)

        # Load BRIR paths
        brir_back_spk1, _ = torchaudio.load(brir_target_back)
        brir_tmic_spk1, _ = torchaudio.load(brir_target_tmic)
        brir_back_spk2, _ = torchaudio.load(brir_noise_back)
        brir_tmic_spk2, _ = torchaudio.load(brir_noise_tmic)

        t_size = max(brir_tmic_spk1.shape[1], brir_tmic_spk2.shape[1], brir_back_spk1.shape[1], brir_back_spk2.shape[1])

        spk1_back = InputFeature.convolve_brir(brir_target_back, norm_target, t_size)
        spk1_tmic = InputFeature.convolve_brir(brir_target_tmic, norm_target, t_size)
        convolved_spk1 = torch.cat((spk1_back, spk1_tmic), axis=1)

        spk2_back = InputFeature.convolve_brir(brir_noise_back, norm_noise, t_size)
        spk2_tmic = InputFeature.convolve_brir(brir_noise_tmic, norm_noise, t_size)
        convolved_spk2 = torch.cat((spk2_back, spk2_tmic), axis=1)

        spk1_tmic_size = spk1_tmic.size(0)
        hrtf_spk1_size = hrtf_spk1.size(0)

        dif = abs(spk1_tmic_size - hrtf_spk1_size)
        hrtf_spk1 = F.pad(hrtf_spk1.T, (0, dif))
        hrtf_spk2 = F.pad(hrtf_spk2.T, (0, dif))

        s1_dry_size = s1_dry.size(1)
        dif_dry = abs(spk1_tmic_size - s1_dry_size)
        s1_dry = F.pad(s1_dry, (0, dif_dry))
        s2_dry = F.pad(s2_dry, (0, dif_dry))

        return convolved_spk1, convolved_spk2, hrtf_spk1, hrtf_spk2, s1_dry, s2_dry

    @staticmethod
    def online_mix_samp_cut_first_reverb_ipd(target, noise_p, targ_snr, brir_num, az_t, az_n, exp):
        random.shuffle(target)  # online mixing
        random.shuffle(noise_p)

        new_targ, new_noise, mix, ipd_, refs, refs_dry = [], [], [], [], [], []

        for i in range(len(target)):
            targ_path = "../" + str(target[i])
            noise_path = "../" + str(noise_p[i])
            targ = scipy.io.loadmat(targ_path)['s1_8k'][:, 0]
            noise = scipy.io.loadmat(noise_path)['s1_8k'][:, 0]

            sr, four_sec = 8000, 8000 * 4
            dur_targ, dur_noise = len(targ), len(noise)

            if dur_targ >= four_sec and dur_noise >= four_sec:
                start_point = random.randint(0, min(dur_targ, dur_noise) - four_sec)
                targ, noise = targ[start_point:start_point + four_sec], noise[start_point:start_point + four_sec]
            else:
                targ, noise = InputFeature.pad_or_truncate(targ, dur_targ, four_sec), InputFeature.pad_or_truncate(noise, dur_noise, four_sec)

            s1_8k, s2_8k = InputFeature.mix_audio_snr(torch.tensor(targ), torch.tensor(noise), targ_snr[i] * 2)
            s1_8k, s2_8k = InputFeature.scale_and_normalize(s1_8k, s2_8k, exp)

            if exp in [-2, -1.5, 0, 1, 3]:
                s1_8k_dry, s2_8k_dry = InputFeature.normalize_tensor_wav(s1_8k, 'h'), InputFeature.normalize_tensor_wav(s2_8k, 'h')
                s1_8k_dry, s2_8k_dry = s1_8k_dry.reshape(1, -1), s2_8k_dry.reshape(1, -1)
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2, s1_8k_dry_, s2_8k_dry_ = InputFeature.spatializer_bi(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i], s1_8k_dry, s2_8k_dry)
                spk1, spk2 = InputFeature.get_multi_ch_audio(s1_8k_.T, exp), InputFeature.get_multi_ch_audio(s2_8k_.T, exp)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], "h"), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], "h")
                hrtf_spk1, hrtf_spk2 = hrtf_spk1.reshape(1, -1), hrtf_spk2.reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))
                out_signals_dry = torch.vstack((s1_8k_dry_, s2_8k_dry_))
                if exp == 0:
                    ipd = InputFeature.get_spatial_feature_bi(mix_8k)
                    ipd_.append(ipd[None, :, :])
            else:
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2 = InputFeature.spatializer(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i])
                spk1, spk2 = InputFeature.normalize_tensor_wav(s1_8k_.T[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(s2_8k_.T[0, :], exp).reshape(1, -1)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], exp).reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))

            new_targ.append(spk1[None, :, :])
            new_noise.append(spk2[None, :, :])
            mix.append(mix_8k[None, :, :])
            refs.append(out_signals[None, :, :])
            if exp in [-2, -1.5, 0, 1, 3]:
                refs_dry.append(out_signals_dry[None, :, :])

        return torch.cat(new_targ), torch.cat(new_noise), torch.cat(mix), torch.cat(refs), torch.cat(refs_dry), torch.cat(ipd_)

    @staticmethod
    def online_mix_samp_cut_first_reverb_ipd_val(target, noise_p, targ_snr, brir_num, az_t, az_n, exp):
        new_targ, new_noise, mix, ipd_, refs, refs_dry = [], [], [], [], [], []

        for i in range(len(target)):
            targ_path = "../" + str(target[i])
            noise_path = "../" + str(noise_p[i])
            targ = scipy.io.loadmat(targ_path)['s1_8k'][:, 0]
            noise = scipy.io.loadmat(noise_path)['s1_8k'][:, 0]

            sr, four_sec = 8000, 8000 * 4
            dur_targ, dur_noise = len(targ), len(noise)

            if dur_targ >= four_sec and dur_noise >= four_sec:
                start_point = random.randint(0, min(dur_targ, dur_noise) - four_sec)
                targ, noise = targ[start_point:start_point + four_sec], noise[start_point:start_point + four_sec]
            else:
                targ, noise = InputFeature.pad_or_truncate(targ, dur_targ, four_sec), InputFeature.pad_or_truncate(noise, dur_noise, four_sec)

            s1_8k, s2_8k = InputFeature.mix_audio_snr(torch.tensor(targ), torch.tensor(noise), targ_snr[i] * 2)
            s1_8k, s2_8k = InputFeature.scale_and_normalize(s1_8k, s2_8k, exp)

            if exp in [-2, -1.5, 0, 1, 3]:
                s1_8k_dry, s2_8k_dry = InputFeature.normalize_tensor_wav(s1_8k, 'h'), InputFeature.normalize_tensor_wav(s2_8k, 'h')
                s1_8k_dry, s2_8k_dry = s1_8k_dry.reshape(1, -1), s2_8k_dry.reshape(1, -1)
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2, s1_8k_dry_, s2_8k_dry_ = InputFeature.spatializer_bi(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i], s1_8k_dry, s2_8k_dry)
                spk1, spk2 = InputFeature.get_multi_ch_audio(s1_8k_.T, exp), InputFeature.get_multi_ch_audio(s2_8k_.T, exp)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], "h"), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], "h")
                hrtf_spk1, hrtf_spk2 = hrtf_spk1.reshape(1, -1), hrtf_spk2.reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))
                out_signals_dry = torch.vstack((s1_8k_dry_, s2_8k_dry_))
                if exp == 0:
                    ipd = InputFeature.get_spatial_feature_bi(mix_8k)
                    ipd_.append(ipd[None, :, :])
            else:
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2 = InputFeature.spatializer(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i])
                spk1, spk2 = InputFeature.normalize_tensor_wav(s1_8k_.T[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(s2_8k_.T[0, :], exp).reshape(1, -1)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], exp).reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))

            new_targ.append(spk1[None, :, :])
            new_noise.append(spk2[None, :, :])
            mix.append(mix_8k[None, :, :])
            refs.append(out_signals[None, :, :])
            if exp in [-2, -1.5, 0, 1, 3]:
                refs_dry.append(out_signals_dry[None, :, :])

        return torch.cat(new_targ), torch.cat(new_noise), torch.cat(mix), torch.cat(refs), torch.cat(refs_dry), torch.cat(ipd_)

    @staticmethod
    def online_mix_samp_cut_first_reverb_ipd_test(target, noise_p, targ_snr, brir_num, az_t, az_n, exp, start_time, pad_choice):
        new_targ, new_noise, mix, ipd_, refs, refs_dry = [], [], [], [], [], []

        for i in range(len(target)):
            targ_path = "../" + str(target[i])
            noise_path = "../" + str(noise_p[i])
            targ = scipy.io.loadmat(targ_path)['s1_8k'][:, 0]
            noise = scipy.io.loadmat(noise_path)['s1_8k'][:, 0]

            sr, four_sec = 8000, 8000 * 4
            dur_targ, dur_noise = len(targ), len(noise)

            if dur_targ > four_sec and dur_noise > four_sec:
                targ, noise = targ[start_time[i]:start_time[i] + four_sec], noise[start_time[i]:start_time[i] + four_sec]
            else:
                targ, noise = InputFeature.pad_or_truncate(targ, dur_targ, four_sec, pad_choice[i]), InputFeature.pad_or_truncate(noise, dur_noise, four_sec, pad_choice[i])

            s1_8k, s2_8k = InputFeature.mix_audio_snr(torch.tensor(targ), torch.tensor(noise), targ_snr[i] * 2)
            s1_8k, s2_8k = InputFeature.scale_and_normalize(s1_8k, s2_8k, exp)

            if exp in [-2, -1.5, 0, 1, 3]:
                s1_8k_dry, s2_8k_dry = InputFeature.normalize_tensor_wav(s1_8k, 'h'), InputFeature.normalize_tensor_wav(s2_8k, 'h')
                s1_8k_dry, s2_8k_dry = s1_8k_dry.reshape(1, -1), s2_8k_dry.reshape(1, -1)
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2, s1_8k_dry_, s2_8k_dry_ = InputFeature.spatializer_bi(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i], s1_8k_dry, s2_8k_dry)
                spk1, spk2 = InputFeature.get_multi_ch_audio(s1_8k_.T, exp), InputFeature.get_multi_ch_audio(s2_8k_.T, exp)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], "h"), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], "h")
                hrtf_spk1, hrtf_spk2 = hrtf_spk1.reshape(1, -1), hrtf_spk2.reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))
                out_signals_dry = torch.vstack((s1_8k_dry_, s2_8k_dry_))
                if exp == 0:
                    ipd = InputFeature.get_spatial_feature_bi(mix_8k)
                    ipd_.append(ipd[None, :, :])
            else:
                s1_8k_, s2_8k_, hrtf_spk1, hrtf_spk2 = InputFeature.spatializer(
                    brir_num[i], s1_8k, s2_8k, az_t[i], az_n[i])
                spk1, spk2 = InputFeature.normalize_tensor_wav(s1_8k_.T[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(s2_8k_.T[0, :], exp).reshape(1, -1)
                mix_8k = spk1 + spk2
                hrtf_spk1, hrtf_spk2 = InputFeature.normalize_tensor_wav(hrtf_spk1[0, :], exp).reshape(1, -1), InputFeature.normalize_tensor_wav(hrtf_spk2[0, :], exp).reshape(1, -1)
                out_signals = torch.vstack((hrtf_spk1, hrtf_spk2))

            new_targ.append(spk1[None, :, :])
            new_noise.append(spk2[None, :, :])
            mix.append(mix_8k[None, :, :])
            refs.append(out_signals[None, :, :])
            if exp in [-2, -1.5, 0, 1, 3]:
                refs_dry.append(out_signals_dry[None, :, :])

        return torch.cat(new_targ), torch.cat(new_noise), torch.cat(mix), torch.cat(refs), torch.cat(refs_dry), torch.cat(ipd_)

    @staticmethod
    def pad_or_truncate(audio, dur, four_sec, choice=None):
        if dur >= four_sec:
            start_point = random.randint(0, dur - four_sec)
            return audio[start_point:start_point + four_sec]
        else:
            diff = four_sec - dur
            choice = random.randint(0, 1) if choice is None else choice
            return np.concatenate((np.zeros(diff), audio)) if choice == 0 else np.concatenate((audio, np.zeros(diff)))

    @staticmethod
    def scale_and_normalize(s1, s2, exp):
        max_amp = max(np.concatenate((np.array([1]), np.abs(s1.squeeze()), np.abs(s2.squeeze()))))
        scale = 0.9 / max_amp
        s1, s2 = s1 * scale, s2 * scale
        return InputFeature.normalize_tensor_wav(s1, exp), InputFeature.normalize_tensor_wav(s2, exp)
    @staticmethod
    def normalize_tensor_wav(wav_tensor, exp=10, eps=1e-8, std=None):
        if exp == 10 or exp == -10 or exp == -1 or exp ==0 or exp =="h": 
            mean = wav_tensor.mean(-1, keepdim=True)
            std = wav_tensor.std(-1, keepdim=True)
            # return (wav_tensor - mean) / (std + eps), mean, std

        else:
            wav_tensor_cat = torch.cat((wav_tensor[0,:],wav_tensor[1,:])) #combine the two channels first
            mean = wav_tensor_cat.mean(-1, keepdim=True)
            std = wav_tensor_cat.std(-1, keepdim=True)

        # mean = wav_tensor.mean(-1, keepdim=True)
        # if std is None:
        #     std = wav_tensor.std(-1, keepdim=True)
        return (wav_tensor - mean) / (std + eps)
    @staticmethod
    def getMultiChAudio(aud, exp):
        if exp == -1.5 or exp == 3:
            mix_ch_0 = aud[0,:] #back mic and t-mic
            mix_ch_1 = aud[2,:] #for unilateral, 2-ch aud
        elif exp == -2 or exp == 1 or exp == 5 or exp == 6 or exp == 0 :
            #use t-mic
            mix_ch_0 = aud[2,:]  #aud[0,:]aud[4,:]
            mix_ch_1 = aud[3,:]  #for bilateralaud[5,:]

        mixed_audio_ = torch.vstack((mix_ch_0,mix_ch_1))
        mixed_audio_ = InputFeature.normalize_tensor_wav(mixed_audio_,exp)

        return mixed_audio_
