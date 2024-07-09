"""
@brief SuDO-RM-RF model
@version 1.0
@date 2024-07-08
@license MIT
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, channel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """Applies forward pass.

        Args:
            x (torch.Tensor): Shape `[batch, chan, *]`

        Returns:
            torch.Tensor: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    """Convolution layer with normalization and PReLU activation."""

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = (kSize - 1) // 2
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """Convolution layer with normalization."""

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = (kSize - 1) // 2
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """Normalization and PReLU activation."""

    def __init__(self, nOut):
        super().__init__()
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """Dilated convolution."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d, padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """Dilated convolution with normalized output."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d, padding=((kSize - 1) // 2) * d, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UBlock(nn.Module):
    """Upsampling block."""

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            stride = 1 if i == 0 else 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=2 * stride + 1, stride=stride, groups=in_channels, d=1))

        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2)

        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

    def forward(self, x):
        """Forward pass."""
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))
        return self.module_act(expanded + x)


class SuDORMRF(pl.LightningModule):
    def __init__(self, out_channels=128, in_channels=512, num_blocks=16, upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512, num_sources=2):
        super(SuDORMRF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources
        self.lcm = abs(self.enc_kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(self.enc_kernel_size // 2, 2 ** self.upsampling_depth)
        self.encoder = nn.Sequential(nn.Conv1d(2, enc_num_basis, enc_kernel_size, stride=enc_kernel_size // 2, padding=enc_kernel_size // 2), nn.ReLU())
        self.new_in_channel = 257 + enc_num_basis
        self.ln = GlobLN(self.new_in_channel)
        self.l1 = nn.Conv1d(self.new_in_channel, out_channels, 1)
        self.sm = nn.Sequential(*[UBlock(out_channels, self.new_in_channel, upsampling_depth) for _ in range(num_blocks)])
        if out_channels != enc_num_basis:
            self.reshape_before_masks = nn.Conv1d(out_channels, enc_num_basis, 1)
        self.m = nn.Conv2d(1, num_sources, (enc_num_basis + 1, 1), padding=(enc_num_basis - enc_num_basis // 2, 0))
        self.decoder = nn.ConvTranspose1d(enc_num_basis * num_sources, num_sources, enc_kernel_size, stride=enc_kernel_size // 2, padding=enc_kernel_size // 2, groups=num_sources)
        self.ln_mask_in = GlobLN(enc_num_basis)

    def forward(self, input_wav, spatial):
        """Forward pass."""
        input_wav = input_wav.cuda()
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x.cuda())
        m = torch.nn.Upsample((x.shape[2]))
        spatial = m(spatial).cuda()
        spatial_enc = nn.Conv1d(spatial.shape[1], spatial.shape[1], 1).cuda()
        spatial = spatial_enc(spatial)
        s = x.clone()
        x = torch.cat((x, spatial), dim=1)
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)
        if self.out_channels != self.enc_num_basis:
            x = self.reshape_before_masks(x)
        x = self.m(x.unsqueeze(1))
        x = torch.sigmoid(x) if self.num_sources == 1 else nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(list(appropriate_shape[:-1]) + [appropriate_shape[-1] + self.lcm - values_to_pad], dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":
    model = SuDORMRF(out_channels=128, in_channels=512, num_blocks=16, upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512, num_sources=2)
    dummy_input = torch.rand(3, 1, 32079)
    estimated_sources = model(dummy_input, dummy_input)  # Added dummy spatial input
    print(estimated_sources.shape)
