import os
import torch
# from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# from Datasets import Datasets
import pytorch_lightning as pl
import math


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)
class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())

class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        # self.act = nn.PReLU(nOut)

        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)

        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        # self.act = nn.PReLU(nOut)

        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UBlock(nn.Module):
    '''
    This class defines the Upsampling block, which is based on the following
    principle:
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)

class SuDORMRF(pl.LightningModule):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Sequential(*[
            nn.Conv1d(in_channels=2, out_channels=enc_num_basis,
                      kernel_size=enc_kernel_size,
                      stride=enc_kernel_size // 2,
                      padding=enc_kernel_size // 2,groups=2),
            nn.ReLU(),
        ])

        #set up to cat ipd before sepaator
        uni_shape = 257 #+ 257 #cause i added ild
        self.new_in_channel = uni_shape + enc_num_basis 

        # Norm before the rest, and apply one more dense layer
        # self.ln = nn.GroupNorm(1, self.new_in_channel, eps=1e-08)
        self.ln = GlobLN(self.new_in_channel)
        self.l1 = nn.Conv1d(in_channels=self.new_in_channel,
                            out_channels=out_channels,
                            kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(*[
            UBlock(out_channels=out_channels,
                   in_channels=self.new_in_channel,
                   upsampling_depth=upsampling_depth)
            for r in range(num_blocks)])

        if out_channels != enc_num_basis:
            self.reshape_before_masks = nn.Conv1d(in_channels=out_channels,
                                                  out_channels=enc_num_basis,
                                                  kernel_size=1)

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=num_sources,
                           kernel_size=(enc_num_basis + 1, 1),
                           padding=(enc_num_basis - enc_num_basis // 2, 0))

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources)
        # self.ln_mask_in = nn.GroupNorm(1, enc_num_basis, eps=1e-08)
        self.ln_mask_in = GlobLN(enc_num_basis)
    # Forward pass
    # def forward(self, input_wav,spatial1,spatial2):
    #     # Front end
    #     input_wav = input_wav.cuda()
    #     x = self.pad_to_appropriate_length(input_wav)
    #     # print("x padded: ",input_wav.shape, x.shape)
    #     x = self.encoder(x.cuda())

    #     #upsample and encode spatial(ipd) with conv 1 x 1
    #     # m = torch.nn.Upsample((x.shape[2]))
    #     # spatial = m(spatial)
    #     # spatial = spatial.cuda()
    #     # print("spatial before enc: ",spatial.shape)
    #     # spatial_enc = nn.Conv1d(spatial.shape[1],spatial.shape[1],1).cuda()
    #     # spatial = spatial_enc(spatial)
    #     # print("spatial no enc: ",x.shape,spatial.shape)
        
    #     # Split paths
    #     s = x.clone()
    #     #concatenate here cause s should be just mixture
    #     x = torch.cat((x,spatial1,spatial2),dim=1)

    #     # Separation module
    #     x = self.ln(x)
    #     x = self.l1(x)
    #     x = self.sm(x)

    #     if self.out_channels != self.enc_num_basis:
    #         # x = self.ln_bef_out_reshape(x)
    #         x = self.reshape_before_masks(x)

    #     # Get masks and apply them
    #     x = self.m(x.unsqueeze(1))
    #     if self.num_sources == 1:
    #         x = torch.sigmoid(x)
    #     else:
    #         x = nn.functional.softmax(x, dim=1)
    #     x = x * s.unsqueeze(1)
    #     # Back end
    #     estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
    #     return self.remove_trailing_zeros(estimated_waveforms, input_wav)
    def forward(self, input_wav,spatial):
        # Front end
        input_wav = input_wav.cuda()
        x = self.pad_to_appropriate_length(input_wav)
        # print("x padded: ",input_wav.shape, x.shape)
        x = self.encoder(x.cuda())#x.cuda()

        #upsample and encode spatial(ipd) with conv 1 x 1
        # m = torch.nn.Upsample((x.shape[2]))
        # spatial = m(spatial)
        # spatial = spatial.cuda()
        # print("spatial before enc: ",spatial.shape)
        # spatial_enc = nn.Conv1d(spatial.shape[1],spatial.shape[1],1).cuda()
        # spatial = spatial_enc(spatial)
        # print("spatial no enc: ",x.shape,spatial.shape)
        
        # Split paths
        s = x.clone()
        #concatenate here cause s should be just mixture
        x = torch.cat((x,spatial.cuda()),dim=1)

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.out_channels != self.enc_num_basis:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.num_sources == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    
    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":
    model = SuDORMRF(out_channels=128,
                     in_channels=512,
                     num_blocks=16,
                     upsampling_depth=4,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     num_sources=2)

    dummy_input = torch.rand(3, 1, 32079)
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)



