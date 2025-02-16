import os
import torch
# from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Datasets import Datasets
# from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningModule
from AudioFeatures import InputFeature
from collections import OrderedDict, defaultdict
import pandas as pd
import scipy
import numpy as np
from asteroid.losses import pairwise_neg_sisdr,PITLossWrapper,PairwiseNegSDR,pairwise_neg_snr
from torchmetrics import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio,signal_distortion_ratio,scale_invariant_signal_noise_ratio,signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
import spyral
import time

class Lightning(LightningModule):
    def __init__(self,config,SuDORMRF,
                 out_channels=128, #256
                 in_channels=512,
                 num_blocks=16 ,#16 big 34
                 upsampling_depth=4, #5 big 4
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 # optimizer
                 in_audio_channels=1,
                 lr=1e-3,
                 scheduler_mode='min',
                 scheduler_factor=0.2,
                 patience=10,
                 sr=8000,
                 # DataLoader
                 batch_size=4):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------
        # self.train_mix_scp = train_mix_scp
        # self.train_ref_scp = train_ref_scp
        # self.val_mix_scp = val_mix_scp
        # self.val_ref_scp = val_ref_scp
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = config["workers"]
        # ----------training&validation&testing Param---------
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        self.config = config
        # -----------------------model-----------------------
        self.sudormf = SuDORMRF(out_channels,in_channels,num_blocks,upsampling_depth,enc_kernel_size,enc_num_basis,num_sources)
        # self.sudormf = CausalSuDORMRF(in_audio_channels,out_channels,in_channels,num_blocks,upsampling_depth,enc_kernel_size,enc_num_basis,num_sources)

    def forward(self, x): #forward(self, x,spatial)
        return self.sudormf(x)

    # ---------------------
    # TRAINING STEP
    # ---------------------

    def training_step(self, batch, batch_idx):
        df = batch['df']
        # print(df)
        refs = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        snr_n = df['Noise_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()
        brir_no = df['BrirListner']
        # spatial = batch["ipd"]

        # new_targ,new_noise,mix,out_signals,refs_dry= InputFeature.OnlineMix_samp_CutFirst_reverb(refs,noise,snr_t,brir_no,az_t,az_n,self.config['Experiment']) #InputFeature.OnlineMix_samp_CutFirst(refs,noise,snr_t,snr_n,self.config["Experiment"])
        
        new_targ,new_noise,mix,out_signals = InputFeature.OnlineMix_samp_CutFirst_reverb(refs,noise,snr_t,brir_no,az_t,az_n,self.config['Experiment']) #InputFeature.OnlineMix_samp_CutFirst(refs,noise,snr_t,snr_n,self.config["Experiment"])

        new_targ =new_targ.cuda()
        new_noise = new_noise.cuda()

        # out_signals = torch.vstack((new_targ,new_noise))
        # print(f"Reference signal is of shape:{refs_dry.shape}")
        
        mix = mix.cuda()
 
        ests = self.forward(mix)

        # loss_sisdr = signal_distortion_ratio(ests, anec_targ.cuda()).mean(axis=1)#for uni just remove .mean()
        # loss = signal_distortion_ratio(ests, anec_targ.cuda()).mean(axis=1)#for uni just remove .mean()
        # loss_sisdr = -torch.sum(loss_sisdr)/len(ests)

        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') #this should completely remove the scale-invariance
        loss = loss_func(ests, out_signals.cuda())

        loss = torch.clamp(loss,min=-30., max=+30.)
        # loss.requires_grad=True
        # print("Loss is: ",loss)
        # tensorboard_logs = {'train_loss': loss}
        self.log("train_loss", loss,on_step=False,on_epoch=True)
        # self.log("train sdr(nonPIT)", loss_sisdr,on_step=False,on_epoch=True)
        return loss #{'loss': loss, 'log': tensorboard_logs}

    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # mix = batch['mix']
        df = batch['df']
        # print(df)
        brir_no = df['BrirListner']
        refs = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        snr_n = df['Noise_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()
        # print(df['Az noise'])
        # new_targ,new_noise,mix,out_signals,refs_dry= InputFeature.OnlineMix_samp_CutFirst_reverb_Val(refs,noise,snr_t,brir_no,az_t,az_n,self.config['Experiment']) #InputFeature.OnlineMix_samp_CutFirst(refs,noise,snr_t,snr_n,self.config["Experiment"])
        
        new_targ,new_noise,mix,out_signals = InputFeature.OnlineMix_samp_CutFirst_reverb_Val(refs,noise,snr_t,brir_no,az_t,az_n,self.config['Experiment']) #InputFeature.OnlineMix_samp_CutFirst(refs,noise,snr_t,snr_n,self.config["Experiment"])


        # print("mix shape, ",mix.shape)
        new_targ =new_targ.cuda()
        new_noise = new_noise.cuda()

        # out_signals = torch.vstack((new_targ,new_noise))

        mix = mix.cuda()
        ests = self.forward(mix) #self.forward(mix,spatial)

        print(f"mixed shape is {mix.shape}, and estimated shape is {ests.shape}, stacked ref: {out_signals.shape}")

        ##checks: play mix and new_targ..especially because you had to create the shape of the array to match the model

        # loss_sisdr_ =  PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')
        # loss_sisdr = loss_sisdr_(ests, out_signals.cuda())
  
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')#pairwise_neg_sisdr
        loss = loss_func(ests, out_signals.cuda())


        loss = torch.clamp(loss,min=-30., max=+30.)
        sr=8000


        # sdr_un = loss_func(mix, out_signals.cuda())
        self.log("val_loss", loss.mean(),on_step=False,on_epoch=True)
        # self.log("val_sdr un-processed", sdr_un.mean(),on_step=False,on_epoch=True)
        # self.log("val sdr", loss_sisdr.mean(),on_step=False,on_epoch=True)


        return {'val_loss': loss}

    def test_step(self,batch,batch_idx):
        df = batch['df']
        # print(df)
        refs_ = df['Target']
        noise = df['Interferer']
        snr_t = df['Targ_snr'].tolist()
        # snr_n = df['Noise_snr'].tolist()
        az_t = df['Az target'].tolist()
        az_n = df['Az noise'].tolist()
        brir_no = df['BrirListner']

        start_time = df["Start time"].tolist()
        # targ_snr = df["Targ_snr"].tolist()
        pad_choice = df["Pad choice"].tolist()
        # spatial = batch["ipd"]

        # new_targ,new_noise,mix,ref_signals,ref_signals_dry= InputFeature.OnlineMix_samp_CutFirst_reverb_Test(refs_,noise,snr_t,brir_no,az_t,az_n,self.config["Experiment"],start_time,pad_choice) #InputFeature.OnlineMix_samp_CutFirst_reverb_Test #InputFeature.OnlineMix_samp_CutFirst_Test_(refs_,noise,snr_t,self.config["Experiment"],start_time,pad_choice)
        
        new_targ,new_noise,mix,ref_signals = InputFeature.OnlineMix_samp_CutFirst_reverb_Test(refs_,noise,snr_t,brir_no,az_t,az_n,self.config["Experiment"],start_time,pad_choice) #InputFeature.OnlineMix_samp_CutFirst_reverb_Test #InputFeature.OnlineMix_samp_CutFirst_Test_(refs_,noise,snr_t,self.config["Experiment"],start_time,pad_choice)
        num_iterations = 100
        inference_times = []


        inference_time = end_time - start_time
        print(f"Inference time on CPU: {inference_time:.6f} seconds\n")
        
        mix= mix.cpu()
        print("starting time\n")
        # start_time = time.time()
        start_event.record()
        ests = self.forward(mix)
        end_event.record()
        # Waits for everything to finish
        torch.cuda.synchronize()

        inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        print(f"Inference time on GPU: {inference_time:.3f} ms")
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time on CPU: {inference_time:.6f} seconds\n")
        print("ending time")

                Waits for everything to finish
        torch.cuda.synchronize()

        inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds

        print(f"Inference time on GPU: {inference_time:.3f} ms")

        fs = 8000
        # mix = spyral(mix, fs, n_electrodes, n_carriers, spread)
        # mix = spyral(mix, fs, n_electrodes, n_carriers, spread)


        # df_test = pd.DataFrame(columns=["Mixed audio","Ref source1 audio","Ref source2 audio","Est source1 audio","Est source2 audio","Start time","Target path","Interferer path","Brir no","Pad choice","Az t","Az n", "snr","Mix type","si-sdr","sdr","stoi","pesq","UN_si-sdr S1","UN_stoi S1","UN_sdr S1","UN_pesq S1","UN_si-sdr S2","UN_stoi S2","UN_sdr S2","UN_pesq S2", "Avg UN_si-sdr","Avg UN_stoi","Avg UN_sdr","Avg UN_pesq"])
        df_test = pd.DataFrame(columns=["Mixed audio","Ref source1 audio","Ref source2 audio","Ref_dry source1 audio","Ref_dry source2 audio","Mix dry","Est source1 audio","Est source2 audio","Start time","Target path","Interferer path","Brir no","Pad choice","Az t","Az n", "snr","Mix type","si-sdr","sdr","stoi","pesq", "si-sdr_noPIT","Avg UN_si-sdr","Avg UN_stoi","Avg UN_sdr","Avg UN_pesq"])       
        sr =8000
        print(f"shape passed into it: {new_targ.shape , new_noise.shape, mix.shape,ests.shape }")
        # mix = mix[:,0,:]
        # # refs = refs[:,0,:]
        # print("length of start time: ", len(start_time),len(choice))
        print("len of test est is: ",len(ests),ests.shape)
        loss_sisdr_ =  PITLossWrapper(PairwiseNegSDR('sisdr'), pit_from='pw_mtx')
        # loss_sdr_ =  PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')



        # ref_signals_dry = np.pad(ref_signals_dry, [(0, 0)] + [(0, 0)] * 2, mode='constant')[:, :, :mix.shape[2]]
        print(f"unp: {mix.shape,ref_signals.shape }")
            # print(df_test.shape, i, mix[i].shape,len(df_test))

        for i in range(len(ests)):
            # print(f"in loop {i}, {ests[i].shape, ests.shape} \n"

            loss_sisdr,reordered_sources = loss_sisdr_(ests[i].reshape(1,ests[i].shape[0],ests[i].shape[1]), ref_signals[i].reshape(1,ref_signals[i].shape[0],ref_signals[i].shape[1]).cuda(),return_est=True)
            # loss_sdr = loss_sdr_(ests[i].reshape(1,ests[i].shape[0],ests[i].shape[1]), ref_signals_dry[i].reshape(1,ref_signals_dry[i].shape[0],ref_signals_dry[i].shape[1]).cuda())

            print(f"re_ordered_source shape is {reordered_sources.shape}, ref_signal shape {ref_signals.shape}, mix shape is {mix.shape}")
            est_source1 = reordered_sources[:,0,:]
            est_source2 = reordered_sources[:,1,:]

            # print(f"separating re-ordered source shapes, s1: {est_source1.shape}, s2: {est_source2.shape}")

            stoi_s1 = short_time_objective_intelligibility(est_source1[0], ref_signals[i][0,:].cuda(),sr)
            stoi_s2 = short_time_objective_intelligibility(est_source2[0], ref_signals[i][1,:].cuda(),sr)
            avg_stoi = np.mean([stoi_s1.item(),stoi_s2.item()])
            
            stoi_s1_up = short_time_objective_intelligibility(mix[i][0], ref_signals[i][0,:].cuda(),sr) #this should be affected by the source snr
            stoi_s2_up = short_time_objective_intelligibility(mix[i][0], ref_signals[i][1,:].cuda(),sr)
            avg_stoi_up = np.mean([stoi_s1_up.item(),stoi_s2_up.item()])

            pesq_s1 = perceptual_evaluation_speech_quality(est_source1[0], ref_signals[i][0,:].cuda(),sr,'nb')
            pesq_s2 = perceptual_evaluation_speech_quality(est_source2[0], ref_signals[i][1,:].cuda(),sr,'nb')
            avg_pesq = np.mean([pesq_s1.item(),pesq_s2.item()])

            pesq_s1_up = perceptual_evaluation_speech_quality(mix[i][0], ref_signals[i][0,:].cuda(),sr,'nb')
            pesq_s2_up = perceptual_evaluation_speech_quality(mix[i][0], ref_signals[i][1,:].cuda(),sr,'nb')
            avg_pesq_up = np.mean([pesq_s1_up.item(),pesq_s2_up.item()]) 

            source1_sidr_unp  = scale_invariant_signal_distortion_ratio(mix[i][0], ref_signals[i][0,:].cuda())
            source2_sidr_unp  = scale_invariant_signal_distortion_ratio(mix[i][0], ref_signals[i][1,:].cuda())
            avg_sidr_unp = np.mean([source1_sidr_unp.item(),source2_sidr_unp.item()])  

            source1_sidr_p  = scale_invariant_signal_distortion_ratio(est_source1[0], ref_signals[i][0,:].cuda())
            source2_sidr_p  = scale_invariant_signal_distortion_ratio(est_source2[0], ref_signals[i][1,:].cuda())
            avg_sidr_p= np.mean([source1_sidr_p.item(),source2_sidr_p.item()])  

            source1_sdr_unp  = signal_distortion_ratio(mix[i][0], ref_signals[i][0,:].cuda())
            source2_sdr_unp  = signal_distortion_ratio(mix[i][0], ref_signals[i][1,:].cuda())
            avg_sdr_unp = np.mean([source1_sdr_unp.item(),source2_sdr_unp.item()])     

            s1_sdr = signal_distortion_ratio(est_source1[0], ref_signals[i][0,:].cuda())
            s2_sdr = signal_distortion_ratio(est_source2[0], ref_signals[i][1,:].cuda())
            avg_sdr = np.mean([s1_sdr.item(),s2_sdr.item()])

            print(f" loss_sisdr are: {loss_sisdr}, loss sisdr(nonpit):{avg_sidr_p}, loss sdr: {avg_sdr}, stoi: {avg_stoi}, pesq:{avg_pesq}\n")
            print(f"sisdr_up: {avg_sidr_unp}, sdr_up: {avg_sdr_unp}, stoi_up: {avg_stoi_up}, pesq: {avg_pesq_up}, reordered_sources shape: {reordered_sources.shape} \n")  

            # print("this is i: ",i,df_test.shape,mix[i].shape,len(mix[i]),len(df_test))
            # print("next mix: ", mix[1].shape,mix[2].shape)

            df_test.loc[i, "Mixed audio"] = mix[i].cpu()
            df_test.loc[i, "Ref source1 audio"] = new_targ[i].cpu()
            df_test.loc[i, "Ref source2 audio"] = new_noise[i].cpu()
            # df_test.loc[i, "Ref_dry source1 audio"] = ref_signals_dry[i][0,:].cpu()
            # df_test.loc[i, "Ref_dry source2 audio"] = ref_signals_dry[i][1,:].cpu()
            # df_test.loc[i,"Mix dry"]  =       ref_signals_dry[i][0,:].cpu() + ref_signals_dry[i][1,:].cpu()
            df_test.loc[i, "Ref_dry source1 audio"] = ref_signals[i][0,:].cpu()
            df_test.loc[i, "Ref_dry source2 audio"] = ref_signals[i][1,:].cpu()
            df_test.loc[i,"Mix dry"]  =       ref_signals[i][0,:].cpu() + ref_signals[i][1,:].cpu()
                        
            df_test.loc[i, "Est source1 audio"] = est_source1.cpu()
            df_test.loc[i, "Est source2 audio"] = est_source2.cpu()
            df_test.loc[i, "Target path"] = refs_[i]
            df_test.loc[i, "Interferer path"] = noise[i]
            df_test.loc[i, "Brir no"] = brir_no[i]
            df_test.loc[i, "Start time"] = start_time[i]
            df_test.loc[i, "Pad choice"] = choice[i]
            df_test.loc[i, "Az t"] = az_t[i]#.cpu()
            df_test.loc[i, "Az n"] = az_n[i]#.cpu()
            # df_test.loc[i, "Mix type"] = "f-m"
            df_test.loc[i, "Mix type"] = "f-m"
            df_test.loc[i, "si-sdr"] = InputFeature.de_neg(loss_sisdr).cpu()
            df_test.loc[i, "Avg UN_si-sdr"] = avg_sidr_unp
            df_test.loc[i, "stoi"] = avg_stoi
            df_test.loc[i, "Avg UN_stoi"] = avg_stoi_up
            df_test.loc[i, "sdr"] = avg_sdr #InputFeature.de_neg(loss_sdr).cpu()
            df_test.loc[i, "Avg UN_sdr"] = avg_sdr_unp
            df_test.loc[i, "si-sdr_noPIT"]= avg_sidr_p
            # df_test.loc[i, "snr"] = snr_p.cpu()
            # df_test.loc[i, "UN_snr"] = snr_up.cpu()
            df_test.loc[i, "pesq"] = avg_pesq
            df_test.loc[i, "Avg UN_pesq"] = avg_pesq_up

            # print(df_test.shape, i, mix[i].shape,len(df_test))

        df_test.reset_index(inplace=True)
        df_test_dict =df_test.to_dict(into=OrderedDict)
        # print(df_test_dict)
        # print("Current working directory:", os.getcwd())
        print(self.config["save_dir_inf"]+self.config["save_name_inf"]+str(batch_idx)+".pt")
        torch.save(df_test_dict, self.config["save_dir_inf"]+self.config["save_name_inf"]+str(batch_idx)+".pt")


    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs,
    #             'progress_bar': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage):
        # Set random seed for weight initialization
        torch.manual_seed(42) 
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2) #step_size: 50 gamma:0.3
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode=self.scheduler_mode, factor=self.scheduler_factor, patience=self.patience, verbose=True, min_lr=1e-8)
        return {"optimizer":optimizer,"lr_scheduler":scheduler,"monitor":"val_loss"}
    def train_dataloader(self):
        df_train = pd.read_csv("../Data/wsj0/"+self.config["train_file"])#Train-20k.csvTrain-30sec Train-8k-posSNR.csv
        # df_test = df_test[:8]
        train_ds = Datasets(df_train,self.config["Experiment"])
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,drop_last=True,pin_memory=True)
        return train_dl
    def val_dataloader(self):
        df_val = pd.read_csv("../Data/wsj0/"+self.config["val_file"])#Val-10k.csv ../Data/WSJ2Mix-2spk/
        # df_val = df_val[:4]
        val_ds = Datasets(df_val,self.config["Experiment"])
        val_dl = torch.utils.data.DataLoader(val_ds,batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers,drop_last=True,pin_memory=True)
        return val_dl
    def test_dataloader(self):
        df_test = pd.read_csv("../Data/wsj0/"+self.config["test_file"])#Test-posSNR
        df_test = df_test[:1]
        print("df test has",df_test)
        test_ds = Datasets(df_test,self.config["Experiment"])
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=self.num_workers,drop_last=True)#self.batch_size
        return test_dl















     
