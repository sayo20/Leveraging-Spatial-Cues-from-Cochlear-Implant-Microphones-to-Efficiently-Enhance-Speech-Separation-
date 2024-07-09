import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from lightning_conf import get_config, get_importer
from Callbacks import EarlyStopping_

import wandb
wandb.login(key="d39daf4c1d304b5fa7f5e1a79958bd85ef105f30")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", help="Experiment number")
args = parser.parse_args()
exp = args.exp

# Import the correct Lightning module based on the experiment
if exp == "10":
    from lightning_baseline import Lightning
elif exp == "-10":
    from lightning_baseline_anech import Lightning
elif exp in ["-1", "-1.5", "-2"]:
    from lightning_reverb import Lightning
elif exp in ["0", "1", "2", "3"]:
    from lightning_reverb_ipds import Lightning
elif exp in ["4", "5"]:
    from lightning_ipd_AF_big import Lightning
elif exp == "6":
    from lightning_bi_AF_big import Lightning

# Import the correct model based on the experiment
if exp in ["10", "-1", "-10"]:
    from model_baseline import SuDORMRF
elif exp in ["-1.5", "-2"]:
    from model_twoChannels import SuDORMRF
elif exp in ["0", "2"]:
    from model_oneChannel_ipd import SuDORMRF
elif exp in ["1", "3"]:
    from model_twoChannels_ipd import SuDORMRF
elif exp == "4":
    from model_uniIPD_AF import SuDORMRF
elif exp == "5":
    from model_BIIPD_AF import SuDORMRF
elif exp == "6":
    from model_BI_AF import SuDORMRF

def Train():
    config_name, run_name, test_name = get_config(exp)
    
    wandb_logger = WandbLogger(project="Finals", config=config_name, name=run_name)
    config = wandb_logger.experiment.config

    early_stop_callback = EarlyStopping_()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["save_dir"],
        save_top_k=3,
        verbose=True,
        filename=config["save_name"] + "-final{epoch:02d}-{val_loss:.2f}",
        mode="min"
    )

    trainer = Trainer(
        gradient_clip_val=5.0,
        max_epochs=config["epochs"],
        fast_dev_run=False,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=wandb_logger,
        profiler="simple"
    )

    model = Lightning(config, SuDORMRF)
    trainer.fit(model=model.cuda(), ckpt_path=config["Best_epoch"])
    trainer.save_checkpoint(os.path.join(config["save_dir"], config["save_name"] + ".ckpt"))

if __name__ == "__main__":
    Train()
