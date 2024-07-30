from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Tuple
import shutil
from pathlib import Path
import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from bsrnn import SourceSeparationDataset, MidSideDataset, collate_fn
from bsrnn import BandSplitRNN, PLModel
import bsrnn.utils as utils

import logging
import traceback

from pytorch_lightning.loggers import WandbLogger
import wandb
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    if cfg.enable_midside:
        train_dataset = MidSideDataset(**cfg.train_dataset)
    else:
        train_dataset = SourceSeparationDataset(**cfg.train_dataset)

    rng = torch.Generator().manual_seed(1204)

    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn,
        generator=rng
    )

    if hasattr(cfg, 'val_dataset'):
        if cfg.enable_midside:
            val_dataset = MidSideDataset(**cfg.val_dataset)
        else:
            val_dataset = SourceSeparationDataset(**cfg.val_dataset)

        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn,
            generator=rng
        )
    else:
        val_loader = None

    return train_loader, val_loader


def initialize_featurizer(
        cfg: DictConfig
) -> Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(
        cfg.featurizer.direct_transform,
    )
    inv_featurizer = instantiate(
        cfg.featurizer.inverse_transform,
    )
    featurizer.to(DEV)
    inv_featurizer.to(DEV)
    return featurizer, inv_featurizer


def initialize_augmentations(
        cfg: DictConfig
) -> nn.Module:
    """
    Initializes augmentations.
    """
    augs = instantiate(cfg.augmentations)
    augs = nn.Sequential(*augs.values())
    return augs


def initialize_model(
        cfg: DictConfig
) -> Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    # initialize model
    model = BandSplitRNN(
        **cfg.model
    )
    model.to(DEV)

    # initialize optimizer
    if hasattr(cfg, 'opt'):
        opt = instantiate(
            cfg.opt,
            params=model.parameters()
        )
    else:
        opt = None

    # initialize scheduler
    if hasattr(cfg, 'sch'):
        if hasattr(cfg.sch, '_target_'):
            # other than LambdaLR
            sch = instantiate(
                cfg.sch,
                optimizer=opt
            )
        else:
            # if LambdaLR
            lr_lambda = lambda epoch: (
                cfg.sch.alpha ** (cfg.sch.warmup_step - epoch)
                if epoch < cfg.sch.warmup_step
                else cfg.sch.gamma ** (epoch - cfg.sch.warmup_step)
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lr_lambda
            )
    else:
        sch = None

    return model, opt, sch


def initialize_utils(
        cfg: DictConfig
):
    # change model and logs saving directory to logging directory of hydra
    if HydraConfig.instance().cfg is not None:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        save_dir = hydra_cfg['runtime']['output_dir']
        cfg.logger.tensorboard.save_dir = save_dir + cfg.logger.tensorboard.save_dir # update tensorboard save_dir
        # cfg.logger.wandb.save_dir = save_dir + cfg.logger.wandb.save_dir # update wandb save_dir
        if hasattr(cfg.callbacks, 'model_ckpt'):
            cfg.callbacks.model_ckpt.dirpath = save_dir + cfg.callbacks.model_ckpt.dirpath
    # delete early stopping if there is no validation dataset
    if not hasattr(cfg, 'val_dataset') and hasattr(cfg.callbacks, 'early_stop'):
        del cfg.callbacks.early_stop

    # initialize loggers
    logger: List = []
    if "logger" in cfg:
        # logger = instantiate(cfg.logger)
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))
                # if lg_conf._target_ == "pytorch_lightning.loggers.WandbLogger":
                #     wandb_logger = WandbLogger(**lg_conf)
                #     logger.append(wandb_logger)
                # else:
                #     logger.append(instantiate(lg_conf))

        # Initialize wandb logger
        for wandb_logger in [l for l in logger if isinstance(l, WandbLogger)]:
                utils.wandb_login(key=cfg.wandb_api_key)
                # utils.wandb_watch_all(wandb_logger, model) # TODO buggy
                break
    
    # initialize callbacks
    callbacks = list(instantiate(cfg.callbacks).values())

    return logger, callbacks


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            featurizer: nn.Module,
            inverse_featurizer: nn.Module,
            augmentations: nn.Module,
            opt: Optimizer,
            sch: lr_scheduler._LRScheduler,
            cfg: DictConfig = None
    ):
        super().__init__()

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.augmentations = augmentations
        self.opt = opt
        self.sch = sch
        self.cfg = cfg
        self.num_iters = 0
        self.cur_epoch = 0

        self.mae_specR = nn.L1Loss()
        self.mae_specI = nn.L1Loss()
        self.mae_time = nn.L1Loss()

        if self.cfg.wandb.name is None:
            self.ckpt_dir = Path('checkpoints/test')
        else:
            self.ckpt_dir = Path(f'checkpoints/{self.cfg.wandb.name}')
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch):
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        loss, loss_dict, usdr = self.step(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       self.cfg.trainer.gradient_clip_val)
        self.opt.step()
        self.opt.zero_grad()

        wandb.log({"train/loss": loss.item()}, step=self.num_iters)
        wandb.log({"train/usdr": usdr.item()}, step=self.num_iters)
        wandb.log({"train/lr": self.opt.param_groups[0]['lr']}, step=self.num_iters)
        self.num_iters += 1

        return loss

    def validation_step(self, batch):
        loss, loss_dict, usdr = self.step(batch)
        return loss.item(), usdr.item()

    def step(self, batchT: torch.Tensor):
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        # augmentations
        batchT = self.augmentations(batchT)

        # STFT
        batchS = self.featurizer(batchT)
        mixS, tgtS = batchS[:, 0], batchS[:, 1]

        # apply model
        predS = self.model(mixS)

        # iSTFT
        batchT = self.inverse_featurizer(
            torch.stack((predS, tgtS), dim=1)
        )
        predT, tgtT = batchT[:, 0], batchT[:, 1]

        # compute loss
        loss, loss_dict = self.compute_losses(
            predS, tgtS,
            predT, tgtT
        )

        # compute metrics
        usdr = self.compute_usdr(predT, tgtT)

        return loss, loss_dict, usdr

    def compute_losses(self,
                       predS: torch.Tensor,
                       tgtS: torch.Tensor,
                       predT: torch.Tensor,
                       tgtT: torch.Tensor):
        # frequency domain
        lossR = self.mae_specR(
            predS.real, tgtS.real
        )
        lossI = self.mae_specI(
            predS.imag, tgtS.imag
        )
        # time domain
        lossT = self.mae_time(
            predT, tgtT
        )
        loss_dict = {
            "lossSpecR": lossR,
            "lossSpecI": lossI,
            "lossTime": lossT
        }
        loss = lossR + lossI + lossT
        return loss, loss_dict

    @staticmethod
    def compute_usdr(
            predT: torch.Tensor,
            tgtT: torch.Tensor,
            delta: float = 1e-7
    ) -> torch.Tensor:
        """Computest the Unweighted Signal to Distortion Ratio (USDR).
        """
        num = torch.sum(torch.square(tgtT), dim=(1, 2))
        den = torch.sum(torch.square(tgtT - predT), dim=(1, 2))
        num += delta
        den += delta
        usdr = 10 * torch.log10(num / den)
        return usdr.mean()

    def configure_optimizers(self):
        return [self.opt], [self.sch]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
    
    def save_checkpoint(self, ckpt_path):
        ckpt = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    def load_checkpoint(self):
        ckpt = torch.load(self.cfg.load_checkpoint)
        self.cur_epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
    
    def train(self):
        best_valid_loss = float('inf')
        best_ckpt_path = None
        last_ckpt_path = None
        for epoch in tqdm(range(self.cfg.trainer.max_epochs), ascii=True):
            self.cur_epoch = epoch
            for batch in tqdm(self.train_loader, ascii=True, leave=False):
                batch = batch.to(DEV)
                self.training_step(batch)
            self.sch.step()

            loss_sum, usdr_sum = 0, 0
            with torch.inference_mode():
                for batch in tqdm(self.val_loader, ascii=True, leave=False):
                    batch = batch.to(DEV)
                    loss, usdr = self.validation_step(batch)
                    loss_sum += loss
                    usdr_sum += usdr
            loss_sum /= len(self.val_loader)
            usdr_sum /= len(self.val_loader)

            if loss_sum < best_valid_loss:
                best_valid_loss = loss_sum
                ckpt_path = self.ckpt_dir / f'best_model_epoch{self.cur_epoch}_{loss_sum:4f}.pt'
                if best_ckpt_path is not None:
                    best_ckpt_path.unlink(missing_ok=True)
                best_ckpt_path = ckpt_path
                self.save_checkpoint(ckpt_path)
            
            if last_ckpt_path is not None:
                last_ckpt_path.unlink(missing_ok=True)
            last_ckpt_path = self.ckpt_dir / f"last_model_epoch{self.cur_epoch}_{loss_sum:4f}.pt"
            self.save_checkpoint(last_ckpt_path)

            wandb.log({"val/loss": loss_sum}, step=self.num_iters)
            wandb.log({"val/usdr": usdr_sum}, step=self.num_iters)
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(
        # device: str,
        cfg: DictConfig
) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    torch.manual_seed(1204)
    random.seed(1204)

    log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders, featurizers.")
    train_loader, val_loader = initialize_loaders(cfg)
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    augs = initialize_augmentations(cfg)

    log.info("Initializing model, optimizer, scheduler.")
    model, opt, sch = initialize_model(cfg)

    trainer = Trainer(
        model,
        train_loader, val_loader,
        featurizer, inverse_featurizer,
        augs,
        opt, sch,
        cfg
    )

    wandb.init(project=cfg.wandb.project,
               name=cfg.wandb.name,
               config=OmegaConf.to_container(cfg, resolve=True))

    log.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        log.error(traceback.format_exc())

    log.info("Training finished!")

    if cfg.trainer.fast_dev_run:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        shutil.rmtree(hydra_cfg['runtime']['output_dir'])


if __name__ == "__main__":
    train()
