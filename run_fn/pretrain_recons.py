import os
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import wandb
import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path
import numpy as np

from configs.algo_config import ReconstructionConfig
from models.lstm_based import LSTMBasedModel
from models.decoder import AutoregressiveDecoder
from models.tracker import PerformanceTracker
from dataset.dataset_load_fn import get_dataloader_fn


def reconstruct_pretrain_fn(config: ReconstructionConfig,
                            model: LSTMBasedModel,
                            decoder: AutoregressiveDecoder,
                            train_dataset: torch.utils.data.Dataset,
                            val_dataset: torch.utils.data.Dataset,
                            write_log: bool = True):
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.lr_decay_steps,
                                                gamma=config.lr_decay_gamma)

    tracker = PerformanceTracker(early_stop_epochs=config.early_stop_epochs,
                                 metric=config.select_metric,
                                 direction="minimize")
    if write_log:
        run = wandb.init(project=config.project, group=config.study_name, name=f"{config.study_name}-reconstruction_train_seed{config.seed}",
                         mode="offline")
    else:
        run = None

    train_loader = get_dataloader_fn(algo_config=config, dataset=train_dataset, mode="train")
    if isinstance(val_dataset, list):
        val_loader = [get_dataloader_fn(algo_config=config, dataset=val, mode="val") for val in val_dataset]
    else:
        val_loader = get_dataloader_fn(algo_config=config, dataset=val_dataset, mode="val")

    for i_epoch in range(config.epochs_num):
        _reconstruction_train_loop(model=model, decoder=decoder, train_loader=train_loader,
                                   optimizer=optimizer, device=config.device, run=run, i_epoch=i_epoch)
        if (i_epoch + 1) > config.lr_warmup_epochs:
            scheduler.step()

        # Validation metrics can be customized as needed, here we log the reconstruction loss
        val_metric = _reconstruction_test_loop(model=model, decoder=decoder, val_loader=val_loader, device=config.device)

        if write_log:
            val_metric["epoch"] = i_epoch
            run.log(val_metric)
        state_dict = {"model": model.state_dict(), "decoder": decoder.state_dict()}
        early_stop_flag = tracker.update(val_metric, state_dict)
        if early_stop_flag:
            break

    best_model_state_dict = tracker.export_best_model_state_dict()
    best_val_metric_dict = tracker.export_best_metric_dict()
    model.load_state_dict(best_model_state_dict["model"])
    decoder.load_state_dict(best_model_state_dict["decoder"])

    if write_log:
        model_path = os.path.join(config.log_dir, f'{config.model}.pth')
        torch.save(best_model_state_dict, model_path)
        run.finish()

    return best_val_metric_dict


def _reconstruction_train_loop(model, decoder, train_loader, optimizer, device: str, run, i_epoch):
    """
    Train loop for reconstruction
    """
    model.train()
    decoder.train()
    i_step = i_epoch * len(train_loader)
    pbar = tqdm(total=len(train_loader), desc=f'Reconstruction Training ({i_epoch+1} epoch)', unit='batch')

    for data in train_loader:
        demo, ts = data['demography'], data['time_series']
        demo, ts = demo.to(device), ts.to(device)
        
        # Reconstruct autoregressively
        optimizer.zero_grad()
        timesteps = ts.size(1)
        reconstruction = torch.zeros_like(ts[:, 1:, :])
        for i in range(timesteps-1):
            emb = model._embed(demo, ts[:, :i+1, :])
            _reconstruction = decoder(emb)
            reconstruction[:, i, :] = _reconstruction

        loss = F.mse_loss(reconstruction, ts[:, 1:, :])
        loss.backward()
        optimizer.step()

        # Display loss
        pbar.set_postfix(**{'loss(batch)': loss.item()})

        lr = optimizer.param_groups[0]['lr']
        log_dict = {"lr": lr, "loss": loss.item()}

        if run is not None:
            run.log(log_dict, step=i_step)

        i_step += 1
        pbar.update()
    pbar.close()


@torch.no_grad()
def _reconstruction_test_loop(model, decoder, val_loader, device: str):
    """
    Test loop for reconstruction
    """
    model.eval()
    decoder.eval()
    loss_list = []
    for data in val_loader:
        demo, ts = data['demography'], data['time_series']
        demo, ts = demo.to(device), ts.to(device)
        
        timesteps = ts.size(1)
        reconstruction = torch.zeros_like(ts[:, 1:, :])
        for i in range(timesteps - 1):
            emb = model._embed(demo, ts[:, :i+1, :])
            _reconstruction = decoder(emb)
            reconstruction[:, i, :] = _reconstruction

        mse_element_wise = F.mse_loss(reconstruction, ts[:, 1:, :], reduction='none')
        mse = mse_element_wise.mean(dim=(1, 2))
        loss_list.extend(mse.cpu().numpy().tolist())

    mse = np.mean(loss_list)
    return {"mse": mse}
