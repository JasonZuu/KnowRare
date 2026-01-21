import os
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb

from configs.algo_config import AdvConfig
from models.lstm_based import LSTMBasedModel
from run_fn.test_fn import _test_loop
from models.tracker import PerformanceTracker
from dataset.dataset_load_fn import get_dataloader_fn
from models import LinearDiscriminator
from utils.misc import set_grad_flag
from dataset.datasets import MIMICDataset


def knowrare_train_fn(config: AdvConfig,
                    model: LSTMBasedModel,
                    dis_model: LinearDiscriminator,
                    train_dataset: MIMICDataset,
                    val_dataset: MIMICDataset,
                    write_log: bool = True,
                    target_icd9=None):
    """
    Adversarial training function for conditional adversarial training at disease level

    Args:
        config: configuration for the training
        model: model to be trained
        dis_model: discriminator model. For traditional adversarial training, please provide Discriminator model with pred_f as input but not used.
        train_dataset: training dataset
        val_dataset: validation dataset
        write_log: whether to write log
        target_icd9: target icd9 code for conditional adversarial training
    """
    # initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.lr_decay_steps,
                                                gamma=config.lr_decay_gamma)
    dis_optimizer = optim.Adam(dis_model.parameters(), config.dis_lr)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer,
                                                    step_size=config.lr_decay_steps,
                                                    gamma=config.lr_decay_gamma)

    tracker = PerformanceTracker(early_stop_epochs=config.early_stop_epochs,
                                 metric=config.select_metric)
    if write_log:
        run = wandb.init(project=config.project, group=config.study_name, name=f"{config.study_name}-train_seed{config.seed}",
                         mode='offline')
    else:
        run = None

    if config.resampling:
        train_loader = get_dataloader_fn(algo_config=config, dataset=train_dataset, mode="resampling")
    else:
        train_loader = get_dataloader_fn(algo_config=config, dataset=train_dataset, mode="train")
    if type(val_dataset) == list:
        val_loader = [get_dataloader_fn(algo_config=config, dataset=val_dataset, mode="val") for val_dataset in val_dataset]
    else:
        val_loader = get_dataloader_fn(algo_config=config, dataset=val_dataset, mode="val")

    for i_epoch in range(config.epochs_num):
        _advdise_train_loop(model=model, dis_model=dis_model, train_loader=train_loader, 
                            optimizer=optimizer, dis_optimizer=dis_optimizer, 
                            dis_step=config.dis_step, dis_coeff=config.dis_coeff,
                            device=config.device, run=run, i_epoch=i_epoch)
        if (i_epoch + 1) > config.lr_warmup_epochs:
            scheduler.step()
            dis_scheduler.step()
        
        # calculate validation metric
        if type(val_loader) == list:
            val_metrics = {}
            val_metric = {}
            for i, loader in enumerate(val_loader):
                _val_metric = _test_loop(model=model, test_loader=loader, device=config.device)
                for metric, value in _val_metric.items():
                    val_metrics[metric] = val_metrics.get(metric, []) + [value]
            for metric, values in val_metrics.items():
                mean_value = np.mean(values)
                val_metric[metric] = mean_value
        else:
            val_metric = _test_loop(model=model, test_loader=val_loader, device=config.device)

        if write_log:
            val_metric["epoch"] = i_epoch
            run.log(val_metric)
        state_dict = {"model": model.state_dict(), 'dis_model': dis_model.state_dict()}
        early_stop_flag = tracker.update(val_metric, state_dict)
        if early_stop_flag:
            break

    best_model_state_dict = tracker.export_best_model_state_dict()
    best_val_metric_dict = tracker.export_best_metric_dict()
    model.load_state_dict(best_model_state_dict["model"])

    if write_log:
        model_path = os.path.join(config.log_dir, 'model.pth') if target_icd9 is None \
                             else os.path.join(config.log_dir, f'model-{target_icd9}.pth')
        torch.save(best_model_state_dict, model_path)
        run.finish()

    return best_val_metric_dict


def _advdise_train_loop(model, dis_model, train_loader,
                        optimizer, dis_optimizer, dis_step, dis_coeff,
                        device:str, run, i_epoch):
    """
    function to train the net
    """
    model.train()
    dis_model.train()
    i_step = i_epoch*len(train_loader)
    train_iter = iter(train_loader)

    pbar = tqdm(total=len(train_loader), desc=f'Disease-level Adv (knowrare) Training ({i_epoch+1} epoch)', unit='batch')
    for data in train_loader:
        # train the discriminator
        set_grad_flag(model, False)
        set_grad_flag(dis_model, True)
        for i_dis_step in range(dis_step):
            dis_data, train_iter = _get_next_dis_batch(train_iter, train_loader)
            demo, ts, icd9_code_idx = dis_data['demography'], dis_data['time_series'], dis_data['icd9_code_idx']
            demo, ts, icd9_code_idx = demo.to(device), ts.to(device), icd9_code_idx.to(device)

            dis_optimizer.zero_grad()

            _, info = model.forward_with_hidden(demo, ts, use_output_activate=True)
            latent_f = info["embed"]
            pred_f = info["pred"]
            dis_scores = dis_model(latent_f, pred_f, use_softmax=True, use_sigmoid=False)
            dis_loss = F.cross_entropy(dis_scores, icd9_code_idx)
            dis_loss.backward()
            dis_optimizer.step()

        # train the model
        set_grad_flag(model, True)
        set_grad_flag(dis_model, False)

        demo, ts, label, icd9_code_idx = data['demography'], data['time_series'], data['label'], data['icd9_code_idx']
        demo, ts, label, icd9_code_idx = demo.to(device), ts.to(device), label.to(device), icd9_code_idx.to(device)

        optimizer.zero_grad()
        y_score, info = model.forward_with_hidden(demo, ts, use_output_activate=True)
        latent_f = info["embed"]
        pred_f = info["pred"]
        dis_scores = dis_model(latent_f, pred_f, use_softmax=True, use_sigmoid=False)
        dis_loss = F.cross_entropy(dis_scores, icd9_code_idx)
        if label.shape[1] == 1:
            cls_loss = F.binary_cross_entropy(y_score, label)
        else:
            cls_loss = F.cross_entropy(y_score, label)
        loss = cls_loss - dis_coeff * dis_loss

        loss.backward()
        optimizer.step()

        # display loss
        loss_dict = {"loss": loss.item(), "cls_loss": cls_loss.item(), "dis_loss": dis_loss.item()}
        pbar.set_postfix(**loss_dict)

        lr = optimizer.param_groups[0]['lr']
        loss_dict['lr'] = lr 

        if run is not None:
            loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
            run.log(loss_dict, step=i_step)

        i_step += 1
        pbar.update()


def _get_next_dis_batch(train_iter, train_loader):
    try:
        train_data = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        train_data = next(train_iter)
    
    return train_data, train_iter

