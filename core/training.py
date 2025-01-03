import torch
import wandb
import pandas as pd

from tqdm import tqdm
from os import environ

IS_LOCAL = bool('LOCAL_TEST' in environ)

def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = batch['label']
        optimizer.zero_grad()
        pred_labels = model(image)
        loss = criterion(pred_labels, label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({f'lr': lr, 'train_step_loss': loss.item()})
        bar.set_postfix_str(f'Epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({f'train_epoch_loss': running_loss})


def valid_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    df_pred = []
    df_sol = []
    for step, batch in bar:
        # B C X Y
        image, label_ = batch['image'], batch['label']
        with torch.no_grad():
            pred_label_ = model(image)
        loss = criterion(pred_label_, label_)
        pred_label, label = accelerator.gather_for_metrics((pred_label_, label_))
        df_pred.extend(pred_label.tolist())
        df_sol.extend(label.tolist())
        running_loss += (loss.item() - running_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, Valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')

    # df_pred = pd.DataFrame(data=df_pred)
    # df_sol = pd.DataFrame(data=df_sol)
    # accelerator.print(df_pred.shape, df_sol.shape)
    # auc_score = score(df_sol, df_pred)
    # accelerator.print(f'Auc Score: {auc_score}')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({f'valid_epoch_loss': running_loss})


def train_one_epoch_w_pos(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_bce_loss = 0.0
    running_l2_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        head = batch['head']
        label = batch['label']
        pos = batch['pos']
        optimizer.zero_grad()
        pred_labels = model(image, head)
        loss, bce_loss, l2_loss = criterion(pred_labels, [label, pos])
        running_loss += (loss.item() - running_loss) / (step + 1)
        running_bce_loss += (bce_loss.item() - running_bce_loss) / (step + 1)
        running_l2_loss += (l2_loss.item() - running_l2_loss) / (step + 1)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({f'lr': lr, 'train_step_loss': loss.item()})
        bar.set_postfix_str(f'Epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({f'train_epoch_loss': running_loss})
        wandb.log({f'train_epoch_bce_loss': running_bce_loss})
        wandb.log({f'train_epoch_l2_loss': running_l2_loss})

def valid_one_epoch_w_pos(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_bce_loss = 0.0
    running_l2_loss = 0.0
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    df_pred = []
    df_sol = []
    for step, batch in bar:
        # B C X Y
        image = batch['image']
        head = batch['head']
        label = batch['label']
        pos = batch['pos']
        with torch.no_grad():
            pred_label = model(image, head)
        loss, bce_loss, l2_loss = criterion(pred_label, [label, pos])
        running_loss += (loss.item() - running_loss) / (step + 1)
        running_bce_loss += (bce_loss.item() - running_bce_loss) / (step + 1)
        running_l2_loss += (l2_loss.item() - running_l2_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, Valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')
    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({f'valid_epoch_loss': running_loss})
        wandb.log({f'valid_epoch_bce_loss': running_bce_loss})
        wandb.log({f'valid_epoch_l2_loss': running_l2_loss})