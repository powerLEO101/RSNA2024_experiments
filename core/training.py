import torch
import wandb
import pandas as pd

from tqdm import tqdm

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
        if accelerator.is_local_main_process:
            wandb.log({f'lr': lr, 'train_step_loss': loss.item()})
        bar.set_postfix_str(f'Epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process:
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

    if accelerator.is_local_main_process:
        wandb.log({f'valid_epoch_loss': running_loss})
