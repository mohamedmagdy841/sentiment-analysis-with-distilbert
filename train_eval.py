import evaluate
import torch
from torch import nn
from tqdm.auto import tqdm, trange
import numpy as np
from collections import defaultdict


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(
  model: torch.nn.Module,
  data_loader: torch.utils.data.DataLoader,
  optimizer: torch.optim.Optimizer,
  scheduler,
  device: torch.device=device
  
):
    train_metric = evaluate.load("accuracy")
    
    train_loss = 0
    model.train()
    for i, batch in enumerate(tqdm(data_loader,desc="train_steps",position=1,leave=False)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_logits = outputs.logits
        train_pred = torch.argmax(train_logits, dim=-1)
        train_metric.add_batch(predictions=train_pred, references=batch["labels"])
    train_loss /= len(data_loader)
    
    return train_metric.compute()['accuracy'], train_loss

def eval_epoch(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               device: torch.device=device):
    
    val_metric = evaluate.load("accuracy")
    val_loss = 0

    model.eval()
    for i, batch in enumerate(tqdm(data_loader,desc="val_steps",position=2,leave=False)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
            val_loss += outputs.loss
            val_logits = outputs.logits
            val_pred = torch.argmax(val_logits, dim=-1)
            val_metric.add_batch(predictions=val_pred, references=batch["labels"])
    val_loss = val_loss.clone() / len(data_loader)
    
    return val_metric.compute()['accuracy'], val_loss


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler,
          epochs):
    
    history = defaultdict(list)
    
    for epoch in tqdm(range(epochs), desc="Epoch", position=0):

        train_acc, train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
          )

        val_acc, val_loss = eval_epoch(
            model,
            val_dataloader,
            device
          )

        print(f"Train loss: {train_loss:.4} | Train accuracy: {train_acc:.0%}")
        print(f"Val   loss: {val_loss:.4} | Val   accuracy: {val_acc:.0%}")

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        history['train_loss'] = torch.tensor(model_results['train_loss'], device='cpu')
        history['val_loss'] = torch.tensor(model_results['val_loss'], device='cpu')
        
    return history