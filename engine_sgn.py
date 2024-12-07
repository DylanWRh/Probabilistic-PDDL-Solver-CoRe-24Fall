from typing import Iterable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F



def train_one_epoch(
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
):
    model.train(True)
    log_header = f'Epoch [{epoch:03d}]'
    total_loss = 0.0
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    
    for batch_idx, (data, labels) in progress_bar:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_description(f'{log_header} Iter [{batch_idx+1:03d}/{len(data_loader):03d}]: Loss: {total_loss / (batch_idx + 1)}')
        
    return total_loss / len(data_loader)
            


@torch.no_grad()
def evaluate(   
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int
):
    model.eval()
    log_header = f'Evaluation [{epoch:03d}]'
    total_loss = 0.0
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    
    for batch_idx, (data, labels) in progress_bar:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        
        progress_bar.set_description(f'{log_header} Iter [{batch_idx+1:03d}/{len(data_loader):03d}]: Loss: {total_loss / (batch_idx + 1)}')
    
    print(f'{log_header}: Average Loss: {total_loss / len(data_loader)}')
    return total_loss / len(data_loader)