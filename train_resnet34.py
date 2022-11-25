import sys
sys.path.append('.')
import numpy as np
import torch
from torch import nn
from torch import optim
import json
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from utils.data import accuracy
from utils.cifar10 import load_cifar10
from multiprocessing import cpu_count
from tqdm import tqdm
from argparse import ArgumentParser
from time import time

argp = ArgumentParser()
argp.add_argument('--data_props', required=True, type=str)

torch.set_num_threads(cpu_count())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '/data/train'
ARGS = {}


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    accs = []
    for inputs, labels in val_loader:
        out = model.forward(inputs)
        losses.append( loss_fn(out, labels).cpu())
        accs.append(accuracy(out, labels).cpu())
    return {'val_loss' : float(np.mean(losses)), 'accuracy' : float(np.mean(accs))}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(epochs, model, train_loader, val_loader):
    start_time = time()
    torch.cuda.empty_cache()
    history = []

    ## Set up cutom optimizer with weight decay
    #optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    ## Set up one-cycle learning rate scheduler
    # this didn't work at all
    #sched = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=.01, max_lr=.1)
    #sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
 
    optimizer = optim.SGD(model.parameters(),
                            lr=.01,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True
                )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_e = time()
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for inputs, labels in tqdm(train_loader):
            pred = model.forward(inputs)       
            loss = loss_fn(pred, labels)
            train_losses.append(loss)
            loss.backward()
            
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        epoch_time = time() - start_e       
        # Validation phase
        result = evaluate(model, val_loader)
        result.update(ARGS)
        result['train_loss'] = float(torch.stack(train_losses).mean().item())
        result['epoch'] = epoch
        result['epoch_time'] = epoch_time
        result['start_time'] = start_time
        print(json.dumps(result))
        result['lrs'] = lrs
        history.append(result)
    return history


def load_resnet34():
    model = models.resnet34(pretrained=False)
    
    #model.fc = nn.Sequential( nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    model.fc = nn.Sequential( nn.Linear(512, 10))
    model.to(device)
    return model


def main(args):
    global ARGS
    ARGS.update(args._get_kwargs())
    ARGS['device'] = device
    props = eval(args.data_props)
    model = load_resnet34()
    # use syn training and real validiation
    train_dl, valid_dl = load_cifar10(*props, device)
    #import pdb; pdb.set_trace()
    epochs = 250

    history = []
    history += train(epochs, model, train_dl, valid_dl)
    

if __name__ == '__main__':
    main(argp.parse_args())
