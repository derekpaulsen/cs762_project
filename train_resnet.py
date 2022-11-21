import sys
sys.path.append('.')
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from utils.data import accuracy
from utils.cifar10 import load_cifar10
from multiprocessing import cpu_count
from tqdm import tqdm

torch.set_num_threads(cpu_count())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '/data/train'


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
    return {'val_loss' : np.mean(losses), 'accuracy' : np.mean(accs)}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
 
    optimizer = optim.SGD(model.parameters(), lr=.01,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    ## Set up cutom optimizer with weight decay
    #optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    ## Set up one-cycle learning rate scheduler
    # this didn't work at all
    #sched = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=.01, max_lr=.1)
    #sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for inputs, labels in tqdm(train_loader):
            pred = model.forward(inputs)       
            loss = loss_fn(pred, labels)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        print(result)
        result['lrs'] = lrs
        history.append(result)
    return history



def load_resnet18():
    model = models.resnet18(pretrained=False)
    
    #model.fc = nn.Sequential( nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    model.fc = nn.Sequential( nn.Linear(512, 10))
    model.to(device)
    return model

def load_resnet34():
    model = models.resnet34(pretrained=False)
    
    #model.fc = nn.Sequential( nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    model.fc = nn.Sequential( nn.Linear(512, 10))
    model.to(device)
    return model

def load_vgg16():
    model = models.vgg16(pretrained=False)
    # model.fc = nn.Sequential( nn.Linear(-1, 10))
    model.classifier[6] = nn.Linear(4096, 10)
    model.to(device)
    return model



def main():
    
    # model = load_resnet34()
    # model = load_resnet18()
    model = load_vgg16()
    # use syn training and real validiation
    train_dl, valid_dl = load_cifar10(0.5, 1, 0.5, 0, device)
    #import pdb; pdb.set_trace()
    epochs = 250
    max_lr = 0.01
    #grad_clip = 0.1
    grad_clip = None
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history = []
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                                 grad_clip=grad_clip, 
                                 weight_decay=weight_decay, 
                                 opt_func=opt_func)
    # torch.save(model, 'cifar.pth')
    # torch.save(model, 'cifar_resnet34.pth')
    # torch.save(model, 'cifar_vgg16_1_1_0_0.pth')
    #torch.save(model, 'cifar_vgg16_0_1_1_0.pth')
    torch.save(model, 'cifar_vgg16_05_1_05_0.pth')


if __name__ == '__main__':
    main()
