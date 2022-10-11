import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


device = 'cuda'
data_dir = '/data/train'

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

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
 
    optimizer = optim.SGD(model.parameters(), lr=.05,
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
        for inputs, labels in train_loader:
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


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class CudaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, device):

        inputs = []
        labels = []
        for i, l in dataset:
            inputs.append(i)
            labels.append(l)

        self._len = len(dataset)
        self.inputs = torch.stack(inputs).to(device)
        self.labels = torch.tensor(labels).to(device)


    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])



def load_cifar10():
    stats = ((0.4914, 0.4822, 0.4465), (0.24705882352941178, 0.24352941176470588, 0.2615686274509804))
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])


    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_tfms)
    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, pin_memory=True, num_workers=16)
    #import pdb; pdb.set_trace()
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=valid_tfms)


    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, pin_memory=True, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return DeviceDataLoader(trainloader, device), DeviceDataLoader(testloader, device)

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader


def load_resnet18():
    model = models.resnet18(weights=None)
    
    #model.fc = nn.Sequential( nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    model.fc = nn.Sequential( nn.Linear(512, 10))
    model.to(device)
    return model


def main():
    
    model = load_resnet18()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)

    train_dl, valid_dl = load_cifar10()
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
    torch.save(model, 'cifar.pth')


if __name__ == '__main__':
    main()
