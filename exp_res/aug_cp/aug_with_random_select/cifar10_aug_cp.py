import sys
sys.path.append('.')
import torch 
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import delayed, Parallel
import multiprocessing
from torchvision import datasets, transforms, models
from torchvision.io import  read_image
import pickle 
import math
from tqdm import tqdm
from utils.data import DeviceDataLoader
from torch.utils.data import Subset, ConcatDataset
import pickle
import PIL

# Note that if you want to run this script or some scripts related to this file, 
# please move it to ./utils folder, otherwise there would be path error.

OUT_IMAGE_SIZE = (32, 32)

_syn_transform = transforms.Resize(OUT_IMAGE_SIZE)
_cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

_dir_to_label = {
    'airplane' : 0,
    'automobile' : 1,
    'bird' : 2,
    'cat' : 3,
    'deer' : 4,
    'dog' : 5,
    'frog' : 6,
    'horse' : 7,
    'ship' : 8,
    'truck' : 9,
}
# 20 MB
MAX_FILE_SIZE = 20 * 2**20
TRAIN_IMAGES_PER_CAT = 5000
#TEST_IMAGES_PER_CAT = 1000


def _preprocess_syn_cifar10_img(img_path):
    with PIL.Image.open(img_path) as img:
        img = img.resize(OUT_IMAGE_SIZE, PIL.Image.Resampling.BILINEAR)
        img = pickle.dumps(img)

        return (img, _dir_to_label[img_path.parent.name.split()[-1]])



def _write_chunks(data, out_dir, name_prefix):
    out_dir.mkdir(parents=True, exist_ok=True)

    nbytes =  data['img_bytes'].apply(len).sum()
    nchunks = math.ceil(nbytes / MAX_FILE_SIZE)
    chunk_size = len(data) // nchunks
    slices = list(range(0, len(data), chunk_size))
    if slices[-1] != len(data):
        slices.append(len(data))

    for i, start, end in tqdm(zip(range(len(slices)), slices[:-1], slices[1:])):
        fname = out_dir / f'{name_prefix}_{i}.parquet'
        data.iloc[start:end].to_parquet(fname, index=False)

def _slice_dataset(dataset, percent):
    if percent == 1.0:
        return dataset
    else:
        indexes = np.random.choice(np.arange(len(dataset)), size=int(len(dataset) * percent), replace=False)
        return Subset(dataset, indexes)


def make_syn_cifar10(in_dir, out_dir):
    file_itr = list(in_dir.glob('**/*.png'))
    pool = Parallel(n_jobs=-1)
    pairs = pool(delayed(_preprocess_syn_cifar10_img)(i) for i in tqdm(file_itr))

    data = pd.DataFrame(pairs, columns=['img_bytes', 'label'])
    train = data.groupby('label')\
                .apply(lambda x: x.head(TRAIN_IMAGES_PER_CAT))\
                .reset_index(drop=True)
    test = data.drop(index=train.index)
    print(test)
    
    _write_chunks(train, out_dir/'train', 'train')
    _write_chunks(test, out_dir/'test', 'test')

class SyntheticCIFAR10(torch.utils.data.Dataset):

    def __init__(self, data_dir, train, transform=None):
        self.transform = transform
        self.train = train
        data_dir = Path(data_dir)
        data_dir = data_dir / 'train' if train else data_dir / 'test'
        data = pd.concat(list(map(pd.read_parquet, data_dir.glob('*.parquet'))))
        self._tensors = data['img_bytes'].apply(pickle.loads).values.tolist()
        self._labels = data['label'].values.tolist()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        X = self._tensors[index]
        if self.transform is not None:
            X = self.transform(X)
        return (X, self._labels[index])
    

def load_cifar10(norm_train_percent, syn_train_percent, syn_2_train_percent, rotation_train_percent, v_flip_train_percent, device='cuda'):
    stats = ((0.4914, 0.4822, 0.4465), (0.24705882352941178, 0.24352941176470588, 0.2615686274509804))
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

    train_tfms_rotation = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomRotation(90), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

    train_tfms_v_flip = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomVerticalFlip(p=1.0), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train_datasets = []
    valid_datasets = []

    if syn_2_train_percent > 0:
        # prompt : "a photo of a {category}"
        trainset = _slice_dataset(SyntheticCIFAR10('./data/synthetic_cifar10_2/', train=True, transform=train_tfms), syn_2_train_percent)
        train_datasets.append(trainset)

    if syn_train_percent > 0:
        # prompt : "{category}"
        trainset = _slice_dataset(SyntheticCIFAR10('./data/synthetic_cifar10/', train=True, transform=train_tfms), syn_train_percent)
        train_datasets.append(trainset)

    if norm_train_percent > 0:
        # the original cifar 10 dataset
        trainset = _slice_dataset(datasets.CIFAR10('./data', train=True, transform=train_tfms, download=True), norm_train_percent)
        train_datasets.append(trainset)

    if rotation_train_percent > 0:
        trainset = _slice_dataset(datasets.CIFAR10('./data/rotation', train=True, transform=train_tfms_rotation, download=True), rotation_train_percent)
        train_datasets.append(trainset)

    if v_flip_train_percent > 0:
        trainset = _slice_dataset(datasets.CIFAR10('./data/v_flip', train=True, transform=train_tfms_v_flip, download=True), v_flip_train_percent)
        train_datasets.append(trainset)

    testset = _slice_dataset(datasets.CIFAR10('./data', train=False, transform=valid_tfms, download=True), 1.0)
    valid_datasets.append(testset)

    trainset = ConcatDataset(train_datasets)
    testset = ConcatDataset(valid_datasets)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, pin_memory=True, num_workers=0)


    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, pin_memory=True, num_workers=0)
    
    return DeviceDataLoader(trainloader, device), DeviceDataLoader(testloader, device)


if __name__ == '__main__':
    in_dir = Path('/arx/cifar10/')
    out_dir = Path('./data/synthetic_cifar10_2/')
    make_syn_cifar10(in_dir, out_dir)

    dataset = SyntheticCIFAR10(out_dir, train=True)
    print(dataset[0])
