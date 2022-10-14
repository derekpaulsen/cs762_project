import torch 
import pandas as pd
from pathlib import Path
from joblib import delayed, Parallel
import multiprocessing
from torchvision import datasets, transforms, models
from torchvision.io import  read_image
import pickle 
import math
from tqdm import tqdm
from .data import DeviceDataLoader

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

def _img_tensor_to_bytes(tensor):
    return tensor.numpy().dumps()

def _bytes_to_img_tensor(bytes):
    return torch.from_numpy(pickle.loads(bytes))

def _preprocess_syn_cifar10_img(img_path):
    img = read_image(str(img_path))
    img = _syn_transform.forward(img)
    img = _img_tensor_to_bytes(img)
    return (img, _dir_to_label[img_path.parent.name])


def make_syn_cifar10(in_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    file_itr = list(in_dir.glob('**/*.png'))
    pool = Parallel(n_jobs=-1)
    pairs = pool(delayed(_preprocess_syn_cifar10_img)(i) for i in tqdm(file_itr))

    data = pd.DataFrame(pairs, columns=['img_bytes', 'label'])
    nbytes =  data['img_bytes'].apply(len).sum()

    nchunks = math.ceil(nbytes / MAX_FILE_SIZE)
    chunk_size = len(data) // nchunks
    slices = list(range(0, len(data), chunk_size))
    if slices[-1] != len(data):
        slices.append(len(data))

    for i, start, end in tqdm(zip(range(len(slices)), slices[:-1], slices[1:])):
        fname = out_dir / f'chunk_{i}.parquet'
        data.iloc[start:end].to_parquet(fname, index=False)


class SyntheticCIFAR10(torch.utils.data.Dataset):

    def __init__(self, data_dir, train, transform=None):
        self.transform = transform
        self.train = train
        data_dir = data_dir / 'train' if train else data_dir / 'test'
        data = pd.concat(list(map(pd.read_parquet, data_dir.glob('*.parquet'))))
        self._tensors = data['img_bytes'].apply(_bytes_to_img_tensor).values
        self._labels = torch.from_numpy(data['label'].values).long()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        X = self._tensors[index]
        if self.transform is not None:
            X = self.transform(X)
        return (X, self._labels[index])
    

def load_cifar10(synthetic=False):
    stats = ((0.4914, 0.4822, 0.4465), (0.24705882352941178, 0.24352941176470588, 0.2615686274509804))

    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    if synthetic:
        trainset = SyntheticCIFAR10('./data/synthetic_cifar10/', train=True, transform=train_tfms)
        testset = SyntheticCIFAR10('./data/synthetic_cifar10/', train=False, transform=valid_tfms)
    else:
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tfms)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_tfms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, pin_memory=True, num_workers=16)


    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, pin_memory=True, num_workers=16)
    
    return DeviceDataLoader(trainloader, 'cuda'), DeviceDataLoader(testloader, 'cuda')


if __name__ == '__main__':
    in_dir = Path('/arx/cifar10/')
    out_dir = Path('./data/synthetic_cifar10')
    make_syn_cifar10(in_dir, out_dir)

    dataset = SyntheticCIFAR10(out_dir)
    print(dataset[0])
    print(dataset[[0, 1]])
