# cs762_project


## Dependencies
```
python=3.8 or higher
```
## Installation

The source code can be installed by simply copying the source code

```bash
git clone https://github.com/derekpaulsen/cs762_project.git
```

Then the dependencies can be installed with pip

```bash
pip install -r ./requirements.txt
```

In the process of generating data, you may need torch package with cuda

```bash
pip uninstall torch
pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```


## Generating Data

To generate synthetic data we used the `./gen_data_cifar10.py`. Before running this script, add your 
huggingface auth token in a file called `auth_token` in the root dir of the project, the script 
will read from this file.

To run execute the command,

```bash
$ python3 ./gen_data_cifar10.py --out_dir <DIRECTORY TO OUTPUT PICTURES>\
								--images_per_class <NUMBER OF IMAGES PER CLASS TO GENERATE>
```

Once the data has been generated, it needs to be formatted into 32x32 images. To do this at the
bottom of `./utils/cifar10.py`, modify the script values to, 

```python 
if __name__ == '__main__':
    in_dir = Path('< --out_dir from previous script >')
    out_dir = Path('<desired output location for formatted data>')
    make_syn_cifar10(in_dir, out_dir)
	
    dataset = SyntheticCIFAR10(out_dir, train=True)
    print(dataset[0])
```

The script can then be run with the command 

```bash
$ python3 ./utils/cifar10.py
```

Note that we have included our generated data, under `data/synthetic_cifar10` and `data/synthetic_cifar10_2`
as small parquet files. The raw output from `./gen_data_cifar10.py` has been omitted because the raw images 
totaled over 50GB in size.

## Training Models

We have provided 5 training scripts to train 5 different models

- `train_densenet121.py`
- `train_resnet18.py`
- `train_resnet34.py`
- `train_resnet50.py`
- `train_vgg16.py`

Each script is run by the following command, 

```bash 
$ python3 train_<MODEL>.py --data_props "<ORIGINAL %>,<SYNTHETIC1 %>,<SYNTHETIC2 %>"
```

Where, 

- `<ORIGINAL %>` = float specifying the percentage of original CIFAR 10 data to be used
- `<SYNTHETIC1 %>` = float specifying the percentage of synthetic CIFAR 10 (generated with just '{label'} as a prompt) data to be used
- `<SYNTHETIC2 %>` = float specifying the percentage of synthetic CIFAR 10 (generated the prompt 'a photo of a {label}') data to be used


For example, to train resnet18 with 100% of the original data, 50% of the synthetic1 data, and 25% of the synthetic2 data we would run,

```bash 
$ python3 train_resnet18.py --data_props "1,0.5,0.25"
```

To run all of the experiments we provide 5 bash scripts,

- `scripts/run_densenet121.sh`
- `scripts/run_resnet18.sh`
- `scripts/run_resnet34.sh`
- `scripts/run_resnet50.sh`
- `scripts/run_vgg16.sh`


These scripts will output json files to the `/tmp` dir with json output for the 
training results


## Results

We include our experimental results in `exp_res/`, both the raw JSON files and the 
aggreagated results in `./exp_res/aggregated_run.parquet`
