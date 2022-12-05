import pandas as pd
import numpy as np
from pathlib import Path
import json




MODELS = [
    'resnet18', 
    'resnet34', 
    'resnet50', 
    'vgg16', 
    'densenet121'
]
N_EPOCHS = 250

def infer_model(fname):
    model = None
    cnt = 0
    for m in MODELS:
        if m in fname:
            model = m
            cnt += 1
    if cnt != 1:
        raise ValueError(f'unable to infer model type for "{fname}"')
    else:
        return model

def read_log(file, run_id_cols, proj_cols):
    file = Path(file)
    df = pd.read_json(file, lines=True)
    with file.open('r') as ifs:
        raw_data = ifs.read()
        if len(raw_data) == 0:
            return pd.DataFrame()

        if raw_data[0] == '{':
            raw_data = list(map(json.loads, filter(lambda x : len(x) != 0, raw_data.split('\n'))))
        else:
            raw_data = pd.DataFrame(json.loads(raw_data))
        df = pd.DataFrame(raw_data)

    data = df.groupby(run_id_cols)\
            .apply(lambda x : pd.Series(x[proj_cols].to_dict('list')))
    data['model'] = infer_model(file.name)
    return data.reset_index(drop=False)

def data_prop_to_str(d):
    t = eval(d)
    return 'orig_train = %f, syn_train = %f' % (t[0], t[2])



def main():
    files = []
    files += Path('./pre-result/').glob('**/*.json')
    files += Path('./exp_res/12_01_partial/').glob('*.json')
    files += Path('./exp_res/12_05_partial/').glob('*.json')

    kwargs = {
            'run_id_cols' : ['start_time', 'data_props'],
            'proj_cols' : ['epoch', 'val_loss', 'accuracy', 'train_loss']
    }
    data = pd.concat([read_log(f, **kwargs) for f in files], ignore_index=True)

    drop = data['accuracy'].apply(len).ne(N_EPOCHS)
    print(f'dropping {drop.sum()} rows due to incomplete runs')
    print(data.loc[drop])
    print('\n\n')
    data = data.loc[~drop]

    print(data)
    stats = data[['model', 'data_props']].value_counts().sort_index()
    print(stats.to_string())
    print(len(stats))
    data.to_parquet('./exp_res/aggregated_run.parquet')


if __name__ == '__main__':
    main()

