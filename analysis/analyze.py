import pandas as pd
import numpy as np

def read_log(file, run_id_cols, proj_cols):
    df = pd.read_json(file, lines=True)
    data = df.groupby(run_id_cols)\
            .apply(lambda x : pd.Series(x[proj_cols].to_dict('list')))
    return data

def data_prop_to_str(d):
    t = eval(d)
    return 'orig_train = %f, syn_train = %f' % (t[0], t[2])




df = read_log('./exp_res/resnet_18.json', ['start_time', 'data_props'], ['epoch', 'val_loss', 'accuracy', 'train_loss'])

stats = df.accuracy.apply(np.max).groupby(level=1).describe()
stats.index = list(map(data_prop_to_str, stats.index))
print(df)
print(stats)
stats = df.accuracy.apply(np.argmax).groupby(level=1).describe()
print(df)
print(stats)
