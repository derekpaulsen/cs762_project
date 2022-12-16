import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

#matplotlib.rc('font', weight='bold')
matplotlib.rc('font', size=30)
    






def aggregate_runs(runs):
    if len(runs) > 1:
        acc = np.column_stack(runs.accuracy.values).mean(axis=1)
    else:
        acc = runs.accuracy.values[0]

    return pd.Series({
        'accuracy' : acc,
        'nruns' : len(runs)
    })

def data_props_to_string(s):
    props = np.array(list(eval(s))) * 100
    l = []
    for t, p in zip(['orig', 'syn1', 'syn2'], props):
        if p > 0:
            l.append(f'{t} % {p}')
    return ' + '.join(l)

def plot_runs(ax, runs, running_max=False):
    ymins = np.column_stack(runs.values).min(axis=1)
    ymaxs = np.column_stack(runs.values).max(axis=1)
    
    START_EPOCH = np.argmax(ymins >= ymaxs.max() - .1)

    for idx, acc in runs.items():

        if running_max:
            acc = np.maximum.accumulate(acc)

        ax.plot(np.arange(1, len(acc)+1), acc, label=data_props_to_string(idx), linewidth=4)
    ax.set_xlabel('Epoch')
    


def make_figure(data, models, data_props, running_max=False):
    scale = 10
    fig, axes = plt.subplots(1, len(models), figsize=(3 * scale,  1 * scale), sharey=True)
    max_acc = data.loc[models].loc[:, data_props].apply(np.max).max()
    min_acc = data.loc[models].loc[:, data_props].apply(np.max).min()
    axes[0].set_ylim(min_acc - .03, max_acc +.01)
    for m, ax in zip(models, axes.flatten()):
        d = data.loc[m]
        plot_runs(ax, d.loc[data_props], running_max)
        ax.set_title(m)

    axes[0].set_ylabel('Accuracy')

    handles, labels = axes.flatten()[0].get_legend_handles_labels()    
    fig.legend(handles, labels, ncol=len(labels), loc='upper center')    
    fig.tight_layout(rect=(0,0,1,.92))
    return fig, axes


def main():
    data = pd.read_parquet('./exp_res/aggregated_run.parquet')\
                .groupby(['model', 'data_props']).apply(aggregate_runs)\
                .sort_index()
    out_dir = Path('./figs/')
    out_dir.mkdir(parents=True, exist_ok=True)
    data_props = [
            ['1,0,0', '1,1,0', '1,0,1', '1,1,1'],
            ['1,0,0', '.5,.5,0', '.5,0,.5', '.5,0,0'],
            ['1,0,0', '0,1,0', '0,0,1']
    ]

    for props, suffix in zip(data_props, ['', '_50_50', '_syn_only' ]): 
        models = ['resnet18', 'resnet34', 'resnet50']
        fig, axes = make_figure(data['accuracy'], models, props)
        fig.savefig(out_dir / f'resnets{suffix}.png')

        fig, axes = make_figure(data['accuracy'], models, props, running_max=True)
        fig.savefig(out_dir / f'resnets_running_max{suffix}.png')

        models = ['resnet18', 'vgg16', 'densenet121']
        
        fig, axes = make_figure(data['accuracy'], models, props)
        fig.savefig(out_dir / f'res_vgg_dense{suffix}.png')
        fig, axes = make_figure(data['accuracy'], models, props, running_max=True)
        fig.savefig(out_dir / f'res_vgg_dense_running_max{suffix}.png')

if __name__ == '__main__':
    main()

    
    


