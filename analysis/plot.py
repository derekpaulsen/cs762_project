import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#matplotlib.rc('font', weight='bold')
matplotlib.rc('font', size=22)
    






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
    
    ax.set_ylim(ymaxs.max() -.1, None)


def make_figure(data, models, data_props, running_max=False):
    scale = 10
    fig, axes = plt.subplots(1, len(models), figsize=(3 * scale,  1 * scale))
    for m, ax in zip(models, axes.flatten()):
        d = data.loc[m]
        plot_runs(ax, d.loc[data_props], running_max)
        ax.set_title(m)

    axes[0].set_ylabel('Accuracy')

    handles, labels = axes.flatten()[0].get_legend_handles_labels()    
    fig.legend(handles, labels, ncol=len(labels), loc='upper center')    
    fig.tight_layout(rect=(0,0,1,.95))
    return fig, axes


def main():
    data = pd.read_parquet('./exp_res/aggregated_run.parquet')\
                .groupby(['model', 'data_props']).apply(aggregate_runs)\
                .sort_index()
    
    models = ['resnet18', 'resnet34', 'resnet50']
    props = ['1,0,0', '1,1,0', '1,0,1', '1,1,1']
    
    fig, axes = make_figure(data['accuracy'], models, props)
    fig.savefig('./exp_res/resnets.png')

    fig, axes = make_figure(data['accuracy'], models, props, running_max=True)
    fig.savefig('./exp_res/resnets_running_max.png')

    models = ['resnet18', 'vgg16', 'densenet121']
    
    fig, axes = make_figure(data['accuracy'], models, props)
    fig.savefig('./exp_res/res_vgg_dense.png')
    fig, axes = make_figure(data['accuracy'], models, props, running_max=True)
    fig.savefig('./exp_res/res_vgg_dense_running_max.png')

if __name__ == '__main__':
    main()

    
    


