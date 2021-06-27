import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_fig(*, ntotal=None, ncols=1, nrows=1, colsize=8, rowsize=6):
    if ntotal is not None:
        nrows = int(np.ceil(ntotal / ncols))
    fig, ax_mat = plt.subplots(
        nrows, ncols, figsize=(colsize*ncols, rowsize*nrows))
    axs = np.array(ax_mat).flatten()
    if ntotal is not None and len(axs) > ntotal:
        for ax in axs[ntotal:]:
            ax.axis('off')
    if len(axs) == 1:
        axs = axs[0]
    return fig, axs


def plot_training_progress(df, x='epoch', y='train_loss', title='Training Progress',
                           xlim=[0, 30], ylim=None, annot=False, annot_nth=5, ax=None):
    if ax is None:
        fig, ax = create_fig(ncols=1, nrows=1)

    assert isinstance(x, str)
    if isinstance(y, str):
        y = [y]

    # create lineplot
    # df.plot(x=x, y=y, kind='line', marker='.',
    #         xlim=xlim, ylim=ylim, title=title, ax=ax)
    for _y in y:
        sns.lineplot(data=df, x=x, y=_y, marker='o', label=_y, ax=ax)
        ax.set(xlim=xlim, ylim=ylim, title=title)
        ax.legend()

    ax.grid()

    if annot:
        # add anotations
        idx = np.arange(len(df))
        cond = (idx % annot_nth == 0) | (idx == idx.max())
        for _, row in df[cond].iterrows():
            for metric in y:
                if ylim is None or row[metric] < ylim[1]:
                    ax.text(row[x], row[metric], f'{row[metric]:.2f}',
                            va='bottom', ha='center', fontsize='large')

    return ax


def plot_score_heatmap(groups_dict,
                       metrics=['accuracy', 'f1_score', 'country_f1_score'],
                       agg='max', cmap='Blues', ax=None):
    if ax is None:
        fig, ax = create_fig(ncols=1, nrows=1)

    # aggregate metrics based on agg argument
    df = pd.DataFrame.from_dict(
        {k: v.agg({m: agg for m in metrics}) for k, v in groups_dict.items()},
        orient='index')

    # plot heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap=cmap, ax=ax)
    ax.set(title='Score Heatmap')
    # plt.yticks(rotation=0)
    ax.tick_params(axis='y', labelrotation=0)

    plt.show()


def compare_training_process(groups_dict, *, xlim=[0, 30]):
    first_group_df = list(groups_dict.values())[0]
    for col in ['epoch', 'train_loss', 'valid_loss',
                'accuracy', 'f1_score', 'country_f1_score']:
        assert col in first_group_df, f'Column "{col}" is missing the the dataframe'

    ngroups = len(groups_dict)
    fig, axs = create_fig(ncols=ngroups, nrows=2)

    for ax, (k, g) in zip(axs[:ngroups], groups_dict.items()):
        params = dict(y=['train_loss', 'valid_loss'], xlim=xlim, ylim=[0.0, 6.0])
        plot_training_progress(g, x='epoch', title=k, ax=ax, **params)

    for ax, (k, g) in zip(axs[ngroups:], groups_dict.items()):
        params = dict(y=['accuracy', 'f1_score', 'country_f1_score'], xlim=xlim, ylim=[0, 1])
        plot_training_progress(g, x='epoch', title=k, ax=ax, **params)

    plt.show()
