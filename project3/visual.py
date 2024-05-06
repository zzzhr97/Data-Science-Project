import os
import numpy as np
from utils import get_result

def plot_line(x, ys, xlabel, ylabels, title, save_path, k=1, logbase=2, **kwargs):
    """
    画折线图，为多条折线，x轴
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("dark") #设立风格
    colors_line = sns.color_palette("bright6", len(ys))

    fig, ax = plt.subplots(layout='constrained')

    if logbase is not None:
        plt.xscale('log', base=logbase)
    else:
        print(x)
        plt.xticks(x, x)

    legends = []

    for i, y in enumerate(ys[:-k]):
        #ax.plot(x, y, label=ylabels[i], color=colors_line[i])
        cur = ax.plot(x, y, label=ylabels[i], color=colors_line[i], marker='o', alpha=0.5, markersize=5)
        legends.append(cur)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Scores', fontsize=10)
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    #ax.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 12})
    ax.set_title(title, fontsize=12)
    #plt.tight_layout()

    # 设置左侧y轴范围
    y_min, y_max = 0.89, 0.93
    ax.set_ylim(y_min, y_max)

    ax2 = ax.twinx()
    for i, y in enumerate(ys[-k:]):
        #ax2.plot(x, y, label=ylabels[-(k-i)], color=colors_line[-(k-i)])
        # 去除 2048 维度的降维时间，因为必定为 0
        if ylabels[-(k-i)] == 'Reduce Time':
            n = -1  
        else:
            n = len(y)
        cur = ax2.plot(x[:n], y[:n], label=ylabels[-(k-i)], color=colors_line[-(k-i)], marker='o', alpha=0.5, markersize=5)
        legends.append(cur)

    ax2.set_ylabel('Time', fontsize=10)
    #ax2.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 12})

    total_legend = legends[0]
    for l in legends[1:]:
        total_legend += l
    labels = [l.get_label() for l in total_legend]
    ax.legend(total_legend, labels, bbox_to_anchor=kwargs['bbox_to_anchor'], prop={'size': 10})

    fig.savefig(save_path, dpi=300)
    plt.show()

def plot_thermodynamic(data, xlabel, ylabels, title, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("dark")
    plt.xticks(np.arange(len(xlabel)), labels=xlabel, 
        rotation=45, rotation_mode="anchor", ha="right", fontsize=13)
    plt.yticks(np.arange(len(ylabels)), labels=ylabels, fontsize=13)    
    # Testing ACC in different hyperparameters by UMAP
    plt.title(title, fontsize=18)

    for i in range(len(ylabels)):
        for j in range(len(xlabel)):
            text = plt.text(j, i, data[i][j], ha="center", va="center", color="black")

    plt.imshow(data, cmap='terrain')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def format_value(value, dtype=None):
    if dtype is None:
        if isinstance(value, float):
            return "{:.2f}".format(value)  
        elif isinstance(value, int):
            return "{}".format(value)
        else:
            return str(value)
    elif dtype == 'str':
        return str(value)
    elif dtype == 'float':
        try:
            return float(value)
        except:
            raise ValueError(f"Value {value} cannot be converted to float")
    
def t1():
    # 'chebyshev', 'cosine', 'manhattan', 'seuclidean', 'euclidean'
    metric_method = 'cosine'
    # metric_method = 'chebyshev'
    # metric_method = 'manhattan'
    # metric_method = 'seuclidean'
    # metric_method = 'euclidean'

    bbox_to_anchor = (0.27, 0.3)

    results = get_result(metric_method, only_distance=True)
    print(results)

    x = [result['K'] for result in results]
    ys = [
        [result['acc'] for result in results], 
        [result['train_time'] for result in results], 
        [result['test_time'] for result in results]
    ]
    xlabel = 'K'
    ylabels = ['ACC', 'Train-Time', 'Test-Time']
    title = f'Performance of Different Hyperparamer K Settings by Metric {metric_method}'
    os.makedirs('./image', exist_ok=True)
    save_path = f'./image/{metric_method}.png'
    plot_line(x, ys, xlabel, ylabels, title, save_path, k=2, bbox_to_anchor=bbox_to_anchor)

def t2():
    bbox_to_anchor = (0.3, 0.15)

    results = get_result('euclidean', only_distance=False)

    x = [result['K'] for result in results]
    x = x[::2]
    ys = [
        [result['acc'] for result in results if result['W'] == 'distance' ],
        [result['acc'] for result in results if result['W'] == 'uniform' ] 
    ]

    print(x)
    print(ys)

    xlabel = 'K'
    ylabels = ['ACC-distance', 'ACC-uniform']
    title = f'Performance of Different Hyperparamers K / weights Settings by Metric euclidean'
    os.makedirs('./image', exist_ok=True)
    save_path = f'./image/euclidean_weights.png'
    plot_line(x, ys, xlabel, ylabels, title, save_path, k=0, bbox_to_anchor=bbox_to_anchor)

def t3():
    # pml_circle, pml_contrastive, pml_triplet
    metric_method = 'pml_triplet'
    results = get_result(metric_method)
    results = [result for result in results if result['epoch'] == 40]
    
    for _ in results:
        print(_)
    
    ylabels = ['dim = 512', 'dim = 1024', 'dim = 1536', 'dim = 2048']
    xlabels = ['lr = 1e-5', 'lr = 5e-5', 'lr = 1e-4', 'lr = 5e-4', 'lr = 1e-3']
    data = []
    for i in range(len(ylabels)):
        cur_line = []
        for j in range(len(xlabels)):
            cur_line.append(format_value(results[i*len(xlabels)+j]['acc'], dtype='float'))
        print(cur_line)
        data.append(cur_line)

    os.makedirs('./image', exist_ok=True)
    save_path = f'./image/{metric_method}.png'
    plot_thermodynamic(data, xlabels, ylabels, f'ACC by {metric_method[4:]}', save_path)

def t4():
    # pml_circle, pml_contrastive, pml_triplet
    metric_method = 'pml_contrastive'

    bbox_to_anchor = (0.27, 0.99)

    results = get_result(metric_method)
    results = [result for result in results if result['epoch'] == 40]
    avg_result = {}
    for dim in [512, 1024, 1536, 2048]:
        avg_result[dim] = {'acc':0.0, 'train_time':0.0, 'test_time':0.0} 
    for result in results:
        avg_result[result['dim']]['acc'] += result['acc'] / 5
        avg_result[result['dim']]['train_time'] += result['train_time'] / 5
        avg_result[result['dim']]['test_time'] += result['test_time'] / 5

    for key in avg_result:
        print(key, avg_result[key])

    x = [512, 1024, 1536, 2048]
    ys = [
        [res['acc'] for res in avg_result.values()],
        [res['train_time'] for res in avg_result.values()],
        [res['test_time'] for res in avg_result.values()],
    ]

    xlabel = 'Dim'
    ylabels = ['ACC', 'Train-Time', 'Test-Time']
    title = f'Performance of Different Dimension by {metric_method[4:]}'
    os.makedirs('./image', exist_ok=True)
    save_path = f'./image/{metric_method}_dim.png'
    plot_line(x, ys, xlabel, ylabels, title, save_path, k=2, logbase=None, bbox_to_anchor=bbox_to_anchor)

    
if __name__ == '__main__':

    t4()