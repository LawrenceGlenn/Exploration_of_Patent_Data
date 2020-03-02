

def plot_feature_importance(ax, features, importances):
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    comb = dict(zip(features, importances))
    comb = {k: v for k, v in sorted(comb.items(), key=lambda item: item[1],reverse=True)}
    ax.bar(list(comb.keys())[:10],list(comb.values())[:10])
    ax.set_xlabel('features', fontsize=34, labelpad=15)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(25)
        
def plot_scatter_matrix(df):
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(34,34))
    plt.show()
    
