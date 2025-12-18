import tools
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 14})

import itertools
import multiprocessing

import os
OUT_FOLDER = 'outputs/'
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)
#

def directed_rectangle(m,n, undirected=False):
    G = nx.generators.grid_graph((n,m))
    G = nx.to_directed(G)
    G = nx.DiGraph(G)
    for j in range(n):
        if j%2==0:
            for i in range(m-1):
                G.remove_edge((i,j), (i+1,j))
        else:
            for i in range(1,m):
                G.remove_edge((i,j), (i-1,j))
            
    return G
#

def comparison(inputs):
    # directed calculation
    G = directed_rectangle(inputs[0], inputs[1])
    A = nx.to_numpy_array(G)
    P = (1/np.sum(A,axis=1)) * A # row-sums one.
    tau_d,_ = tools.markov_metrics(P)
    
    # undirected
    G_D = nx.to_undirected(G)
    A = nx.to_numpy_array(G_D)
    P = (1/np.sum(A,axis=1)) * A # row-sums one.
    tau_u,_ = tools.markov_metrics(P)
    return tau_u,tau_d
#

def plot_example(m,n):
    pair = (m,n)
    fig,ax = plt.subplots(constrained_layout=True)

    G = directed_rectangle(m,n)
    
    nx.draw(G, ax=ax, pos={n:n for n in G.nodes()},
        arrowstyle = '-|>',
        arrowsize=30,
        node_color=[[0.1, 0.2, 0.6, 0.5]],
        edge_color=[[0.3,0.3,0.3,0.8]]
    )
    
    tu,td = comparison((m,n))
    _r = tu/td
    
    ax.set_title(f'{pair} : ' +  r'$\tau_u =$' + f'{tu:.3f};  ' + r'$\tau_d =$' + f' {td:.3f};  ' + r'$\mathcal{F}=$' + f'{_r:.2f}')
    fig.show()
    return fig,ax
#


if __name__=="__main__":
    # plot and export some examples.
    for example in [(2,2), (2,3), (3,2), (8,3), (8,4)]:
        fig,ax = plot_example(*example)
        fig.savefig(OUT_FOLDER + f'grid_{example}.png', dpi=300, bbox_inches='tight')


    #########
    ms = np.arange(2,21)
    ns = np.arange(2,21)

    if False:
        inputs = itertools.product(ms,ns)       

        p = multiprocessing.Pool(1)
        results = p.map(comparison, inputs)
        ratios = [r[0]/r[1] for r in results]
        R = np.reshape(ratios, (len(ms), len(ns)))

    else:
        R = np.zeros((len(ms), len(ns)))
        for i,m in enumerate(ms):
            for j,n in enumerate(ns):
                ratio = comparison((m,n))
                R[i,j] = ratio[0]/ratio[1]
        #

    fig2,ax2 = plt.subplots()

    # TODO: better visualization which properly labels the axes.
    cax = ax2.matshow(R.T, cmap=plt.cm.PRGn, vmin=0, vmax=2)

    fig2.colorbar(cax)
    fig2.show()
    fig2.savefig(OUT_FOLDER+'oneway_grids_paramsweep.png', dpi=300, bbox_inches='tight')

