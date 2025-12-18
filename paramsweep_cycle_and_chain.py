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


def cycle_and_chain(cycle_length=3, chain_length=0, undirected=False):
    edgelist = [(k,(k+1)%cycle_length) for k in range(cycle_length)]

    if undirected:
        edgelist = edgelist + [((k+1)%cycle_length,k) for k in range(cycle_length)]
    #
    s = (1+undirected)*cycle_length - 1
    
    edgelist = edgelist + [(k,k+1) for k in range(s,s+chain_length)]
    edgelist = edgelist + [(k+1,k) for k in range(s,s+chain_length)]
    
    G = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    return G
#

def comparison(inputs):
    # directed calculation
    G = cycle_and_chain(inputs[0], inputs[1])
    A = nx.to_numpy_array(G)
    P = (1/np.sum(A,axis=1)) * A # row-sums one.
    tau_d,_ = tools.markov_metrics(P)
    
    # undirected
    G_D = nx.to_undirected(G)
    A = nx.to_numpy_array(G_D)
    P = (1/np.sum(A,axis=1)) * A # row-sums one.
    tau_u,_ = tools.markov_metrics(P)
    return tau_u/tau_d
#

# parameter sweep over cycle lengths and chain lengths.
cycle_lengths = np.arange(3,51, dtype=int)
chain_lengths = np.arange(0, 51, dtype=int)

if False:
    # Multiprocessing: not even once.
    p = multiprocessing.Pool(16)

    all_inputs = itertools.product(cycle_lengths, chain_lengths)
    all_inputs = list(all_inputs)

    results = p.map(comparison, all_inputs)
    R = np.reshape(results, (len(cycle_lengths), len(chain_lengths)))
else:
    R = np.zeros((len(cycle_lengths), len(chain_lengths)))
    for i,y in enumerate(cycle_lengths):
        for j,h in enumerate(chain_lengths):
            R[i,j] = comparison((y,h))
#



###

fig,ax = plt.subplots(constrained_layout=True, figsize=(7,6))

levels = np.arange(0,17,2)

ax.pcolormesh(cycle_lengths, chain_lengths, R[1:,1:].T, vmin=0, vmax=10)
cax = ax.contour(cycle_lengths, chain_lengths, R.T, levels=levels, linewidths=2, cmap=plt.cm.Reds)
ax.clabel(cax, cax.levels, fontsize=12)


ax.set(xlabel='Cycle length', ylabel='Chain length', aspect='equal')
ax.set_title(r'$\tau(G)/\tau(\vec{G})$')
fig.show()

fig.savefig(OUT_FOLDER+'cycle_chain_paramsweep.png', bbox_inches='tight', dpi=300)

# Dump demo images of a few cycle-and-chains.
G1 = cycle_and_chain(20, 5)
G2 = cycle_and_chain(20, 30)
G3 = cycle_and_chain(45, 10)
G4 = cycle_and_chain(5, 30)
for _G,name in zip([G1, G2, G3, G4], ['(20,5)', '(20,30)', '(45,10)', '(5,30)']):
    _f,_a = plt.subplots(1,1, constrained_layout=True)
    nx.draw(_G, ax=_a)
    _f.savefig(OUT_FOLDER+f'cnc_{name}.png', dpi=300, bbox_inches='tight')


