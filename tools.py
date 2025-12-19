# -*- coding: utf-8 -*-
# =============================================================================
# Description:  Computes the global time tau and kemeny's constant using the 
#               fundamental matrix and the mean first passage matrix
# =============================================================================
# Import libraries
import numpy as np
from scipy.linalg import eig
import networkx as nx
import torch


def markov_metrics(P):
    '''
    Input: P, row-stochastic matrix
    
    Outputs:
        tau : float, (stationary-averaged mean travel time 
            with i as the finish)
            
        K: float, Kemeny's constant (stationary-averaged mean travel time 
            with i as the start)
    '''
    n = len(P) # size of square nxn matrix
    I = np.identity(n, dtype=int)
    e = np.ones(n, dtype=int)

    # calculate the stationary vector pi
    ev,lv = eig(P,left=True, right=False) # eig values and left eig vectors
    ix = np.argsort(ev)[-1] # dominant eigvalue index
    dom_eigenvector = lv[:,ix].real
    pi = (1/np.sum(dom_eigenvector))*dom_eigenvector # probabiltiy vector
    
    # calculate the fundamental matrix Z
    W = np.outer(e,pi) # pi in every row

    # could go wrong
    Z = np.linalg.inv(I - P + W)


    # construct the mean first passage matrix M
    E = np.ones((n,n), dtype=int)
    Z_diag = np.diag(np.diag(Z))
    D = np.diag(1/pi)
    mat1 = (I - Z + np.matmul(E,Z_diag))
    M = np.matmul(mat1,D)
    np.fill_diagonal(M, 0) # set M[i][i]=0

    # calculate the global time tau
    pi_mat = np.outer(pi,e) # pi in every column
    tau_mat = pi_mat * M 
    tau_i = np.sum(tau_mat,axis=0) # vector of tau values
    tau_global = np.sum(tau_i) / n
    #print('tau: ' + str(tau_global))
    
    # calculate kemeny's constant
    kc_mat = W * M 
    kc_i = np.sum(kc_mat,axis=1) # vector of kc values (should all be the same)
    #print('kc: ' + str(kc_i[0]))
    return tau_global,kc_i[0]
#

class network_passage_analyzer:
    def __init__(self,G):
        self.G = G
        self.n = len( self.G.nodes() )
        return
    def preprocess(self):
        '''
        computes/stores:
            adjacency matrix A as numpy array (full)
            adjacency matrix as scipy array (sparse)
            transition matrix P (full)
            left eigenvectors of P
        '''
        self.A = nx.to_numpy_array(self.G)
        self.P = (1/np.sum(self.A,axis=1)) * self.A # make each row sum to one.
        ev,lv = eig(self.P,left=True, right=False) # eig values and left eig vectors
        self.eigenvalues = ev
        self.eigenvectors_l = lv
        return
        
    def markov_metrics(self):
        '''
        Outputs:
            tau : float, (stationary-averaged mean travel time 
                with i as the finish)
                
            K: float, Kemeny's constant (stationary-averaged mean travel time 
                with i as the start)
        '''
        try:
            ev = getattr(self,'eigenvalues')
            lv = getattr(self,'eigenvectors_l')
        except:
            raise Exception('Could not fetch eigenvectors/values; run preprocess() first')
        
        n = self.n # size of square nxn matrix
        #I = np.identity(n, dtype=int)
        I = torch.eye(n)
        #e = np.ones(n, dtype=int)
        e = torch.ones(n)

        # calculate the stationary vector pi
        #ev,lv = eig(P,left=True, right=False) # eig values and left eig vectors
        ev,lv = torch.linalg.eig(torch.Tensor(self.P.T))
        # cast to real...?
        ev = torch.real(ev)
        lv = torch.real(lv)
        
        #ix = np.argsort(ev)[-1] # dominant eigvalue index
        ix = torch.argsort(ev)[-1]
        #dom_eigenvector = lv[:,ix].real
        dom_eigenvector = lv[:,ix]
        pi = (1/torch.sum(dom_eigenvector))*dom_eigenvector # probabiltiy vector
        self.steady_real_resid = max(abs(dom_eigenvector - lv[:,ix]))
        
        # calculate the fundamental matrix Z
        #W = np.outer(e,pi) # pi in every row
        W = torch.outer(e,pi)
        #try:
        #    Z = np.linalg.inv(I - P + W)
        #except:
        #    print(n)
        #    import pdb
        #    pdb.set_trace()
        
        #Z = np.linalg.inv(I - self.P + W)
        Z = torch.linalg.inv(I - torch.Tensor(self.P) + W)

        # construct the mean first passage matrix M
        #E = np.ones((n,n), dtype=int)
        E = torch.ones((n,n))
        #Z_diag = np.diag(np.diag(Z))
        Z_diag = torch.diag(torch.diag(Z))
        #D = np.diag(1/pi)
        D = torch.diag(1/pi)
        #mat1 = (I - Z + np.matmul(E,Z_diag))
        mat1 = (I - Z + E@Z_diag)
        #M = np.matmul(mat1,D)
        M = mat1@D
        #np.fill_diagonal(M, 0) # set M[i][i]=0
        M = M - torch.diag(torch.diag(M))

        self.M = M
        
        # calculate the global time tau
        pi_mat = torch.outer(pi,e) # pi in every column
        tau_mat = pi_mat * M 
        tau_i = torch.sum(tau_mat,axis=0) # vector of tau values
        tau_global = torch.sum(tau_i) / n
        #print('tau: ' + str(tau_global))
        
        # calculate kemeny's constant
        #if False:
        #kc_mat = W * M #entrywise product
        #kc_i = torch.sum(kc_mat,axis=1) # vector of kc values (should all be the same)
        
        #self.kemeny = kc_i[0]
        self.tau_mat = tau_mat
        self.tau_i = tau_i
        self.tau_global = tau_global
        
        return tau_global.item()

#

#
###############
#



def cycle_and_chain(cycle_length=3, chain_length=0, undirected=False):
    '''
    inputs: cycle_length, chain_length, integers
    outputs: networkx DiGraph of a chain connected to a directed cycle of 
        associated sizes.
    '''
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

def directed_rectangle(m,n, undirected=False):
    '''
    inputs: m,n; integers. number of columns and rows in the grid.
    outputs: networkx DiGraph of alternating one-ways.
    '''
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

def plot_weighted_graph(adj_matrix,
                        ax=None,
                        node_labels=None,
                        pos=None,
                        figsize=(8, 6),
                        min_width=0.5,
                        max_width=8.0,
                        with_labels=True,
                        with_edge_labels=False,
                        edge_label_format="{:.2g}"):
    """
    Credit: ChatGPT slop code.
    
    Create a NetworkX plot where edge thickness is proportional to edge weight.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square adjacency matrix. Zero entries are treated as no-edge.
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None, a new figure and axis are created.
    node_labels : dict or None
        Optional mapping {node_index: label} to use for node labels.
    pos : dict or None
        Optional node positions as {node: (x,y)}. If None, a spring layout is used.
    figsize : tuple
        Figure size used only if ax is None.
    min_width, max_width : float
        Minimum and maximum line width for edges.
    with_labels : bool
        Whether to draw node labels.
    with_edge_labels : bool
        Whether to draw edge labels (defaults to False).
    edge_label_format : str
        Format string used to convert edge weights to labels.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the drawn graph.
    """
    # Validate adjacency matrix
    adj = np.array(adj_matrix, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj_matrix must be a square 2D numpy array")

    # Create axis if needed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Choose directed vs undirected based on symmetry
    directed = not np.allclose(adj, adj.T)
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph() if directed else nx.Graph())

    # Remove zero-weight edges defensively
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                       if d.get("weight", 0) == 0]
    if edges_to_remove:
        G.remove_edges_from(edges_to_remove)

    # Prepare positions
    if pos is None:
        pos = nx.spring_layout(G)

    # Handle empty-edge case (nodes only)
    if G.number_of_edges() == 0:
        nx.draw_networkx_nodes(G, pos=pos, ax=ax)
        if with_labels:
            labels = node_labels if node_labels else {n: n for n in G.nodes()}
            nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax)
        ax.set_axis_off()
        return ax

    # Extract and normalize weights
    edge_data = list(G.edges(data=True))
    weights = np.array([d.get("weight", 1.0) for _, _, d in edge_data], dtype=float)

    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        widths = min_width + (weights - w_min) / (w_max - w_min) * (max_width - min_width)
    else:
        widths = np.full_like(weights, (min_width + max_width) / 2.0)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=pos, ax=ax)

    # Draw edges
    edgelist = [(u, v) for u, v, _ in edge_data]
    if directed:
        nx.draw_networkx_edges(
            G, pos=pos, ax=ax,
            edgelist=edgelist,
            width=widths,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=12
        )
    else:
        nx.draw_networkx_edges(
            G, pos=pos, ax=ax,
            edgelist=edgelist,
            width=widths
        )

    # Draw labels
    if with_labels:
        labels = node_labels if node_labels else {n: n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax)
    
    if with_edge_labels:
        edge_labels = {
            (u, v): edge_label_format.format(d.get("weight", 1.0))
            for u, v, d in edge_data
        }
        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels=edge_labels,
            ax=ax
        )
    
    ax.set_axis_off()
    return ax
#


if __name__=='__main__':
    def comparison(inputs):
        # directed calculation
        G = directed_rectangle(inputs[0], inputs[1])
        #A = nx.to_numpy_array(G)
        #P = (1/np.sum(A,axis=1)) * A # row-sums one.
        po_d = network_passage_analyzer(G)
        po_d.preprocess()
        tau_d = po_d.markov_metrics()
        
        # undirected
        G_U = nx.to_undirected(G)
        po_u = network_passage_analyzer(G_U)
        po_u.preprocess()
        tau_u = po_u.markov_metrics()
        return tau_u,tau_d
    #
    R = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            tau_u, tau_d = comparison((i+2,j+2))
            R[i,j] = tau_u/tau_d
    #
    from matplotlib import pyplot as plt
    
    fig,ax = plt.subplots()
    ax.matshow(R.T, cmap=plt.cm.PRGn, vmin=0., vmax=2)
    fig.show()
    
