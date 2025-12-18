# -*- coding: utf-8 -*-
# =============================================================================
# Description:  Computes the global time tau and kemeny's constant using the 
#               fundamental matrix and the mean first passage matrix
# =============================================================================
# Import libraries
import numpy as np
from scipy.linalg import eig
import networkx as nx



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
        I = np.identity(n, dtype=int)
        e = np.ones(n, dtype=int)

        # calculate the stationary vector pi
        #ev,lv = eig(P,left=True, right=False) # eig values and left eig vectors
        
        ix = np.argsort(ev)[-1] # dominant eigvalue index
        dom_eigenvector = lv[:,ix].real
        pi = (1/np.sum(dom_eigenvector))*dom_eigenvector # probabiltiy vector
        
        # calculate the fundamental matrix Z
        W = np.outer(e,pi) # pi in every row
        #try:
        #    Z = np.linalg.inv(I - P + W)
        #except:
        #    print(n)
        #    import pdb
        #    pdb.set_trace()
        
        Z = np.linalg.inv(I - self.P + W)

        # construct the mean first passage matrix M
        E = np.ones((n,n), dtype=int)
        Z_diag = np.diag(np.diag(Z))
        D = np.diag(1/pi)
        mat1 = (I - Z + np.matmul(E,Z_diag))
        M = np.matmul(mat1,D)
        np.fill_diagonal(M, 0) # set M[i][i]=0

        self.M = M
        
        # calculate the global time tau
        pi_mat = np.outer(pi,e) # pi in every column
        tau_mat = pi_mat * M 
        tau_i = np.sum(tau_mat,axis=0) # vector of tau values
        tau_global = np.sum(tau_i) / n
        #print('tau: ' + str(tau_global))
        
        # calculate kemeny's constant
        kc_mat = W * M 
        kc_i = np.sum(kc_mat,axis=1) # vector of kc values (should all be the same)
        
        self.kemeny = kc_i[0]
        self.tau_mat = tau_mat
        self.tau_i = tau_i
        self.tau_global = tau_global
        
        return tau_global,kc_i[0]

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


