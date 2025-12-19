import torch
from torch import nn
import numpy as np
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

class damn_edge(nn.Module):
    def __init__(self, adj):
        super().__init__()
        
        
        self.flatten = nn.Flatten() # needed?
        self.A0 = torch.Tensor(adj)
        self.n = self.A0.shape[0]
        
        # make optimizeable via nn.Parameter
        self.B = nn.Parameter(torch.zeros((self.n,self.n)), requires_grad=True)
        
        self.P0 = torch.reshape((1/torch.sum(self.A0, axis=1)),(self.n,1)) * self.A0
        return

    def forward(self, x):
        '''
        in: an adjacency matrix of modifications (can/should be negative for 
        edge deletions)
        
        out: directed travel time.
        '''
        
        I = torch.eye(self.n)
        #e = np.ones(n, dtype=int)
        e = torch.ones(self.n)
        
        _A = torch.Tensor(self.A0 + x)
        _P = torch.reshape((1/torch.sum(_A, axis=1)),(self.n,1)) * _A
        ev,lv = torch.linalg.eig(_P) # eig values and left eig vectors
        ev = torch.real(ev)
        lv = torch.real(lv)
        
        ix = torch.argsort(ev)[-1]
        dom_eigenvector = lv[:,ix]
        pi = (1/torch.sum(dom_eigenvector))*dom_eigenvector # probabiltiy vector
        
        
        # calculate the fundamental matrix Z
        W = torch.outer(e,pi)
        
        Z = torch.linalg.inv(I - _P + W)

        # construct the mean first passage matrix M
        E = torch.ones((self.n,self.n))
        Z_diag = torch.diag(torch.diag(Z))
        D = torch.diag(1/pi)
        mat1 = (I - Z + E@Z_diag)
        M = mat1@D
        M = M - torch.diag(torch.diag(M))

        # calculate the global time tau
        pi_mat = torch.outer(pi,e) # pi in every column
        tau_mat = pi_mat * M 
        tau_i = torch.sum(tau_mat,axis=0) # vector of tau values
        
        tau_global = torch.sum(tau_i) / self.n
        
        # total mass penalty.
        _l = torch.abs(torch.abs(torch.sum(x.flatten())) - 2)
        
        #return tau_global + _l
        return _l
    #
    
    def train(self, nit=1000, lr=1e-6):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        for k in range(nit):
            tau = self(self.B)
            tau.backward()
            optimizer.step() # descend
            
            
            optimizer.zero_grad()
            if k%100==0:
                print(f'{k} : {tau:.2f}')
        return self.B


if __name__=="__main__":
    from matplotlib import pyplot as plt
    import tools

    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})

    G = tools.directed_rectangle(2,3)
    A = nx.to_numpy_array(G)
    model = damn_edge(A)
    model.train(1000, lr=1e-1)
    
