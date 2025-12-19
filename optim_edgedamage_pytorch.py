import torch
from torch import nn
import numpy as np
import networkx as nx
import itertools
import copy

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
    def __init__(self, adj, ne):
        super().__init__()
        
        self.flatten = nn.Flatten() # needed?
        self.A0 = torch.Tensor(adj)
        self.n = self.A0.shape[0]
        self.ne = ne
        
        # make optimizeable via nn.Parameter
        self.B = nn.Parameter(torch.zeros((self.n,self.n)), requires_grad=True)
        
        self.P0 = torch.reshape((1/torch.sum(self.A0, axis=1)),(self.n,1)) * self.A0
        return

    def tau_global(self,x):
        I = torch.eye(self.n)
        #e = np.ones(n, dtype=int)
        e = torch.ones(self.n)
        
        _A = torch.Tensor(self.A0 + x)
        _P = torch.reshape((1/torch.sum(_A, axis=1)),(self.n,1)) * _A
        ev,lv = torch.linalg.eig(_P.T) # eig values and left eig vectors
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
        return tau_global
    #
    def forward(self, x, record=False):
        '''
        in: an adjacency matrix of modifications (can/should be negative for 
        edge deletions)
        
        out: directed travel time.
        '''
        
        _tau = self.tau_global(x)
        _B_vec1norm = self.adjnorm(x)
        
        _l = torch.abs(_B_vec1norm - self.ne)

        return _tau + self.mu*_l # todo: arbitrary multiplier
    #
    
    def adjnorm(self,x):
        return torch.linalg.norm(x.flatten(),1)
    #
    
    def train(self, nit=1000, lr=1e-3, print_every=100, record_every=100, record_jumps=False, mu=1e-1):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        self.mu=mu
        
        self.steps = [0]
        print(torch.Tensor(self.B.detach()))
        
        import copy
        
        self.snapshots = [copy.deepcopy(torch.Tensor(self.B.detach()))]
        self.taus = [self.tau_global(self.B)]
        self.Bnorms = [self.adjnorm(self.B)]
        self._prev = self(self.B) # initial
        self.objectives = [self.forward(self.B)]
        
        for k in range(1,nit):
            try:
                obj = self(self.B)
                obj.backward()
                optimizer.step() # descend
                
                optimizer.zero_grad()
            except:
                # expect the linalg_eig_backward error happened, record and exit.
                print('OH NO', k, abs(self._prev - obj)/abs(self._prev))
                self.steps.append(k)
                self.snapshots.append( copy.deepcopy(torch.Tensor(self.B.detach())) )
                self.taus.append( self.tau_global(self.B) )
                self.Bnorms.append( self.adjnorm(self.B) )
                self.objectives.append( self(self.B) )
                print('Eig error, halting.')
                break
            # save checkpoints, print, etc.
            if k%record_every==0:
                self.steps.append(k)
                self.snapshots.append( copy.deepcopy(torch.Tensor(self.B.detach())) )
                self.taus.append( self.tau_global(self.B) )
                self.Bnorms.append( self.adjnorm(self.B) )
                self.objectives.append( self(self.B) )
                self._prev = self(self.B) # initial
                
                #import pdb
                #pdb.set_trace()
            elif record_jumps and abs(self._prev - obj)/abs(self._prev) > 0.5:
                print('OH NO', k, abs(self._prev - obj)/abs(self._prev))
                self.steps.append(k)
                self.snapshots.append( torch.Tensor(self.B).detach() )
                self.taus.append( self.tau_global(self.B) )
                self.Bnorms.append( self.adjnorm(self.B) )
                self.objectives.append( self(self.B) )
                self._prev = self(self.B) # initial
            #
            
            if k%print_every==0:
                print(f'iter: {k:7} : tau: {self.tau_global(self.B):3.2f} -- norm(B): {self.adjnorm(self.B):4.1f}')
            # update
            self._prev = obj
        #
        
        return self.B
    #

if __name__=="__main__":
    from matplotlib import pyplot as plt
    import tools

    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    
    m,n = 2,2
    G = tools.directed_rectangle(2,2)
    target_num_edges = 2
    
    A = nx.to_numpy_array(G)
    
    # random initialization
    torch.manual_seed(0)
    #B0 = torch.rand(A.shape)
    B0 = torch.zeros(A.shape)
    #B0 = -target_num_edges*B0/torch.linalg.norm(B0.flatten(),1)
    
    model = damn_edge(A, target_num_edges)
    #model.B = nn.Parameter(B0, requires_grad=True)
    nepoch = 10000
    model.train(nepoch, 
        mu=1,
        lr=1e-4, 
        record_every=nepoch//100, 
        print_every=nepoch//100, 
        record_jumps=True
    )
    
    ptrs={idx:i for i,idx in enumerate(model.steps)}
    
    # visualize
    fig,ax = plt.subplot_mosaic('''
    TTTAD
    TTTBE
    TTTCF
    NNN..
    ''',
    figsize=(10,6),
    constrained_layout=True)
    
    bn=torch.Tensor(model.Bnorms).detach().numpy()
    objective=torch.Tensor(model.objectives).detach().numpy()
    taus=torch.Tensor(model.taus).detach().numpy()
    
    ax['T'].plot(model.steps, taus, label=r'$\tau(\vec{G})$')
    #ax['T'].set_yscale('log')
    
    ax['N'].plot(model.steps, bn, label=r'$||B||_\mathrm{vec}$')
    
    kmax = len(model.steps)
    
    for idx,l in zip(np.arange(0,kmax,kmax//6)[:6],'ABCDEF'):
        #A_damaged = (model.A0 + model.snapshots[idx]).detach().numpy()
        
        # threshold perturb.
        perturb = (model.snapshots[idx]).detach().numpy()
        perturb[abs(perturb)<1e-2]=0
        
        pos = {i+n*j:(i,j) for i,j in itertools.product(range(m),range(n))}
        
        #if idx==0:
        #    to_vis = model.A0
        #else:
        #    to_vis = perturb
        #
        to_vis = model.A0 + perturb
        
        tools.plot_weighted_graph(
            to_vis, ax=ax[l], 
            min_width=0.0, max_width=2, pos=pos,
            with_edge_labels=True
        )
        
        ax[l].text(0,1, l, ha='left', va='top', bbox={'facecolor':'#ffd'})
        ax['T'].scatter(model.steps[idx],taus[idx], marker='*', c='#333', s=200)
        ax['T'].text(model.steps[idx],taus[idx], l, fontsize=16, ha='left', va='bottom', bbox={'facecolor':'#ffd'}, zorder=-100)
    #
    
    ax['T'].set(xticklabels=[], ylabel=r'$\tau$')
    ax['T'].legend()
    ax['N'].set(xlabel='epoch', ylabel='matrix_norm')
    ax['N'].legend()
    ax['N'].axhline(target_num_edges, c='#333', lw=0.5, ls='--')
    
    fig.show()
    OUT_FOLDER = 'outputs/'
    fig.savefig(OUT_FOLDER+'edge_damage_optimization_out.png', dpi=300, bbox_inches='tight')
    
