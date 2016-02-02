import networkx as nx
import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import resample
class PathAccumCoordinates:

    def __init__(self, g_raw, vertex_filter = None, p_decay=1e-5, max_path_len=10, weight='weight'):
        self.max_path_len=max_path_len
        self.p_decay = p_decay
        self.weight = weight
        self.g = g_raw
        self.vertex_filter = vertex_filter

    @staticmethod
    def ConstructCoorSys(g_raw, lmrks, Dim, **kwargs):
        coorsys = PathAccumCoordinates(g_raw, **kwargs)
        coorsys.__set_landmarks__(lmrks, Dim)
        return coorsys

    def __set_landmarks__(self, lmrks, Dim):
        vertices = None
        if self.vertex_filter:
            vertices = [v for v in self.g.nodes() if self.vertex_filter(v)]

        L2all = self.proximity_to(lmrks,dests=vertices)
        L2L = L2all.loc[lmrks]
        U,S,_ = randomized_svd(L2L.as_matrix(), n_components = Dim)
        
        lmrk_coors = np.dot(U, np.sqrt(np.diag(S)))
        self.lmrk_coors = pd.DataFrame(lmrk_coors.transpose(),columns=lmrks)
        self.L2all = L2all

    def nodes(self):
        if self.vertex_filter:
            return [v for v in self.g.nodes() if self.vertex_filter(v)]
        else:
            return self.g.nodes()

    def coordinate(self, ordinaries):
        Prox_i2L = self.L2all.loc[ordinaries].as_matrix() #shape: ordinaries * landmarks
        Coor_L = self.lmrk_coors.as_matrix()    #shape: dimensions * landmarks
        Coor_L_trans = Coor_L.transpose()       #shape: landmarks * dimensions
        
        #Coor_i: shape: oridnaries * dimensions
        #Prox_i2L = Coor_i * Coor_L 
        # ==> Coor_i = Prox_i2L*transpose(Coor_L) * inv(Coor_L *transpose(Coor_L))
        Coor_L_sym_inv = np.linalg.inv(np.dot(Coor_L,Coor_L_trans))        
        Coor_i = np.dot(np.dot(Prox_i2L, Coor_L_trans),Coor_L_sym_inv)
        return pd.DataFrame(Coor_i.transpose(),columns=ordinaries)
    
    def proximity_to(self, sources, dests=None):
        node_names =[n for n in self.g.nodes()]
        A = nx.adjacency_matrix(self.g)
        res = pd.DataFrame(index=node_names)
        for s in sources:
            u = A[:,node_names.index(s)]   

            for l in range(1, self.max_path_len):
                u = u + (A.dot(u))*self.p_decay
            res[s] = u.toarray()

        if dests:
            res = res.loc[dests]
        return res

def find_content_landmarks(g, nlmrk, seeds = None, **kwargs):
    coorsys=PathAccumCoordinates(g,kwargs)
    candidates = [n for n in g.nodes() if n.startswith('p')]

    seed = [n for n in g.nodes()][0]
    prox2Candidates = coorsys.proximity_to(sources=[seed], dests=candidates).iloc[:,0]

    lmrk = prox2Candidates.argmin()
    L2candidates = pd.DataFrame()    
    L2candidates[lmrk]=coorsys.proximity_to(sources=[lmrk], dests=candidates).iloc[:,0]

    for i in range(nlmrk -1):        
        lmrk = L2candidates.sum(axis=1).argmin() #sum the columns
        L2candidates[lmrk] = coorsys.proximity_to(sources = [lmrk], dests=candidates).iloc[:,0]        

    return L2candidates.columns.tolist()

def find_user_landmarks(g, content_landmarks, **kwargs):
    coorsys=PathAccumCoordinates(g,kwargs)
    g = coorsys.g
    
    candidates = [n for n in g.nodes() if n.startswith('u')]
    L2candidates = coorsys.proximity_to(sources=content_landmarks, dests=candidates)
    user_landmarks=[]
    for l in L2candidates:
        user_landmarks.append(L2candidates[l].argmax())
    return user_landmarks

def algin_coor_sys(cs_static, cs):
    #A,B have shape (landmark-by-dimension)
    A = cs.lmrk_coors.as_matrix().transpose()
    B = cs_static.lmrk_coors.as_matrix().transpose()
    U,Sigma,V = randomized_svd(np.dot(A.transpose(),B), n_components=A.shape[0])
    Q = np.dot(U,V.transpose())
    W = Q
    A = np.dot(A,W)
    cs.lmrk_coors = pd.DataFrame(A.transpose(), columns=cs.lmrk_coors.columns)
    return W

class PAC_Eval:
    def __init__(self, cs):
        self.cs = cs
        self.coor_all = cs.coordinate(cs.nodes())
    
    def eval_prox_random(self, n_sample_node=5, sample_nodes=[]):
        cs = self.cs
        measurements = {}
        nodes = cs.nodes()

        test_nodes=[]
        if len(sample_nodes):
            if type(sample_nodes[0]) is str: 
                test_nodes = sample_nodes
            elif type(sample_nodes[0]) is int:
                test_nodes = [nodes[i] for i in sample_nodes]
        else:
            test_nodes = resample(nodes, n_samples=n_sample_node)

        #nae of coordinate-based proximity vs ground-proximity
        coor_test = self.coor_all[test_nodes]

        ground_prox = cs.proximity_to(sources=test_nodes,dests=cs.nodes()).as_matrix().transpose() #shape: test_nodes x all_nodes
        coor_prox = np.dot(coor_test.as_matrix().transpose(), self.coor_all.as_matrix())

        nae = pd.Series.combine(pd.Series(coor_prox.flatten()),pd.Series(ground_prox.flatten()), lambda c,g: abs(c-g)/g)    
        nae_plot = pd.Series(np.linspace(0.,1.,num=len(nae)),index=nae.order())       
        measurements['nae']=nae
        measurements['nae_plot']=nae_plot

        return measurements        
