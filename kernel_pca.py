from networkx.algorithms.kernels import __matrix_power_kernel as matrix_power_kernel, normalize, normalize_sin
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import numpy as np
from math import sqrt
import collections
from sklearn.utils import resample

class CoorSys:

	def __init__(self, Adj, node_names):
		'''
		Adj: Adjacency matrix 
		'''
		self.Adj = Adj
		self.Name2Id={name:idx for idx,name in enumerate(node_names)}

	def atonce(Adj, node_names, lmrk_candidates, k_lmrk, k_dim, kernel,**kwargs):
		cs = CoorSys(Adj,node_names)
		lmrks = resample(lmrk_candidates,n_samples=k_lmrk, replace=False)
		if type(k_dim) is float:
			k_dim =k_dim*k_lmrk
		cs.setlandmarks(lmrks, k_dim,kernel=kernel, **kwargs)
		return cs

	def setlandmarks(self, lmrks, dim, kernel='von_neumann',**kwargs):
		idx_lmrks = [self.Name2Id[l] for l in lmrks]
		#self.idx_lmrks = idx_lmrks
		self.K_lmrk2all = matrix_power_kernel(self.Adj, method=kernel, idx_from = idx_lmrks, **kwargs)
		
		#Mitigate diagonal
		L2L = self.K_lmrk2all[:,idx_lmrks]		
		L2L = L2L.toarray()
		eig_vals, eig_vecs = eigh(L2L)

		n_row = L2L.shape[1]
		data_I = np.ones(n_row)
		col_I = idx_lmrks
		row_I = np.arange(n_row)
		L_I = csr_matrix((data_I,(row_I,col_I)), shape=self.K_lmrk2all.shape)
		self.K_lmrk2all = self.K_lmrk2all + (1e-6 - eig_vals[0])*L_I
		print('removed',eig_vals[0])
		
		L2L = self.K_lmrk2all[:,idx_lmrks]
		sqrt_d = np.sqrt(L2L.diagonal())		
		L2L = L2L / sqrt_d
		L2L = L2L / sqrt_d[:,None]
		#Find eigen pairs.		
		eig_vals, eig_vecs = eigh(L2L)
		eig_vals = eig_vals[::-1]
		eig_vecs = np.fliplr(eig_vecs)

		#Half-Normalize
		self.K_lmrk2all = self.K_lmrk2all / sqrt_d[:,None]				
		
		self.dim = dim
		self.eig_vals = eig_vals[eig_vals > 0]
		self.eig_vecs = eig_vecs[:,eig_vals > 0]
		
		dim_full = len(self.eig_vals)		
		if dim > dim_full:
			dim = dim_full

	def setlandmarks_no_truncate(self, lmrks, dim, kernel='von_neumann',**kwargs):
		idx_lmrks = [self.Name2Id[l] for l in lmrks]
		#self.idx_lmrks = idx_lmrks
		self.K_lmrk2all = matrix_power_kernel(self.Adj, method=kernel, idx_from = idx_lmrks, **kwargs)
		
		L2L = self.K_lmrk2all[:,idx_lmrks]
		sqrt_d = np.sqrt(L2L.diagonal())		
		L2L = L2L / sqrt_d
		L2L = L2L / sqrt_d[:,None]
		#Find eigen pairs.		
		eig_vals, eig_vecs = eigh(L2L)
		eig_vals = eig_vals[::-1]
		eig_vecs = np.fliplr(eig_vecs)

		#Half-Normalize
		self.K_lmrk2all = self.K_lmrk2all / sqrt_d[:,None]				
		
		self.dim = dim
		self.eig_vals = eig_vals[eig_vals > 0]
		self.eig_vecs = eig_vecs[:,eig_vals > 0]
		
		dim_full = len(self.eig_vals)		
		if dim > dim_full:
			dim = dim_full

	def __getitem__(self,item):
		if item is None or isinstance(item,collections.Iterable):
			return self.__coor_batch__(item)
		else:
			return self.__coor__(item)

	def __coor__(self,name):
		idx = self.Name2Id[name]
		L2x = self.K_lmrk2all[:,idx]

		dim_full = len(self.eig_vals)
		
		coor = L2x.transpose().dot(self.eig_vecs).getA1()
		coor = coor / np.sqrt(self.eig_vals)

		coor = coor / np.linalg.norm(coor)

		return coor[:self.dim]

	def __coor_batch__(self,names):
		
		if names:
			L2x = self.K_lmrk2all[:,[self.Name2Id[i] for i in names]]
		else: # coors for all nodes
			L2x = self.K_lmrk2all

		#coors shape: src * lmrk_count
		coors = L2x.transpose().dot(self.eig_vecs)
		coors = coors / np.sqrt(self.eig_vals)
		norms = np.linalg.norm(coors,axis=1)
		coors = coors / norms[:,None]

		return coors[:,:self.dim]

	def proximity(self, src, dest=None):

		if src is None or type(src) is list:
			coors_src = self[src]			
		else:
			coors_src = self[[src]]

		coors_dest = self[dest]

		proximity = coors_src.dot(coors_dest.transpose())

		return proximity

class CoorSys_Eval:
	def __init__(self,cs,wrapped_kernel,test_nodes):
		self.cs = cs				
		self.idx_test = [cs.Name2Id[n] for n in test_nodes]
		self.kerTest2All = wrapped_kernel.similarity(src=test_nodes)
		self.coorTest2All = self.cs.proximity(src=test_nodes)

	def eval(self, test_name):
		from scipy.stats import kendalltau

		if test_name == 'NRMSE':
			
			Err = (self.kerTest2All - self.coorTest2All)
			Err[:,self.idx_test] = 0
			Err = np.sum(np.square(Err))/(Err.shape[0]*Err.shape[1])
			Err = sqrt(Err)
			return Err
		elif test_name == 'kendalltau':			
			return np.mean([kendalltau(self.kerTest2All[i,:].toarray().flatten(), 
										self.coorTest2All[i,:].flatten())[0] \
							for i in range(self.kerTest2All.shape[0])])
		else:
			raise Exception("Unknown test:",test_name)

'''class CoorSys_Eval:
	
	def create_coorsys(Adj, node_names, lmrk_candidates, k_lmrk, k_dim, kernel,**kwargs):
		cs = CoorSys(Adj,node_names)
		lmrks = resample(lmrk_candidates,n_samples=k_lmrk, replace=False)
		if type(k_dim) is float:
			k_dim =k_dim*k_lmrk
		cs.setlandmarks(lmrks, k_dim,kernel=kernel, **kwargs)
		return cs

	def __init__(self, cs):
		self.cs = cs

	def eval(self, test_name):

		return None

	def test(self,lmrks, name):

		idx_lmrks = [self.Name2Id[l] for l in lmrks]
		L2L = self.K_lmrk2all[:,idx_lmrks]
		sqrt_d = np.sqrt(L2L.diagonal())
		L2L = L2L / sqrt_d

		idx = lmrks.index(name)		
		L2x = L2L[:,idx]
		dim_full = len(self.eig_vals)
		
		coor = L2x.transpose().dot(self.eig_vecs)
		coor = coor / np.sqrt(self.eig_vals)

		return coor / np.linalg.norm(coor)
'''