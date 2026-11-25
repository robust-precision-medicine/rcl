# Required dependencies in addition to the robust_sc env:
# pip install contrastive
# conda install -c conda-forge qpsolvers
# conda install dask -c conda-forge

from typing import Union, Literal, List, Callable, Dict
from anndata._core.anndata import AnnData
import numpy as np
import os
from qpsolvers import solve_qp
import time
from dask import compute, delayed
from dask.distributed import Client
from logger import SimpleLogger
from scipy.sparse._csr import csr_matrix


class rcl:
    
    # robust: setting to False will fit a non-robust objective
    # inplace: setting to False will return a new copy of anndata 
    def __init__(self,
                 adata: AnnData,
                 sample_key: str,
                 G0: list[str],
                 G1: list[str],
                 subset_size: int = 2,
                 parallel: bool = True,
                 robust: bool = True,
                 ratio: bool = True,
                 inplace: bool = True,
                 logfile: str = "rcl.log"):
        
        self.subset_size = subset_size
        self.G0 = G0
        self.G1 = G1
        self.parallel = parallel
        self.robust = robust
        self.ratio = ratio
        self.sample_key = sample_key
        self.n_genes = adata.shape[1]
        self.X1 = adata[adata.obs[self.sample_key].isin(self.G1),:].X.toarray()
        self.X0 = adata[adata.obs[self.sample_key].isin(self.G0),:].X.toarray()
        if inplace:
            self.adata = adata
        else:
            self.adata = adata.copy()
        assert isinstance(adata.X,csr_matrix), "adata.X must be of type csr_matrix."
        if self.robust:
            self.n0 = len(G0)
            self.n1 = len(G1)
        else:
            self.n0 = 1
            self.n1 = 1
        self.n = self.n0+self.n1
        self.dim = self.n+self.n_genes
        self._calc_QP_matrices()
        self._calc_covariances()
        self.logfile = logfile
        self.logger = None

    
    def _calc_covariances(self):
        # create a 3D array with the (X_i^T X_i) m by m matrices of each sample
        Z0 = []
        Z1 = []
        if self.robust:
            for i in range(self.n0):
                X_i = self.adata[self.adata.obs[self.sample_key] == self.G0[i],:].X.toarray()
                Z0.append(np.dot(X_i.T,X_i))
            Z0 = np.stack(Z0)
            for i in range(self.n1):
                X_i = self.adata[self.adata.obs[self.sample_key] == self.G1[i],:].X.toarray()
                Z1.append(np.dot(X_i.T,X_i))
            Z1 = np.stack(Z1)
        else:
            # simply treat G0 as a single sample and G1 as a single sample
            Z0 = np.stack([np.dot(self.X0.T,self.X0)])
            Z1 = np.stack([np.dot(self.X1.T,self.X1)])
        self.Z = np.concatenate((Z0,Z1),axis=0)
        self.Z_T = self.Z.T.copy()
    
    # Encode constraints for the projection via QP
    def _calc_QP_matrices(self):
        self.G = None
        self.h = None
        self.lb = None
        self.ub = None
        if self.robust:
            self.P = np.identity(self.dim)
            self.G0_l1_norm = np.concatenate((np.zeros((1,self.n_genes)),np.ones((1,self.n0)), np.zeros((1,self.n1))),axis=1)
            self.G1_l1_norm = np.concatenate((np.zeros((1,self.n_genes)),np.zeros((1,self.n0)), np.ones((1,self.n1))),axis=1)
            self.lb = np.zeros((self.dim,1)) 
            self.ub = np.ones((self.dim,1))
            self.lb[0:self.n_genes] -= np.inf
            self.ub[0:self.n_genes] += np.inf
        else:
            self.P = np.identity(self.n_genes)

        ### below a version with inequality constraints instead of lb and ub
        # if self.robust:
        #     self.P = np.identity(self.dim)
        #     self.G0_l1_norm = np.concatenate((np.zeros((1,self.n_genes)),np.ones((1,self.n0)), np.zeros((1,self.n1))),axis=1)
        #     self.G1_l1_norm = np.concatenate((np.zeros((1,self.n_genes)),np.zeros((1,self.n0)), np.ones((1,self.n1))),axis=1)
        #     self.G = np.concatenate((np.zeros((self.n_genes,self.n0+self.n1)),-np.identity(self.n0+self.n1))).T
        #     self.h = np.zeros((self.n0+self.n1,1))
        #     if self.subset_size > 1:
        #         # add constraints that the entries of w0, w1 are all <=1
        #         self.G = np.vstack((self.G, np.concatenate((np.zeros((self.n_genes,self.n0+self.n1)),np.identity(self.n0+self.n1))).T))
        #         self.h = np.concatenate((self.h,np.ones((self.n0+self.n1,1))))
        # else:
        #     self.P = np.identity(self.n_genes)
        #     self.G = None
        #     self.h = None


    def get_rand_point(dim, n_genes, n0, rng):
        y0 = rng.random((dim,))
        y0[:n_genes] = y0[:n_genes] / np.sqrt(np.sum(y0[:n_genes]**2)) # set l2 norm of v to be 1
        y0[n_genes:n_genes+n0] = y0[n_genes:n_genes+n0] / np.sum(y0[n_genes:n_genes+n0]) # set l1 norm of w0 to be 1
        y0[n_genes+n0:] = y0[n_genes+n0:] / np.sum(y0[n_genes+n0:]) # set l1 norm of w1 to be 1
        return y0
    
    # returns the gradient of the objective at point z
    def _F(self, z):
        if self.robust:
            v = z[0:self.n_genes]
            w0 = z[self.n_genes:self.n_genes+self.n0]
            w1 = z[self.n_genes+self.n0:]

            Az = np.dot(np.dot(self.Z_T[:,:,0:self.n0],w0),v)
            Bz = np.dot(np.dot(self.Z_T[:,:,self.n0:],w1),v)
            zAz = np.dot(v,Az)
            zBz = np.dot(v,Bz)
            
            if self.ratio:
                Fz_v = 2*(zBz*Az-zAz*Bz)/(zBz**2)
                Fz_w0 = np.dot(np.dot(self.Z[0:self.n0,:,:],v),v)/zBz
                Fz_w1 = -zAz*(np.dot(np.dot(self.Z[self.n0:,:,:],v),v)/(zBz**2))
            else:
                Fz_v = 2*(Az-Bz)
                Fz_w0 = np.dot(np.dot(self.Z[0:self.n0,:,:],v),v)
                Fz_w1 = -np.dot(np.dot(self.Z[self.n0:,:,:],v),v)
            Fz = np.concatenate((Fz_v,-Fz_w0,-Fz_w1)) # negate the w0, w1 components for a gradient descent-ascent
            return Fz
        else:
            # calculate the gradient of the objective at point z for a non-robust objective
            Az = np.dot(self.Z_T[:,:,0],z)
            Bz = np.dot(self.Z_T[:,:,1],z)
            if self.ratio:
                zAz = np.dot(z,Az)
                zBz = np.dot(z,Bz)
                return 2*(zBz*Az-zAz*Bz)/(zBz**2)
            else:
                return 2*(Az-Bz)
            
            
        
    # Returns the objective.
    # For a robust objective, this calculation does not use w0, w1 but rather use the combination of self.subset_size samples from G0 and self.subset_size samples from G1 such that the objective is maximized for the given v.
    def calc_obj_fixed_v(self, v):
        if self.robust:
            local_obj = np.dot(np.dot(self.Z,v),v)
            g0_val = np.sum(local_obj[np.argsort(-local_obj[0:self.n0])[0:self.subset_size]]) # sum across self.subset_size samples with largest values in w0
            g1_val = np.sum(local_obj[self.n0+np.argsort(local_obj[self.n0:])[0:self.subset_size]]) # sum across self.subset_size samples with smallest values in w1
        else:
            g0_val = np.dot(v,np.dot(self.Z_T[:,:,0],v))
            g1_val = np.dot(v,np.dot(self.Z_T[:,:,1],v))
        if self.ratio:
            return g0_val/g1_val
        else:
            return g0_val-g1_val
    
    
    def _train_explore(self, lrs, n_iters, rng, A, b, z_init = None):
        # learning rates to consider
        z_inits = []
        if z_init is None:
            for i in range(len(lrs)):
                z_inits.append(rcl.get_rand_point(self.dim, self.n_genes, self.n0, rng))
            if not self.robust:
                for i in range(len(lrs)):
                    z_inits[i] = z_inits[i][0:self.n_genes]
        
        if self.parallel:
            if len(z_inits)>0:
                tasks = [delayed(self._train_single_lr)(A = A, b = b, z_init = z_inits[i], lr = lrs[i], n_iters = n_iters) for i in range(len(lrs))]
            else:
                tasks = [delayed(self._train_single_lr)(A = A, b = b, z_init = z_init.copy(), lr = lrs[i], n_iters = n_iters) for i in range(len(lrs))]
            solutions = compute(*tasks)
        else:
            solutions = []
            for i in range(len(lrs)):
                lr = lrs[i]
                if len(z_inits)>0:
                    z_init_i = z_inits[i]
                else:
                    z_init_i = z_init.copy()
                solutions.append(self._train_single_lr(A = A, b = b, z_init = z_init_i, lr = lr, n_iters = n_iters))

        objs = [self.calc_obj_fixed_v(z[0:self.n_genes]) for z in solutions]
        i = np.argmin(objs)
        lr = lrs[i]
        z = solutions[i]
        min_obj = objs[i]
        return z, lr, min_obj
    
    def _train_single_lr(self, A, b, z_init, lr, n_iters):
        P = self.P
        G = self.G
        h = self.h
        lb = self.lb
        ub = self.ub
        # QP for projecting a point onto the feasible space of solutions that satisfies the constraints on w_0, w_1
        # This is achived by finding the closest point to the unconstrained solution z that satisfies the constraints.
        # This is equivalent to solving: min_x |x^T x - z^T x|_2,  with constraints on the components corresponding to w.
        def proj(z):
            return solve_qp(P, -z, G, h, A, b, lb = lb, ub = ub, solver="daqp")
        z = z_init.copy()
        if self.robust:
            for t in range(n_iters):
                grad = self._F(z)
                lr_adjusted = np.zeros((len(grad),)) + lr # adjust the learning rate for the w0 and w1 paramters
                lr_adjusted[self.n_genes:] *= np.sqrt(np.sum(grad[0:self.n_genes]**2))/np.sqrt(np.sum(grad[self.n_genes:]**2))
                z = proj(z - lr_adjusted*grad)  # update z and project it to the space of constraints on w_0, w_1
                z[0:self.n_genes] = z[0:self.n_genes]/np.sqrt(np.sum(z[0:self.n_genes]**2))    # constrain |v|_2=1
        else:
            for t in range(n_iters):
                if A is None:
                    z = z - lr*self._F(z)
                else:
                    z = proj(z - lr*self._F(z))
                z = z/np.sqrt(np.sum(z**2))
        return z
    
    def train(self,
              n_components = 1,
              n_iters_main = 50,
              n_iters_explore = 50,
              n_iters_lr = 250,
              min_lr = 1e-8,
              max_lr = 1e-2,
              n_lrs = 20,
              rand_seed = None):

        self.logger = SimpleLogger(filename = self.logfile)
        self.logger.add_to_log("Starting optimization. n0={},n1={},n_genes={}.".format(len(self.G0),len(self.G1),self.n_genes))

        if self.ratio:
            convergence_cutoff = 1e-3
        else:
            convergence_cutoff = 1

        lrs = np.logspace(np.log2(max_lr), np.log2(min_lr), n_lrs, base = 2)

        if rand_seed is None:
            rand_seed = 1
        rng = np.random.default_rng(rand_seed)
        components = []
        w0 = []
        w1 = []
        A = None
        b = None
        for n_component in range(1,n_components+1):
            
            self.logger.add_to_log("Learning component number {} out of {} components...".format(n_component,n_components))

            # Equality constraints for the projection via QP
            if n_component > 1:
                # If previous components were calculated then A and b further include orthogonality constraints. Specifically, for every previously calculated component z', let p be the projection Xz', then we wish np.dot(p.T, np.dot(X,v)) to be 0. We rewrite to require np.dot(p.T,X) to be orthogonal to v.
                orthogonal_subspace = []
                for n_component_prev in range(n_component-1):
                    orthogonal_condition = np.dot(np.dot(self.X1,components[n_component_prev][0:self.n_genes]).T,self.X1)
                    if self.robust:
                        a = np.zeros((1,self.dim))
                    else:
                        a = np.zeros((1,self.n_genes))
                    a[0,0:self.n_genes] = orthogonal_condition
                    orthogonal_subspace.append(a)
                if self.robust:
                    l1_norm_constraints = [self.G0_l1_norm,self.G1_l1_norm]
                    n_eq_constraints = 1+n_component
                else:
                    l1_norm_constraints = []
                    n_eq_constraints = n_component-1
                A = np.concatenate(l1_norm_constraints + [a for a in orthogonal_subspace], axis=0)
                b = np.zeros((n_eq_constraints,1))
                if self.robust:
                    b[0:2] = self.subset_size # for the G0, G1 l1 norms
            else:
                if self.robust:
                    A = np.concatenate((self.G0_l1_norm,self.G1_l1_norm), axis=0)
                    b = np.ones((2,1)) * self.subset_size
            
            z = None
            prev_obj = np.Inf
            intermediate_zs = []
            intermediate_objs = []
            for i in range(n_iters_main):
                self.logger.add_to_log("Iteration number {}...".format(i+1))
                self.logger.add_to_log("\tExploring learning rates")
                z, lr, min_obj = self._train_explore(lrs, n_iters = n_iters_explore, rng = rng, A = A, b = b, z_init = z)
                # save the current z and its objective value
                intermediate_zs.append(z)
                intermediate_objs.append(min_obj)

                # check for convergence
                if np.abs(prev_obj - min_obj) < convergence_cutoff:
                    self.logger.add_to_log("\tConverged.")
                    break
                prev_obj = min_obj

                self.logger.add_to_log("\tLearning rate selected: {}; current objective: {}".format("{:.2e}".format(lr), round(min_obj,4)))
                self.logger.add_to_log("\tTraining model for {} iterations...".format(n_iters_lr))
                z = self._train_single_lr(A, b, z_init = z, lr = lr, n_iters = n_iters_lr)
                obj_i = self.calc_obj_fixed_v(z[0:self.n_genes])
                self.logger.add_to_log("\tCurrent objective: {}".format(round(obj_i,4)))
                
                # save the current z and its objective value
                intermediate_zs.append(z)
                intermediate_objs.append(obj_i)

            z = intermediate_zs[np.argsort(intermediate_objs)[0]]
            self.logger.add_to_log("Objective of final solution: {}".format(round(self.calc_obj_fixed_v(z[0:self.n_genes]), 4)))
            components.append(z[0:self.n_genes])
            if self.robust:
                w0.append(z[self.n_genes:self.n_genes+self.n0])
                w1.append(z[self.n_genes+self.n0:])
            self.logger.add_to_log("Completed optimization of component {}.".format(n_component))
        self.logger.add_to_log("Saving components and projections...")
        self.components = components
        self.G0_projections = [np.dot(self.X0,components[i][0:self.n_genes]) for i in range(n_components)]
        self.G1_projections = [np.dot(self.X1,components[i][0:self.n_genes]) for i in range(n_components)]
        self.w0 = w0
        self.w1 = w1
        self.adata.varm["RCL"] = np.stack([i[0:self.n_genes] for i in components]).T
        self.adata.obsm["RCL"] = np.dot(self.adata.X.toarray(), self.adata.varm["RCL"])
        self.logger.add_to_log("Done.")

