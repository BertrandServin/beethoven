"""
A set of models to population structure parameters from poolseq data 
"""
import sys
import time
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import binom, norm
from scipy.special import logit, expit, ndtr

## use namedtuple for parameters


## Use abstract class
class StructModel(ABC):
    """Generic Structure Model class for queen genotype prediction.


    This is an abstract class that cannot be instanciated. Required abstract methods are:

    -- _init_parameters : initialize parameters of the model for EM algorithm 
    -- update_parameters : Compute EM updates of the parameters 
    -- init_data_structures : initialize data structures for EM computations 
    -- P_X_giv_G : Compute marginal probabilities of the data (X) conditional on queen genotype (G) and model parameters 
    -- predict : method for predicting posterior probabilities of queen genotype (G)

    """
    
    def __init__(self, name):
        if name is None:
            name = 'sm'
        self.name=str(name)

    def estimate_parameters(self, X, D, F, nstep, delta_l = 0.1, minvar = 0.01, verbose=False):
        """ Estimate model parameter on data (X, D, F) using EM algorithm

        Parameters
        ----------
        X : numpy array
           Reference allele counts
        D : numpy array
           Sequencing depths
        F : numpy array
            Matrix of allele frequencies in reference groups
        nstep : int
            Maximum number of EM iterations 
        delta_l : float, optional
            Stop EM when loglikelihood increase is < `delta_l`
        minvar : float, positive, optional
            Minimum variance across group for a SNP to be used
        verbose : bool, optional, default False
            Be verbose

        
        Returns
        -------
        dict
            Estimated parameters
        """
        assert minvar>0
        vf = F.var(axis = 1)
        snpsel = np.isfinite(vf) & (vf > minvar)
        X = X[snpsel]
        D = D[snpsel]
        F = F[snpsel,]
        self.init_data_structures(X, D, F)
        pars = self._init_parameters()

        curlik = -np.inf
        for iem in range(nstep):
            Q = pars['Q']
            #### P( Z | Q )
            P_Z_giv_Q = self.calc_P_Z_giv_Q(Q)
            #### P( X | Z ) = Sum_g { P( X | g)  P( g | Z ) }
            P_X_giv_G = self.P_X_giv_G(X, D, F, pars)
            P_X_giv_Z = np.einsum('lg, lgk -> lk', P_X_giv_G, self.P_G_giv_Z)
            #### P(X , Z|Q ) = P( X | Z ) * P( Z | Q )
            P_X_Z_giv_Q = P_X_giv_Z * P_Z_giv_Q
            #### P(Z | X,Q ) = P( X, Z | Q) / P( X | Q)
            lik = P_X_Z_giv_Q.sum(axis=1, keepdims=True)
            loglik =  np.sum(np.log10( lik))
            P_Z_giv_XQ = P_X_Z_giv_Q / lik
            
            self.update_parameters(P_Z_giv_XQ, pars)
            if verbose:
                print('** Iteration', iem, loglik, np.array2string(np.around(Q, decimals=3))[1:-1])
            if iem>10 and (loglik - curlik) < delta_l:
                break
            curlik = loglik
        return pars
    
    @abstractmethod
    def predict(self, X, D, F, pars):
        """
        Predict queen genotypes from data (X, F, D) using parameters pars
        """
        pass
    
    @abstractmethod
    def _init_parameters(self):
        Q = np.random.random(self.K)
        Q /= sum(Q)
        pars = {'Q' : Q}
        return pars

    @abstractmethod
    def update_parameters(self, P_Z_giv_XQ, pars ):
        ### Q* | Q = E( P( Z | X, Q)) 
        newQ = np.zeros(self.K)
        for k1 in range(self.K):
            for k2 in range(k1, self.K):
                ik = self.Kdicts['p2i'][(k1,k2)]
                newQ[ k1 ] += 0.5*P_Z_giv_XQ[ : , ik].sum()
                newQ[ k2 ] += 0.5*P_Z_giv_XQ[ : , ik].sum()
        pars['Q'] = newQ/newQ.sum()
    
    @abstractmethod
    def init_data_structures(self, X, D, F):
        self.L = F.shape[0]
        self.K = F.shape[1]
        self.Ksq = int(self.K*(self.K+1)/2)
        self.Kdicts = self.make_dicts( self.K)
        ## P(G|Z)
        self.P_G_giv_Z = np.zeros((self.L, 3, self.Ksq), dtype = np.float)
        for k1 in range(self.K):
            for k2 in range(k1, self.K):
                ik = self.Kdicts['p2i'][(k1,k2)]
                self.P_G_giv_Z[:,0,ik] = (1 - F[:,k1]) * ( 1 - F[:,k2])
                self.P_G_giv_Z[:,1,ik] = (1 - F[:,k1]) * F[:,k2] + (1 - F[:,k2]) * F[:,k1]
                self.P_G_giv_Z[:,2,ik] = F[:,k1] * F[:,k2]

    @abstractmethod
    def P_X_giv_G(self, X, D, F, pars):
        pass
    
    @staticmethod
    def make_dicts(K):
        """make dictionaries of idx values for cluster pairs

        Parameters
        ----------
        K : int
            Number of clusters
        
        Returns
        -------
        dict 
            { 
              'p2i' : dict { (k1,k2) -> idx },
              'i2p' : dict { idx -> (k1,k2) } 
            }
        """
        idx=0
        p2i ={}
        i2p = {}
        for k1 in range(K):
            for k2 in range(k1,K):
                p2i[(k1,k2)] = idx
                p2i[(k2,k1)] = idx
                i2p[idx] = (k1,k2)
                idx+=1
        assert idx == K*(K+1)/2
        return { 'p2i':p2i, 'i2p':i2p }

    @staticmethod
    def calc_P_Z_giv_Q(Q):
        K = Q.shape[0]
        Ksq = int(K*(K+1)/2)
        res = np.zeros(Ksq, dtype = np.float)
        Kdicts = StructModel.make_dicts(K)
        for k1 in range(K):
            ik = Kdicts['p2i'][(k1, k1)]
            res[ik] = Q[k1]**2
            for k2 in range(k1+1, K):
                ik = Kdicts['p2i'][(k1, k2)]
                res[ik] = 2*Q[k1]*Q[k2]
        return res

class HeterogeneousModelBayes(StructModel):
    """A supervised structure model with different genetic mixtures in the queen and drones.

    The model places a prior on the allele frequency in drones constructed from 
    the F matrix (default) or using a uniform distribution.
    
    To use a uniform prior distribution use:
    mod = HeterogeneousModelBayes(uniform_prior = True)

    The model is fitted / applied to data that consist of:

    1. X : a vector of reference allele counts (integer)
    2. D : a vector of sequencing depth (integer)
    3. F : a matrix of reference allele frequencies in different genetic types (clusters)

    X,D are assumed to be obtained from a pool-seq experiment of a large number of worker bees.
    F is assumed to be computed from single individuals within each reference cluster

    """
    def __init__(self,name = 'sm_het_bayes',uniform_prior = False):
        StructModel.__init__(self, name)
        self.uniform_prior = uniform_prior

    def _init_parameters(self):
        """Initialize parameters
        
        This model has a single parameter to estimate : the admixture proportion 
        of the queen ( Q ). 
        """
        pars = StructModel._init_parameters(self)
        return pars

    def init_data_structures(self, X, D, F, tol = 1e-3, nbins = 100):
        """Initialize data structures needed for fit / prediction of the model

        Parameters
        ----------

        - tol : float, optional
            F values < tol or > (1-tol) are set to tol ( 0<tol<1 but typically small)
        - nbins : int
            number of bins for f prior

        Notes 
        ----- 
        Using stupid values for `tol` and `nbins` is not
        recommended as they may increase computing time and affect the
        accuracy of the model.

        """
        
        StructModel.init_data_structures(self, X, D, F)

        fseg = np.linspace( 0, 1, num = nbins+1)
        self.fbins = 0.5*(fseg[1:]+fseg[:-1])
        self.B = len( self.fbins)
        
        if not self.uniform_prior:
            Fp = np.where( F < tol, tol, F)
            Fp = np.where( Fp > 1-tol, 1-tol, Fp)
            logit_Fp = logit(Fp)
            logit_fseg = logit(fseg)
        
            mf = np.average( logit_Fp, axis=1).reshape( ( self.L, 1))
            sf = np.std( logit_Fp, axis=1)
            sf = np.where( sf >0 , sf, 0.5)
            self.f_prior = np.zeros( ( self.B, self.L))
            ## optimize prior calculations
            z = np.zeros( ( nbins+1, self.L))
            for l in range( self.L):
                ## default to uniform prior if NA
                if np.isnan(mf[l]) or np.isnan(sf[l]):
                    z[:,l] = 1
                else:
                    z[:,l] = (logit_fseg - mf[l])/sf[l]
            ##cdf_seg = ndtr(z) ## Gaussian cdf
            pdf_seg = norm.pdf(z) ## Gaussian pdf
            for l in range( self.L):
                self.f_prior[:,l] = 0.5*(pdf_seg[1:,l]+pdf_seg[:-1,l])/nbins
        else:
            self.f_prior = np.ones( ( self.B, self.L))
        self.f_prior /= self.f_prior.sum(axis=0, keepdims=True)
        ## P(X|G) = Sum_f (P(X|f,g)P(f)
        fd = np.array( [ 0.5 * self.fbins, 0.5 * ( self.fbins + 0.5), 0.5 * ( self.fbins + 1)])
        self.fgrid = np.zeros( ( 3, self.B, self.L)) ## 3 geno, B freq, L loci
        for _ in range(self.L): 
            self.fgrid[:,:,_] = fd
        ## likelihood
        self._p_x_giv_fg = binom.pmf( X, D, self.fgrid) ## shape is 3, B, L
        ## marginal likelihoods
        self._p_x_giv_g = np.einsum( 'gbl, bl -> lg', self._p_x_giv_fg, self.f_prior)
        self._p_x_giv_fz = np.einsum( 'gbl, lgk -> lbk', self._p_x_giv_fg, self.P_G_giv_Z)
            
    def P_X_giv_G(self, X, D, F, pars):
        try:
            return self._p_x_giv_g
        except AttributeError:
            self.init_data_structures(X,D,F)
            return self._p_x_giv_g

    def genotype_loglikelihoods(self, X, D, F):
        return np.log(self.P_X_giv_G(X,D,F,None))
        
    def update_parameters(self, P_Z_giv_XQ, pars ):
        StructModel.update_parameters( self, P_Z_giv_XQ, pars)

    def posterior_distribution( self, X, D, F, pars):
        """Calculate posterior distribution P(G,F | X,Q)

        Parameters
        ----------
        X : numpy array
            reference allele counts
        D : numpy array
            sequencing depths
        F : numpy array
            Matrix of reference allele counts
        
        Returns
        -------
        dict 
            {
                'postdist' :
                    numpy array of posterior probabilities P(F,G | X,Q) of dimensions (L, 3, B). 
                    Where B is the number of  bins for the prior on f.
                'fbins' :
                      numpy array of values of f over which f is integrated
            }
        """
        self.init_data_structures( X, D, F)
        ## G prior from Q : P(G|Q)
        p_z_giv_q = self.calc_P_Z_giv_Q( pars['Q'])
        p_g_giv_q = np.dot(self.P_G_giv_Z, p_z_giv_q)
        
        # ## P(F,G|X, Q) propto [ P(X|F,G) P(F) ] P(G|Q)
        _a = np.einsum( 'gbl,  bl -> gbl', self._p_x_giv_fg, self.f_prior)
        p_fg_giv_xq = np.einsum( 'gbl, lg -> lgb', _a, p_g_giv_q)
        p_fg_giv_xq /= p_fg_giv_xq.sum(axis=(1,2), keepdims=True)
        return {
                 'lik' : np.einsum('gbl -> lgb', self._p_x_giv_fg),
                 'gprior' : p_g_giv_q,
                 'fprior' : self.f_prior,
                 'postdist' : p_fg_giv_xq,
                 'fbins' : self.fbins,
               }
    
    def predict(self, X, D, F, pars, chunksize=100000):
        """Predict queen genotypes from data (X, D, F) using parameters pars.
        Calculations are processes in chunks of 'chunksize' SNPs.
        
        Parameters
        ----------
        X : numpy array
            reference allele counts
        D : numpy array
            sequencing depths
        F : numpy array
            Matrix of reference allele counts
        pars: dictionary
            Parameters of the model 
        chunksize : 
            Size of SNP chunks to process. Lower value means lesser memory footprint but possibly
            greater computation time.
        Returns
        -------
        dict
            { 
              'postG' :
                  numpy array of posterior probabilities of genotypes (AA/ AR / RR) where 
                  A : alternative and R : reference alleles.
              'Fdrone' :
                  numpy array of maximum a posteriori estimate of allele frequency in drones
              'egs' :
                  numpy array of expected genotypes
              'bgs' :
                  numpy array of maximum a posteriori genotype (best guess)
            }
        """
        c_postg = []
        c_Fdrone = []
        c_egs = []
        c_bgs = []
        for istart in np.arange(0,X.shape[0],chunksize):
            iend = np.min([X.shape[0],istart+chunksize])
            sys.stderr.write('Processing SNPs : {} - {}\r'.format(istart,iend))
            sys.stderr.flush()
            c_X = X[istart:iend,]
            c_D = D[istart:iend,]
            c_F = F[istart:iend,]
            c_pprob = self.posterior_distribution(c_X, c_D, c_F, pars)
            c_pggivx =  np.sum(c_pprob['postdist'], axis=2)
            c_postg.append( c_pggivx)
            c_best_guess = [r for r in map(lambda M: np.unravel_index(M.argmax(),M.shape), c_pprob['postdist'])]
            c_bgs.append( np.array([ bg[0] for bg in c_best_guess]))
            c_egs.append( np.dot( c_pggivx, np.array([0, 0.5, 1])))
            c_Fdrone.append( np.array([ c_pprob['fbins'][bg[1]] for bg in c_best_guess]))
        print("\nDone")
        return { 'postG' : np.vstack( c_postg),
                 'Fdrone' : np.concatenate( c_Fdrone),
                 'egs' : np.concatenate( c_egs),
                 'bgs' : np.concatenate( c_bgs)
        }
    
            
    def predict_mem(self, X, D, F, pars):
      """Predict queen genotypes from data (X, D, F) using parameters pars. Memory intensive version.
        
        Parameters
        ----------
        X : numpy array
            reference allele counts
        D : numpy array
            sequencing depths
        F : numpy array
            Matrix of reference allele counts
        
        Returns
        -------
        dict
            { 
              'postG' :
                  numpy array of posterior probabilities of genotypes (AA/ AR / RR) where 
                  A : alternative and R : reference alleles.
              'Fdrone' :
                  numpy array of maximum a posteriori estimate of allele frequency in drones
              'egs' :
                  numpy array of expected genotypes
              'bgs' :
                  numpy array of maximum a posteriori genotype (best guess)
            }
      """
      pprob = self.posterior_distribution( X, D, F, pars)
      best_guess = [r for r in map(lambda M: np.unravel_index(M.argmax(),M.shape), pprob['postdist'])]
      P_G_giv_X =  np.sum( pprob['postdist'], axis=2)
      return { 'postG' : P_G_giv_X,
               'Fdrone' : np.array([ pprob['fbins'][bg[1]] for bg in best_guess]),
               'egs' : np.dot( P_G_giv_X, np.array([ 0, 0.5, 1])),
               'bgs' : np.array([ bg[0] for bg in best_guess])
      }

    def _predict_old(self, X, D, F, pars):
        """Predict queen genotypes from data (X, D, F) using parameters pars.
        
        Parameters
        ----------
        X : numpy array
            reference allele counts
        D : numpy array
            sequencing depths
        F : numpy array
            Matrix of reference allele counts
        
        Returns
        -------
        dict
            { 
              'postG' :
                  numpy array of posterior probabilities of genotypes (AA/ AR / RR) where 
                  A : alternative and R : reference alleles.
              'fbins' :
                  numpy array of values of f over which f is integrated
              'postF' : 
                  numpy array of posterior probabilities for each f in fbins
            }
        """
        self.init_data_structures( X, D, F)
        p_z_giv_q = self.calc_P_Z_giv_Q( pars['Q'])
        ## P(G | X, Q) = Sum_z P(G|X,Z) P(Z|Q)
        ### P( G|X,Z) \propto P(X|G) P(G|Z)
        P_G_giv_XZ = np.einsum( 'lg, lgk -> lgk',
                               self.P_X_giv_G(None, None, None, None),
                               self.P_G_giv_Z)
        normG = np.sum(P_G_giv_XZ,axis=1,keepdims=True)
        P_G_giv_XZ /= normG
        P_G_giv_X = np.dot(P_G_giv_XZ, p_z_giv_q)
        ## P(F | X,Q) \propto P(X | F,Q) P(F)
        ### P(X | F,Q) = Sum_Z P(X | F, Z) P(Z|Q)
        ### P(X | F, Z) = Sum_g P(X|F,g) P(g|Z)
        P_X_giv_F = np.einsum( 'lbk, k -> lb', self._p_x_giv_fz, p_z_giv_q) ## L x B
        P_XF = P_X_giv_F * self.f_prior.T ##np.einsum( 'lb, b -> lb', P_X_givF, f_prior)
        P_X = P_XF.sum(axis=1, keepdims=True)
        P_F_giv_X = P_XF/P_X ## L x B
        ##Postmean_F = np.dot( P_F_giv_X, self.fbins)
        return { 'postG' : P_G_giv_X,
                 'fbins' : self.fbins,
                 'postF' : P_F_giv_X
               }


class HomogeneousModel( StructModel):
    """
     A supervised structure model with common mixture in the queen and drones.

    The model is fitted / applied to data that consist of:

    1. X : a vector of reference allele counts (integer)
    2. D : a vector of sequencing depth (integer)
    3. F : a matrix of reference allele frequencies in different genetic types (clusters)

    X,D are assumed to be obtained from a pool-seq experiment of a large number of worker bees.
    F is assumed to be computed from single individuals within each reference cluster
    """

    def __init__(self, name='sm_hom'):
        StructModel.__init__(self, name)

    def _init_parameters( self):
        """
        This model has a single parameter to estimate : the admixture proportion 
        of the queen ( Q ). 
        """
        pars = StructModel._init_parameters( self )
        return pars
    
    def update_parameters(self, P_Z_giv_XQ, pars ):
        StructModel.update_parameters( self, P_Z_giv_XQ, pars)
    
    def init_data_structures(self, X, D, F):
        """
        Initialize data structures needed for fit / prediction of the model
        """
        StructModel.init_data_structures(self, X, D, F)
     
    def P_X_giv_G(self, X, D, F, pars):
        Fdrone = np.dot( F, pars['Q'])
        fh = np.array( [0.5 *   Fdrone,
                        0.5 * ( Fdrone + 0.5),
                        0.5 * ( Fdrone + 1)] )
        return binom.pmf( X, D, fh).T

    def predict(self, X, D, F, pars):
        """
        Predict queen genotypes from data (X, D, F) using parameters pars.

        Return value
        ------------

        A dictionary :

        - postG : posterior probabilities of genotypes (AA/ AR / RR) where
                  A : alternative and R : reference alleles.
        - Fdrone : drone allele frequencies = dot( F, pars[Q])
        """
        self.init_data_structures( X, D, F)
        Fdrone = np.dot( F, pars['Q'])
        
        p_z_giv_q = self.calc_P_Z_giv_Q( pars['Q'])
        ## P(G | X, Q) = Sum_z P(G|X,Z) P(Z|Q)
        ### P( G|X,Z) \propto P(X|G) P(G|Z)
        _p_x_giv_g = self.P_X_giv_G(X, D, F, pars) ## L x 3
        P_G_giv_XZ = np.einsum( 'lg, lgk -> lgk', _p_x_giv_g, self.P_G_giv_Z) 
        normG = np.sum(P_G_giv_XZ,axis=1,keepdims=True)
        P_G_giv_XZ /= normG
        P_G_giv_X = np.dot(P_G_giv_XZ, p_z_giv_q)
        return { 'postG' : P_G_giv_X,
                 'Fdrone' : Fdrone,
                 'egs' : np.dot( P_G_giv_X, np.array([ 0, 0.5, 1])),
                 'bgs' : np.argmax( P_G_giv_X, axis=1)
               }


    
class HeterogeneousModelMax( StructModel):
    """
    A supervised structure model with different genetic mixtures in the queen and drones.

    The model estimates allele frequency in drones by maximum likelihood.

    The model is fitted / applied to data that consist of:

    1. X : a vector of reference allele counts (integer)
    2. D : a vector of sequencing depth (integer)
    3. F : a matrix of reference allele frequencies in different genetic types (clusters)

    X,D are assumed to be obtained from a pool-seq experiment of a large number of worker bees.
    F is assumed to be computed from single individuals within each reference cluster

    """
    def __init__( self, name = 'sm_het_max'):
        StructModel.__init__( self, name)

    def _init_parameters( self ):
        """
        This model has a single parameter to estimate : the admixture proportion 
        of the queen ( Q ). 
        """
        pars = StructModel._init_parameters( self )
        pars['Fdrone'] = np.random.random( self.L)
        return pars

    def init_data_structures(self, X, D, F, tol = 1e-3, nbins = 100):
        """
        Initialize data structures needed for fit / prediction of the model

        Extra parameters
        ----------------

        - tol : F values < tol or > (1-tol) are set to tol ( 0<tol<1 but typically small)
        - nbins : controls the granularity of the prior on f (integer, # bins)

        Using stupid values for these is not recommended as they may increase computing time
        and affect the accuracy of the model.
        """
        
        StructModel.init_data_structures(self, X, D, F)
        
        self.fbins = np.linspace( 0, 1, num = nbins+1)
        self.B = len( self.fbins)

        fd = np.array( [ 0.5 * self.fbins, 0.5 * ( self.fbins + 0.5), 0.5 * ( self.fbins + 1)])
        self.fgrid = np.zeros( ( 3, self.B, self.L)) ## 3 geno, B freq, L loci
        for _ in range(self.L): 
            self.fgrid[:,:,_] = fd
        ## likelihood and marginal on f
        self._p_x_giv_fg = binom.pmf( X, D, self.fgrid) ## shape is 3, B, L
        self._p_x_giv_fz = np.einsum( 'gbl, lgk -> lbk', self._p_x_giv_fg, self.P_G_giv_Z)
            
    def P_X_giv_G(self, X, D, F, pars):
        fh = np.array( [0.5 *   pars['Fdrone'],
                        0.5 * ( pars['Fdrone'] + 0.5),
                        0.5 * ( pars['Fdrone'] + 1)] )
        return binom.pmf( X, D, fh).T

    def update_parameters(self, P_Z_giv_XQ, pars ):
        StructModel.update_parameters( self, P_Z_giv_XQ, pars)
        P_X_giv_f = np.einsum( 'lbk, lk -> lb', self._p_x_giv_fz, P_Z_giv_XQ) ## L x B
        ## cas indep des f
        argfmax = np.argmax(P_X_giv_f, axis=1)
        pars['Fdrone'] = self.fbins[ argfmax]
        
    def predict(self, X, D, F, pars):
        """
        Predict queen genotypes from data (X, D, F) using parameters pars.

        Return value
        ------------

        A dictionary :

        - postG : posterior probabilities of genotypes (AA/ AR / RR) where
                  A : alternative and R : reference alleles.
        - Fdrone : MLE (?) of Fdrone
        """
        self.init_data_structures( X, D, F)
        ## this needs to be checked as it means the model must have been fitted
        ## on the whole dataset
        assert len(pars['Fdrone']) == self.L
        
        p_z_giv_q = self.calc_P_Z_giv_Q( pars['Q'])
        ## P(G | X, Q) = Sum_z P(G|X,Z) P(Z|Q)
        ### P( G|X,Z) \propto P(X|G) P(G|Z)
        _p_x_giv_g = self.P_X_giv_G(X, D, F, pars) ## L x 3
        P_G_giv_XZ = np.einsum( 'lg, lgk -> lgk', _p_x_giv_g, self.P_G_giv_Z) 
        normG = np.sum(P_G_giv_XZ,axis=1,keepdims=True)
        P_G_giv_XZ /= normG
        P_G_giv_X = np.dot(P_G_giv_XZ, p_z_giv_q)
        return { 'postG' : P_G_giv_X,
                 'Fdrone' : pars['Fdrone'] 
               }
