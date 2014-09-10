import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import scipy.stats as st
import scipy.special as ss
from mingrid import *
from util import *
import time

class LMM(object):
    '''
    linear mixed model with up to two kernels
    N(y | X*beta + covariates*alpha; sigma2(h2*K + (1-h2)*I),
    where
    K = G*G^T
    '''
    __slots__ = ["linreg","G","Y","X","K","U","S","UX","UY","UUX","UUY","forcefullrank","regressX","numcalls"]

    def __init__(self, forcefullrank=False, X=None, linreg=None, Y=None, G=None, K=None, regressX=True, inplace=False):
        '''
        Input:
        forcefullrank   : if True, then the code always computes K and runs cubically
                            (False)
        '''
        self.numcalls=0
        self.setX(X=X, regressX=regressX, linreg=linreg)    #set the covariates (needs to be first)
        self.forcefullrank=forcefullrank
        self.setK(K=K, G=G, inplace=inplace)                 #set the kernel, if available
        self.setY(Y=Y)                      #set the phenotypes

    def setY(self, Y):
        '''
        set the phenotype y.
        --------------------------------------------------------------------------
        Input:
        y       : [N] 1-dimensional array of phenotype values
        --------------------------------------------------------------------------
        '''
        self.Y = Y
        self.UY = None
        self.UUY = None

    def setX(self, X=None, regressX=True, linreg=None):
        self.X = X
        self.UX = None
        self.UUX = None
        self.linreg=linreg
        self.regressX=regressX
        if self.linreg is None and regressX:
            self.linreg=Linreg(X=self.X)


    def setSU_fromK(self):
        N=self.K.shape[0]
        D=self.linreg.D
        ar = np.arange(self.K.shape[0])
        self.K[ar,ar]+=1.0
        K_ = self.linreg.regress(Y=self.K)
        K_ = self.linreg.regress(Y=K_.T)
        [self.S,self.U] = la.eigh(K_)
        self.U=self.U[:,D:N]
        self.S=self.S[D:N]-1.0


    def setSU_fromG(self):
        k = self.G.shape[1]
        N = self.G.shape[0]
        if k:
            if ((not self.forcefullrank) and (k<N)):
                #it is faster using the eigen decomposition of G.T*G but this is more accurate
                PxG = self.linreg.regress(Y=self.G)
                try:
                    [self.U,self.S,V] = la.svd(PxG,False,True)
                    inonzero = self.S>1E-10
                    self.S=self.S[inonzero]
                    self.S=self.S*self.S
                    self.U=self.U[:,inonzero]
                
                except la.LinAlgError:  # revert to Eigenvalue decomposition
                    print "Got SVD exception, trying eigenvalue decomposition of square of G. Note that this is a little bit less accurate"
                    [S,V] = la.eigh(PxG.T.dot(self.PxG))
                    inonzero=(S>1E-10)
                    self.S = S[inonzero]
                    #self.S*=(N/self.S.sum())
                    self.U=self.G.dot(V[:,inonzero]/np.sqrt(self.S))
            else:
                K=self.G.dot(self.G.T);
                self.setK(K=K)
                self.setSU_fromK()
            pass
        else:#rank of kernel = 0 (linear regression case)
            self.S = np.zeros((0))
            self.U = np.zeros_like(self.G)


    def getSU(self):
        """
        get the spectral decomposition of K.
        """
        if self.U is None or self.S is None:
            if self.K is not None:
                self.setSU_fromK()
            elif self.G is not None:
                self.setSU_fromG()
            else:
                raise Exception("No Kernel is set. Cannot return U and S.") 
        return self.S, self.U

    def rotate(self, A):
        S,U=self.getSU()
        N=A.shape[0]
        D=self.linreg.D
        if (S.shape[0]<N-D):#lowrank case
            A=self.linreg.regress(A)
            UA =self.U.T.dot(A)
            UUA = A-U.dot(UA)
        else:
            #A=self.linreg.regress(A)
            UA=U.T.dot(A)
            #A1 = UA=U.T.dot(A)
            #diff = np.absolute(A1-UA).sum()
            #print diff
            #print UA.shape
            UUA=None
        return UA,UUA


    def getUY(self):
        if self.UY is None:
            self.UY,self.UUY=self.rotate(A=self.Y)
        return self.UY,self.UUY


    def setK(self, K=None, G=None, inplace=False):
        '''
        set the Kernel K.
        --------------------------------------------------------------------------
        Input:
        K : [N*N] array, random effects covariance (positive semi-definite)
        --------------------------------------------------------------------------
        '''
        self.clear_cache()
        if K is not None:
            if inplace:
                self.K = K
            else:
                self.K = K.copy()
        elif G is not None:
            if inplace:
                self.G = G
            else:
                self.G = G.copy()
        

    def clear_cache(self):
        self.U=None
        self.S=None
        self.UY=None
        self.UUY=None
        self.UX=None
        self.UUX=None           
        self.G=None
        self.K=None


    def innerLoop_2K(self, h2 = 0.5, nGridA2=10, minA2=0.0, maxA2=1.0, i_up=None, i_G1=None, UW=None, UUW=None, **kwargs):
        '''
        For a given h2, finds the optimal a2 and returns the negative log-likelihood
        --------------------------------------------------------------------------
        Input:
        a2      : mixture weight between K0 and K1
        nGridA2 : number of a2-grid points to evaluate the negative log-likelihood at
        minA2   : minimum value for a2 optmization
        maxA2   : maximum value for a2 optmization
        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal a2
        --------------------------------------------------------------------------
        '''

        #TODO: ckw: is this method needed? seems like a wrapper around findA2_2K!
        if self.Y.shape[1]>1:
            print "not implemented"
            raise NotImplementedError("only single pheno case implemented")

        #if self.K0 is not None:
        #    self.setK(K0 = self.K0, K1 = self.K1, a2 = a2)
        #else:
        #    self.setG(G0 = self.G0, G1 = self.G1, a2 = a2)
        #self.setX(self.X)
        #self.sety(self.y)
        return self.findA2_2K(nGridA2=nGridA2, minA2=minA2, maxA2=maxA2, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, h2=h2, **kwargs)


    def findA2_2K(self, nGridA2=10, minA2=0.0, maxA2=1.0, verbose=False, i_up=None, i_G1=None, UW=None, UUW=None, h2=0.5, **kwargs):
        '''
        Find the optimal a2 and h2, such that K=(1.0-a2)*K0+a2*K1. Performs a double loop optimization (could be expensive for large grid-sizes)
        (default maxA2 value is set to 1 as loss of positive definiteness of the final model covariance only depends on h2, not a2)
        --------------------------------------------------------------------------
        Input:
        nGridA2 : number of a2-grid points to evaluate the negative log-likelihood at
        minA2   : minimum value for a2 optmization
        maxA2   : maximum value for a2 optmization
        
        #TODO: complete doc-string!

        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal a2
        --------------------------------------------------------------------------
        '''
        if self.Y.shape[1]>1:
            print "not implemented"
            raise NotImplementedError("only single pheno case implemented")
        
        self.numcalls=0
        resmin=[None]
        def f(x,resmin=resmin, **kwargs):
            self.numcalls+=1
            t0=time.time()
            #pdb.set_trace()
            h2_1=(1.0-h2)*x
            res = self.nLLeval_2K(h2_1=h2_1, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, h2=h2, **kwargs)
            
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            t1=time.time()
            #print "one objective function call took %.2f seconds elapsed" % (t1-t0)
            #import pdb; pdb.set_trace()
            return res['nLL']
        if verbose: print "finda2"
        min = minimize1D(f=f, nGrid=nGridA2, minval=minA2, maxval=maxA2,verbose=False)
        #print "numcalls to innerLoopTwoKernel= " + str(self.numcalls)
        return resmin[0]

    def findH2_2K(self, nGridH2=10, minH2=0.0, maxH2=0.99999, nGridA2=10, minA2=0.0, maxA2=1.0, i_up=None, i_G1=None, UW=None, UUW=None, **kwargs):
        '''
        Find the optimal h2 and a2 for a given K. Note that this is the single kernel case. So there is no a2.
        (default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
        --------------------------------------------------------------------------
        Input:
        nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at
        minH2   : minimum value for h2 optmization
        maxH2   : maximum value for h2 optmization
        nGridA2 : number of a2-grid points to evaluate the negative log-likelihood at
        minA2   : minimum value for a2 optmization
        maxA2   : maximum value for a2 optmization

        #TODO: complete doc-string

        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal h2 and a2
        --------------------------------------------------------------------------
        '''
        #f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
        if self.Y.shape[1]>1:
            print "not implemented"
            raise NotImplementedError("only single pheno case implemented")
        resmin=[None]
        noG1=True
        if i_G1.any():
            noG1=False
        def f(x,resmin=resmin,**kwargs):
            if noG1:
                res = self.nLLeval_2K(h2_1=0.0, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, h2=x, **kwargs)
            else:
                res = self.innerLoop_2K(h2=x, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, nGridA2=nGridA2, minA2=minA2, maxA2=maxA2, **kwargs)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            return res['nLL']
        min = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2 )
        return resmin[0]

    def find_log_delta(self, sid_count, min_log_delta=-5, max_log_delta=10, nGrid=10, **kwargs):
        '''
        #Need comments
        '''
        #f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
        resmin=[None]
        def f(x,resmin=resmin,**kwargs):
            h2 = 1.0/(np.exp(x)*sid_count+1) #We convert from external log_delta to h2 and then back again so that this code is most similar to findH2

            res = self.nLLeval(h2=h2,**kwargs)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            #logging.info("search\t{0}\t{1}".format(x,res['nLL']))
            return res['nLL']
        min = minimize1D(f=f, nGrid=nGrid, minval=min_log_delta, maxval=max_log_delta )
        res = resmin[0]
        internal_delta = 1.0/res['h2']-1.0
        ln_external_delta = np.log(internal_delta / sid_count)
        res['log_delta'] = ln_external_delta
        return res

    def findH2(self, nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False, **kwargs):
        '''
        Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no a2.
        (default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
        --------------------------------------------------------------------------
        Input:
        nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at
        minH2   : minimum value for h2 optmization
        maxH2   : maximum value for h2 optmization

        #TODO: complete doc-string
        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal h2
        --------------------------------------------------------------------------
        '''
        #f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
        resmin=[None]
        if estimate_Bayes or self.Y.shape[1]>1:
            def f(x):
                res = self.nLLeval(h2=x,**kwargs)
                return res['nLL']
            (evalgrid,resultgrid)=evalgrid1D(f, evalgrid = None, nGrid=nGridH2, minval=minH2, maxval = maxH2, dimF=self.Y.shape[1])
            lik=np.exp(-resultgrid)
            evalgrid = lik*evalgrid[:,np.newaxis]

            posterior_mean = evalgrid.sum(0)/lik.sum(0)
            return posterior_mean
        else: 

            def f(x,resmin=resmin):
                res = self.nLLeval(h2=x,**kwargs)
                if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                    resmin[0]=res
                return res['nLL'][0]   
            min = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2 )
            return resmin[0]

    def nLLeval_2K(self, h2=0.0, h2_1=0.0, dof=None, scale=1.0, penalty=0.0, snps=None, UW=None, UUW=None, i_up=None, i_G1=None, subset=False):
        '''
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where ((1-a2)*K0 + a2*K1) = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        REML    : boolean
                  if True   : compute REML
                  if False  : compute ML
        dof     : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        logdelta: log(delta) allows to optionally parameterize in delta space
        delta   : delta     allows tomoptionally parameterize in delta space
        scale   : Scale parameter the multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'dof'       : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        '''

        N=self.Y.shape[0]-self.linreg.D
        
        P=self.Y.shape[1]
        S,U = self.getSU()
        k=S.shape[0]
        if (h2<0.0) or (h2+h2_1>=0.99999) or (h2_1<0.0):
            return {'nLL':3E20,
                    'h2':h2,
                    'h2_1':h2_1,
                    'scale':scale}
        Sd = (h2*self.S + (1.0-h2-h2_1))*scale#?(1.0-h2-h2_1)?
        #Sd = (h2*self.S + (1.0-h2))*scale#?(1.0-h2-h2_1)?
        denom = (1.0-h2-h2_1)*scale      # determine normalization factor
        if subset: #if G1 is a subset of G, then we don't need to 
            h2_1=h2_1-h2

        #UY,UUY = self.getUY()
        #YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
        #logdetK = np.log(Sd).sum()
        #
        #if (UUY is not None):#low rank part
        #    logdetK+=(N-k) * np.log(denom)
        
        if UW is not None:
            weightW=np.zeros(UW.shape[1])
            weightW[i_up] = -h2
            weightW[i_G1] = h2_1
        else:
            weightW=None

        Usnps,UUsnps = None,None
        if snps is not None:
            
            if snps.shape[0] != self.Y.shape[0]:
                #pdb.set_trace()
                print "shape mismatch between snps and Y"
            Usnps,UUsnps = self.rotate(A=snps)
                
        result = self.nLLcore(Sd=Sd, dof=dof, scale=scale, penalty=penalty, UW=UW, UUW=UUW, weightW=weightW, denom=denom, Usnps=Usnps, UUsnps=UUsnps)
        result['h2']=h2
        result['h2_1']=h2_1
        return result


    def nLLeval(self, h2=0.0, logdelta = None, delta = None, dof = None, scale = 1.0, penalty=0.0, snps=None, Usnps=None, UUsnps=None, UW=None, UUW=None, weightW=None):
        '''
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where ((1-a2)*K0 + a2*K1) = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        REML    : boolean
                  if True   : compute REML
                  if False  : compute ML
        dof     : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        logdelta: log(delta) allows to optionally parameterize in delta space
        delta   : delta     allows tomoptionally parameterize in delta space
        scale   : Scale parameter the multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'dof'       : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        '''

        N=self.Y.shape[0]-self.linreg.D
        
        P=self.Y.shape[1]
        S,U = self.getSU()
        k=S.shape[0]

        if logdelta is not None:
            delta = np.exp(logdelta)

        if delta is not None:
            Sd = (self.S+delta)*scale
            denom = delta*scale         # determine normalization factor
            h2=1.0/(1.0+delta)
        else:
            Sd = (h2*self.S + (1.0-h2))*scale
            denom = (1.0-h2)*scale      # determine normalization factor
        if (h2<0.0) or (h2>=1.0):
            return {'nLL':3E20,
                    'h2':h2,
                    'scale':scale}
        UY,UUY = self.getUY()
        YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
        logdetK = np.log(Sd).sum()

        if (UUY is not None):#low rank part
            logdetK+=(N-k) * np.log(denom)
        
        if (snps is not None) and (Usnps is None):
            if snps.shape[0] != self.Y.shape[0]:
                pdb.set_trace()
                print "shape missmatch between snps and Y"
            Usnps,UUsnps = self.rotate(A=snps)
        
        if weightW is not None:
            #multiply the weight by h2
            weightW=weightW*h2#TODO: remove
        
        result = self.nLLcore(Sd=Sd, dof=dof, scale=scale, penalty=penalty, UW=UW, UUW=UUW, weightW=weightW, denom=denom, Usnps=Usnps, UUsnps=UUsnps)
        result['h2']=h2
        return result

    def nLLcore(self, Sd=None, dof=None, scale=1.0, penalty=0.0, UW=None, UUW=None, weightW=None, denom=1.0, Usnps=None, UUsnps=None):
        '''
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where ((1-a2)*K0 + a2*K1) = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        REML    : boolean
                  if True   : compute REML
                  if False  : compute ML
        dof     : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        logdelta: log(delta) allows to optionally parameterize in delta space
        delta   : delta     allows tomoptionally parameterize in delta space
        scale   : Scale parameter the multiplies the Covariance matrix (default 1.0)

        #TODO: complete doc-string (clean up!)

        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'dof'       : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        '''

        N=self.Y.shape[0]-self.linreg.D
        
        P=self.Y.shape[1]
        S,U = self.getSU()
        k=S.shape[0]

        UY,UUY = self.getUY()
        YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
        logdetK = np.log(Sd).sum()

        if (UUY is not None):#low rank part
            logdetK+=(N-k) * np.log(denom)
        
        if Usnps is not None:
            
            snpsKsnps = computeAKA(Sd=Sd, denom=denom, UA=Usnps, UUA=UUsnps)[:,np.newaxis]
            snpsKY = computeAKB(Sd=Sd, denom=denom, UA=Usnps, UB=UY, UUA=UUsnps, UUB=UUY)
        
        if weightW is not None:
            absw=np.absolute(weightW)
            weightW_nonz=absw>1e-10
        if (UW is not None and weightW_nonz.any()):#low rank updates
            #pdb.set_trace()
            multsign=False
            absw=np.sqrt(absw)
            signw=np.sign(weightW)
            #make sure that the identity works and if needed remove any W with zero weight:
            if (~weightW_nonz).any():
                weightW=weightW[weightW_nonz]
                absw=absw[weightW_nonz]
                signw=signw[weightW_nonz]
                UW=UW[:,weightW_nonz]
                if UUW is not None:
                    UUW=UUW[:,weightW_nonz]
            UW = UW * absw[np.newaxis,:]
            if multsign:
                UW_ = UW * signw[np.newaxis,:]
            if UUW is not None:
                UUW = UUW * absw[np.newaxis,:]
            if multsign:
                UUW_ = UUW * signw[np.newaxis,:]
            num_exclude = UW.shape[1]
            

            #WW = np.diag(1.0/weightW) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UW, UUB=UUW)
            if multsign:
                WW = np.eye(num_exclude) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UW_, UUB=UUW_)
            else:
                WW = np.diag(signw) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UW, UUB=UUW)
            
            # compute inverse efficiently
            [S_WW,U_WW] = la.eigh(WW)
             # compute S_WW^{-1} * UWX
                        
            WY = computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UY, UUB=UUY)
            UWY = U_WW.T.dot(WY)
            WY = UWY / np.lib.stride_tricks.as_strided(S_WW, (S_WW.size,UWY.shape[1]), (S_WW.itemsize,0))
            # compute S_WW^{-1} * UWy

            # perform updates (instantiations for a and b in Equation (1.5) of Supplement)
            YKY -= (UWY*WY).sum(0)
            
            if Usnps is not None:
                Wsnps = computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=Usnps, UUB=UUsnps)
                UWsnps = U_WW.T.dot(Wsnps)
                Wsnps = UWsnps / np.lib.stride_tricks.as_strided(S_WW, (S_WW.size,UWsnps.shape[1]), (S_WW.itemsize,0))

                snpsKY -= UWsnps.T.dot(WY)
                # perform updates (instantiations for a and b in Equation (1.5) of Supplement)
                snpsKsnps -= (UWsnps * Wsnps).sum(0)[:,np.newaxis]
            
            # determinant update
            prod_diags=signw*S_WW
            if np.mod((prod_diags<0).sum(),2):
                raise FloatingPointError("nan log determinant")
            logdetK += np.log(np.absolute(S_WW)).sum()
            
            ########

        if Usnps is not None:            
            if penalty:
                beta = snpsKY / (snpsKsnps+penalty)
                r2 = -(snpsKY*beta-YKY[np.newaxis,:])
                variance_beta = r2/(N-1) * snpsKsnps / ((snpsKsnps+penalty)*(snpsKsnps+penalty))#note that we assume the loss in DOF is 1 here, even though it is less, so the variance estimate is coservative
            else:
                beta = snpsKY / snpsKsnps
                r2 = -(snpsKY*beta-YKY[np.newaxis,:])
                variance_beta = r2/(N-1) / snpsKsnps
        else:
            r2 = YKY
            beta=None
            variance_beta=None

        if dof is None:#Use the Multivariate Gaussian
            sigma2 = r2 / N
            nLL =  0.5 * ( logdetK + N * ( np.log(2.0*np.pi*sigma2) + 1 ) )
        else:#Use multivariate student-t
            nLL =   0.5 * ( logdetK + (dof + N) * np.log(1.0+r2/dof) )
            nLL +=  0.5 * N*np.log( dof*np.pi ) + SS.gammaln( 0.5*dof ) - SS.gammaln( 0.5* (dof + N))
        result = {
                'nLL':nLL,
                'dof':dof,
                'beta':beta,
                'variance_beta':variance_beta,
                'scale':scale
                }
        
        if np.isnan(nLL).any():
            raise FloatingPointError("nan likelihood")
        return result


class Linreg(object):
    __slots__ = ["X", "Xdagger", "beta", "N", "D"]

    def __init__(self,X=None, Xdagger=None):
        self.N=0
        self.setX(X=X, Xdagger=Xdagger)
        
    def setX(self, X=None, Xdagger=None):
        self.beta = None
        self.Xdagger = Xdagger
        self.X = X
        if X is not None:
            self.D=X.shape[1]
        else:
            self.D=1

    def set_beta(self,Y):
        self.N=Y.shape[0]
        if Y.ndim == 1:
            P=1
        else:
            P = Y.shape[1]    
        if self.X is None:
            self.beta=Y.mean(0)
        else:        
            if self.Xdagger is None:
                self.Xdagger = la.pinv(self.X)       #SVD-based, and seems fast
            self.beta = self.Xdagger.dot(Y)

    def regress(self, Y):
        self.set_beta(Y=Y)
        if self.X is None:
            RxY=Y-self.beta
        else:
            RxY = Y-self.X.dot(self.beta)
        return RxY

    def predict(self,Xstar):
        return Xstar.dot(self.beta)

def computeAKB(Sd, denom, UA, UB, UUA=None, UUB=None):
    UAS = UA / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UA.shape[1]), (Sd.itemsize,0))
    AKB = UAS.T.dot(UB)
    if UUA is not None:
        AKB += UUA.T.dot(UUB)/denom
    return AKB

def computeAKA(Sd, denom, UA, UUA=None):
    UAS = UA / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UA.shape[1]), (Sd.itemsize,0))
    AKA = (UAS*UA).sum(0)
    if UUA is not None:
        AKA += (UUA*UUA).sum(0)/denom
    return AKA

if 0:
    import scipy as sp
    import scipy.linalg as la
    N=7
    D=2
    X = sp.randn(N,D)

    X_K= sp.randn(N,N)
    K=sp.dot(X_K,X_K.T)+sp.eye(N)

    Kinv = la.inv(K)

    linreg=linreg(X=X)
    Kinv_=linreg.regress(Kinv)
    Kinv_=linreg.regress(Kinv_.T)
    P_=Kinv_#this one does not match with P

    X_K_=linreg.regress(X_K)
    S_x = linreg.regress(sp.eye(N))
    S_x = linreg.regress(S_x.T)
    K_=X_K_.dot(X_K_.T)+S_x
    [u,s,v]=la.svd(X_K_)
    inonz=s>1e-10
    s=s[inonz]*s[inonz]+1

    u=u[:,inonz]
    #s+=1
    P__=u.dot(sp.diag(1.0/s)).dot(u.T)#matches with P
    
    P___=la.pinv(K_)#matches with P
    
    KX=Kinv.dot(X)
    XKX=X.T.dot(KX)
    P=Kinv - KX.dot(la.inv(XKX)).dot(KX.T)#matches with P



if __name__ == "__main__":
    from fastlmm.association.gwas import *
    #from fastlmm.pyplink.snpreader.Bed import Bed
    #import time

    delta = 1.0
    num_pcs = 100
    mixing = 0.5

    #bed_fn = "G:\Genetics/dbgap/ARIC/autosomes.genic"
    #pheno_fn = "G:\Genetics/dbgap/ARIC/all-ldlsiu02.phe"

    #bed_fn = "../data/autosomes.genic"
    #pheno_fn = "../all-ldlsiu02.phe"

    bed_fn = "../feature_selection/examples/toydata"
    pheno_fn = "../feature_selection/examples/toydata.phe"

    selected_snp_pos_fn = "../feature_selection/examples/test_snps.txt"
    selected_snp_pos = np.loadtxt(selected_snp_pos_fn,comments=None)


    snp_reader = Bed(bed_fn)
    snp_reader.run_once()

    G, y, rs = load_intersect(snp_reader, pheno_fn)

    # get chr names/id
    chr_ids = snp_reader.pos[:,0]
    snp_pos = snp_reader.pos[:,2]

    #snp_name = geno['rs']

    #loco = LeaveOneChromosomeOut(chr_ids, indices=True)
    loco = [[range(0,5000), range(5000,10000)]]

    if 0:
        #TODO: wrap up results using pandas
        for train_snp_idx, test_snp_idx in loco:

            print len(train_snp_idx), len(test_snp_idx)

        
            int_snp_idx = argintersect_left(snp_pos[train_snp_idx], selected_snp_pos)
            sim_keeper_idx = np.array(train_snp_idx)[int_snp_idx]

            print sim_keeper_idx

            G_train = G[:,train_snp_idx]
            G_sim = G[:,sim_keeper_idx]
            G_test = G[:,test_snp_idx]
        
            import pdb
            pdb.set_trace()

            logging.info("computing pca...")
        
            t0 = time.time()
        
            pca = PCA(n_components = num_pcs)
            pcs = pca.fit_transform(G_train)

            t1 = time.time()

            logging.info("done after %.4f seconds" % (t1 - t0))
        
            gwas = Gwas(G_sim, G_test, y, delta, train_pcs=pcs, mixing_weight=mixing)
            gwas.run_gwas()

    if 1:

        i_min = np.array([[ 576],
            [2750],
            [4684],
            [7487],
            [3999],
            [4742],
            [ 564],
            [9930],
            [6252],
            [5480],
            [8209],
            [3829],
            [ 582],
            [6072],
            [2237],
            [7051],
            [  71],
            [8590],
            [5202],
            [6598]])
        N=G.shape[0]
        S=G.shape[1]
        
        t0=time.time()
        Gup = np.hstack((G[:,i_min[17:18,0]],G[:,18:27])).copy()
        Gdown=G[:,20:25]
        Gback=np.hstack((G[:,0:12],G[:,i_min[10:12,0]],0*Gdown)).copy()
        Gback_=np.hstack((Gup,G[:,0:12],G[:,i_min[10:12,0]])).copy()
        
        Gcovar = G[:,[9374,1344]]
        covariates = np.hstack([Gcovar,np.ones((N,1))]).copy()
        fullr=False
        K=None

        weightW=np.ones(Gup.shape[1]+Gdown.shape[1])*0.0
        weightW[0:Gup.shape[1]]=-1.0
        W = np.hstack((Gup,Gdown)).copy()

        #W = G_snp
        lmm =LMM(X=covariates,Y=y[:,np.newaxis],G=Gback_,K=K,forcefullrank=fullr)
        UGup,UUGup=lmm.rotate(W)
        #UGup,UUGup=None,None
        
        opt = lmm.findH2(nGridH2=10,UW=UGup,UUW=UUGup,weightW=weightW)
        h2=opt['h2']
        
        
        delta = None#=(1.0/h2-1.0)
        #REML=False
        REML=False
        #lmm.set_snps(snps=G)
        i_up=weightW==-1.0
        i_G1=weightW==4

        res2 = lmm.nLLeval_2K(h2=h2, h2_1=(4.0*h2), dof = None, scale = 1.0, penalty=0.0, snps=G, UW=UGup, UUW=UUGup, i_up=i_up, i_G1=i_G1, subset=False)
        res = lmm.nLLeval(h2=h2, logdelta = None, delta = None, dof = None, scale = 1.0, penalty=0.0, snps = G, UW=UGup, UUW=UUGup, weightW=weightW)
        chi2stats = res['beta']*res['beta']/res['variance_beta']
        
        pv = st.chi2.sf(chi2stats,1)
        pv_ = st.f.sf(chi2stats,1,G.shape[0]-3)#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+Snp)
        
        chi2stats2 = res2['beta']*res2['beta']/res2['variance_beta']
        
        pv2 = st.chi2.sf(chi2stats2,1)
        
        opt_2K = lmm.findH2_2K(nGridH2=10, minH2 = 0.0, maxH2 = 0.99999, i_up=i_up, i_G1=i_G1, UW=UGup, UUW=UUGup)
        res_2K_ = lmm.nLLeval_2K(h2=opt['h2'], h2_1=0, dof = None, scale = 1.0, penalty=0.0, snps=G, UW=UGup, UUW=UUGup, i_up=i_up, i_G1=i_G1, subset=False)
        res_2K = lmm.nLLeval_2K(h2=opt_2K['h2'], h2_1=opt_2K['h2_1'], dof = None, scale = 1.0, penalty=0.0, snps=G, UW=UGup, UUW=UUGup, i_up=i_up, i_G1=i_G1, subset=False)
        t1=time.time()
        i_pv = pv[:,0].argsort()

        if 0:
            #lmm.findH2()
            
            gwas = Gwas(Gback, G, y, mixing_weight=mixing, cov=covariates, delta=delta, REML=REML)
            gwas.run_gwas()
            t2=time.time()

            timing1=t1-t0
            timing2=t2-t1
            print "t1 = %.5f   t2 = %.5f"%(timing1,timing2)

            #import pylab as PL
            PL.ion()
            PL.figure()
            PL.plot([0,8],[0,8])
            PL.plot(-np.log10(pv[gwas.p_idx,0]),-np.log10(gwas.p_values),'.g')
            
            PL.plot(-np.log10(pv),-np.log10(pv2),'.r')
