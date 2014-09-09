import scipy as sp
import numpy as np
import scipy.stats as st
import pdb
import warnings
import logging

def thin_results_file(myfile,dup_postfix="v2"):
    '''
    Used in score vs lrt to remove any lines in the results
    ending with "v2", as these were replicate gene set entries.
    '''    
    sets = np.loadtxt(myfile,dtype=str,comments=None)	
    nodup_ind = []
    dup_ind = []

    #indexes of non-duplicates, as indicated by dup_postfix
    for i in range(0,sets.shape[0]):
        tmpset=sets[i,0]
        if tmpset[-2:]!=dup_postfix:
            nodup_ind.append(i)
        else:
            dup_ind.append(i)	    

    sets_nodup = sets[nodup_ind]
    print "%i reps, and %i non-reps" % (len(dup_ind),len(nodup_ind))
    return sets_nodup

def compare_files(file1,file2,tol=1e-8,delimiter="\t"):
    '''
    Given two files, compare the contents, including numbers up to absolute tolerance, tol
    Returns: val,msg 
    where val is True/False (true means files to compare to each other) and a msg for the failure.
    '''
    dat1=sp.loadtxt(file1,dtype='str',delimiter=delimiter,comments=None)
    dat2=sp.loadtxt(file2,dtype='str',delimiter=delimiter,comments=None)

    ncol1=dat1[0].size
    ncol2=dat2[0].size

    if ncol1!=ncol2:         
        return False,"num columns do not match up"

    try:
        head1=dat1[0,:]
        head2=dat2[0,:]
    except:
        #file contains just a single column.
        return sp.all(dat1==dat2), "single column result doesn't match exactly ('{0}')".format(file1)

    #logging.warn("DO headers match up? (file='{0}', '{1}' =?= '{2}')".format(file1, head1,head2))
    if not sp.all(head1==head2):         
        return False, "headers do not match up (file='{0}', '{1}' =?= '{2}')".format(file1, head1,head2)
        
    for c in range(ncol1):
        checked=False
        col1=dat1[1:,c]
        col2=dat2[1:,c]        
        try:
            #if it is numeric
            col1=sp.array(col1,dtype='float64')
            col2=sp.array(col2,dtype='float64')                    
        except Exception:
            # if it is a string
            pass
            if not sp.all(col1==col2):     
                return False, "string column %s does not match" % head1[c]
            checked=True

        #if it is numeric
        if not checked:
            absdiff=sp.absolute(col1-col2)
            if sp.any(absdiff>tol):
                try:                
                    return False, "numeric column %s does diff of %e not match within tolerance %e" % (head1[c],absdiff,  tol)
                except:
                    return False, "Error trying to print error message while comparing '{0}' and '{1}'".format(file1,file2)
        
    return True, "files are comparable within abs tolerance=%e" % tol
    

#could make this more efficient by reading in blocks of SNPs, as in 
#FastLmmSet.py:KfromAltSnps()
def write_kernel(iid,K,fileout):
    '''
    writes out kernel
    assumes that iid contains a list of the ids, or else a list of [famid, personid] which
    then get merged with a space in between
    '''
    nInd = K.shape[0]
    header = 'var'
    iid_merged = []

    # first line contains iids
    for i in range(nInd):
        if iid.ndim==1 or iid.shape[1]==1:
            header += '\t%s'%(iid[i])
            iid_merged.append('%s'%(iid[i]))
        else:
            header += '\t%s %s'%(iid[i,0],iid[i,1])
            iid_merged.append('%s %s'%(iid[i,0],iid[i,1]))

    # each row of the matrix is one line
    f = open(fileout,'w')
    f.write(header+'\n')
    for i in range(nInd):
        row = ['\t%.4f'%x for x in K[i,:]]
        f.write('%s%s\n'%(iid_merged[i],''.join(row)))
    f.close()
    
def write_plink_covariates(iid,X,fileout):
    '''
    writes out plink-style covariates/phen file
    assuming that X is [N,M] for N individuals and M features
    assumes that iid contains a list of [famid, personid] 
    '''
    [nInd,M] = X.shape

    # each row of the matrix is one line
    f = open(fileout,'w')    
    for i in range(nInd):
        row = ['\t%.4f'%x for x in X[i,:]]        
        f.write('%s\t%s%s\n'%(iid[i,0],iid[i,1],''.join(row)))
    f.close()


def combineseeds(seed1,seed2):
    import hashlib
    import sys
    seed=int(hashlib.md5(str(seed1) + "_" + str(seed2)).hexdigest(), 16)    
    seed = int(seed % sys.maxint)
    return seed


def standardize_col(dat,meanonly=False):
    '''
    Mean impute each columns of an array.
    '''           
    colmean=st.nanmean(dat)
    if ~meanonly:
        colstd=st.nanstd(dat)
    else:
        colstd=None
    ncol=dat.shape[1]           
    nmissing=sp.zeros((ncol))    
    datimp=sp.empty_like(dat); datimp[:]=dat
    for c in sp.arange(0,ncol):        
        datimp[sp.isnan(datimp[:,c]),c]=colmean[c] 
        datimp[:,c]=datimp[:,c]-colmean[c]        
        if not meanonly:
            if colstd[c]>1e-6:
                datimp[:,c]=datimp[:,c]/colstd[c]
            else:
                print "warning: colstd=" + colstd[c] + " during normalization"
        nmissing[c]=float(sp.isnan(dat[:,c]).sum())
    fracmissing=nmissing/dat.shape[0]         
    return datimp,fracmissing

def extractcols(filein,colnameset=None,dtypeset=None):
    if colnameset is None: raise Exception("must specify column names to read")
    import pandas as pd          
    data=pd.read_csv(filein,delimiter = '\t',dtype=dtypeset,usecols=colnameset)    
    r={}
    for j in sp.arange(0,len(colnameset)):
        name=colnameset.pop()
        r[name]=(data[name].values)
    return r


def argintersect_left(a, b):
    """
    find indices in a, whose corresponding values are in b
    ----------------------------------------------------------------------
    Input:
    a        : array, for which indices are returned that are in the intersect with b
    b        : array to be intersected with a
    ----------------------------------------------------------------------
    Output:
    the indices of elements of a, which are in intersect of a and b
    ----------------------------------------------------------------------
    """
    return sp.arange(a.shape[0])[sp.in1d(a,b)]


def intersect_ids(idslist,sep="Q_Q"):
    '''
    Takes a list of 2d string arrays of family and individual ids.
    These are intersected.
    "sep" is used to concatenate the family and individual ids into one unique string
    Returns: indarr, an array of size N x L, where N is the number of
             individuals in the intersection, and L is the number of lists in idslist, and which
             contains the index to use (in order) such that all people will be identical and in order
             across all data sets.
    If one of the lists=None, it is ignored (but still has values reported in indarr, all equal to -1),
    but the first list must not be None.
    '''
    #!!!cmk04072014 warnings.warn("This intersect_ids is deprecated. Pysnptools includes newer versions of intersect_ids", DeprecationWarning)
    id2ind={}    
    L=len(idslist)
    observed=sp.zeros(L,dtype='bool')

    for l, id_list in enumerate(idslist):
            if id_list is not None:
                observed[l]=1
                if l==0:
                    if ~observed[l]:
                        raise Exception("first list must be non-empty")
                    else:
                        for i in xrange(id_list.shape[0]):
                            id=id_list[i,0] +sep+ id_list[i,1]
                            entry=sp.zeros(L)*sp.nan #id_list to contain the index for this id, for all lists provided
                            entry[l]=i                 #index for the first one
                            id2ind[id]=entry
                elif observed[l]:
                    for i in xrange(id_list.shape[0]):
                        id=id_list[i,0] +sep+ id_list[i,1]
                        if id2ind.has_key(id):
                            id2ind[id][l]=i

    indarr=sp.array(id2ind.values(),dtype='float')  #need float because may contain NaNs
    indarr[:,~observed]=-1                          #replace all Nan's from empty lists to -1
    inan = sp.isnan(indarr).any(1)                  #find any rows that contain at least one Nan
    indarr=indarr[~inan]                            #keep only rows that are not NaN
    indarr=sp.array(indarr,dtype='int')             #convert to int so can slice 
    return indarr

def indof_constfeatures(X,axis=0):
    '''
    Assumes features are columns (by default, but can do rows), and checks to see if all features are simply constants,
    such that it is equivalent to a bias and nothing else
    '''
    featvar=sp.var(X,axis=axis)
    badind = sp.nonzero(featvar==0)[0]
    return badind

def constfeatures(X,axis=0):
    '''
    Assumes features are columns (by default, but can do rows), and checks to see if all features are simply constants,
    such that it is equivalent to a bias and nothing else
    '''
    featmeans=sp.mean(X,axis=axis)
    return (X-featmeans==0).all()


def appendtofilename(filename,midfix,sep="."):
        import os
        dir, fileext = os.path.split(filename)
        file, extension = os.path.splitext(fileext)
        infofilename = dir + os.path.sep + file + sep + midfix + extension
        return infofilename

def datestamp(appendrandom=False):
    import datetime
    now = datetime.datetime.now()
    s = str(now)[:19].replace(" ","_").replace(":","_")
    if appendrandom:
        import random
        s += "_" + str(random.random())[2:]
    return s
           


#not needed, just use the sp RandomState.permutation
#def permute(numbersamples):
#    perm = sp.random.permutation(numbersamples)
#    return perm

#Not needed because enumerate is built in to the language
#def appendindex(iter):
#    index = -1;
#    for item in iter:
#        index += 1
#        yield item, index

def create_directory_if_necessary(name, isfile=True):    
    import os
    if isfile:
        directory_name = os.path.dirname(name)
    else:
        directory_name = name

    if directory_name != "":
        try:
            os.makedirs(directory_name)
        except OSError, e:
            if not os.path.isdir(directory_name):
                raise Exception("not valid path: '{0}'. (Working directory is '{1}'".format(directory_name,os.getcwd()))

def which(vec):
    '''
    find the True from the index 0 with bool vector vec
    ----------------------------------------------------------------------
    Input:
    vec        : vector of bool
    ----------------------------------------------------------------------
    Output:
    index of the first True from the bool vector vec
    ----------------------------------------------------------------------
    '''
    for i, item in enumerate(vec):
        if (item):
            return(i)
    return(-1)

def which_opposite(vec):
    '''
    find the True from the index 0 with bool vector vec
    ----------------------------------------------------------------------
    Input:
    vec        : vector of bool
    ----------------------------------------------------------------------
    Output:
    index of the last True from the bool vector vec
    ----------------------------------------------------------------------
    '''
    for i in reversed(xrange(len(vec))):
        item = vec[i]
        if (item):
            return(i)
    return(-1)


def generatePermutation(numbersamples,randomSeedOrState):
    from numpy.random import RandomState

    if isinstance(randomSeedOrState,RandomState):
        randomstate = randomSeedOrState
    else:
        randomstate = RandomState(randomSeedOrState)

    perm = randomstate.permutation(numbersamples)
    return perm

def excludeinds(pos0, pos1, mindist = 10.0,idist = 2):
    '''
    get the indices of SNPs that have to be excluded from the set of null SNPs when testing alternative SNPs to correct for proximal contamination.
    --------------------------------------------------------------------------
    Input:
    pos0        : [S0*3] array of null-model SNP positions
    pos1        : [S0*3] array of alternative-model SNP positions
    idist       : index in pos array that the exclusion is based on.
                  (1=genetic distance, 2=basepair distance)
    --------------------------------------------------------------------------
    Output:
    i_exclude   : [S] 1-D boolean array indicating excluson of SNPs
                  (True: exclude, False: do not exclude)
    --------------------------------------------------------------------------
    '''
    chromosomes1 = sp.unique(pos1[:,0])
    i_exclude = sp.zeros(pos0[:,0].shape[0],dtype = 'bool')
    if (mindist>=0.0):
        for ichr in xrange(chromosomes1.shape[0]):
            i_SNPs1_chr=pos1[:,0] == chromosomes1[ichr]
            i_SNPs0_chr=pos0[:,0] == chromosomes1[ichr]
            pos1_ = pos1[i_SNPs1_chr,idist]
            pos0_ = pos0[i_SNPs0_chr,idist]
            distmatrix = pos1_[sp.newaxis,:] - pos0_[:,sp.newaxis]
            i_exclude[i_SNPs0_chr] = (sp.absolute(distmatrix)<=mindist).any(1)
    return i_exclude


def dotDotRange(dotDotString):
    '''
    A method for generating integers.
    For example:

> for i in util.dotDotRange("1..4,100,-1..1"): print i
 1
 2
 3
 4
 100
 -1
 0
 1

    '''
    for intervalString in dotDotString.split(","):
        parts = intervalString.split("..")
        if len(parts) > 2 : raise Exception("Expect at most one '..' between commas. (see {0})".format(intervalString))
        start = int(parts[0])
        if len(parts) == 1:
            yield start
        else:
            lastInclusive = int(parts[1])
            for i in xrange(start,lastInclusive+1):
                yield i
