import numpy as np
import scipy.stats
import npp

def model_pvalue(wts, stim, resp, nboot=1e4, randinds=None):
    """Computes a bootstrap p-value by resampling the [wts] of the model, which
    is [wts] * [stim] ~ [resp].
    """
    origcorr = np.corrcoef(resp, np.dot(stim, wts))[0,1]
    if randinds is None:
        #randinds = np.random.randint(0, len(wts), (len(wts), nboot))
        randinds = make_randinds(len(wts), nboot)
    pwts = wts[randinds]
    pred = np.dot(stim, pwts)
    
    ## Compute correlations using vectorized method and bootstrap p-value
    zpred = (pred-pred.mean(0))/pred.std(0)
    zresp = (resp-resp.mean())/resp.std()
    bootcorrs = np.dot(zpred.T, zresp).ravel()/resp.shape[0]
    #bootcorrs = np.array([np.corrcoef(resp, p.T)[0,1] for p in pred.T])
    bspval = np.mean(bootcorrs>origcorr)
    
    ## Compute parametric p-value based on transformed distribution
    zccs = ztransformccs(bootcorrs)
    zorig = ztransformccs(origcorr)
    ppval = 1-scipy.stats.norm.cdf(zorig, loc=zccs.mean(), scale=zccs.std())
    
    print("Boostrap p-value: %0.3f, parametric p-value: %0.03f"%(bspval, ppval))
    return bspval, ppval

def make_randinds(nwts, nboot, algo="randint", maxval=None):
    if maxval is None:
        maxval = nwts
    
    if algo=="randint":
        return np.random.randint(0, maxval, (nwts, nboot))
    
    elif algo=="bytes":
        N = nwts*nboot*2
        return np.mod(np.frombuffer(np.random.bytes(N), dtype=np.uint16), maxval).reshape((nwts, nboot))

    elif algo=="bytes8":
        N = nwts*nboot
        return np.mod(np.frombuffer(np.random.bytes(N), dtype=np.uint8), maxval).reshape((nwts, nboot))

def ztransformccs(ccs):
    """Transforms the given correlation coefficients to be vaguely Gaussian.
    """
    return ccs/np.sqrt((1-ccs**2))

def exact_correlation_pvalue(corr, N, alt="greater"):
    """Returns the exact p-value for the correlation, [corr] between two vectors of length [N].
    The null hypothesis is that the correlation is zero. The distribution of
    correlation coefficients given that the true correlation is zero and both
    [a] and [b] are gaussian is given at 
    http://en.wikipedia.org/wiki/Pearson_correlation#Exact_distribution_for_Gaussian_data

    Parameters
    ----------
    corr : float
        Correlation value
    N : int
        Length of vectors that were correlated
    alt : string
        The alternative hypothesis, is the correlation 'greater' than zero,
        'less' than zero, or just 'nonzero'.
        
    Returns
    -------
    pval : float
        Probability of sample correlation between [a] and [b] if actual correlation
        is zero.
    """
    f = lambda r,n: (1-r**2)**((n-4.0)/2.0)/scipy.special.beta(0.5, (n-2)/2.0)
    pval = scipy.integrate.quad(lambda r: f(r, N), corr, 1)[0]
    if alt=="greater":
        return pval
    elif alt=="less":
        return 1-pval
    elif alt=="nonzero":
        return min(pval, 1-pval)

def correlation_pvalue(a, b, nboot=1e4, confinterval=0.95, method="pearson"):
    """Computes a bootstrap p-value for the correlation between [a] and [b].
    The alternative hypothesis for this test is that the correlation is zero or less.
    This function randomly resamples the timepoints in the [a] and [b] and computes
    the correlation for each sample.

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    nboot : int, optional
        Number of bootstrap samples to compute, default 1e4
    conflevel : float, optional
        Confidence interval size, default 0.95
    method : string, optional
        Type of correlation to use, can be "pearson" (default) or "robust"
        
    Returns
    -------
    bspval : float
        The fraction of bootstrap samples with correlation less than zero.
    bsconf : (float, float)
        The [confinterval]-percent confidence interval according to the bootstrap.
    ppval : float
        The probability that the correlation is zero or less according to parametric
        computation using Fisher transform.
    pconf : (float, float)
        The parametric [confinterval]-percent confidence interval according to
        parametric computation using Fisher transform.
    bootcorrs : array_like, shape(nboot,)
        The correlation for each bootstrap sample
    """
    ocorr = np.corrcoef(a, b)[0,1]
    conflims = ((1-confinterval)/2, confinterval/2+0.5)
    confinds = list(map(int, (conflims[0]*nboot, conflims[1]*nboot)))

    N = len(a)
    inds = make_randinds(N, nboot, algo="bytes")
    rsa = a[inds] ## resampled a
    rsb = b[inds] ## resampled b

    if method=="pearson":
        za = (rsa-rsa.mean(0))/rsa.std(0)
        zb = (rsb-rsb.mean(0))/rsb.std(0)
        bootcorrs = np.sum(za*zb, 0)/(N-1) ## The correlation between each pair
    elif method=="robust":
        bootcorrs = np.array([robust_correlation(x,y)[0] for (x,y) in zip(rsa.T, rsb.T)])
    else:
        raise ValueError("Unknown method: %s"%method)

    ## Compute the bootstrap p-value
    bspval = np.mean(bootcorrs<0) ## Fraction of correlations smaller than zero
    #bspval = np.mean(bootcorrs>ocorr)
    bsconf = (np.sort(bootcorrs)[confinds[0]], np.sort(bootcorrs)[confinds[1]])

    ## Compute the parametric bootstrap p-value using Fisher transform
    zccs = np.arctanh(bootcorrs)
    ppval = scipy.stats.norm.cdf(0, loc=zccs.mean(), scale=zccs.std())
    pconf = tuple(map(lambda c: np.tanh(scipy.stats.norm.isf(1-c, loc=zccs.mean(), scale=zccs.std())), conflims))

    ## return things!
    return bspval, bsconf, ppval, pconf, bootcorrs

def robust_correlation(a, b, cutoff=2.5):
    """Computes a robust estimate of the correlation between [a] and [b] using the
    least mean squares (LMS) based method defined in: 
    Abdullah, 1990, "On a robust correlation coefficient"

    First, outliers are removed based on the residual of the linear regression of [a]
    on [b] and the [cutoff]. Then the correlation is computed on non-outliers.

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    cutoff : float, default=2.5
        The cutoff for outlier detection.
    
    Returns
    -------
    goodcorr : float
        The correlation between non-outliers in [a] and [b]
    """
    assert a.size == b.size
    rho = np.corrcoef(a, b)[0,1]
    zscore = lambda v: (v-v.mean())/v.std()
    res = b - (a*np.linalg.lstsq(np.atleast_2d(zscore(a)).T, b)[0])
    s = 1.4826*(1+5/(a.size-rho))*np.sqrt(np.median(res**2))
    goodpts = np.abs(res/s)<cutoff
    goodcorr = np.corrcoef(a[goodpts], b[goodpts])[0,1]

    return goodcorr, goodpts

from numpy.lib.stride_tricks import as_strided

def block_bootstrap_correlation(a, b, blocklen, nboots):
    """Computes the block-bootstrap correlation between [a] and [b] with
    block length [blocklen] and [nboots] bootstrap samples. The block
    bootstrap is preferable for autocorrelated time series, as it preserves
    the autocorrelation structure of the data. If the data are generated
    by an AR(1) process with parameter r, the autocorrelation of the
    block bootstrap time series will be (blocklen-1)/blocklen * r. E.g.
    with blocklen=10, the autocorrelation of the bootstrap samples will
    be 90% of that in the real data. This formulation comes from
    Vogel & Shallcross, 'The moving blocks bootstrap versus parametric
    time series models', 1996.

    This function implements uses overlapping blocks.

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    blocklen : int
        The length of block to use.
    nboots : int
        Number of bootstrap samples to compute

    Returns
    -------
    bscorrs : array_like, shape (nboots,)
        Correlation between samples of a and b for each bootstrap.
    """
    N = len(a)
    if N != len(b):
        raise Exception("Inputs a and b must have same length!")
    
    ## Break a and b into overlapping blocks
    ## Using a numpy striding trick, this is very efficient
    block_a = as_strided(a.copy(), shape=(N-blocklen+1, blocklen), strides=(a.itemsize, a.itemsize))
    block_b = as_strided(b.copy(), shape=(N-blocklen+1, blocklen), strides=(b.itemsize, b.itemsize))

    ## Figure out how many blocks should go into each bootstrap sample
    ## If number isn't an integer, let's round up
    nblocks = int(np.ceil(float(N)/blocklen))
    totalblocks = block_a.shape[0]
    
    ## Choose which blocks will appear in bootstrap samples with replacement
    if totalblocks<255:
        ## Can use super-efficient algo for generating single-byte random ints
        bsinds = make_randinds(nblocks, nboots, "bytes8", maxval=totalblocks)
    else:
        ## Generate 2-byte ints..
        bsinds = make_randinds(nblocks, nboots, "bytes", maxval=totalblocks)

    ## Create block bootstrap sampled a and b
    sample_a = np.hstack(block_a[bsinds,:])[:,:N]
    sample_b = np.hstack(block_b[bsinds,:])[:,:N]

    ## Compute correlation for each sample
    bscorrs = mcorr(sample_a.T, sample_b.T)
    
    return bscorrs

def block_permutation_correlation(a, b, blocklen, nperms):
    """Computes the block-permutation correlation between [a] and [b] with
    block length [blocklen] and [nperms] samples. Block
    permutation is preferable for autocorrelated time series, as it preserves
    the autocorrelation structure of the data. If the data are generated
    by an AR(1) process with parameter r, the autocorrelation of the
    block bootstrap time series will be (blocklen-1)/blocklen * r. E.g.
    with blocklen=10, the autocorrelation of the bootstrap samples will
    be 90% of that in the real data. This formulation comes from
    Vogel & Shallcross, 'The moving blocks bootstrap versus parametric
    time series models', 1996.

    This function implements uses overlapping blocks.

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    blocklen : int
        The length of block to use.
    nperms : int
        Number of permutation samples to compute

    Returns
    -------
    permcorrs : array_like, shape (nboots,)
        Correlation between samples of a and b for each permutation.
    """
    N = len(a)
    if N != len(b):
        raise Exception("Inputs a and b must have same length!")
    
    ## Break a and b into overlapping blocks
    ## Using a numpy striding trick, this is very efficient
    block_a = as_strided(a.copy(), shape=(N-blocklen+1, blocklen), strides=(a.itemsize, a.itemsize))
    #block_b = as_strided(b.copy(), shape=(N-blocklen+1, blocklen), strides=(b.itemsize, b.itemsize))

    ## Figure out how many blocks should go into each bootstrap sample
    ## If number isn't an integer, let's round up
    nblocks = int(np.ceil(float(N)/blocklen))
    totalblocks = block_a.shape[0]
    
    ## Choose which blocks will appear in bootstrap samples with replacement
    ## Create shuffled indices for one of the datasets
    shuf_inds = np.vstack([np.random.permutation(totalblocks)[:nblocks] for _ in range(nperms)]).T

    ## Create block bootstrap sampled a and b
    sample_a = np.hstack(block_a[shuf_inds,:])[:,:N]
    #sample_b = np.hstack(block_b[inds,:])[:,:N]

    ## Compute correlation for each sample
    bscorrs = npp.mcorr(sample_a.T, np.atleast_2d(b).T)
    
    return bscorrs


### FROM https://stackoverflow.com/a/39544572

from scipy.fftpack import rfft, irfft

def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave the real and imaginary components. 
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr

### END

def phase_permutation_correlation(a, b, nperms):
    """Computes the phase-randomized correlation between [a] and [b] over
    [nperms] samples. 

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    nperms : int
        Number of permutation samples to compute

    Returns
    -------
    permcorrs : array_like, shape (nboots,)
        Correlation between samples of a and b for each bootstrap.
    """
    N = len(a)
    if N != len(b):
        raise Exception("Inputs a and b must have same length!")
    
    
    sample_a = np.vstack([phaseScrambleTS(a) for _ in range(nperms)])

    ## Compute correlation for each sample
    permcorrs = npp.mcorr(sample_a.T, np.atleast_2d(b).T)
    
    return permcorrs