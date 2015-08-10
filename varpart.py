import numpy as np
from matplotlib.pyplot import *
import npp

from itertools import combinations, chain
import ridge
import scipy.optimize

def make_data(N_R, N_P, P_parts, M, true_variances, noise_variance, combs, Pnoise_models,
              P_models, use_mixing=True, orthogonalize=True, noise_scale=1.0, **etc):
    # Generate timecourses for each partition
    X_parts = [np.random.randn(p, N_R + N_P) for p in P_parts]
    print "X_parts[0].shape", X_parts[0].shape
    XX = np.corrcoef(np.vstack(X_parts))

    # Orthogonalize timecourses across and within partitions?
    if orthogonalize:
        cat_orthog_X_parts, _, _ = np.linalg.svd(np.vstack(X_parts).T, full_matrices=False)
        X_parts = np.vsplit(npp.zs(cat_orthog_X_parts).T, np.cumsum(P_parts)[:-1])
        XX_orthog = np.corrcoef(np.vstack(X_parts))

    # Generate "true" weights used to construct Y
    Btrue_parts = [np.random.randn(p, M) for p in P_parts]
    print "Btrue_parts[0].shape", Btrue_parts[0].shape

    # Generate output timecourses for each partition
    Y_parts = [B.T.dot(X).T for X,B in zip(X_parts, Btrue_parts)]
    print "Y_parts[0].shape", Y_parts[0].shape

    # Rescale timecourses for each partition to have appropriate variance
    scaled_Y_parts = [Y / Y.std(0) * np.sqrt(tv) for Y,tv in zip(Y_parts, true_variances)]
    print "scaled_Y_parts[0].shape", scaled_Y_parts[0].shape

    # Generate noise timecourses scaled to have appropriate variance
    Y_noise = np.random.randn(N_R + N_P, M)
    scaled_Y_noise = Y_noise / Y_noise.std(0) * np.sqrt(noise_variance)
    print "scaled_Y_noise.shape", scaled_Y_noise.shape

    # Construct Y from combination of partition timecourses
    Y_total = np.array(scaled_Y_parts).sum(0) + scaled_Y_noise
    zY_total = npp.zs(Y_total)
    print "Y_total.shape", Y_total.shape

    # Generate feature timecourses
    # Stack together partition features to make "true" features for each feature space
    Xtrue_feats = [np.vstack([X_parts[c] for c in comb]) for comb in combs]
    print "Xtrue_feats[0].shape", Xtrue_feats[0].shape

    # Generate noise features to round out each feature space
    Xnoise_feats = [noise_scale * np.random.randn(Pnoise, N_R + N_P) for Pnoise in Pnoise_models]
    print "Xnoise_feats[0].shape", Xnoise_feats[0].shape

    # Generate matrices to mix real and noise features in each space
    mixing_mats = [np.random.randn(P, P) for P in P_models]
    print "mixing_mats[0].shape", mixing_mats[0].shape

    # Use mixing matrices to generate feature timecourses
    if use_mixing:
        X_feats = [m.dot(np.vstack([Xt, Xn])) for m,Xt,Xn in zip(mixing_mats, Xtrue_feats, Xnoise_feats)]
    else:
        X_feats = [np.vstack([Xt, Xn]) for m,Xt,Xn in zip(mixing_mats, Xtrue_feats, Xnoise_feats)]
    print "X_feats[0].shape", X_feats[0].shape

    # Bulk up skinny feats with extra shit to mash bias
    Pmax = max(P_models)
    bulked_X_feats = [np.vstack([X, np.random.randn(Pmax - P, N_R + N_P)]) for X,P in zip(X_feats, P_models)]
    print "bulked_X_feats[0].shape", bulked_X_feats[0].shape

    return locals()

rsq_corr = lambda c: (c ** 2) * np.sign(c) # r -> r^2

def compare(args1, args2, name1, name2, data_params, data, **kwargs):
    allargs1, allargs2 = dict(kwargs), dict(kwargs)

    allargs1.update(data)
    allargs2.update(data)

    allargs1.update(args1)
    allargs2.update(args2)

    corr1 = fit_models(data_params=data_params, **allargs1)
    corr2 = fit_models(data_params=data_params, **allargs2)

    c1_bias, c1_orig_parts, c1_fix_parts = correct_rsqs(corr1, **allargs1)
    c2_bias, c2_orig_parts, c2_fix_parts = correct_rsqs(corr2, **allargs2)

    plot_part_comparison(c1_orig_parts, c1_fix_parts, c2_orig_parts, c2_fix_parts,
                         name1 + " orig", name1 + " fixed", name2 + " orig", name2 + " fixed",
                         errtype=kwargs.get("errtype", "perc"), **data_params)

    print_stats(c1_orig_parts, c1_fix_parts, c2_orig_parts, c2_fix_parts,
                name1, name2, **data_params)


def fit_models(X_feats, zY_total, bulked_X_feats, data_params,
               use_ols=False,
               use_features="raw",
               metric="corr",
               ridge_optimize_corr=True,
               alphas=np.logspace(-3, 3, 10),
               verbose=True,
               nboots=5,
               **etc):
    feature_combs = list(chain(*[combinations(range(3), n) for n in [1, 2, 3]])) # feature spaces to use in each model

    B_est = [] # estimated weights (not used for anything currently)
    corr_est = [] # estimated r^2 from correlation
    rsq_est = [] # estimated R^2 from sum of squared error

    N_R = data_params['N_R']
    N_P = data_params['N_P']
    true_variances = data_params['true_variances']
    combs = data_params['combs']

    if not use_ols and verbose: figure()
    Psum = sum(data_params['P_models'])

    for comb in feature_combs:
        if verbose: print "\nFitting model %s" % ", ".join([['A','B','C'][c] for c in comb])
        thisP = np.array(data_params['P_models'])[list(comb)].sum()
        
        if use_features == "raw":
            Xcomb = npp.zs(np.vstack([X_feats[c] for c in comb]).T).T
        elif use_features == "bulked":
            Xcomb = npp.zs(np.vstack([bulked_X_feats[c] for c in comb]).T).T # <- bulked gives best results!! ??!!?!
        elif use_features == "same":
            Xcomb = npp.zs(np.vstack([X_feats[c] for c in comb] + [np.random.randn(Psum - thisP, N_R + N_P)]).T).T
        else:
            raise ValueError(use_features)
        
        if verbose: print Xcomb.shape
        
        if use_ols:
            wts, res, ranks, sings = np.linalg.lstsq(Xcomb.T[:N_R], zY_total[:N_R])
        else:
            wts, vcorrs, valphas, bscorrs, valinds = ridge.bootstrap_ridge(Xcomb.T[:N_R], zY_total[:N_R],
                                                                           Xcomb.T[N_R:], zY_total[N_R:],
                                                                           alphas=alphas,
                                                                           nboots=nboots,
                                                                           chunklen=1,
                                                                           nchunks=int(N_R * 0.2),
                                                                           use_corr=ridge_optimize_corr,
                                                                           single_alpha=True)
        if not use_ols and verbose: semilogx(alphas, npp.zs(bscorrs.mean(2).mean(1)))
        
        B_est.append(np.vstack(wts).T)
        preds = np.dot(Xcomb.T[N_R:], wts)
        corrs = [np.corrcoef(pred, Y[N_R:])[0,1] for pred,Y in zip(preds.T, zY_total.T)]
        rsqs = [1 - (Y[N_R:] - pred).var() / Y[N_R:].var() for pred,Y in zip(preds.T, zY_total.T)]
        corr_est.append(corrs)
        rsq_est.append(rsqs)
        
        theoretical_rsq = true_variances[list(set.union(*[set(combs[c]) for c in comb]))].sum()
        avg_corr_rsq = rsq_corr(np.array(corrs)).mean()
        avg_rsq = np.array(rsqs).mean()
        if verbose: print "Theor. rsq: %0.3f, corr-based: %0.3f, rsq: %0.3f" % (theoretical_rsq, avg_corr_rsq, avg_rsq)

    if not use_ols and verbose: xlabel("Alpha"); title("Ridge Regularization Path");

    if metric == "corr":
        return rsq_corr(np.array(corr_est))
    elif metric == "rsq":
        return np.array(rsq_est)
    else:
        raise ValueError(metric)


# Correcting variance partitions to nearest legal value
##  B . x = partition areas, where x is vector of model R^2 for models (A, B, C, AB, AC, BC, ABC)
B = np.array([[0, 0, 0, 0, 0, -1, 1], # Abc
              [0, 0, 0, 0, -1, 0, 1], # aBc
              [0, 0, 0, -1, 0, 0, 1], # abC
              [0, 0, -1, 0, 1, 1, -1], # ABc
              [0, -1, 0, 1, 0, 1, -1], # AbC
              [-1, 0, 0, 1, 1, 0, -1], # aBC
              [1, 1, 1, -1, -1, -1, 1], # ABC
             ])

def correct_rsqs(b, neg_only=False, minimize="l2", verbose=True, **etc):
    maxs = B.dot(np.nan_to_num(b))
    if minimize == "l2":
        minfun = lambda x: (x ** 2).sum()
    elif minimize == "l1":
        minfun = lambda x: np.abs(x).sum()
    else:
        raise ValueError(minimize)

    biases = np.zeros((maxs.shape[1], 7)) + np.nan
    for vi in range(b.shape[1]):
        if not (vi % 1000) and verbose:
            print "%d / %d" % (vi, b.shape[1])
        
        if neg_only:
            bnds = [(None, 0)] * 7
        else:
            bnds = [(None, None)] * 7
        res = scipy.optimize.fmin_slsqp(minfun, np.zeros(7),
                                        f_ieqcons=lambda x: maxs[:,vi] - B.dot(x),
                                        bounds=bnds, iprint=0)
        biases[vi] = res
    
    # compute fixed (legal) variance explained values for each model
    fixed_b = np.array(b) - np.array(biases).T

    orig_parts = B.dot(b)
    fixed_parts = B.dot(fixed_b)
    
    return biases, orig_parts, fixed_parts

def plot_part_comparison(est11, est12, est21, est22, name11, name12, name21, name22,
                         true_variances, part_names, errtype="perc", **etc):
    # Plot actual vs. estimated variance in each partition
    figure(figsize=(10,6))
    bar(range(7), true_variances, align='center', edgecolor='none', facecolor="0.7", label="actual")

    #errtype = "perc" # "std": standard deviation, "perc": 10th-90th percentiles, "sem": standard error of the mean
    if errtype == "std":
        errfun = lambda x: x.std(1)
    elif errtype == "perc":
        errfun = lambda x: np.abs(np.percentile(x, [25, 75], axis=1) - x.mean(1))
    elif errtype == "sem":
        errfun = lambda x: x.std(1) / np.sqrt(x.shape[1])

    eargs = dict(capsize=0, mec='none')
    x = np.arange(7)
    errorbar(x - 0.2, est11.mean(1), yerr=errfun(est11), fmt='bo', label=name11, **eargs)
    errorbar(x - 0.1, est12.mean(1), yerr=errfun(est12), fmt='ro', label=name12, **eargs)

    errorbar(x + 0.1, est21.mean(1), yerr=errfun(est21), fmt='bs', label=name21, **eargs)
    errorbar(x + 0.2, est22.mean(1), yerr=errfun(est22), fmt='rs', label=name22, **eargs)

    xticks(range(7), part_names);
    grid(axis='y'); ylabel('Variance in partition'); legend(); xlabel("Partition");

def plot_biases():
    # Plot actual vs. estimated biases for each partition
    theoretical_rsqs = [true_variances[list(set.union(*[set(combs[c]) for c in comb]))].sum() for comb in feature_combs]
    actual_corr_biases = rsq_corr(np.array(corr_est)).mean(1) - theoretical_rsqs
    actual_rsq_biases = np.array(rsq_est).mean(1) - theoretical_rsqs

    bar(x, actual_corr_biases, align='center', edgecolor='none', facecolor="0.7", label="corr actual")
    #bar(x, actual_rsq_biases, align='center', edgecolor='none', facecolor="0.4", label="rsq actual", width=0.4)
    errorbar(x - 0.1, corr_biases.mean(0), yerr=errfun(corr_biases.T), fmt='bo', label="corr est", **eargs)
    #errorbar(x + 0.1, rsq_biases.mean(0), yerr=errfun(rsq_biases.T), fmt='ro', label="rsq est", **eargs)
    xticks(range(7), part_names);
    grid(axis='y');
    xlim(-1, 7); legend(loc="lower left");
    ylabel("Bias in model variance estimate");
    xlabel("Model");

def print_stats(est11, est12, est21, est22, name1, name2, true_variances, **etc):
    # Compute and compare error, variance and bias of each estimate

    # Mean squared error (MSE): how wrong, in total, is the estimate?
    est11_errs = ((est11.T - true_variances) ** 2).mean()
    est12_errs = ((est12.T - true_variances) ** 2).mean()

    est21_errs = ((est21.T - true_variances) ** 2).mean()
    est22_errs = ((est22.T - true_variances) ** 2).mean()

    print "%s MSE: Orig: %f, Fixed: %f, Ratio: fixed %0.3fx better" % (name1, est11_errs, est12_errs, est11_errs / est12_errs)
    print "%s MSE:  Orig: %f, Fixed: %f, Ratio: fixed %0.3fx better\n" % (name2, est21_errs, est22_errs, est21_errs / est22_errs)
    print "%s vs. %s fixed MSE: %s %0.3fx better\n" % (name1, name2, name1, est22_errs / est12_errs)

    # Variance: how variable is the estimate across voxels?
    est11_var = est11.var(1).mean()
    est12_var = est12.var(1).mean()

    est21_var = est21.var(1).mean()
    est22_var = est22.var(1).mean()

    print "%s Variance: Orig: %f, Fixed: %f, Ratio: fixed %0.3fx better" % (name1, est11_var, est12_var, est11_var / est12_var)
    print "%s Variance: Orig: %f, Fixed: %f, Ratio: fixed %0.3fx better\n" % (name2, est21_var, est22_var, est21_var / est22_var)
    print "%s vs. %s fixed variance: %s %0.3fx better\n" % (name1, name2, name1, est22_var / est12_var)

    # Bias: how biased is the estimate? (i.e. how far is mean estimate from true value)
    est11_bias = est11.mean(1) - true_variances
    est12_bias = est12.mean(1) - true_variances
    corr_bias_ratio = np.abs(est11_bias).sum() / np.abs(est12_bias).sum()

    est21_bias = est21.mean(1) - true_variances
    est22_bias = est22.mean(1) - true_variances
    rsq_bias_ratio = np.abs(est21_bias).sum() / np.abs(est22_bias).sum()

    print "%s Bias: \nOrig: %s, \nFixed: %s, \nRatio: fixed %0.3fx better\n" % (name1, est11_bias, est12_bias, corr_bias_ratio)
    print "%s Bias: \nOrig: %s, \nFixed: %s, \nRatio: fixed %0.3fx better\n" % (name2, est21_bias, est22_bias, rsq_bias_ratio)
    print "%s vs. %s fixed bias: %s %0.3fx better" % (name1, name2, name1, np.abs(est22_bias).sum() / np.abs(est12_bias).sum())

