# -*- coding: utf-8 -*-
# opts.py

"""Compute classification accuracy and consistency statistics.
"""
import os
import sys
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import stats
import damon1.core as dmn
import damon1.tools as dmnt
np.seterr(all='ignore')


import matplotlib.pyplot as plt

# Field names for scale score frequencies file
RS = 'RS'
SS = 'SS'
SE = 'SEM'
N = 'N'
GRADE = 'Grade'
DOMAIN = 'Domain'
PL = 'PL'

### Field names for cut score file
##MIN = 'Min'
##MAX = 'Max'

# Domains
RD = 'RD'
WR = 'WR'
LI = 'LI'
SP = 'SP'

SS_COLS = [RS, SS, SE, N]
SS_COLS_EDS = [GRADE, DOMAIN, RS, SS, SE, PL, N]



def load_ss(filename, names=SS_COLS, usecols=SS_COLS, index_col=RS, sep=','):
    """Load scale score file.

    Format
    ------
    RS	SS	SEM	N
    0	220	117	166
    1	245	110	174
    2	255	105	417
    3	262	101	743
    etc.
 
    """
    df = pd.read_csv(filename,
                     sep,
                     header=0,
                     names=names,
                     usecols=usecols,
                     index_col=index_col)
    return df
    

def load_ss_eds(filename, grade, domain, names=SS_COLS_EDS, usecols=SS_COLS_EDS,
                index_col=[GRADE, DOMAIN], sep=','):
    """Load scale score file in EDS format.

    Format
    ------
    Grade   Domain  RS	SS	SEM	PL	N
    0	  L	0	220	117	1	166
    0	  L	1	245	110	1	174
    0	  L	2	255	105	1	417
    0	  L	3	262	101	1	743
    etc.
    
    Gotcha
    ------
    Loading grades as a mix of string and int ('k', 1, 2) creates
    problems.  Either make them all int (0, 1, 2) or all string
    ('k', 'g1', 'g2').
    
    """
    # Load data
    df_ = pd.read_csv(filename,
                      sep,
                      header=0,
                      names=names,
                      usecols=usecols,
                      index_col=index_col)
    df_.sortlevel(inplace=True)

    # Extract desired test and columns
    try:
        df = df_.loc[grade, domain].loc[:, SS_COLS]
    except KeyError:
        exc = ('Sorry, the grade values have to be all integers. Edit your '
               'data file accordingly.')
        raise KeyError(exc)
            
    df.set_index(RS, inplace=True)

    return df



def load_cuts_eds(filename, grade, domain, sep=','):
    """Load scale score file in EDS format.

    Format
    ------
    Grade   Domain  B	EI	I	EA	A	Max
    0  L	220	362	409	455	502	570
    0  S	140	353	405	457	509	630
    0  R	220	232	300	380	468	570
    0  W	220	255	327	383	430	600
    1	 L	220	362	409	455	502	570
    1	 S	140	353	405	457	509	630
    1	 R	220	357	393	468	570	570
    1	 W	220	372	406	444	518	600
    etc.

    Gotcha
    ------
    Loading grades as a mix of string and int ('k', 1, 2) creates
    problems.  Either make them all int (0, 1, 2) or all string
    ('k', 'g1', 'g2').


    """
    # Load data
    df_ = pd.read_csv(filename,
                      sep,
                      header=0,
                      index_col=[0, 1])
    df_.sortlevel(inplace=True)

    try:
        df = df_.loc[grade, domain]
    except KeyError:
        exc = ('Sorry, the grade values have to be all integers. Edit your '
               'data file accordingly.')
        raise KeyError(exc)

    return df


def cat_p(cdf):
    "Get probabilities of each category from cumulative distribution."
    p = [0] + list(cdf) + [1]
    cat_ps = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    
    return np.array(cat_ps)


def ss_to_z(ss, se, cuts):
    "Convert scale scores to z-scores relative to cuts."
    z = (cuts - ss) / float(se)
    return z


def pl_probs(ss_se, cuts):
    """Get probability of each category for each scale score.

    """
    nrows = ss_se.shape[0]
    ncols = len(cuts) - 1
    acc = np.zeros((nrows, ncols))

    for i in range(nrows):
        ss = ss_se.loc[i, SS]
        se = ss_se.loc[i, SE]
        cuts_ = cuts.values[1:-1]
        zs = ss_to_z(ss, se, cuts_)
        cdf = stats.norm.cdf(x=zs)
        acc[i] = cat_p(cdf)

    consist = acc**2

    return {'acc':acc, 'consist':consist}
    

def acc_consist(ss_se, cuts):
    """Calculate classification accuracy.

    Returns
    -------
        acc_consist() returns a dictionary of statistics:

        {'acc':accuracy, 'consist':consistency, 'kappa':Cohen's kappa}

    Comments
    --------
        Accuracy and consistency are computed using Rudner's IRT-based
        method.  Cohen's kappa is the unweighted kappa as traditionally
        calculated.  However, it will differ from the CTT-based kappa
        derived using the Livingston and Lewis method.
    
    References
    ----------
    Livingston, Samuel A., & Lewis, Charles (1993). Estimating the 
    Consistency and Accuracy of Classifications Based on Test Scores.
    Education Testing Service, Research Report.
    https://www.ets.org/Media/Research/pdf/RR-93-48.pdf

    Rudner, Lawrence M. (2001). Computing the expected proportions 
    of misclassified examinees. Practical Assessment, Research &
    Evaluation, 7(14). 
    Available online: http://PAREonline.net/getvn.asp?v=7&n=14.
    
    Cohen's kappa. (2016, October 4). In Wikipedia, The Free Encyclopedia. 
    Retrieved 13:39, October 4, 2016, 
    from https://en.wikipedia.org/w/index.php?title=Cohen%27s_kappa&oldid=742569319
    """
    probs = pl_probs(ss_se, cuts)
    a, c = probs['acc'], np.sum(probs['consist'], axis=1)
    cuts_ = cuts.values[:-1]
    ncats = len(cuts_)
    accs = np.zeros((ncats))
    consists = np.zeros(np.shape(accs))
    counts = np.zeros((ncats))
    tab = np.zeros((ncats, ncats))
    
    # Get accuracy and consistency
    for i, cut in enumerate(cuts_):
        ss = ss_se.loc[:, SS]
        n = ss_se.loc[:, N].values
        
        # Max value needs to be included when counting kids in top cat
        if cut == cuts_[-1]:
            ix = (ss >= cut) & (ss <= cuts[i + 1])
        else:
            ix = (ss >= cut) & (ss < cuts[i + 1])
        ix = ix.values
        nix = n[ix]
        accs[i] = np.sum(a[ix, i] * nix) / float(np.sum(nix))
        consists[i] = np.sum(c[ix] * nix) / float(np.sum(nix))
        counts[i] = np.sum(nix)

        # tab used to calculate Cohen's kappa
        tab[i] = np.sum(a[ix, :] * nix[:, np.newaxis], axis=0)

    acc = np.sum(accs * counts) / np.sum(counts)
    consist = np.sum(consists * counts) / np.sum(counts)
    kappa = get_kappa(tab)
    
    norm_tab = tab / np.sum(tab)
#    print 'a=\n', a, np.shape(a)
#    print 'tab=\n', tab #norm_tab
#    print 'row sums=\n', np.sum(norm_tab, axis=1)
#    print 'col sums=\n', np.sum(norm_tab, axis=0)
#    print 'sum all=', np.sum(norm_tab)
#    sys.exit()
    

    return {'acc':acc, 'consist':consist, 'kappa':kappa}



def get_kappa(tab):
    "Calculate Cohen's kappa statistic."
    
    s_all = np.sum(tab)
    s_rows = np.sum(tab, axis=1)
    s_cols = np.sum(tab, axis=0)
    ncats = np.size(tab, axis=1)

    agree = 0
    exp_freq = 0
    for i in range(ncats):
        exp_freq += (s_rows[i] * s_cols[i]) / float(s_all)
        agree += tab[i, i]
        
    kappa = (agree - exp_freq) / (s_all - exp_freq)
    
    return kappa









