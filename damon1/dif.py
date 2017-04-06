# -*- coding: utf-8 -*-
# opts.py

"""Compute differential item functioning (DIF).
"""
import sys
import numpy as np
import damon1 as damon1
import damon1.core as dmn
import damon1.tools as dmnt
from scipy.special import ndtr
from scipy import stats

class dif_stats_Error(Exception): pass


# Load file
def load_scores(filename,
                getrows = {'Get':'NoneExcept', 'Labels':'key', 'Cols':['All']},
                getcols = {'Get':'NoneExcept', 'Labels':'key', 'Cols':['All']},
                labelcols = ['Student_ID', 'Group'],
                key4rows = ['Student_ID', 'S60', 'warn_dups'],
                delimiter = '\t'
                ):
    "Load score file and build stratified array of group counts"

    d = dmn.TopDamon(filename,
                     getrows = getrows,
                     getcols = getcols,
                     labelcols = labelcols,
                     key4rows = key4rows,
                     validchars = ['All', ['All'], 'Num'],
                     delimiter = delimiter,
                     verbose = True)

    return d


def count_cats(strat_vals, groups, scores, stratum, group, cat):
    "Count number of score values for a given stratum and group"

    strat_loc = strat_vals == stratum
    groups_x = groups[strat_loc]
    scores_x = scores[strat_loc]
    scores_xx = scores_x[groups_x == group]
    count = np.sum(scores_xx == cat)

    return count


def build_strat_table(loaded,
                      item,
                      raw_score = 'RawScore',
                      group = 'Sex',
                      strata = 'all_scores'
                      ):
    "Build ability stratified table of score counts by group"

    # Get scores, raw scores, groups
    d = loaded
    scores = d.core_col[item]
    sum_scores = d.core_col[raw_score]
    groups = d.rl_col[group]
    unique_raws = np.unique(sum_scores[sum_scores != d.nanval])

    # Associate raw scores with strata
    if strata == 'all_scores':
        interval = 1
        strata = np.arange(len(unique_raws))
    else:
        interval = int(len(unique_raws) / float(strata))
        strata = np.arange(strata)

    # Strata bins are of equal length, except the bottom bin which captures the remainder.
    strat_vals = np.zeros(np.shape(unique_raws))
    svals = np.repeat(strata, interval)
    strat_vals[-len(svals):] = svals
    strat_lookup = dict(zip(unique_raws, strat_vals))

    # Get stratum for each person
    person_strat = np.zeros(np.shape(sum_scores))

    for score in unique_raws:
        person_strat[sum_scores == score] = strat_lookup[score]

    # Get rating categories
    cats = np.unique(scores[scores != d.nanval])
    
    # Build stratum and group arrays
    groups_ = np.unique(groups)
    stratum = np.repeat(strata, len(groups_))
    group = np.tile(groups_, len(strata))

    # Labels
    corner = np.array([['ID', 'Stratum', 'Group']])
    collabels = np.append(corner, np.array([cats.astype(int)]), axis=1)
    rowlabels = np.zeros((len(stratum) + 1, 3), dtype='S20')
    rowlabels[0, :] = corner
    rowlabels[1:, 0] = np.arange(len(stratum))
    rowlabels[1:, 1] = stratum
    rowlabels[1:, 2] = group

    # Build counts table
    core = np.zeros((len(stratum), len(cats)))

    for row, strat in enumerate(stratum):
        for col, cat in enumerate(cats):
            gr = group[row]
            core[row, col] = count_cats(person_strat, groups, scores, strat, gr, cat)

    # Build Damon object
    counts = {'rowlabels':rowlabels, 'collabels':collabels, 'coredata': core,
              'key4rows':0, 'rowkeytype':int,
              'key4cols':0, 'colkeytype':'S60',
              'nanval':d.nanval, 'validchars':['All', ['All'], 'Num']}

    counts = dmn.Damon(counts, 'datadict', 'RCD_dicts_whole', verbose=None)

    # Check that each stratum has sufficient counts, > 1
    for strat in strata:
        cats = counts.extract(counts,
                              getrows = {'Get':'NoneExcept', 'Labels':'Stratum', 'Rows':[strat]})
        if np.sum(cats['coredata']) <= 1:
            exc = 'Insufficient data for one of the strata for an item.'
            raise dif_stats_Error(exc)
        
    return counts


def mean_score(scores, kg_counts, dif_type='SMD'):
    "Sum product of scores divided by total"

    if dif_type == 'SMD':
        out = np.sum(scores * kg_counts) / float(np.sum(kg_counts))

    if dif_type == 'M_a':
        out = np.sum(scores * kg_counts[1])

    elif dif_type == 'M_b':
        kscore_sum = np.sum(kg_counts, axis=0)
        w = np.sum(kg_counts[1]) / np.sum(kg_counts)
        out = w * np.sum(scores * kscore_sum)   

    return out


def stratum_variance(scores, kg_counts, dif_type='SMD'):
    """Get stratum variance

    See Zwick & Thayer, 1996

    """

    k_sum = float(np.sum(kg_counts, axis=None))
    kscore_sum = np.sum(kg_counts, axis=0)
    kg_sum = np.sum(kg_counts, axis=1)
    
    term0 = k_sum * np.sum(scores**2 * kscore_sum)
    term1 = np.sum(scores * kscore_sum)**2

    fact0 = (kg_sum[0] * kg_sum[1]) / ((k_sum**2) * (k_sum - 1))
    fact1 = ((1. / kg_sum[0]) + (1. / kg_sum[1]))**2

    if dif_type == 'SMD':
        svar = fact1 * fact0 * (term0 - term1)

    if dif_type == 'M':
        svar = fact0 * (term0 - term1)


    if 'MH' in dif_type:
        n11 = kg_counts[0, 0]
        n12 = kg_counts[0, 1]
        n21 = kg_counts[1, 0]
        n22 = kg_counts[1, 1]

        if dif_type == 'MH_a1':
            svar = (n11 * n22) / k_sum

        if dif_type == 'MH_a2':
            svar = ((n11 + n22) * n11 * n22) / k_sum**2

        if dif_type == 'MH_b1':
            svar = (n12 * n21) / k_sum

        if dif_type == 'MH_b2':
            svar = ((n12 + n21) * n12 * n21) / k_sum**2

        if dif_type == 'MH_c1':
            svar = (n11 * n22) / k_sum

        if dif_type == 'MH_c2':
            svar = (n12 * n21) / k_sum

        if dif_type == 'MH_c3':
            svar = (((n12 + n21) * n12 * n21) + ((n11 + n22) * n11 * n22)) / k_sum**2

    return svar

    
def dif_smd(strat_tab, focal_group, ref_group):
    """Get standardized mean difference (SMD) DIF statistic for a polytomous item.

    See Dorans & Schmitt, 1991

    """

    d = strat_tab
    strata = np.unique(d.rl_col['Stratum'])
    groups = [focal_group, ref_group]
    ncats = np.size(d.coredata, axis=1)
    scores = np.arange(ncats)

    # Initialize arrays:  smd for each stratum, focal group total count per stratum
    k_smd = np.zeros((len(strata)))
    k_var = np.zeros((len(strata)))
    kf_sum = np.zeros((len(strata)))
    kr_sum = np.zeros((len(strata)))

    # Get SMD and variance for each stratum
    for k, stratum in enumerate(strata):

        # Extract stratum
        stratum_x = d.extract(d,
                              getrows = {'Get':'NoneExcept',
                                         'Labels':'Stratum',
                                         'Rows':[stratum]})

        # Get group stats per stratum.  Foc = index 0, Ref = index 1
        kg_mean_score = np.zeros((2))
        kg_sum = np.zeros((2))
        kg_counts = np.zeros((2, ncats))

        for g, group in enumerate(groups):

            # Extract counts for ref and focal group
            kg_counts[g] = d.extract(stratum_x,
                                     getrows = {'Get':'NoneExcept',
                                                'Labels':'Group',
                                                'Rows':[group]})['coredata']

            kg_mean_score[g] = mean_score(scores, kg_counts[g], 'SMD')
            kg_sum[g] = np.sum(kg_counts)

        # Get stratum smd
        k_smd[k] = kg_mean_score[0] - kg_mean_score[1]
        kf_sum[k] = kg_sum[0]
        kr_sum[k] = kg_sum[1]

        # Get stratum variance
        k_var[k] = stratum_variance(scores, kg_counts, 'SMD')
        
    # Get overall SMD and variance
    smd_weights = kf_sum / np.sum(kf_sum)
    smd = np.sum(smd_weights * k_smd)
    smd_var = np.sum(smd_weights**2 * k_var)
    z = smd / np.sqrt(smd_var)
    pval = 1 - ndtr(np.abs(z))
    chisq_pval = 1 - stats.chi2.cdf(z**2, 1)

    return {'SMD_dif':smd, 'SMD_var':smd_var, 'SMD_z':z, 'SMD_pval':pval,
            'SMD_chisq':z**2, 'SMD_chisq_pval':chisq_pval}


def dif_MH(strat_tab, focal_group, ref_group):
    """Get Mantel-Haenszel DIF statistic for a dichotomous item.

    See Wood, 2011

    """
    d = strat_tab    
    strata = np.unique(d.rl_col['Stratum'])
    groups = [ref_group, focal_group]
    k_num = np.zeros((len(strata)))
    k_denom = np.zeros((len(strata)))
    scores = np.arange(2)

    # Strata-specific variance formula components
    k_var_a1 = np.zeros((len(strata)))
    k_var_a2 = np.zeros((len(strata)))
    k_var_b1 = np.zeros((len(strata)))
    k_var_b2 = np.zeros((len(strata)))
    k_var_c1 = np.zeros((len(strata)))
    k_var_c2 = np.zeros((len(strata)))
    k_var_c3 = np.zeros((len(strata)))

    for k, stratum in enumerate(strata):
        
        # Extract stratum
        stratum_x = d.extract(d,
                              getrows = {'Get':'NoneExcept',
                                         'Labels':'Stratum',
                                         'Rows':[stratum]})

        kg_counts = np.zeros((2, 2))
        
        for g, group in enumerate(groups):
            
            # Extract counts for ref and focal group
            kg_counts[g] = d.extract(stratum_x,
                                     getrows = {'Get':'NoneExcept',
                                                'Labels':'Group',
                                                'Rows':[group]})['coredata']

        # M-H table: k=ability stratum, r=ref, f=foc, 1=correct, 0=incorrect, rf=sum(r, f), n=sum
        r_k = kg_counts[0]
        f_k = kg_counts[1]

        rk1 = r_k[0]
        rk0 = r_k[1]
        fk1 = f_k[0]
        fk0 = f_k[1]
        knn = np.sum(kg_counts)

        # MH for k
        k_num[k] = (rk1 * fk0) / float(knn)
        k_denom[k] = (rk0 * fk1) / float(knn)

        # Variance components for k
        k_var_a1[k] = stratum_variance(scores, kg_counts, dif_type='MH_a1')
        k_var_a2[k] = stratum_variance(scores, kg_counts, dif_type='MH_a2')
        k_var_b1[k] = stratum_variance(scores, kg_counts, dif_type='MH_b1')
        k_var_b2[k] = stratum_variance(scores, kg_counts, dif_type='MH_b2')
        k_var_c1[k] = stratum_variance(scores, kg_counts, dif_type='MH_c1')
        k_var_c2[k] = stratum_variance(scores, kg_counts, dif_type='MH_c2')
        k_var_c3[k] = stratum_variance(scores, kg_counts, dif_type='MH_c3')

    # Calculate a = MH odds ratio estimate
    alpha = np.sum(k_num) / np.sum(k_denom)
    log_alpha = np.log(alpha)

    # Variance of log_alpha
    term_a = (1 / (2 * np.sum(k_var_a1)**2)) * np.sum(k_var_a2)
    term_b = (1 / (2 * np.sum(k_var_b1)**2)) * np.sum(k_var_b2)
    term_c = (1 / (2 * np.sum(k_var_c1) * np.sum(k_var_c2)) * np.sum(k_var_c3))
    var = term_a + term_b + term_c

    # Convert to ETS delta scale (formula is -2.35, but positive is what matches others' values !!!)
    d_dif = 2.35 * log_alpha
    d_var = 2.35 * var
    
    # z statistic
    chisq = (log_alpha**2) / var
    z = np.sqrt(chisq)
    pval = 1 - ndtr(np.abs(z))
    chisq_pval = 1 - stats.chi2.cdf(chisq, 1)

    return {'MH_alpha':alpha, 'MH_dif':log_alpha, 'MH_d-dif':d_dif,
            'MH_var':var, 'MH_d-var':d_var,
            'MH_z':z, 'MH_pval':pval,
            'MH_chisq':chisq, 'MH_chisq_pval':chisq_pval}


def dif_M(strat_tab, focal_group, ref_group, continuity_correction=False):
    "Get Mantel DIF statistic for a polytomous item."

    d = strat_tab
    strata = np.unique(d.rl_col['Stratum'])
    groups = [ref_group, focal_group]
    ncats = np.size(d.coredata, axis=1)
    scores = np.arange(ncats)
    k_num_a = np.zeros((len(strata)))
    k_num_b = np.zeros((len(strata)))
    k_var = np.zeros((len(strata)))
    
    for k, stratum in enumerate(strata):

        # Extract stratum
        stratum_x = d.extract(d,
                              getrows = {'Get':'NoneExcept',
                                         'Labels':'Stratum',
                                         'Rows':[stratum]})

        # Get group stats per stratum.  Foc = index 0, Ref = index 1
        kg_counts = np.zeros((2, ncats))
        kg_sum = np.zeros((2))
        
        for g, group in enumerate(groups):
       
            # Extract counts for ref and focal group
            kg_counts[g] = d.extract(stratum_x,
                                     getrows = {'Get':'NoneExcept',
                                                'Labels':'Group',
                                                'Rows':[group]})['coredata']

            kg_sum[g] = np.sum(kg_counts)

        # Get terms
        Fk = mean_score(scores, kg_counts, dif_type='M_a')
        EFk = mean_score(scores, kg_counts, dif_type='M_b')
        k_num_a[k] = Fk
        k_num_b[k] = EFk
        
        # Get stratum variance
        k_var[k] = stratum_variance(scores, kg_counts, 'M')
        
    # Get overall Z-statistic
    dif = np.sum(k_num_a) - np.sum(k_num_b)
    var = np.sum(k_var)
    z = dif / np.sqrt(var)
    pval = 1 - ndtr(np.abs(z))

    if continuity_correction is True:
        M_chisq = (np.abs(dif) - .5)**2 / var
    else:
        M_chisq = z**2
    
    chisq_pval = 1 - stats.chi2.cdf(M_chisq, 1)

    return {'M_dif':dif, 'M_var':var, 'M_z':z, 'M_pval':pval,
            'M_chisq':M_chisq, 'M_chisq_pval':chisq_pval}


def dif_stats(filename,   # [<'my/file.txt',...> => name of scored data file]
              student_id = 'Student_ID',    # [<'Student_ID', ...> => student id column label]
              group = ['Sex', {'focal':0, 'ref':1}],  # [<e.g.'Sex', {'focal':'female', 'ref':'male'}]> => column label with assignment to focal and reference]
              raw_score = 'RawScore',  # [<'RawScore',...> => raw score column label]
              items = 'All',  # [<'All', ['item1', 'item3',...]> => items for which to get stats]
              stats = 'All',  # [<'All', [see list in docs]> => desired statistics]
              strata = ('all_scores', 4),   # [<'all_scores', int> => number of raw score strata to apply]
              getrows = None, # [<None, {'Get':_,'Labels':_,'Rows':_}> => select rows using extract() syntax]
              getcols = None, # [<None, {'Get':_,'Labels':_,'Cols':_}> => select cols using extract() syntax]
              delimiter = '\t',   # [<',', '\t'> => column delimiter]
              ):
    "Calculate DIF stats for each in a range of items"

    # Load data
    d = load_scores(filename = filename,
                    getrows = getrows,
                    getcols = getcols,
                    labelcols = [student_id, group[0]],
                    key4rows = [student_id, 'S60', 'warn_dups'],
                    delimiter = delimiter
                    )

    if items == 'All':
        items = dmnt.getkeys(d, 'Col', 'Core', 'Auto', None)
        items = items[items != raw_score]
    else:
        items = np.array(items)

    if stats == 'All':
        stats = ['MH_alpha', 'MH_dif', 'MH_d-dif', 'MH_var', 'MH_d-var',
                 'MH_z', 'MH_pval', 'MH_chisq', 'MH_chisq_pval',
                 'M_dif', 'M_var', 'M_z', 'M_pval', 'M_chisq', 'M_chisq_pval',
                 'SMD_dif', 'SMD_var', 'SMD_z', 'SMD_pval', 'SMD_chisq', 'SMD_chisq_pval',
                 'SD', 'SMD/SD', 'Flag', 'Counts']

    if 'Flag' in stats:
        flag_stats = ['MH_d-dif', 'MH_var', 'MH_pval', 'SMD_dif',
                      'SD', 'SMD/SD', 'M_chisq_pval'] 
        for stat in flag_stats:
            if stat not in stats:
                stats.append(stat)

    if 'SMD/SD' in stats:
        smd_sd_stats = ['SMD_dif', 'SD']
        for stat in smd_sd_stats:
            if stat not in stats:
                stats.append(stat)

    if 'Counts' in stats:
        count_stats = ['Count_Ref', 'Count_Focal', 'Count_All', ]
        for stat in count_stats:
            if stat not in stats:
                stats.insert(0, stat)
        stats.remove('Counts')

    # Initialize DIF table
    corner = np.array([['Item', 'N_Cats']])
    collabels = np.append(corner, np.array([stats]), axis=1)
    rowlabels = np.zeros((len(items) + 1, 2), dtype='S60')
    rowlabels[0, :] = corner[0]
    rowlabels[1:, 0] = np.array(items)
    core = np.zeros((len(items), len(stats)))

    # Get stats for each item
    for i, item in enumerate(items):
        try:
            tab = build_strat_table(loaded = d,
                                    item = item,
                                    raw_score = raw_score,
                                    group = group[0],
                                    strata = strata[0]
                                    )           
        except (damon1.utils.Damon_Error, dif_stats_Error):
            # Try with backup strata parameter
            try:
                tab = build_strat_table(loaded = d,
                                        item = item,
                                        raw_score = raw_score,
                                        group = group[0],
                                        strata = strata[1]
                                        )
            except (damon1.utils.Damon_Error, dif_stats_Error):
                print ('Warning in tools.dif_stats(): Unable to build a '
                       'stratification table for: '
                       'stratum=', strata, 'item=', item)
                core[i, :] = d.nanval
                continue

        ncats = np.size(tab.coredata, axis=1)
        continuity_correction = True if ncats == 2 else False
        rowlabels[i + 1, 1] = ncats

        # Flag needed DIF functions
        run_dif_MH = False
        MH_stats = []
        for stat in stats:
            if 'MH_' in stat and ncats <= 2:
                MH_stats.append(stat)
                run_dif_MH = True

        run_dif_M = False
        M_stats = []
        for stat in stats:
            if 'M_' in stat:
                M_stats.append(stat)
                run_dif_M = True

        run_dif_smd = False
        smd_stats = []
        for stat in stats:
            if 'SMD' in stat and ncats > 2:
                smd_stats.append(stat)
                run_dif_smd = True

        run_sd = False
        for stat in stats:
            if 'SD' in stat:
                run_sd = True

        run_counts = False
        for stat in stats:
            if 'Count' in stat:
                run_counts = True

        # Get item standard deviation
        stat_ = {}
        if run_sd is True:
            ivals = d.core_col[item]
            item_sd = np.std(ivals[ivals != d.nanval])
            stat_['SD'] = item_sd

        # Get counts
        if run_counts is True:
            ivals = d.core_col[item]
            gvals = d.rl_col[group[0]]
            valid = ivals != d.nanval
            stat_['Count_All'] = np.sum(valid)
            stat_['Count_Focal'] = np.sum((valid) & (gvals == str(group[1]['focal'])))
            stat_['Count_Ref'] = np.sum((valid) & (gvals == str(group[1]['ref'])))

        # Calculate MH DIF
        if run_dif_MH is True:
            dif_MH_out = dif_MH(tab, group[1]['focal'], group[1]['ref'])
            for stat in MH_stats:
                stat_[stat] = dif_MH_out[stat]

        # Calculate M DIF
        if run_dif_M is True:
            dif_M_out = dif_M(tab, group[1]['focal'], group[1]['ref'], continuity_correction)
            for stat in M_stats:
                stat_[stat] = dif_M_out[stat]

        # Calculate SMD DIF
        if run_dif_smd is True:
            dif_smd_out = dif_smd(tab, group[1]['focal'], group[1]['ref'])
            
            for stat in smd_stats:
                if stat != 'SMD/SD':
                    stat_[stat] = dif_smd_out[stat]
                else:
                    stat_[stat] = dif_smd_out['SMD_dif'] / item_sd

        # Calculate DIF flag
        if 'Flag' in stats:
            if ncats == 2:
                d_dif = np.abs(stat_['MH_d-dif'])
                se = np.sqrt(stat_['MH_var'])
                pval = stat_['MH_pval']
                z_crit = (d_dif - 1.0) / se
                
                if d_dif > 1.5 and z_crit > 1.645:
                    stat_['Flag'] = 2
                elif d_dif < 1.0 or pval > 0.05:
                    stat_['Flag'] = 0
                else:
                    stat_['Flag'] = 1
            else:
                smd_sd = np.abs(stat_['SMD/SD'])
                p_val = stat_['M_chisq_pval']

                if smd_sd > 0.25 and p_val < 0.05:
                    stat_['Flag'] = 2
                else:
                    stat_['Flag'] = 0

        # Populate table
        for j, stat in enumerate(stats):
            if 'MH_' in stat and ncats > 2:
                core[i, j] = d.nanval
            elif 'SMD' in stat and ncats <= 2:
                core[i, j] = d.nanval
            else:
                core[i, j] = stat_[stat]

    # Build table
    tab_dict = {'rowlabels':rowlabels, 'collabels':collabels, 'coredata':core,
               'key4rows':0, 'rowkeytype':'S60', 'key4cols':0, 'colkeytype':'S60',
               'nanval':d.nanval, 'validchars':['All', ['All'], 'Num']}

    tab_obj = dmn.Damon(tab_dict, 'datadict', 'RCD_dicts_whole', verbose=None)

    return tab_obj
        
           
                    
                   
















