# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:03:10 2016
@author: Mark H. Moulton, Educational Data Systems, Inc.
Dependencies:  Numpy, Pandas

Stand-alone utility (not integrated with the main Damon workflow) for creating 
raw-to-scale score conversion tables from item parameters.  Supports Rasch, 
2PL, 3PL, and GPC models, including mixed.  Requires an item-by-parameter 
input file with information about model, number of categories, scale
multiplication factor (1.0 or 1.7), and relevant item parameters.


Workflow Example
----------------
import sys
import numpy as np
import damon1.raw_to_scale as rs

# Get file information
path = ('/Users/markhmoulton/anaconda/lib/python2.7/site-packages/damon1'
        '/tests/play/')
data = 'raw_K1_RD_ITM.txt' 

# Specify parameters for load_item_params()
sep = '\t'
index_col = 'name'

# {file_label:internal_field_name}
cols = {'aparam':rs.A,
        'bparam':rs.B,
        'cparam':rs.C,
        'step1':rs.D1,
        'step2':rs.D2,
        'step3':rs.D3,
        'model':rs.MODEL,
        'name':rs.ITEM,
        'ncat':rs.NCAT,
        'scale':rs.SCALE,
        'test':rs.TEST
        }

# {model_label:internal_model_type}
models = {'L3':rs.PL3,
          'PC1':rs.GPC}

params = rs.load_item_params(data, sep, index_col, cols, models)
thetas = np.arange(-4.0, 4.1, 0.10)
title = data[:-4]
conv_tab = rs.build_conversion_table(thetas, params, title, path)

"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
TEST = 'test'
ITEM = 'item'
STUDENT = 'student'
MODEL = 'model'
NCAT = 'ncat'
A = 'a'
B = 'b'
C = 'c'
D1 = 'd1'
D2 = 'd2'
D3 = 'd3'
D4 = 'd4'
D5 = 'd5'
D6 = 'd6'
PL1 = '1PL'
PL2 = '2PL'
PL3 = '3PL'
GPC = 'GPC'
SCALE = 'scale_factor'
THETA_SS = 'theta_ss'
THETA_SE = 'theta_se'
TCC = {'pct':False, 'score_line':'b-', 'err_line':'r-', 'title':None,
       'score_label':None, 'err_label':None, 'xlabel':'Theta',
       'ylabel':'True Score', 'ylabel_err':'Standard Error',
       'adj_box':None, 'score_legend':{'loc':'upper left'}, 
       'err_legend':{'loc':'upper center'}, 'theta_se':True}
       

def load_thetas(data, sep, index_col, cols):
    """Load theta values and format as pandas dataframe.
    
    """
    usecols = cols.keys()    
    if isinstance(usecols[0], int):
        header = None
    elif isinstance(usecols[0], str):
        header = 0
    
    df = pd.read_csv(data, sep,
                     index_col=index_col,
                     header=header,
                     usecols=usecols)
                             
    df.rename(columns=cols, inplace=True)
    df.index.name = cols[index_col]

    return df

    
def load_item_params(data, sep, index_col, cols, models, select=None):
    """Load and format item parameters file as pandas dataframe.
    
    Returns
    -------
        Item parameters and information formatted as a Pandas dataframe.
    
    Comments
    --------
        The item-by-parameter file needs a header row.  Required fields (in 
        no particular order, more allowed) should include:
            * item id
            * model type
            * number of score categories
            * scale multiplication factor (1.0 for logistic, 1.7 for normal)
            * item parameters
        
        You don't need to load the standard errors of the item parameters.
        
        Example Data

        name,model,ncat,scale,aparam,bparam,cparam,step1,step2,step3
        sf_rd1,L3,2,1.7,1.009,-0.584,0.400,,,
        sf_rd2,L3,2,1.7,0.829,0.010,0.316,,,
        sf_rd3,L3,2,1.7,1.345,1.026,0.271,,,
        sf_rd4,L3,2,1.7,1.079,0.641,0.317,,,
        sf_rd5,L3,2,1.7,0.343,-0.598,0.015,,,
        sf_rd6,L3,2,1.7,1.169,-0.075,0.328,,,
        sf_rd7,L3,2,1.7,2.094,0.565,0.332,,,
        sf_rd8,L3,2,1.7,1.027,1.399,0.318,,,
        sf_rd9,L3,2,1.7,2.234,1.000,0.267,,,
        sf_rd12,L3,2,1.7,1.283,-0.488,0.269,,,
        sf_rd13,L3,2,1.7,1.560,0.819,0.361,,,
        sf_rd14,L3,2,1.7,2.561,0.346,0.241,,,
        sf_rd15,L3,2,1.7,2.304,0.329,0.345,,,
        sf_rd16,L3,2,1.7,2.580,0.331,0.230,,,
        sf_rd10,PC1,4,1.7,1.851,,,-1.118,-0.864,-0.776
        sf_rd11,PC1,4,1.7,2.410,,,-1.114,-0.923,-0.742
        sf_rd19,L3,2,1.7,1.131,0.660,0.144,,,
        sf_rd20,L3,2,1.7,0.939,0.326,0.184,,,
        sf_rd17,L3,2,1.7,1.110,0.572,0.000,,,
        sf_rd18,L3,2,1.7,1.456,0.879,0.000,,,

    Parameters
    ----------
        "data" is either the name or path of the item parameter file or it
        is an already loaded pandas dataframe.
            data = 'my_params.txt'
            data = my_params   # pandas dataframse

        --------------
        "sep" is the column delimiter.
            sep = ','               =>  columns are comma-delimited.
            sep = '\t'              =>  columns are tab-delimited.

        --------------        
        "index_col" is the column containing unique item ids.  It should be a
        field name if there is a header row, otherwise an integer index.
             index_col = 0          =>  item ids are in the 0'th column
             index_col = 'name'     =>  item ids are in the 'name' column

        --------------        
        "cols" is a dictionary matching existing field names to Damon's
        constant names as given in the irt_tools module.  In IRT parlance, 
        A means the 'a' parameter, B the 'b' parameter, 'c' the C parameter.  
        The D parameters are the step parameters.  If D is specified, B is 
        not specified.  Only those columns are loaded that are specified in 
        your cols dictionary.  Columns that are listed in 'cols' but are
        not present in the data header are ignored, saving you the trouble
        of having to build a new cols list for every test.

            import irt_tools as it
            
            cols = {'test':rs.TEST,     =>  if selecting items for a test
                    'aparam':it.A,
                    'bparam':it.B,
                    'cparam':it.C,
                    'step1':it.D1,
                    'step2':it.D2,
                    'step3':it.D3,
                    'model':it.MODEL,
                    'name':it.ITEM,
                    'ncat':it.NCAT,
                    'scale':it.SCALE    =>  scale factor 1.0 or 1.7
                    }
            
            This means, for example, that the column that is labeled
            'aparam' corresponds to the irt_tools.A constant.  The column
            labeled 'bparam' corresponds to the irt_tools.B constant.
            the column labeled 'model' corresponds to the irt_tools.MODEL
            constant.  And so on.  The original column headers are replaced
            by their Damon equivalents.
            
        --------------
        "models" is a dictionary for renaming model names to Damon constant
        names.  Like the column names, model names are stored in the
        irt_tools module.

            import irt_tools as it
            
            models = {'L3':it.PL3,  =>  3PL model
                      'PC1':it.GPC} =>  Generalized Partial Credit Model

        --------------
        "select" allows you to select a subset of rows (items) that belong
        to a particular test in the column corresponding to 
        irt_tools.TEST.
        
            select = None (default) =>  Use all rows.
        
            select = 'test_LI_3-5'  =>  Use only items that have 'test_LI_3-5'
                                        in the TEST field.


    """
    if isinstance(data, pd.DataFrame):
        if select:
            df = data.loc[data[TEST] == select]
        else:
            df = data
        return df
        
    elif isinstance(data, str):
    
        # Get usecols, header parameters
        usecols = cols.keys()    
        if isinstance(usecols[0], int):
            header = None
        elif isinstance(usecols[0], str):
            header = 0
        
            # Remove from usecols labels that are not in file header
            df_ = pd.read_csv(data, sep,
                              index_col=index_col,
                              header=header,
                              usecols=None)
            usecols_ = []
            for c in usecols:
                if c in df_.columns.values:
                    usecols_.append(c)
            
            usecols = usecols_.append(index_col)

        df = pd.read_csv(data, sep,
                         index_col=index_col,
                         header=header,
                         usecols=usecols)
                                 
        df.rename(columns=cols, inplace=True)
        df.index.name = cols[index_col]
        
        # Rename models
        df.replace({MODEL:models}, inplace=True)

        # Select subset of rows
        if select:
            df = df.loc[df[TEST] == select]

        return df
        
    else:
        exc = 'data must be a string path/file name or a Pandas dataframe.'
        raise ValueError(exc)
    

def build_conversion_table(thetas, params, title, dir_path, sep=',',
                           nraw=None, min_max=None, theta_se=True,
                           tcc=None):
    """Build raw-to-scale-score conversion table with TCC chart.
    
    Returns
    -------
        Returns a Python dict containing a raw-to-scale-score
        conversion table as a Pandas dataframe as well as a dataframe
        relating theta scale scores to IRT "true scores":
        
            out = build_conversion_table(...)
            conv_tab = out['ctab']
            true_scores = out['true']
        
        The latter is used as an input to the compare_tccs() function.
        
        In addition, the function outputs the conversion table as a text file
        and a test characteristic chart as a .png file.
    
    Comments
    --------
        Test scoring frequently requires a simple table for converting
        raw scores to scale scores in some desired metric.  This is
        done straightfowardly with unidimensional models in which there is
        no need for "pattern scoring" (where each response pattern
        has its own score).  All that is required is a one-to-one monotonic
        relationship between raw scores (summed across items) and scale
        scores.
        
        In addition to a table for scoring, the chart of scale scores
        to "true scores", known as a Test Characteristic Curve (TCC), is useful
        for comparing tests and validating their comparability.  If the TCCs
        for two tests are not parallel to each other, their scale scores are 
        not comparable.
        
        Each person's "true score" is simply their expected score on a test
        based on their ability and the item parameters.  It is in the same
        metric as person "raw scores", but continuous rather than discrete.
                
    Parameters
    ----------
        "thetas" is a numpy array containing a range of equally spaced student 
        ability values (called "theta" in IRT jargon).  They may be in any
        metric (e.g., -3 to +3 in increments of 0.10 or 100 to 300 in 
        increments of 1, the smaller the increment the better), but it is
        important that the theta metric match the metric of the difficulty
        'b' parameters or 'd' step parameters -- all in logits or all in some
        other scale score metric.
        
        The thetas array is usually built using numpy's arange() function.  
        
            thetas = np.arange(-3.0, 3.1, 0.1)
                                        =>  Define the theta ability range from
                                            -3 up to, but not including, 3.1,
                                            in increments of 0.1.
            
            thetas = np.arange(200, 601, 1)
                                        =>  Define the ability range from 200 
                                            up to, but not including, 600 in 
                                            unit increments.
        
        --------------
        "params" is a dataframe of item parameters obtained by running
        irt_tools.load_item_params().
            
            import irt_tools as it
            
            params = it.load_item_params('params.txt', ...)

        --------------        
        "title" is the description to put on top of the TCC chart and in
        the output conversion table file name.
            
            title = 'Spring 2016 Reading'

        --------------        
        "dir_path" is the directory in which the conversion table and TCC
        chart should be placed.
        
            dir_path = '/Users/markhmoulton/test_results/'

        --------------        
        "sep" is the delimiter to use when outputting the file.  It also 
        controls the file extension.
        
            sep = ','           =>  Comma-delimited '.csv' file
            sep = '\t'          =>  Tab-delimited '.txt' file

        -------------- 
        "nraw" <int> is the maximum raw score.
        
            nraw = 20           =>  A maximum of 20 raw score points
        
        
        -------------- 
        "min_max" <int> is the minimum and maximum allowed scale score.
        
            min_max=[200, 800]  =>  Scale scores below 200 will be clipped at
                                    200.  Scale score above 800 will be
                                    clipped at 800.

        --------------    
        "theta_se" (True, False) specifies whether to report "conditional 
        standard errors" -- the standard error of measurement at each value
        of theta.  Expect the standard error to rise toward infinity at the 
        upper and lower extremes of the scale.
        
        -------------- 
        "tcc" is a dictionary of parameters used for building charts of
        the test characteristic curve with standard error.  To see the
        most current default tcc parameters, type:
            
            import irt_tools as it
            print it.TCC
        
        For more details refer to the matplotlib docs.  The following will
        give you most of what you need to know.
            
            import matplotlib.pyplot as plt
            help(plt.plot)
        
        This will help with the "legend" parameters (where to place, etc.):
            
            help(plt.legend)
        
        The tcc parameters are (as of this writing):
            
          {'pct':False,         => <bool> report y-axis as a percentage
           'theta_se':True,     => <bool> display conditional standard errors
           'score_line':'b-',   => display the score curve as a blue line
           'err_line':'r-',     => display the error curve as a red line
           'title':None,        => don't provide a chart title
           'score_label':None,  => <str> If None, test + 'Score'
           'err_label':None,    => <str> If None, test + 'SE'
           'xlabel':'Theta',    => <str> x-axis label
           'ylabel':'True Score', => <str> left-hand y-axis label
           'ylabel_err':'Standard Error', => <str> right-hand y-axis label
           'adj_box':None,      => <None, [pWidth, pHeight]> adjust size of box
           'score_legend':{'loc':'top left'}, => dict of score 'legend' params
           'err_legend':{'loc':'upper center'} => dict of error 'legend' params
           }
        
        It is easy to place the 'score_legend' and 'err_legend' inside the
        chart using the legend() loc parameter.  To place them outside the
        box (e.g. below the x-axis label), you need to adjust the size of
        the box (e.g., shrink the chart vertically) to make room for the
        legends and place them using the legend() 'bbox_to_anchor' parameter:
        
            {other tcc params, ... ,
             'adj_box':[1.0, 0.9],
             'score_legend':{'bbox_to_anchor':(0.5, -0.15), 'fontsize':'medium'}
             'err_legend:{'bbox_to_anchor':(0.8, -0.15), 'fontsize':'medium'}
             }
        
        In this case, adj_box contains two parameters -- a factor by which
        to multiply the width of the box and a factor by which to multiply
        its height.  Here we keep the width unchanged and shrink its height 
        to 0.9 of the original.
        
        If you don't use the 'adj_box' parameter in this scenario, the
        output .png file will cut off the legends.  This is the only
        scenario where you will need 'adj_box'.
        
        The 'bbox_to_anchor' parameters (see matplotlib.pyplot.legend docs)
        consist of a tuple of numbers representing the x-value and y-value
        of where each legend should be placed relative to the bottom left of
        the chart.  These values tend to range from -0.20 to 1.2 but you
        have to experiment to see where the legends end up.
        
        The legend parameters reflect the matplotlib.pyplot.legend() method
        and are therefore quite flexible, if a bit tricky to master.
        
        All parameters you do NOT specify will refer to the it.TCC default.
        
            tcc = None          =>  Do not build a test characteristic curve.
            
            tcc = {'pct':True}  =>  Defaults will be used for all parameters
                                    except for 'pct', which specifies that
                                    the left-hand y-axis run from 0 - 100.

        Paste Function
        --------------
        build_conversion_table(thetas, 
                               params, 
                               title, 
                               dir_path, 
                               sep=',',
                               nraw=None, 
                               min_max=None,
                               theta_se=True,
                               tcc=None)
    """
    # Get cell estimates for theta x item
    est, info = estimates(thetas, params)
    conv_tab = conversion_table(est, info, nraw, min_max, theta_se)
    ctab = conv_tab['ctab']
    true = conv_tab['true']
    
    # Export table
    ext = '.csv' if sep == ',' else '.txt'
    export(ctab, dir_path + title + '_convtab' + ext, sep)
    
    # Export chart
    if tcc is not None:
        tcc_ = TCC.copy()
        for k in tcc:
            tcc_[k] = tcc[k]
                
        chart_tcc(true, tcc_, dir_path + title + '_tcc.png')
    
    return {'ctab':ctab, 'true':true}
    

def estimates(thetas, params):
    """Calculate expected values for each theta and item.
    
    Returns
    -------
        two thetas x items dataframes containing cell estimates (also
        called "expected values" and cell information:
            
            est, info = estimates(thetas, params)
    
    Comments
    --------
        estimates() is used by build_conversion_table() to calculate
        a raw-to-scale score conversion table with standard errors
        as well as test characteristic and standard error curves.
        
        It draws on the IRT probability formulas implemented in
        est_pl() and est_gpc(), i.e., the formulas for the 1PL,
        2PL, and 3PL models for dichotomous data and the Generalized
        Partial Credit Model (de Ayala, 2009).
        
    Parameters
    ----------
        "thetas" is a list or array of possible theta values (person scale
        scores) in either a transformed or untransformed metric.
        
        --------
        "params" is a dataframe of item parameters and other item
        information, generated by load_item_params().
        
    References
    ----------
    de Ayala, R.J. (2009). "The Theory and Practice of Item Response
    Theory".  The Guilford Press.

     """
    
    # Initialize estimates (expected values) dataframe
    items = params.index.values
    est = pd.DataFrame(index=thetas, columns=items)
    info = pd.DataFrame(index=thetas, columns=items)
    
    # For each item, get an array of estimates across thetas
    for item in items:
        est_, info_ = estimates_for_item(thetas, params, item, pcats=None)
        est.loc[:, item] = est_
        info.loc[:, item] = info_
        
    return est, info


def estimates_for_item(thetas, params, item, pcats=False):
    "Calculate expected values for each theta and specified item."

    mod = params.loc[item, MODEL]
    est, info, pcats_ = est_func[mod](thetas, params.loc[item, :])
    
    if pcats:
        return est, info, pcats_
    else:
        return est, info    
    
        
# Model-specific expected value (cell estimate) functions
def est_pl(thetas, params):
    """Calcuate expected value for a dichotomous 1-, 2- or 3PL item.
    
    References
    ----------
    de Ayala, R.J. (2009). "The Theory and Practice of Item Response
    Theory".  The Guilford Press.
    
        See p. 124, eq 6.3, for 3PL probability formula.
    """
    
    mod = params[MODEL]
    
    # Define parameters
    if mod == PL3:
        
        try:
            params[C]
        except KeyError:
            params[C] = 0
            
        if np.isnan(params[C]):
            params[C] = 0
            
        a, b, c, k, th, ncat = (params[A], params[B], params[C], params[SCALE], 
                                thetas, params[NCAT])

    elif mod == PL2:
        a, b, c, k, th, ncat = (params[A], params[B], 0, params[SCALE], thetas, 
                                params[NCAT])

    elif mod == PL1:
        a, b, c, k, th, ncat = (1, params[B], 0, params[SCALE], thetas, 
                                params[NCAT])
    
    else:
        exc = 'Unable to figure out the model.'
        raise ValueError(exc)
    
    # Calculate cell estimates across thetas
    y = np.exp(a*k*(th - b))
    est = p = c + ((1 - c) * (y / (1 + y)))
    
    # Calculate cell information across thetas
    if mod == PL3:
        info = k**2 * (a**2 * (p - c)**2 * (1 - p)) / ((1 - c)**2 * p)
    elif mod == PL2:
        info = k**2 * a**2 * p * (1 - p)
    elif mod == PL1:
        info = k**2 * p * (1 - p)
    
    cat_probs = {0:(1 - p), 1:p}
                 
    return est, info, cat_probs


def est_gpc(thetas, params):
    """Calculate expected value for a polytomous Generalized Partial Credit 
    item.
    
    References
    ----------
    de Ayala, R.J. (2009). "The Theory and Practice of Item Response
    Theory".  The Guilford Press.
    
        See p. 200, eq 7.6, for information function.
    """
    
    ncat = params[NCAT]
    nrows = len(thetas)
    a, th, k = params[A], thetas, params[SCALE]
    steps = [D1, D2, D3, D4, D5, D6][:ncat - 1]
    st = params[steps]
    cats = range(ncat)
    cat_probs = {}
    cat_prob_denom = np.zeros(nrows)
    
    # Get numerators and denominator
    for cat in cats:
        if cat == 0:
            cat_probs[cat] = 1.0 #np.exp(cat)
        else:
            cat_probs[cat] = np.exp(a*k*(cat*th - np.sum(st[:cat].values)))
        cat_prob_denom += cat_probs[cat]
    
    # Get cat probabilities
    for cat in cats:
        cat_probs[cat] = cat_probs[cat] / cat_prob_denom
        
    # Expected values
    est = np.zeros((nrows))
    for cat in cats:
        est += cat * cat_probs[cat]
    
    # Get information 
    term1 = np.zeros((nrows))
    term2 = np.zeros((nrows))
    
    for cat in cats:
        p = cat_probs[cat]
        term1 += cat**2 * p
        term2 += cat * p
    
    info = k**2 * a**2 * (term1 - term2**2)
    
    return est, info, cat_probs
    
# Dictionary to store model estimation functions
est_func = {PL3:est_pl,
            PL2:est_pl,
            PL1:est_pl,
            GPC:est_gpc}

            
def conversion_table(estimates, info, nraw=None, min_max=None, theta_se=True):
    "Get nearest raw score for each theta."
    
    # Load "true scores" and scale score se into dataframe
    true_ = estimates.sum(axis=1)
    
    if theta_se:
        se_ = 1.0 / np.sqrt(info.sum(axis=1))

    true = true_.to_frame('true_score')
    true.index.name = 'theta'
    if theta_se:
        true['se'] = se_
    
    # Get range of raw scores
    if nraw:
        min_raw, max_raw = np.amin(np.around(true.values)), nraw
    else:
        min_raw, max_raw = (np.amin(np.around(true.values)), 
                            np.amax(np.around(true.values)))
    raws = np.arange(max_raw + 1)
    
    # Initialize conversion table
    columns = ['theta', 'se'] if theta_se else ['theta']
    ctab = pd.DataFrame(index=raws, columns=columns)
    ctab.index.name = 'raw_score'

    if min_max is not None:
        min_, max_ = min_max
    
    # Get scale score corresponding to true score nearest each raw score
    for raw_score in ctab.index:
        if min_max is not None:
            if raw_score < min_raw:
                ss = min_
            elif raw_score > max_raw:
                ss = max_
            else:
                ss = np.clip([find_nearest(true.loc[:, 'true_score'], 
                                           raw_score)], min_, max_)[0]
        else:
            ss = find_nearest(true, raw_score)
        
        if theta_se:
            se = se_[ss]
            ctab.loc[raw_score] = [ss, se]
        else:
            se = None
            ctab.loc[raw_score] = ss

    return {'ctab':ctab, 'true':true}

    
def find_nearest(array, value):
    "Get value in array nearest to that specified for conversion_table()."
    
    pdix = (np.abs(array - value)).argmin()
    return pdix
    

def export(data, save_as, sep='\t'):
    "Export dataframe as a tab-delimited text file for build_conversion_table."
    
    # Export data
    data.to_csv(save_as, sep=sep)

    return None
    
    
def chart_tcc(true, tcc, savefig):
    ("Chart 'true scores' against theta for build_conversion_table() "
     "and compare_tccs().")

    if isinstance(true, dict):
        tests = true.keys()
        trues = true
        if tcc.keys()[0] not in trues.keys():
            exc = ("Unable to figure out tcc parameter. It should be nested "
                   "by test, e.g., {'test1':{tcc1...}, 'test2':{tcc2...}}")
            raise ValueError(exc)
        else:
            tccs = tcc
    else:
        tests = ['test']
        trues = {}
        trues['test'] = true
        tccs = {}
        tccs['test'] = tcc

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    thetas = trues[tests[0]].index.values
    
    for i, test in enumerate(tests):
        tcc = tccs[test]
        true = trues[test]
        
        if not tcc['theta_se']:
            errs = None
        else:
            try:
                errs = true.loc[:, 'se']
            except KeyError:
                errs = None

        scores = true.loc[:, 'true_score']
        if tcc['pct']:
            scores = 100 * scores / np.float(max(scores))
            
        # Main plot
        slabel, elabel = tcc['score_label'], tcc['err_label']
        slabel = test+' score' if slabel is None else slabel
        elabel = test+' SE' if elabel is None else elabel
        ax1.plot(thetas, scores, tcc['score_line'], label=slabel)
        ax2.plot(thetas, errs, tcc['err_line'], label=elabel)
    
    # Add formatting
    ax1.set_xlabel(tcc['xlabel'])
    ax1.set_ylabel(tcc['ylabel'])
    ax1.set_xlim((min(thetas) - 1, max(thetas + 1)))
    ax1.set_ylim((0, max(scores) + 3))

    if errs is not None:
        errs_ylim = (min(errs) - 1, max(errs) + 1)
        ax2.set_ylim(errs_ylim)
        ax2.set_ylabel(tcc['ylabel_err'])

        
#        box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Adjust size of charting box to make room for legend
    if tcc['adj_box'] is not None:
        box = ax1.get_position()
        start_x = box.x0
        start_y = box.y0 + box.height * (1.0 - tcc['adj_box'][1])
        stop_x = box.width * tcc['adj_box'][0]
        stop_y = box.height * tcc['adj_box'][1]

        ax1.set_position([start_x, start_y, stop_x, stop_y])
        ax2.set_position([start_x, start_y, stop_x, stop_y])



    # Add legend
    if tcc['score_legend'] is not None:
        ax1.legend(**tcc['score_legend'])
    if tcc['err_legend'] is not None and errs is not None:
        ax2.legend(**tcc['err_legend'])
        
    # Format and save
    if tcc['title'] is not None:
        plt.title(tcc['title'])
        
    fig.savefig(savefig)


def compare_tccs(trues, tccs=None, savefig='compare_tccs.png'):
    """Compare multiple TCC's with standard errors on one chart.
    
    Returns
    -------
        Outputs a .png chart containing multiple test characteristic
        curves with their standard errors.
    
    Comments
    --------
        The build_conversion_table() function returns a raw to
        scale score conversion table with a chart of a single
        test characteristic curve (TCC). TCC's compare each
        possible person ability theta with the corresponding
        "true score" (like continuous raw scores) and are generally
        S-curves or ogives.
        
        Test characteristic curves are used in IRT to determine whether
        the scales associated with two tests behave the same way.  When
        data fit the Rasch (1PL) model, the TCCs will necessarily be 
        comparable.  They will be parallel and have the same upper and lower
        asymptotes.  
        
        However, because the 2PL and 3PL models permit varying item 
        discrimination parameters, which yield different slopes per item, 
        there is no guarantee, when their item characteristic curves are
        combined to create test characteristic curves (TCCs), that the TCCs
        will be parallel.  Variation in the c-parameter also means there
        is no guarantee that all curves will have the same lower asymptote.
        Since comparability of scales is only possible when the TCCs are
        parallel, use of the 2PL and 3PL models requires selecting items
        specifically for their capacity, when combined, to yield parallel
        TCCs.  
        
        Thus, with Rasch, achieving comparability is done up front by
        using analysis of fit and other diagnostics to selecting items that 
        fit the model and thus have parallel TCCs.  With 2PL and 3PL, 
        comparability is achieved on the back end by selecting items that,
        when combined, yield parallel TCCs.
        
        This generally requires a visual inspection -- placing two or
        more TCCs side by side on a graph and seeing whether they are
        parallel.  That's what compare_tccs() is used for.
        
        compare_tccs() is usually run following build_conversion_table(),
        since it relies on one of the outputs of that function.  However,
        that output (theta with true score equivalents) can be pulled
        from any source.  Here is the expected workflow:
        
        Workflow
        --------
        
        import irt_tools as it
        
        trues = {}
        
        for test in tests:
            params = it.load_item_params(...)
            conv_tab = it.build_conversion_table(...)
            
            # Here is the conversion table
            ctab = conv_tab['ctab']
            
            # Here is the theta/true_score table, saved in the trues dict
            trues[test] = conv_tab['true']
        
        # Now, having collected multiple true_score tables, compare TCCs
        it.compare_tccs(trues, ...)
    
    Parameters
    ----------
        "trues" is a Python dictionary of test labels and the 
        "theta/true-score" dataframes associated with each test.  These
        dataframes are one of the outputs of the build_conversion_table()
        function, so this will generally (though not necessarily) be run first.
        
            trues = {'test_2015':df_2015, 'test_2016':df_2016}
                            =>  We are comparing the TCCs for the 2015 and
                                2016 versions of a given test.  The thetas
                                and true scores for each year are contained
                                in the "true score" dataframe output by 
                                build_conversion_table().
        
         ----------
         "tccs" is a nested dictionary of parameters for building the test 
         characteristic and error curves.  It is just like the tcc parameter
         in build_conversion_table() except that you specify a separate 
         dictionary for each set of test curves.  Parameters that are
         not specified fall back on the default irt_tools.TCC dictionary.
         None, in this case, tells the function to use only the defaults.
         
             tccs = {'test_2015':{'pct':True,
                                  'xlabel':'thetas for 2015'},
                     'test_2016':{'pct':Tru,
                                  'xlabel':'thetas for 2016'}}

            tccs = None     => Use default parameters
        
        For more documentation, go to:
            
            import irt_tools as it
            help(it.build_conversion_table)
            
            import matplotlib.pyplot as plt
            help(plt.plot)
            help(plt.legend)

        ----------
        "savefig" (default = 'compare_tcc.png') is the name or path to
        which the output TCC chart is to be saved.
        
            save_fig = '/my/path/compare_2015_16_tcc.png'
                            =>  Save the chart to the specified path.
            
            save_fig = 'compare_2015_16_tcc.png'
                            =>  Saves to the current working directory.

    Paste Function
    --------------
    compare_tccs(trues, 
                 savefig='compare_tccs.png', 
                 tccs=None)
    """    
    tests = trues.keys()
    tccs_ = {}
    score_lines = ['b-', 'g-', 'k-']
    err_lines = ['r-', 'y-', 'm-']
    
    # Use defaults
    if tccs is None:
        tccs_ = {}
        for i, test in enumerate(tests):
            tccs_[test] = TCC.copy()
            tccs_[test]['score_line'] = score_lines[i]
            tccs_[test]['err_line'] = err_lines[i]
            
    # Overwrite defaults with specified params
    else:
        for test in tests:
            tccs_[test] = TCC.copy()
            for k in tccs[test]:
                tccs_[test][k] = tccs[test][k]
    
    chart_tcc(trues, tccs_, savefig)
        


###############################################################################
    
def dif_stats(filename,   # [<'my/file.txt',...> => name of scored data file]
              student_id = 'Student_ID',    # [<'Student_ID', ...> => student id column label]
              group = ['Sex', {'focal':0, 'ref':1}],  # [<e.g.'Sex', {'focal':'female', 'ref':'male'}]> => column label with assignment to focal and reference]
              raw_score = 'RawScore',  # [<'RawScore',...> => raw score column label]
              items = 'All',  # [<'All', ['item1', 'item3',...]> => items for which to get stats]
              stats = 'All',  # [<'All', [see list in docs]> => desired statistics]
              strata = ('all_scores', 4),   # [(<'all_scores', int>, int) => number of raw score strata, with backup if insufficient]
              getrows = None, # [<None, {'Get':_,'Labels':_,'Rows':_}> => select rows using extract() syntax]
              getcols = None, # [<None, {'Get':_,'Labels':_,'Cols':_}> => select cols using extract() syntax]
              delimiter = '\t',   # [<',', '\t'> => column delimiter]
              ):
    """Calculate DIF stats for each in a range of items.

    Returns
    -------
        dif() returns an item by statistic Damon object with
        a column containing number of score categories. Display
        results using:

        >>>  print tabulate(dif(...).whole, 'firstrow')

    Comments
    --------
        "dif" (DIF) stands for "differential item functioning" and reflects
        the degree to which items have different difficulties for two
        groups of persons, a "focal" and a "reference" group, after
        adjusting for the ability of each person.  It is used to flag
        items that "play favorites" with student groups, e.g., that are
        easy for girls and hard for boys even though the two groups 
        otherwise have similar ability.

        There are a profusion of DIF statistics, organized mainly by whether
        they are intended for dichotomous or polytomous items.  The Rasch
        model has its own way of estimating DIF (not included in this
        function) which yields similar results. dif() supports three 
        categories of DIF statistics plus related variances, z-scores,
        chi-squares and so on.  Any number of combinations of these statistics
        have been proposed for flagging DIF items.

            'MH'    =>  Mantel-Haenszel, for dichotomous data
            'M'     =>  Mantel, for dichotomous and polytomous data
            'SMD'   =>  standardized mean difference, usually for polytomous

        Formulas are pulled from Zwick & Thayer (1996) and Wood (2011).

        A commonly used statistic is the 'Flag' statistic, which gives a code 
        for whether an item should be flagged.  ETS's a, b, c DIF flags 
        are reported numerically as 0, 1, 2.  See discussion below.

        The dif_stats() function applies only to unidimensional data.
        Multidimensional DIF can be evaluated in Damon to a limited
        degree using the "stability" statistic in conjunction with
        coord()'s seed parameters.

        dif() requires a student-by-item data file or array with a group
        membership column and a column of student raw scores.  Thus, column
        headers should contain a student id column, a group column, a raw score
        column, and a series of item columns.  Any other columns in your
        dataset should be filtered out using the getcols parameter.

    References
    ----------
        Zwick, R., Thayer, D. (Autumn, 1996).  "Evaluating the Magnitude of Differential
            Item Functioning in Polytomous Items".  Journal of Educational and
            Behavioral Statistics, Vol. 21, No. 3, pp 187-201.
            http://www.jstor.org/stable/1165267

        Wood, S. W. (2011). "Differential item functioning procedures for polytomous
            items when examinee sample sizes are small." doctoral PhD diss, University
            of Iowa, 2011.
            http://ir.uiowa.edu/etd/1110.

    Parameters
    ----------
        "filename" is the string name of a person x item file containing
        integer scores of how each student did on each item, a column
        containing test-level raw scores for each student, and a column
        assigning each student to a group.  All non-numerical cells are
        treated as missing.  All numerical scores are treated as valid.
        Numerical scores must be integers whose minimum value is zero.
        Data must be tabular and field-delimited.
        
            filename = '/path/to/my_file.txt'
                                        =>  file is 'my_file.txt'

        -----------
        "student_id' is the header label of the column containing unique
        student identifiers.

            student_id = 'Student_ID'   =>  Student identifiers are in the
                                            column labels 'Student_ID'.

        -----------
        "group" contains the header label of the group column and
        assigns one group to be "focal" and the other to be the "reference".

            group = ['Sex', {'focal':'female', 'ref':'male'}]
                                        =>  Student gender identifiers are
                                            in the column labeled 'Sex'.
                                            Students labeled "female" will
                                            be the focal group.  Students
                                            labeled "male" will be the
                                            reference group.

            Note: As is typical with DIF statistics, while there can be
            more than two groups, only two are compared at a time.
        
        -----------
        "raw_score" is the header label of the raw score column.

            raw_score = 'RawScore'      =>  Test-level student raw scores
                                            are in the column labeled
                                            'RawScore'

            
        -----------
        "items" is the list of items for which DIF statistics should be
        calculated.

            items = 'All'               =>  Calculate DIF for all items
                                            in the dataset.

            items = ['item1', 'item5']  =>  Calculate DIF for only items
                                            1 and 5.

        -----------
        "stats" is the list of DIF stats to be calculated for each
        item.  If a given statistic cannot be calculated for a given
        item, the cell is left blank.

            stats = 'All'               =>  Calculate all possible DIF
                                            statistics for all items (see
                                            list below).

            stats = ['MH_d-dif', 'MH_z', 'M_z', 'SMD_z']
                                        =>  Calculate just the Mantel-Haenszel
                                            delta-DIF (defined by ETS), the
                                            Mantel-Haenszel z statistic (both
                                            for dichotomous items), the Mantel
                                            z-statistic (for dichotomous and
                                            polytomous items), and the
                                            standardized mean difference
                                            z-statistic.

        List of available DIF-related statistics ("MH" means Mantel-
        Haenszel, "M" means Mantel, "SMD" means standardized mean difference.

        Mantel-Haenszel (dichotomous data)
            'MH_alpha'      =>  odds ratio, dich, 0 -> +inf
            'MH_dif'        =>  log-odds ratio, dich, -inf -> +inf
            'MH_d-dif'      =>  delta-DIF = -2.35*log-odds, dich, -inf -> +inf,
                                negative implies bias toward reference group.
                                (d-dif > 1.5 implies DIF)
            'MH_var'        =>  variance of MH_dif (SE = sqrt(var))
            'MH_d-var'      =>  variance of MH_d-dif
            'MH_z'          =>  absolute z-statistic (dif/sqrt(var)), z > 2.0 => p < 0.05
            'MH_pval'       =>  p-value associated with z, pval < 0.05 => significance
            'MH_chisq'      =>  chi-square = z^2.  chisq > 3.84 => p < 0.05
            'MH_chisq_pval' =>  p-value associated with chisq, pval < 0.05 => significance

        Mantel (dichotomous and polytomous data)
            'M_dif'         =>  observed - expected frequencies
            'M_var'         =>  variance of M_diff (SE = sqrt(var))
            'M_z'           =>  signed z-statistic, dif/sqrt(var), z > 2.0 => p < 0.05
            'M_pval'        =>  p-value associated with z, pval < 0.05 => significance
            'M_chisq'       =>  chi-square = z^2.  chisq > 3.84 => p < 0.05
            'M_chisq_pval'  =>  p-value associated with chisq, pval < 0.05 => significance

        Standardized mean difference (mainly for polytomous data)
            'SMD_dif'       =>  difference between reference and focal groups
            'SMD_var'       =>  variance of SMD_dif (SE = sqrt(var))
            'SMD_z'         =>  signed z-statistic, dif/sqrt(var), z > 2.0 => p < 0.05
            'SMD_pval'      =>  p-value associated with z, pval < 0.05 => significance
            'SMD_chisq'     =>  chi-square = z^2.  chisq > 3.84 => p < 0.05
            'SMD_chisq_pval'=>  p-value associated with chisq, pval < 0.05 => significance

        Other stats
            'SD'            =>  standard deviation of person scores for that item
            'SMD/SD'        =>  absolute SMD/SD > 0.25 implies DIF if SMD_chisq_pval < 0.05
            'Flag'          =>  flag a DIF item based on the rules described below. 
            'Counts'        =>  Count valid scores for each item, overall and by group.

        As mentioned, all statistics that are dependent on sample size (e.g., z,
        chi-square) will show larger values as sample size increases and their
        standard errors go to zero.  Therefore, DIF decisions should be based
        on other considerations.

        One useful rule suggested by Zwick, Thayer, and Mazzeo and used by
        ETS is as follows.  Flag DIF:

            for dichotomous items:
                Flag = 2 if:
                    'MH_d-dif' is greater than 1.5 and significantly greater than 1.0.
                Flag = 0 if:
                    'MH_d-dif' is less than 1.0 or the p-value is greater than 0.05.
                Flag = 1, otherwise.

                These correspond to ETS a, b, c DIF flags:
                    'a'=>0, 'b'=>1, 'c'=>2
 
            for polytomous items:
                Flag = 2 if:
                    'SMD/SD' is greater than 0.25 and 'M_chisq_pval' is less than 0.05.
                Flag = 0, otherwise.

                There is no flag = 1 here.

                (Note: Zwick refers to this as a Mantel-Haenszel chi-square p-value
                 but the formula resembles the polytomous Mantel chi-square p-value,
                 which is what is used here.)

        -----------
        "strata" is the number of ability strata or levels into which
        to divide student test raw scores for purposes of matching
        students of similar abilities.  If the number of strata do
        not divide evenly into the number of potential raw scores,
        the remainder are stuck in the lowest stratum.  "strata" requires
        a backup strata specification in case the primary specification
        leads to a count of one or less for a given item:

            strata = (primary, backup)

        Examples:

            strata = ('all_scores', 4)  =>  Let each possible raw
                                            score be its own stratum.
                                            This is desirable so long as
                                            the sample of persons is large
                                            enough that all cells in
                                            the resulting stratum x score
                                            table have fairly large counts.

                                            If 'all_scores' yields insufficient
                                            data for a given item, use a
                                            stratum of 4 for that item.

            strata = (20, 10)           =>  Divide the raw scores into
                                            20 strata and match students
                                            who belong to the same stratum.
                                            If this leads to insufficient data,
                                            use 10 for that item.

            Some DIF programs allow no more than five or so stratification
            levels in order to avoid insufficient counts. This degrades the
            DIF statistics a little, but not generally enough to be a problem.
            
        -----------
        "getrows" controls the rows that are loaded from the datafile,
        making it possible to filter out unneeded rows, e.g., to get a
        student subsample.  The syntax is drawn from Damon's extract()
        method and can be a bit fancy. To get a full description of
        what you can do with getrows, see:

            >>>  help(core.Damon.extract)
    
        Simple examples:
    
            getrows = None      =>  Retain all rows as they are.
                                    Non-intuitively, this really means
                                    "get all rows".
    
            getrows = {'Get':'AllExcept','Labels':'key','Rows':['row_x', 'row_y']}
                                =>  Extract all rows except those
                                    labeled 'row_x' and 'row_y'.

            getrows = {'Get':'NoneExcept','Labels':'index','Rows':[range(1, 20, 2)]}
                                =>  Extract only row 1 up to, but not
                                    including, row 20.  2 is a step parameter, and
                                    means get every other row within the range.
                                    Counting starts from 0. The 'index' parameter
                                    means 'Rows' refers to positions, not 'keys'.
                                    
        -----------
        "getcols" controls the columns that are loaded from the datafile,
        making it possible to filter out unneeded columns, e.g., data
        columns that are not items or the student raw score.  The syntax
        is drawn from Damon's extract() method and can be a bit fancy.
        To get a full description of what you can do with getcols, see:

            >>>  help(core.Damon.extract)
    
        Simple examples:
    
            getcols = None      =>  Retain all columns as they are.
                                    Non-intuitively, this really means
                                    "get all columns".
    
            getcols = {'Get':'AllExcept','Labels':'key','Cols':['col_x', 'col_y']}
                                =>  Extract all columns except those
                                    labeled 'col_x' and 'col_y'.

            getcols = {'Get':'NoneExcept','Labels':'index','Cols':[range(2, 41)]}
                                =>  Extract only columns 2 up to, but not
                                    including, 41.  Counting starts from 0.
                                    Note the 'index' parameter.

        -----------
        "delimiter" is the character used to delimit columns in
        the dataset.

            delimiter = ','             =>  File is comma-delimited.

            delimiter = '\t'            =>  File is tab-delimited.



    Examples
    --------

        [under construction]

    Paste Function
    --------------
        dif_stats(filename,   # [<'my/file.txt',...> => name of scored data file]
                  student_id = 'Student_ID',    # [<'Student_ID', ...> => student id column label]
                  group = ['Sex', {'focal':0, 'ref':1}],  # [<e.g.'Sex', {'focal':'female', 'ref':'male'}]> => column label with assignment to focal and reference]
                  raw_score = 'RawScore',  # [<'RawScore',...> => raw score column label]
                  items = 'All',  # [<'All', ['item1', 'item3',...]> => items for which to get stats]
                  stats = 'All',  # [<'All', [see list in docs]> => desired statistics]
                  strata = ('all_scores', 4),   # [(<'all_scores', int>, int) => number of raw score strata, with backup if insufficient]
                  getrows = None, # [<None, {'Get':_,'Labels':_,'Rows':_}> => select rows using extract() syntax]
                  getcols = None, # [<None, {'Get':_,'Labels':_,'Cols':_}> => select cols using extract() syntax]
                  delimiter = '\t',   # [<',', '\t'> => column delimiter]
                  )

    """
    args = locals()

    from dif import dif_stats

    return dif_stats(**args)
    
    
###############################################################################

def classify(ss_file,  # [<'my/scorefile.csv'> scores, errors, counts]
             cut_file, # [<'my/cutfile.csv'> => cut-points and labels]
             grade,    # [<int> => grade as a number]
             domain,    # [<str> => domain label]
             at_cuts=False,  # [<bool> => report stats at each cut-point]
             load_ss=None,  # [<func> => function for loading ss_file]
             load_cuts=None,  # [<func> => function for loading cut_file]
             sep=',' # [<str> column delimiter]
             ):
    """Calculation classification accuracy and consistency statistics.
    
    Returns
    -------
        classify() returns a nested dictionary giving classification
        and accuracy statistics for each cut-point and overall, e.g.:
            
            {'summary':{'acc':0.68,
                        'consist':0.56,
                        'kappa':0.55},
             'Basic':{'acc':0.60,
                      'consist':0.43,
                      'kappa':0.51},
             'Proficient':{'acc':0.73,
                           'consist':0.69,
                           'kappa':0.65}
             }

    Comments
    --------
        classify() answers the questions:
            
            1)  How accurate is the test overall in assigning students to their 
                "true" performance level?
            
            2)  How consistent is that assignment, i.e., how often would
                students be assigned to the same performance level on
                repeated testing?
            
            3)  What is the accuracy and consistency with regard to a
                particular performance level cut-point?
        
        The psychometric literature offers a variety of methods to answer
        these questions, some based on classical test theory, some on item 
        response theory, with variations that differ according to the
        item parameter estimation program.  classify() uses the IRT-based 
        method proposed by Lawrence Rudner (2001).  In this method, the
        standard error associated with each scale score, which is assumed to 
        be normally distributed, is used to calculate the probability that
        a student with that scale score will fall into each of the
        performance levels.  The resulting probabilities are used to 
        compute the desired accuracy, consistency, and Cohen's kappa
        statistics (Cohen's kappa is an alternative measure of consistency).
        The method is fast, reliable, and (relatively) easy to understand.
        
        Note that Cohen's kappa as computed here is calculated using a
        traditional unweighted formula, but the results look quite different 
        from those calculated  using the CTT-based Livingston & Lewis method.  
        I believe that's because the L&L method applies the kappa formula to 
        a hypothetical test whose length is adjusted to account for test 
        reliability.  In the IRT-based Rudner method, there is no need to 
        create such a hypothetical test.
        
    Parameters
    ----------
        "ss_file" is a path or filename containing fields for grade, domain,
        raw score, scale score, standard error of measure, performance level,
        and count of students.  Thus for a given grade with a test of 25 
        items, you will have 25 rows.  Each row will have a raw score
        from 0 to 25, the corresponding scale score and standard error,
        and the number of students who got that score.  The raw score,
        scale score, and standard error of measurement will be an output
        of your IRT package.  You may have to get the counts yourself.
        
        Here is an example:
        
            Grade   Domain  RS	SS	SEM	PL	N
            0	      L	0	220	117	1	166
            0	      L	1	245	110	1	174
            0	      L	2	255	105	1	417
            0	      L	3	262	101	1	743
            etc.
            
            Gotcha
            ------
            Loading grades as a mix of string and int ('k', 1, 2) creates
            problems.  Either make them all int (0, 1, 2) or all string
            ('k', 'g1', 'g2').
        
        This file format is here termed the "EDS format", since it is
        the format we use at Educational Data Sytems, but classify() supports 
        other file formats if you're willing to write the requisite
        load function (see "load_ss" below).
        
        -------------
        "cut_file" is a file of cut-scores, the scale scores used to
        demarcate adjoining performance level.  Here is an example in
        EDS format:
            
            Grade   Domain  B	EI	I	EA	A	Max
            0       L	      220	362	409	455	502	570
            0       S	      140	353	405	457	509	630
            0       R	      220	232	300	380	468	570
            0       W	      220	255	327	383	430	600
            1       L	      220	362	409	455	502	570
            1       S	      140	353	405	457	509	630
            1       R	      220	357	393	468	570	570
            1       W	      220	372	406	444	518	600
            etc.
        
            Gotcha
            ------
            Loading grades as a mix of string and int ('k', 1, 2) creates
            problems.  Either make them all int (0, 1, 2) or all string
            ('k', 'g1', 'g2').
          
        Each scale score is the lowest scale score for that performance
        level.  So 220 is the lowest possible score in kindergarten 
        Listening as well as the lowest in the 'B'asic performance level.
        
        Note the "Max" column.  This is the maximum possible scale score,
        at the top of the highest performance level.
        
        As with ss_file, you can read cut files stored in alternative
        formats by specifying your own load function (see "load_cuts"
        below).
        
        -------------
        "grade" is an integer representing the grade (kindergarten is "0").
        If your load method doesn't need grade, you can set grade=None.
        
        -------------
        "domain" is a string representing the content domain. If your load 
        method doesn't need domain, you can set domain=None.
        
        -------------
        "at_cuts" <bool>, if True, adds a separate set of accuracy,
        consistency, and kappa statistics for each cut-point.
        
        -------------        
        "load_ss" allows you to specify a (function, kwargs) tuple for 
        loading a file of scale scores and frequencies that are in a
        different format than the "EDS" format assumed here.  Whatever
        the inputs, the output should be a Pandas Dataframe consisting
        minimally of:
            
            RS	SS	SEM	N
            0	220	117	166
            1	245	110	174
            2	255	105	417
            3	262	101	743
            etc.

        in which the RS column (Raw Score) is the index.  Open the classify.py
        module to see two examples of load functions:  load_ss() and
        load_ss_eds().  If your function returns similar dataframes, classify()
        can analyze them.
        
            Examples:
                
                
                load_ss = None      =>  Use the default classify.load_ss_eds()
                                        load function.
                
                import damon1.classify as cl
                load_ss = (cl.load_ss, {'filename':_, 'names':_, etc})
                                    =>  Use the classify.load_ss() function
                                        with the accompanying keyword
                                        arguments.  (Positional arguments
                                        won't work.)
                
                load_ss = (my_func, {my_kwargs})
                                    =>  To load your file, use my_func
                                        and accompanying keyword arguments.

        -------------
        "load_cuts" allows you to specify a (function, kwargs) tuple for
        loading a file of cut-scores (with maximum score) that are in a
        different format that the "EDS" format assumed here.  Whatever
        the inputs, the output should be a pandas Series consisting 
        minimally of:
            
            B	EI	I	EA	A	Max
            220	362	409	455	502	570
            
        that is, a scale score cut-point giving the start of each 
        performance level cut-point, plus a maximum scale score.
        Open the classify.py module for an example.
            
        -------------
        "sep" is the column delimiter for both the ss_file and cut_file
        files.
        
            sep = ','       =>  Both files are comma-delimited
            
            sep = '\t'      =>  Both files are tab-delimited

    References
    ----------
        Cohen's kappa. (2016, October 4). In Wikipedia, The Free Encyclopedia. 
        Retrieved 13:39, October 4, 2016, from 
        https://en.wikipedia.org/w/index.php?title=Cohen%27s_kappa&oldid=742569319

        Livingston, Samuel A., & Lewis, Charles (1993). Estimating the 
        Consistency and Accuracy of Classifications Based on Test Scores.
        Education Testing Service, Research Report.
        https://www.ets.org/Media/Research/pdf/RR-93-48.pdf

        Rudner, Lawrence M. (2001). Computing the expected proportions 
        of misclassified examinees. Practical Assessment, Research &
        Evaluation, 7(14). 
        http://PAREonline.net/getvn.asp?v=7&n=14.

    Paste Function
    --------------
        classify(ss_file,  # [<'my/scorefile.csv'> scores, errors, counts]
                 cut_file, # [<'my/cutfile.csv'> => cut-points and labels]
                 grade,    # [<int> => grade as a number]
                 domain,    # [<str> => domain label]
                 at_cuts=False,  # [<bool> => report stats at each cut-point]
                 load_ss=None,  # [<func> => function for loading ss_file]
                 load_cuts=None,  # [<func> => function for loading cut_file]
                 sep=',' # [<str> column delimiter]
                 )
    """
    import damon1.classify as cl
    
    # Get function to load raw and scale scores
    if load_ss is None:
        load_ss = cl.load_ss_eds
        load_ss_args = {'filename':ss_file, 'grade':grade, 'domain':domain,
                        'names':cl.SS_COLS_EDS, 'usecols':cl.SS_COLS_EDS,
                        'index_col':[cl.GRADE, cl.DOMAIN], 'sep':sep}
    else:
        load_ss = load_ss[0]
        load_ss_args = load_ss[1]
        
    # Get function to load cutpoints
    if load_cuts is None:
        load_cuts = cl.load_cuts_eds
        load_cuts_args = {'filename':cut_file, 'grade':grade, 'domain':domain,
                          'sep':sep}
    else:
        load_cuts = load_cuts[0]
        load_cuts_args = load_cuts[1]
    
    # Load scores and cuts
    ss_se = load_ss(**load_ss_args)
    cuts = load_cuts(**load_cuts_args)
    
    # Get summary stats (across all cutpoints)
    out = {}
    out['summary'] = cl.acc_consist(ss_se, cuts)
    
    # Get stats at each cut
    if at_cuts:
        max_ = len(cuts) - 1
        
        for i in range(1, max_):
            label = cuts.index[i]
            cuts_ = cuts.iloc[[0, i, max_]]
            out[label] = cl.acc_consist(ss_se, cuts_)
       
    return out




###############################################################################

def fit_for_celdt(scored, thetas, params, skipna=True):
    """Calculate item fit statistics using a formula specific to the EDS 
    CELDT contract.
    
    This procedure is based on the following explanation in the CELDT Tech
    Report and an email by Brad Mulder at ETS.

    8.6.1	IRT Model Fit Analyses. Because the CELDT makes use of item 
    response theory (IRT) to equate successive forms of the test, evaluating 
    the extent to which the model is appropriate for the CELDT data is an 
    important part of evaluating the validity of the test. Goodness-of-fit 
    statistics were computed for each item to examine how closely an item’s 
    data conform to the item response models. For each item, a comparison 
    of the observed proportions of examinees in each response category with 
    the expected proportion based on the model parameters yields a 
    chi-square-like goodness-of-fit test (with degrees of freedom equal 
    to mj -1, one less than the number of response categories for an item) 
    for each item, the Q statistic.
    
    This statistic is directly dependent on sample size, and for the large 
    samples of the CELDT, the Q values need to be modified to take this 
    dependency into account. Consistent with past practice, we calculated 
    a Z statistic as
    
    Z[j] = (Q[j] - df[Qj]) / sqrt(2(df))
    
    where df = mj -1.
    
    This statistic is useful for flagging items that fit relatively poorly. 
    Zj is sensitive to sample size, and cutoff values for flagging an item 
    based on Zj have been developed and were used to identify items for the 
    item review. The cutoff value is (N/1,500 x 4) for a given test, where 
    N is the sample size.
    
    Brad's explanation:
        
    Here are the steps for evaluating fit:
     
    1) Multilog provides [from Speaking K-2]:
            
      OBSERVED AND EXPECTED COUNTS/PROPORTIONS IN
      CATEGORY(K):  1      2
      OBS. FREQ.  3721  66165
      OBS. PROP.  0.0532 0.9468
      EXP. PROP.  0.0651 0.9349
      
    2) Get the expected frequency from the expected proportions.
            
    3) Calculate Pearson chi-square using only those 4 counts (I get 161.4102 
       for item 5 in Listening K-2)
            
    4) Calculate fit value=(_PCHI_-DF_PCHI)/(sqrt(2*DF_PCHI)) (I get 113.4272)
            
    5) Calculate cutoff=N/1500*4; (I get 186.3627)
            
    6) If fit>=cutoff then Fit_Flag='Y';
     
    Because this isn’t evaluating anything beyond that the item calibration 
    recovering the number of examinees earning each score point, it isn’t 
    going to match something that is picking up deviations in particular 
    ranges of theta. We’d have replaced them were we using them to choose 
    items.
     
    Brad
    
    Mark Moulton's Opinion:  I don't believe this procedure offers a sound
    measurement of fit:
        
        1)  The Multilog expected values (probabilities) assume complete
            data.  When the observed array contains missing cells, this
            the percentage of missing biases the fit statistics.
            
        2)  In my understanding of the Rasch-based joint maximum likelihood
            algorithm, iteration occurs until the sum of observed values
            equals the sum of expected values.  If jMetrik or Multilog
            use the same iterative approximations, we would then expect
            the difference between the observed and expected sums
            to approach zero.  Misfit, in that case, would be driven by 
            number of iterations.
        
        3)  These fit statistics don't look anything like those calculated
            by jMetrik or Rasch programs.
            
    Therefore, these fit stats are reported at the request of the client to 
    maintain continuity with the past.  They are not used to make decisions 
    about items.
    
    """
    
    thetas_ = thetas.loc[:, THETA_SS].values
    items = scored.columns
#    items = ['CES00818_14']
#    items = ['CEL00899_5']
#    items = ['CEL00526_2']
    fit = {'fit':[], 'flag':[], 'cut':[], 'xsq':[], 'n':[]}
    cat0 = []
    cat1 = []
    
    for item in items:        
        obs = scored.loc[:, item]
        obs_freq = obs.value_counts().sort_index()
        
        pcats = estimates_for_item(thetas_, params, item, pcats=True)[2]
        pcats = pd.DataFrame(pcats, index=thetas.index.values)
        n = np.sum(obs_freq) #if skipna else len(thetas_)     
        
        if skipna is True:
            nix = obs.isnull()
            pcats.iloc[nix.values, :] = np.nan

        if skipna is True:
            exp_freq = pcats.sum(0, skipna).sort_index()
        else:
            exp_freq = (pcats.mean(0, skipna) * n).sort_index()
#        print '\n', item
#        print 'first=\n', exp_freq
#        
#        n_ = n
#        ef = n_ * np.array([0.2625, 0.7375])
#        exp_freq[0] = ef[0]
#        exp_freq[1] = ef[1]
#        print 'second=\n', exp_freq
#        sys.exit()
        
        
        
        
        xsq = np.sum((obs_freq - exp_freq)**2 / exp_freq)
        df = len(obs_freq) - 1

        fit_ = (xsq - df) / np.sqrt(2 * df)
        cut = (n / 1500.0) * 4.0
        flag = fit_ >= cut
        fit['xsq'].append(xsq)
        fit['n'].append(n)
        fit['fit'].append(fit_)
        fit['cut'].append(cut)
        fit['flag'].append(flag)
        
        
        cat0.append(pcats.mean(0, skipna).sort_index()[0])
        cat1.append(pcats.mean(0, skipna).sort_index()[1])
        
#        print 'obs=\n', obs_freq
#        print '\nexp=\n', item, pcats.mean(0, skipna).sort_index()
#        print '\nxsq=', xsq
#        print 'df=', df
#        print 'n=', n
#        print 'cut=', cut

    results = pd.DataFrame(fit, index=items).round(2)
    
    results['cat0'] = np.around(cat0, 3)
    results['cat1'] = np.around(cat1, 3)
    
    return results
    

###############################################################################

def rasch_fit(scored, fit='in', samp=300, n_samps=5, cut=1.2):
    """Calculate item fit statistics using a Rasch model.
    """
    import damon1.core as dmn
    import damon1.tools as dmnt
    
    t = {'in':['fac1_infit', 'infit'],
         'out':['fac1_outfit', 'outfit']}
    
    infits = []
    
    for i in range(n_samps):
        scored_ = scored.sample(samp, replace=False, random_state=i, axis=0)
        d = dmn.Damon(scored_, 'dataframe', #'RCD_dicts_whole',
                      validchars=['All', range(0, 10), 'Num', 'Guess'],
                      verbose=None)

        d.extract_valid(5, 5, 0.1)
        
        # Define groups.  Each non-dich item gets its own group (GPC)
        vcdict = d.extract_valid_out['validchars'][1]
        groups = {'dich':[]}
        gpc = 0
        
        if isinstance(vcdict, list):
            groups['dich'] = dmnt.getkeys(d, 'Col', 'Core')
        elif isinstance(vcdict, dict):
            for item in vcdict:
                if len(vcdict[item]) == 2:
                    groups['dich'].append(item)
                else:
                    groups['gpc'+str(gpc)] = [item]
                    gpc += 1
    
#        print 'vcdict=\n', vcdict
#        print 'groups=\n', groups
#        
#        sys.exit()
    
        

        d.rasch(['key', groups])
        
#        print d.to_dataframe(d.rasch_out[t[fit][0]])
        
        
        
        infits.append(d.to_dataframe(d.rasch_out[t[fit][0]]))
    
    df = pd.concat(infits, axis=1)
    mean_fits = df.mean(axis=1).to_frame(t[fit][1])
    flag = mean_fits.loc[:, t[fit][1]] >= cut
    mean_fits['flag'] = flag

    return mean_fits




































    

    
    