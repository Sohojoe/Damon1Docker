# -*- coding: utf-8 -*-
# opts.py

"""tests.py
unit tests for tools and methods related to Damon.

Copyright (c) 2016, Mark H. Moulton

"""
#234567890123456789012345678901234567890123456789012345678901234567890123456789

import os
import sys

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.ma as npma
import pandas as pd
np.seterr(all='ignore')

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# If damon1 is not stored in your Python's site-packages folder, specify path
#damon1_path = '/Users/markhmoulton/Google Drive/'
#if damon1_path:
#    sys.path.append(damon1_path)


import damon1
import damon1.core as core
import damon1.utils as utils
import damon1.tools as tools
import damon1.tester as ut


TEMP_PATH = (os.path.dirname(os.path.realpath(__file__)) + 
             '/test_asserts/temp/')

# Arguments available to set up data and run methods for Damon
CREATE_ARGS = {'nfac0':10, 'nfac1':8, 'ndim':2, 'seed':1,
               'facmetric':[4, -2], 'noise':None,
               'validchars':['All', ['All'], 'Num'],
               'mean_sd':None, 'p_nan':0.05, 'nanval':-999,
               'condcoord_':None, 'nheaders4rows':1, 'nheaders4cols':1,
               'extra_headers':2, 'input_array':None, 'apply_zeros':None,
               'output_as':'Damon', 'outfile':None, 'delimiter':None,
               'bankf0':None, 'bankf1':None, 'verbose':None}
          
def setup_damon(create_args, meths=None, output='data'):
    """Sets up a Damon object with desired method attributes for unit
    testing.

    Parameters
    ----------
    create_args : dict
        Dictionary of Damon.create_data arguments for which you would
        like to override the CREATE_ARGS defaults. This is used to
        create the Damon object to which Damon methods can be applied.
    meths : None, [('meth', {meth_args}),...]
        List of tuples containing the string name of a method and
        an {arg:parameter} dict.  Unspecified parameters fall back
        on the default value in the core.Damon method, if given.
    output : {'data', 'All', str(output)} (optional)
        create_data() returns a dict containing 'data', 'model', 'anskey',
        'fac0coord', and 'fac1coord'.  `output` specifies which of
        these you want. 'All' returns the dict.
        
    """        
    # Create the Damon object
    cargs = CREATE_ARGS.copy()
    for key in create_args:
        cargs[key] = create_args[key]
    d = core.create_data(**cargs)

    # Run Damon methods
    if meths is not None:
        for meth in meths:
            getattr(d['data'], meth[0])(**meth[1])

    if output == 'All':
        return d
    else:
        return d[output]
    
          
def test_create_data(check='run', asserts=np.array_equal, printout=True):
    "Test damon1.create_data()"

    def create_data(**kwargs):
        x = core.create_data(**kwargs)['data']
        if kwargs['output_as'] == 'datadict':
            out = x['coredata']
        elif kwargs['output_as'] == 'Damon':
            out = x.coredata
        else:
            out = x
        return out
    
    out = ut.test(create_data,
            args={'nfac0':[10],
                  'nfac1':[8],
                  'ndim':[2],
                  'seed':[1],
                  'facmetric':[[4, -2], [1, 0]],
                  'noise':[None, 1.0, {'Rows':1.0, 'Cols':{3:4.0, 4:8.0}},
                             {'Rows':{3:4.0, 4:8.0},
                              'Cols':{3:4.0, 4:8.0}}],
                  'validchars':[['All', ['All'], 'Num'],
                                  ['All',['All']],
                                  ['All',['0 -- 3']],
                                  ['All',[0,1,2,3]],
                                  ['All',['a','b','c']],
                                  ['All',[0,1]],
                                  ['All',['0.0 -- 1.0']],
                                  ['All',['0.0 -- ']],
                                  ['Cols',{1:['0 -- 3'],
                                           2:['0.0 -- 3.0'],
                                           3:['a','b','c','d'],
                                           4:[0,1],
                                           5:['-2 -- 2'],
                                           6:['0.0 -- '],
                                           7:['-3.5 -- 9.5'],
                                           8:['All']}]],
                  'mean_sd':[None,
                               ['All', [50,25]],
                               ['All', [0,1]],
                               ['Cols', {1:[5, 2.5],
                                         2:[0, 1],
                                         3:[-10, 2.5],
                                         4:[0, 1],
                                         5:[0, 1],
                                         6:[0, 1],
                                         7:[0, 1],
                                         8:[0, 1]}]],
                  'p_nan':[0.05],
                  'nanval':[-999],
                  'condcoord_':[None],
                  'nheaders4rows':[1],
                  'nheaders4cols':[1],
                  'extra_headers':[0],
                  'input_array':[None],
                  'apply_zeros':[None],
                  'output_as':['Damon', 'datadict', 'array', 'textfile'],
                  'outfile':[TEMP_PATH + 'test_create_data.csv'],
                  'delimiter':[','],
                  'bankf0':[None],
                  'bankf1':[None],
                  'verbose':[None]},
            check=check,
            asserts=asserts,
            printout=printout
            )
    return out
            

def test_TopDamon(check='run', asserts=ut.damon_equal, printout=True):
    "Test TopDamon function."

    def setup(args):
        setup_damon(args)
        filename = TEMP_PATH + 'a_data_test_TopDamon.txt'
        return filename

    outfile = TEMP_PATH + 'test_TopDamon.txt'
    validchars = ['All', ['a', 'b', 'c']]
    delimiter = '\t'

    # To create data d.
    d_args = {'validchars':validchars, 'nheaders4rows':1, 'nheaders4cols':1,
              'extra_headers':0, 'output_as':'Damon_textfile',
              'outfile':outfile, 'delimiter':delimiter}

    # To create data e
    e_args = {'validchars':validchars, 'nheaders4rows':2, 'nheaders4cols':2,
              'extra_headers':2, 'output_as':'Damon_textfile',
              'outfile':outfile, 'delimiter':delimiter}
    
    # run the unit tests given 'd' data
    print('\ntest for d')
    d = ut.Setup('d', setup, [d_args])    
    x_d = ut.test(core.TopDamon,
                  args={'data':[d],
                        'recode':[None],
                        'collabels':[[1, 0, 'S60']],
                        'getcols':[{'Get':'AllExcept',
                                    'Labels':'key',
                                    'Cols':[3, 4]}, None],
                        'labelcols':[['id', 7, 5], 3, None],
                        'rename':[{8:28}, None],
                        'key4rows':[['id', 'S3'], ['Auto'],
                                    ['id', 'S3', 'warn_dups']],
                        'getrows':[{'Get':'AllExcept',
                                    'Labels':1,
                                    'Rows':['0', '1', '2']}, None],
                        'validchars':[validchars],
                        'delimiter':[delimiter],
                        'pytables':[None],
                        'verbose':[None]},
                  check=check,
                  asserts=asserts,
                  suffix='d',
                  printout=printout
                  )
    
        
    # run the unit tests given 'e' data
    print('test for e')
    e = ut.Setup('e', setup, [e_args])
    x_e = ut.test(core.TopDamon,
                  args={'data':[e],
                        'recode':[None],
                        'collabels':[[2, 0, 'S4']],
                        'getcols':[{'Get':'AllExcept',
                                    'Labels':'key',
                                    'Cols':[3, 4]}, None],
                        'labelcols':[['id', 7, 5], 3, None],
                        'rename':[{8:28}, None],
                        'key4rows':[['id', 'S3'], ['Auto'],
                                    ['id', 'S3', 'warn_dups']],
                        'getrows':[{'Get':'AllExcept',
                                    'Labels':1,
                                    'Rows':['0', '1', '2']}, None],
                        'validchars':[validchars],
                        'delimiter':[delimiter],
                        'pytables':[None],
                        'verbose':[None]},
                  check=check,
                  asserts=asserts,
                  suffix='e',
                  printout=printout
                  )

    return {'d':x_d, 'e':x_e}
                 

def test_Damon(check='run', asserts=ut.damon_equal, printout=True):
    "Test Damon's __init__() method."
    outfile = TEMP_PATH + 'test_Damon.txt'
     
    def setup(args, format_):
        args['output_as'] = format_
        if format_ == 'textfile':
            args['outfile'] = TEMP_PATH + 'test_Damon.txt'
            if args['delimiter'] is None:
                args['delimiter'] = '\t'
            setup_damon(args)
            return TEMP_PATH + 'a_data_test_Damon.txt'
        else:
            d = setup_damon(args)
            return d

    # Args needing a different setup are permuted outside test()
    formats = ['textfile', 'datadict', 'array', 'dataframe', 'Damon']
    vc_dict = {str(i):[0, 1, 2] for i in range(1, 5)}
    vc2 = {str(i):['a', 'b', 'c'] for i in range(5, 9)}
    vc_dict.update(vc2)
    validchars = [['All', ['a', 'b', 'c']],
                  ['All', [0, 1, 2], 'Num'],
                  ['Cols', vc_dict]]
    
    nheaders4rows = [1]
    nheaders4cols = [1]
    delimiter = [',', '\t']
    outs = {}

    i = 0
    for a in formats:
        for b in validchars:
            for c in nheaders4rows:
                for d in nheaders4cols:
                    for e in delimiter:
                        cargs = {'validchars':b,
                                 'nheaders4rows':c,
                                 'nheaders4cols':d,
                                 'delimiter':e}
                        data = ut.Setup(str(i), setup, [cargs, a])
                        
                        if printout:
                            print(str(tuple([a, b, c, d, e])))

                        
                        x = ut.test(core.Damon,
                                    {'data':[data],
                                     'format_':[a],
                                     'workformat':['RCD', 'RCD_dicts_whole'],
                                     'validchars':[b],
                                     'nheaders4rows':[c],
                                     'key4rows':[0],
                                     'rowkeytype':['S60'],
                                     'nheaders4cols':[d],
                                     'key4cols':[0],
                                     'colkeytype':['S60'],
                                     'check_dups':[None],
                                     'dtype':[[object, None],
                                              ['S10', None],
                                              ['S60', 8]],
                                     'nanval':[-999],
                                     'missingchars':[None],
                                     'miss4headers':[None],
                                     'recode':[None],
                                     'cols2left':[None],
                                     'selectrange':[None],
                                     'delimiter':[e],
                                     'pytables':[None],
                                     'verbose':[None]},
                                    check=check,
                                    asserts=asserts,
                                    suffix=str(i),
                                    printout=printout)
                        outs[str(i)] = x
                        i += 1
    return outs

                                 
def test_merge_info(check='run', asserts=np.array_equal, printout=True):
    "Test Damon's merge_info() method."

    def setup(args):
        d = setup_damon(args, output='All')
        return d
    
    def merge_info(data_info, target_axis, get_validchars):
        d = data_info['data']            
        info = data_info['anskey']
        d.merge_info(info, target_axis, get_validchars)
        d = core.Damon(d.merge_info_out, 'datadict', 'whole', verbose=None)
        return d.whole

    args = {'validchars':['All', ['a', 'b', 'c']]}
    data_info = ut.Setup('data', setup, [args])

    x = ut.test(merge_info,
                {'data_info':[data_info],
                 'target_axis':['Row', 'Col'],
                 'get_validchars':[None, 'ValidResp']},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x


def test_extract_valid(check='run', asserts=np.array_equal, printout=True):
    "Test Damon's extract_valid() method."

    def setup(args):
        d = setup_damon(args)
        return d

    def extract_valid(data, **kwargs):
        d = data
        d.extract_valid(**kwargs)
        d = core.Damon(d.extract_valid_out, 'datadict', 'whole', verbose=None)
        return d.whole

    args = {'validchars':['All', [0, 1]]}
    d = ut.Setup('d', setup, [args])

    x = ut.test(extract_valid,
                {'data':[d],
                 'minperrow':[3, 0.30],
                 'minpercol':[3, 0.30],
                 'minsd':[None, 0.001],
                 'rem_rows':[None, ['1']],
                 'rem_cols':[None, ['1']],
                 'iterate':[True, False]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
    

def test_score_mc(check='run', asserts=np.array_equal, printout=True):
    "Test Damon's score_mc() method."

    def setup(args):
        d_ak = setup_damon(args, output='All')
        return d_ak

    def score_mc(d_ak, **kwargs):
        d = d_ak['data']
        ak_ = d_ak['anskey']
        keys = tools.getkeys(ak_, 'Row', 'Core')
        correct = ak_.core_col['Correct']
        ak = ['Cols', dict(zip(keys, correct))]
        d.score_mc(ak, **kwargs)
        d = core.Damon(d.score_mc_out, 'datadict', 'whole', verbose=None)
        return d.whole

    def getargs(validchars, cargs):
        args = cargs.copy()
        args['validchars'] = validchars
        return args

    args_0 = {'validchars':['All', ['a', 'b', 'c']],
              'nheaders4rows':2,
              'extra_headers':2}
    dak_0 = ut.Setup('All', setup, [args_0])

    args_1 = {'validchars':['Cols', {str(k):['a', 'b', 'c']
                                     for k in range(2, 10)}],
              'nheaders4rows':2,
              'extra_headers':2}
    dak_1 = ut.Setup('Cols', setup, [args_1])

    x = ut.test(score_mc,
                {'d_ak':[dak_0, dak_1],
                 'report':[None, ['All'], ['RowScore', 'ColScore', 'RowFreq',
                                           'ColFreq', 'MostFreq', 'AnsKey',
                                           'MatchKey', 'PtBis']],
                 'getrows':[{'Get':'AllExcept', 'Labels':'key', 'Rows':[None]},
                            {'Get':'NoneExcept', 'Labels':1, 'Rows':['0']}],
                 'usecols':[{'Scores':'All', 'Freqs':'All'},
                            {'Scores':'All', 'Freqs':'Scored'},
                            {'Scores':'Scored', 'Freqs':'All'},
                            {'Scores':'Scored', 'Freqs':'Scored'}],
                 'score_denom':['All', 'NonMissing'],
                 'nanval':[-999.]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x


def test_rasch(check='run', asserts=np.array_equal, printout=True):
    "Test Damon's rasch() method."

    def setup(args):
        d = setup_damon(args)
        return d

    def rasch(data, **kwargs):
        d = data
        bankfile = TEMP_PATH + 'ibank.pkl'

        # Build bank first
        if kwargs['anchors'] is not None:
            try: os.remove(bankfile)
            except: pass
            kwargs_ = kwargs.copy()
            kwargs_['anchors'] = None
            d.rasch(**kwargs_)
            d.bank(bankfile)

        d.rasch(**kwargs)
        d = core.Damon(d.rasch_out['estimates'], 'datadict', 'whole',
                       verbose=None)
        return d.whole

    args = {'nfac0':20, 'nfac1':10, 'facmetric':[1, 0.001], 'noise':0.5,
            'nheaders4cols':2, 'extra_headers':{'0':0.50, '1':0.50}}

    args_0 = args.copy()
    args_0['validchars'] = ['All', [0, 1], 'Num']
    d_0 = ut.Setup('0', setup, [args_0])

    args_1 = args.copy()
    args_1['validchars'] = ['All', [0, 1, 2], 'Num']
    d_1 = ut.Setup('1', setup, [args_1])

    args_2 = args.copy()
    args_2['validchars'] = ['Cols', {str(i):[0, 1] for i in range(1, 11)}]
    d_2 = ut.Setup('2', setup, [args_2])
     
    x = ut.test(rasch,
                {'data':[d_0, d_1, d_2],
                 'groups':[None,
                           {'row':1},
                           ['key', {'group0':[str(i) for i in range(1, 6)],
                                   'group1':[str(i) for i in range(6, 11)]}],
                           ['index', {'group0':range(0, 5),
                                     'group1':range(5, 10)}]
                           ],
                 'anchors':[None, {'Bank':TEMP_PATH + 'ibank.pkl',
                                   'row_ents':[None],
                                   'col_ents':['All']}],
                 'runspecs':[[0.001, 20]],
                 'minvar':[0.001],
                 'maxchange':[10],
                 'labels':[{'row_ents':'Person', 'col_ents':'Item'}]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
                           
                           
def test_coord(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's coord() method."
    
    def setup(*args):
        d = setup_damon(*args)
        return d

    def coord(data, **kwargs):
        d = data
        bankfile = TEMP_PATH + 'ibank.pkl'

        if kwargs['anchors'] is not None:
            try: os.remove(bankfile)
            except: pass
            kwargs_ = kwargs.copy()
            kwargs_['anchors'] = None
            d.coord(**kwargs_)
            d.bank(bankfile)
        d.coord(**kwargs)
        d = core.Damon(d.coord_out['fac0coord'], 'datadict', 'RCD',
                       verbose=None)
        return d.coredata

    d = ut.Setup('d', setup, [{'validchars':['All', ['All']]},
                              [('standardize', {})]])
    e = ut.Setup('e', setup, [{'validchars':['All', [0, 1, 2, 3]]},
                              [('standardize', {})]])

    # Seed parameters
    c_ents = ['1', '3', '5', '7']
    seed_0 = {'MinR':0.90,
              'MaxIt':[3, 10],
              'Facet':1,
              'Stats':['Acc','Stab'],
              'Group1':{'Get':'NoneExcept', 'Labels':'key',
                        'Entities':c_ents},
              'Group2':{'Get':'AllExcept', 'Labels':'key',
                        'Entities':c_ents}}
    r_ents = c_ents
    seed_1 = {'MinR':0.90,
              'MaxIt':10,
              'Facet':0,
              'Stats':['Obj'],
              'Group1':{'Get':'NoneExcept', 'Labels':'key',
                        'Entities':r_ents},
              'Group2':{'Get':'AllExcept', 'Labels':'key',
                        'Entities':r_ents}}
    bankfile = TEMP_PATH + 'ibank.pkl'
    
    x = ut.test(coord,
                {'data':[d, e],
                 'ndim':[[[2]], [range(1, 5), 'Acc'],
                         [range(1, 5), 'search', 'homogenize']],
                 'runspecs':[[0.0001, 10]],
                 'seed':['Auto', seed_0, seed_1],
                 'homogenize':[None, {'ApplyAncs':True}],
                 'anchors':[None,
                            {'Bank':bankfile,
                             'Facet':1,
                             'Coord':'ent_coord',
                             'Entities':['All'],
                             'Refresh_All':False}],
                 'quickancs':[None],
                 'startercoord':[None],
                 'pseudomiss':[None, True],
                 'miss_meth':['IgnoreCells'],
                 'solve_meth':['LstSq'],
                 'solve_meth_specs':[None],
                 'condcoord_':[None, {'Fac0':'Orthonormal', 'Fac1':None}],
                 'weightcoord':[True],
                 'jolt_':[None],
                 'feather':[None]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x


def test_sub_coord(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's sub_coord() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def sub_coord(data, **kwargs):
        d = data
        d.sub_coord(**kwargs)
        return d.sub_coord_out['estimates']['coredata']

    cargs = {'nheaders4cols':2,
             'extra_headers':{'0':0.50, '1':0.50}}
    d = ut.Setup('d', setup, [cargs,
                              [('standardize', {})]])

    x = ut.test(sub_coord,
                {'data':[d],
                 'subspaces':[{'row':1},
                              ['key', {'0':['1', '2', '3', '4'],
                                       '1':['5', '6', '7', '8']}]],
                 'coord_subs':[{'All':{'ndim':[[2]]}}],
                 'coord_resids':[{'All':{'ndim':[[1]]}}],
                 'unique_weights':[{'0':'Auto', '1':'Auto'}, {'All':0.50}],
                 'share_if':[{'targ_<':30, 'pred_>': 4}],
                 'min_rel':[0.02],
                 'rpt_optimal':[None, True]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)

    return x
                

def test_base_est(check='run', asserts=ut.allclose, printout=True):
    "test Damon's base_est() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def base_est(data, **kwargs):
        d = data
        d.base_est(**kwargs)
        return d.base_est_out['coredata']

    # Set up two scenarios
    args_0 = {'validchars':['All', ['All']],
              'mean_sd':['Cols', 'Refer2VC']}
    d_0 = ut.Setup('d_0', setup, [args_0,
                                  [('standardize', {}),
                                   ('coord', {'ndim':[[2]]})]])

    args_1 = {'validchars':['All', [0, 1, 2, 3, 4]]}
    d_1 = ut.Setup('d_1', setup, [args_1,
                                  [('standardize', {}),
                                   ('coord', {'ndim':[[2]]})]])

    # Setup fac_coords
    def fc(args):
        nanval = -999
        fac_coords = [np.random.RandomState(seed=1).rand(10, 5) * 5,
                      np.random.RandomState(seed=1).rand(8, 5) * 5,
                      nanval]
        return fac_coords
    fac_coords = ut.Setup('fac_coords', fc, [{}])
    
    x = ut.test(base_est,
                {'data':[d_0, d_1],
                 'fac_coords':['Auto', fac_coords],
                 'ecutmaxpos':[None, ['All', ['Med', 'Max']],
                               ['Cols', ['Med', 'Max']]],
                 'refit':[None, 'Lstsq'],
##                 'nondegen':[None, True],
                 }, # These work but trigger warnings
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
                 

def test_base_resid(check='run', asserts=ut.allclose, printout=True):
    "test Damon's base_resid() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def base_resid(data, **kwargs):
        d = data
        d.base_resid(**kwargs)
        return d.base_resid_out['coredata']
    
    # Setup method args
    margs = [('pseudomiss', {'rand_nan':0.80}),
             ('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {})]

    # Set up two scenarios
    args_0 = {'validchars':['All', ['All']]}
    d_0 = ut.Setup('d_0', setup, [args_0, margs])

    args_1 = {'validchars':['All', [0, 1, 2, 3, 4]]}
    d_1 = ut.Setup('d_1', setup, [args_1, margs])
    
    x = ut.test(base_resid,
                {'data':[d_0, d_1],
                 'nearest_val':[None, 'Nearest', 'ECut'],
                 'psmiss':[None, True]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
                 

def test_base_ear(check='run', asserts=ut.allclose, printout=True):
    "test Damon's base_ear() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def base_ear(data, **kwargs):
        d = data
        d.base_ear(**kwargs)
        return d.base_ear_out['coredata']

    # Setup method args
    margs = [('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {})]

    # Set up two scenarios
    args_0 = {'validchars':['All', ['All']]}
    d_0 = ut.Setup('d_0', setup, [args_0, margs])

    args_1 = {'validchars':['All', [0, 1, 2, 3, 4]]}
    d_1 = ut.Setup('d_1', setup, [args_1, margs])
    
    x = ut.test(base_ear,
                {'data':[d_0, d_1],
                 'ndim':[2]
                 },
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x


def test_base_se(check='run', asserts=ut.allclose, printout=True):
    "test Damon's base_se() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def base_se(data, **kwargs):
        d = data
        d.base_se(**kwargs)
        return d.base_se_out['coredata']

    # Setup method args
    margs_0 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {})]

    margs_1 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {'ecutmaxpos':['All', ['Max', 'Med']]}),
               ('base_resid', {}),
               ('base_ear', {})]

    margs_2 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {'ecutmaxpos':['All', [0.50, 1.4]]}),
               ('base_resid', {'nearest_val':'ECut'}),
               ('base_ear', {})]

    # Set up two scenarios
    args_0 = {'validchars':['All', ['All']]}
    d_0 = ut.Setup('d_0', setup, [args_0, margs_0])

    args_1 = {'validchars':['All', [0, 1, 2]]}
    d_1 = ut.Setup('d_1', setup, [args_1, margs_1])
    
    args_2 = {'validchars':['All', [0, 1, 2, 3]]}
    d_2 = ut.Setup('d_2', setup, [args_2, margs_2])

    x = ut.test(base_se,
                {'data':[d_0, d_1, d_2],
                 'obspercellmeth':['PickMinFac', 'CombineFacs']},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
                        

def test_base_fit(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's base_fit() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def base_fit(data, **kwargs):
        d = data
        d.base_fit(**kwargs)
        return d.base_fit_out['coredata']

    # Setup method args
    margs = [('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {}),
             ('base_ear', {})]

    d = ut.Setup('d', setup, [{}, margs])

    x = ut.test(base_fit,
                {'data':[d], 'ear':[None, 2, 'median', 'mean']},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x

                
def test_equate(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's equate() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def equate(data, **kwargs):
        d = data
        d.equate(**kwargs)
        return d.equate_out['Construct']['coredata']

    # Setup method args
    margs_0 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})]

    args_0 = {'validchars':['All', ['All'], 'Num'],
              'nheaders4rows':2, 'nheaders4cols':2,
              'extra_headers':{'0':0.50, '1':0.50}}
              
    d_0 = ut.Setup('d_0', setup, [args_0, margs_0])

    print('\nTest for workflow 0 (no bank)')
    x_0 = ut.test(equate,
                  {'data':[d_0],
                   'construct_ents':[{'Get':'AllExcept', 'Ents':[None]},
                                     {'Get':'AllExcept', 'Ents':['4']},
                                     ],
                   'label':['construct'],
                   'subscales':[None,
                                {'Get':'AllExcept', 'Labels':1, 'Subs':[None]}
                                ],
                   'facet':[0, 1],
                   'logits':[True, False],
                   'rescale':[None,
                              {'All':{'straighten':True, 'mean_sd':[0.0, 2.0]}},

# TODO: Figure out why this doesn't work
#                              {'0':{'mean_sd':[0.0, 2.0]}, '1':{'m_b':[1, 0]}}
                              ],
                   'refresh':[None],
                   'cuts':[None,
                           {'All':[-1, 0, 1]},
                           {'construct':[-1, 0, 1],
                            '0':[-1, 0, 1],
                            '1':[-1, 0, 1]}
                           ],
                   'stats':[True],
                   'group_se':[None, [1, '1/sqrt(2d)*0.75']],
                   },
                  check=check,
                  asserts=asserts,
                  suffix='0',
                  printout=printout)

    # Test for bank workflow
    bankfile = TEMP_PATH + 'ibank.pkl'
    margs_1 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {}),
               ('equate', {}),
               ('bank', {'filename':bankfile}),
               ('standardize', {'std_params':bankfile}),
               ('coord', {'anchors':{'Bank':bankfile,
                                     'Facet':1}}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})
               ]

    args_1 = {'validchars':['All', ['All'], 'Num'],
              'nheaders4rows':2, 'nheaders4cols':2,
              'extra_headers':{'0':0.50, '1':0.50}}
              
    d_1 = ut.Setup('d_1', setup, [args_1, margs_1])

    print('\nTest for workflow 1 (from bank)')
    x_1 = ut.test(equate,
                  {'data':[d_1],
                   'construct_ents':['Bank'],
                   'label':['Bank'],
                   'subscales':['Bank'],
                   'facet':['Bank'],
                   'correlated':['Bank'],
                   'logits':['Bank'],
                   'rescale':['Bank'],
                   'refresh':['Bank'],
                   'cuts':['Bank']},
                  check=check,
                  asserts=asserts,
                  suffix='1',
                  printout=printout)

    return {'0':x_0, '1':x_1}
             

def test_fin_est(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's fin_est() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def fin_est(data, **kwargs):
        d = data
        d.fin_est(**kwargs)
        return d.fin_est_out['coredata']

    cargs_0 = {'validchars':['All', ['All'], 'Num']}
    margs_0 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})]
    d_0 = ut.Setup('d_0', setup, [cargs_0, margs_0])

    cargs_1 = {'validchars':['All', ['a', 'b', 'c']]}
    margs_1 = [('parse', {}),
               ('standardize', {'referto':'Cols'}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})]
    d_1 = ut.Setup('d_1', setup, [cargs_1, margs_1])

    cargs_2 = {'validchars':['Cols', {'1':['a', 'b', 'c'],
                                      '2':['a', 'b', 'c'],
                                      '3':['a', 'b', 'c'],
                                      '4':[0, 1],
                                      '5':[0, 1],
                                      '6':[0, 1, 2],
                                      '7':[0, 1, 2],
                                      '8':['All']}]}
    margs_2 = [('parse', {'items2parse':['NoneExcept', ['1', '2', '3', '6']]}),
               ('standardize', {'referto':'Cols'}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})]
    d_2 = ut.Setup('d_2', setup, [cargs_2, margs_2])

    cargs_3 = {'validchars':['All', [0, 1], 'Num']}
    margs_3 = [('standardize', {}),
               ('coord', {'ndim':[[2]]}),
               ('base_est', {}),
               ('base_resid', {}),
               ('base_ear', {}),
               ('base_se', {})]
    d_3 = ut.Setup('d_3', setup, [cargs_3, margs_3])

    x_0 = ut.test(fin_est,
                  {'data':[d_0, d_3],
                   'stdmetric':['Auto', 'PreLogit'],
                   'orig_data':['data_out'],
                   'ents2restore':['All'],
                   'referto':['Whole'],
                   'continuous':['Auto', True],
                   'std_params':[None],
                   'alpha':[None]},
                  check=check,
                  asserts=asserts,
                  suffix='0',
                  printout=printout)

    x_1 = ut.test(fin_est,
                  {'data':[d_1, d_2],
                   'stdmetric':['Auto', 'PreLogit'],
                   'orig_data':['data_out'],
                   'ents2restore':['All'],
                   'referto':['Cols'],
                   'continuous':['Auto', True],
                   'std_params':[None],
                   'alpha':[None]},
                  check=check,
                  asserts=asserts,
                  suffix='1',
                  printout=printout)
    
    return {'0':x_0, '1':x_1}
                                    

def test_est2logit(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's est2logit() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def est2logit(data, **kwargs):
        d = data
        d.est2logit(**kwargs)
        return d.est2logit_out['coredata']

    cargs_0 = {'validchars':['All', ['All'], 'Num']}
    cargs_1 = {'validchars':['All', [0, 1, 2, 3], 'Num']}
    margs = [('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {}),
             ('base_ear', {}),
             ('base_se', {}),
             ('fin_est', {}),
             ('equate', {})]
    d_0 = ut.Setup('d_0', setup, [cargs_0, margs])
    d_1 = ut.Setup('d_1', setup, [cargs_1, margs])

    x = ut.test(est2logit,
                {'data':[d_0, d_1],
                 'estimates':['base_est_out', 'fin_est_out', 'equate_out'],
                 'ecutmaxpos':['Auto',
                               ['All', ['Med', 'Max']],
                               ['Cols', ['Med', 'Max']]],
                 'logitform':['Metric', 'Statistical'],
                 'obspercellmeth':['PickMinFac', 'CombineFacs']},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
               
                 
def test_fin_resid(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's fin_resid() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def fin_resid(data, **kwargs):
        d = data
        d.fin_resid(**kwargs)
        return d.fin_resid_out['coredata']

    cargs = {'validchars':['All', [0, 1, 2, 3]]}
    margs = [('standardize', {}),
             ('pseudomiss', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {}),
             ('fin_est', {})]
    d = ut.Setup('d', setup, [cargs, margs])
    
    x = ut.test(fin_resid,
                {'data':[d],
                 'resid_type':[['All', ['Diff']],
                               ['All', ['Nearest']],
                               ['All', ['ECut', 1.0]],
                               ['All', ['Match']],
                               ['Cols', {'1':['ECut', 1.0],
                                         '2':['Diff'],
                                         '3':['Nearest'],
                                         '4':['Diff'],
                                         '5':['Diff'],
                                         '6':['Diff'],
                                         '7':['Diff'],
                                         '8':['Diff']}]],
                 'psmiss':[None, True]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x
                

def test_fin_fit(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's fin_fit() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def fin_fit(data, **kwargs):
        d = data
        d.fin_fit(**kwargs)
        return d.fin_fit_out['coredata']

    cargs = {'validchars':['All', [0, 1, 2, 3]]}
    margs = [('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {}),
             ('base_ear', {}),
             ('base_se', {}),
             ('fin_est', {}),
             ('fin_resid', {})]
    d = ut.Setup('d', setup, [cargs, margs])
    
    x = ut.test(fin_fit,
                {'data':[d]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x                              


def test_restore_invalid(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's restore_invalid() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def restore_invalid(data, **kwargs):
        d = data
        d.restore_invalid(**kwargs)
        return d.base_est_out['coredata']  # Restores base_est_out

    cargs = {'validchars':['All', [0, 1]]}
    margs = [('extract_valid', {'minperrow':2, 'minpercol':2, 'minsd':0.001,
                                'rem_rows':['4']}),
             ('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('fin_est', {})]

    # This 'd' results in two missing columns and one missing row
    d = ut.Setup('d', setup, [cargs, margs])
    
    x = ut.test(restore_invalid,
                {'data':[d],
                 'outputs':[['base_est_out', 'fin_est_out']],
                 'getrows':[None, True],
                 'getcols':[True]},
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x   


def test_summstat(check='run', asserts=ut.allclose, printout=True):
    "Test Damon's summstat() method."

    def setup(*args):
        d = setup_damon(*args)
        return d

    def summstat(data, **kwargs):
        d = data
        d.summstat(**kwargs)
        if kwargs['outname'] is None:
            return d.summstat_out['col_ents']['coredata']
        else:
            return d.summstat_out['cat']['col_ents']['coredata']

    cargs_0 = {'validchars':['All', ['All'], 'Num']}
    cargs_1 = {'validchars':['All', [0, 1, 2, 3], 'Num']}
        
    margs = [('standardize', {}),
             ('coord', {'ndim':[[2]]}),
             ('base_est', {}),
             ('base_resid', {}),
             ('base_ear', {}),
             ('base_se', {}),
             ('base_fit', {}),
             ('fin_est', {}),
             ('fin_resid', {}),
             ('fin_fit', {})]
    d_0 = ut.Setup('d_0', setup, [cargs_0, margs])
    d_1 = ut.Setup('d_1', setup, [cargs_1, margs])
    
    x = ut.test(summstat,
                {'data':[d_0, d_1],
                 'getstats':[['All']],
                 'getrows':[{'Get':'AllExcept', 'Labels':'key', 'Rows':[None]}],
                 'getcols':[{'Get':'AllExcept', 'Labels':'key', 'Cols':[None]}],
                 'itemdiff':[None, True],
                 'outname':[None, 'cat'],
                 'labels':[{'row_ents':'person', 'col_ents':'item'}],
                 'group_se':[['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75']]
                 },
                check=check,
                asserts=asserts,
                suffix=None,
                printout=printout)
    return x   



            


                       
            
##CREATE_ARGS = {'nfac0':10, 'nfac1':8, 'ndim':2, 'seed':1,
##               'facmetric':[4, -2], 'noise':None,
##               'validchars':['All', ['All'], 'Num'],
##               'mean_sd':None, 'p_nan':0.05, 'nanval':-999,
##               'condcoord_':None, 'nheaders4rows':1, 'nheaders4cols':1,
##               'extra_headers':2, 'input_array':None, 'apply_zeros':None,
##               'output_as':'Damon', 'outfile':None, 'delimiter':None,
##               'bankf0':None, 'bankf1':None, 'verbose':None}

        

##d = create_data()['data']       =>  Create artificial Damon objects
##d = TopDamon()                  =>  Create a Damon object from an existing dataset
##d = Damon(data,'array',...)     =>  More generic low-level way to create a Damon object
##d.merge_info()                  =>  Merge row or column info into labels
##d.extract_valid()               =>  Extract only valid rows/cols
##d.pseudomiss()                  =>  Create index of pseudo-missing cells
##d.score_mc()                    =>  Score multiple-choice data
##d.subscale()                    =>  Append raw scores for item subscales
##d.parse()                       =>  Parse response options to separate columns
##d.standardize()                 =>  Convert all columns into a standard metric
##d.rasch()                       =>  Rasch-analyze data (in place of coord())
##d.coord()                       =>  Calculate row and column coordinates
##d.sub_coord()                   =>  Calculate coordinates given multiple subspaces (in place of coord)
##d.objectify()                   =>  Maximize objectivity of specified columns (in place of coord)
##d.base_est()                    =>  Calculate cell estimates
##d.base_resid()                  =>  Get residuals (observation - estimate)
##d.base_ear()                    =>  Get expected absolute residuals
##d.base_se()                     =>  Get standard errors for all cells
##d.equate()                      =>  Equate two datasets using a bank
##d.base_fit()                    =>  Get cell fit statistics
##d.fin_est()                     =>  Get final estimates, original metric
##d.est2logit()                   =>  Convert estimates to logits
##d.item_diff()                   =>  Get probability-based item difficulties
##d.fillmiss()                    =>  Fill missing cells of original dataset
##d.fin_resid()                   =>  Get final cell residuals, original metric
##d.fin_fit()                     =>  Get final cell fit, original metric
##d.restore_invalid()             =>  Restores invalid rows/cols to output arrays
##d.summstat()                    =>  Get summary row/column/range statistics
##d.merge_summstats()             =>  Merge multiple summstat() runs
##d.plot_two_vars()               =>  Plot two variables to create bubble chart
##d.wright_map()                  =>  Plot person and item distributions
##d.bank()                        =>  Save row/column coordinates in "bank" file
##d.export()                      =>  Export specified outputs as files





















