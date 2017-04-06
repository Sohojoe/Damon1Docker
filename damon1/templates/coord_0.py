# -*- coding: utf-8 -*-
# opts.py

"""template.py
Template for writing Damon programs.

Copyright (c) 2009 - 2011, [Developer Name] for [Company Name].

Purpose:

Damon Version:
Python Version:
Numpy Version:

License
-------
This program references one or more software modules that are
under copyright to Pythias Consulting, LLC.  Therefore, it is subject
to the Apache 2.0 license.  See the damon1 folder for a copy of the
license.

How To Use
----------
You can run Damon from the prompt interactively on IDLE, or
you can run it from scripts.  This template is for writing Damon
scripts.  It doesn't contain much, but saves the time of
writing the necessary import statements.  It also contains
a handy lookup reference of the various methods you will need.

Open the template from the Python shell menu by selecting
file/Open Module.  type_:  damon1.template

Save the template under a different name to a directory of
your choice.  (If you forget this step, you will overwrite
the template and have to create a new one.)

Write code and run using the F5 key.

If you use the template to define functions, you can run
the functions from this module using the "if __name__ == "__main__":
trick at the bottom of the module.  (See Python docs.)

A tutorial can be found on the Pythias website.  Most of the
documentation is found at the level of individual functions
and methods and can be accessed using help().

Top-level documentation can be accessed using:

>>>  import damon1
>>>  help(damon1)

That will get you started.



Cheatsheet of Damon Methods
---------------------------
In (approximate) order of application:

d = create_data()['data']       =>  Create artificial Damon objects
d = TopDamon()                  =>  Create a Damon object from an existing dataset
d = Damon(data,'array',...)     =>  More generic low-level way to create a Damon object
d.merge_info()                  =>  Merge row or column info into labels
d.extract_valid()               =>  Extract only valid rows/cols
d.pseudomiss()                  =>  Create index of pseudo-missing cells
d.score_mc()                    =>  Score multiple-choice data
d.subscale()                    =>  Append raw scores for item subscales
d.parse()                       =>  Parse response options to separate columns
d.standardize()                 =>  Convert all columns into a standard metric
d.rasch()                       =>  Rasch-analyze data (in place of coord())
d.coord()                       =>  Calculate row and column coordinates
d.sub_coord()                   =>  Calculate coordinates given multiple subspaces (in place of coord)
d.objectify()                   =>  Maximize objectivity of specified columns (in place of coord)
d.base_est()                    =>  Calculate cell estimates
d.base_resid()                  =>  Get residuals (observation - estimate)
d.base_ear()                    =>  Get expected absolute residuals
d.base_se()                     =>  Get standard errors for all cells
d.equate()                      =>  Equate two datasets using a bank
d.base_fit()                    =>  Get cell fit statistics
d.fin_est()                     =>  Get final estimates, original metric
d.est2logit()                   =>  Convert estimates to logits
d.item_diff()                   =>  Get probability-based item difficulties
d.fillmiss()                    =>  Fill missing cells of original dataset
d.fin_resid()                   =>  Get final cell residuals, original metric
d.fin_fit()                     =>  Get final cell fit, original metric
d.restore_invalid()             =>  Restores invalid rows/cols to output arrays
d.summstat()                    =>  Get summary row/column/range statistics
d.merge_summstats()             =>  Merge multiple summstat() runs
d.plot_two_vars()               =>  Plot two variables to create bubble chart
d.wright_map()                  =>  Plot person and item distributions
d.bank()                        =>  Save row/column coordinates in "bank" file
d.export()                      =>  Export specified outputs as files

"""
import os
import sys

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.ma as npma
np.seterr(all='ignore')

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import damon1 as damon1
import damon1.core as dmn
import damon1.tools as dmnt

"""
This template creates three dimensional data, runs coord() to determine best
dimensionality, computes estimates and errors, converts estimates to logits,
computes summary statistics, and generates plots of the objectivity curve
and of success in predicting true values.
"""

data_source = 'create_data'     # <'create_data', 'path/to/my/data.csv'>

if data_source == 'create_data':
    
    # Create data as Damon object
    cd = dmn.create_data(nfac0 = 100,  # [Number of facet 0 elements -- rows/persons]
                        nfac1 = 80,  # [Number of facet 1 elements -- columns/items]
                        ndim = 3,   # [Number of dimensions to create]
                        seed = None,  # [<None => randomly pick starter coordinates; int => integer of "seed" random coordinates>]
                        facmetric = [4,-2],  # [[m,b] => rand() * m + b, to set range of facet coordinate values]
                        noise = 10, # [<None, noise, {'Rows':<noise,{1:noise1,4:noise4}>,'Cols':<noise,{2:noise2,5:noise5}> => add error to rows/cols]
                        validchars = ['All', ['All'], 'Num'],   # [<None, ['All',[valid chars]]; or ['Cols', {1:['a','b','c','d'],2:['All'],3:['1.2 -- 3.5'],4:['0 -- '],...}]> ]
                        mean_sd = None, # [<None, ['All',[Mean,SD]], or ['Cols', {1:[Mean1,SD1],2:[Mean2,SD2],3:'Refer2VC',...}]> ]
                        p_nan = 0.10,  # [Proportion of cells to make missing at random]
                        nanval = -999.,  # [Numeric code for designating missing values]
                        condcoord_ = None,  # [< None, 'Orthonormal'>]
                        nheaders4rows = 1,  # [Number of header column labels to put before each row]
                        nheaders4cols = 1,  # [Number of header row labels to put before each column]
                        extra_headers = 0,  # [<0, int, {'0':0.25, '1':0.75}> => If headers > 1, range of ints for labels, randomly assigned or in blocks]
                        input_array = None,   # [<None, name of data array, {'fac0coord':EntxDim row coords,'fac1coord':EntxDim col coords}>]
                        apply_zeros = None, # [<None, [row, {'sub1':[0,1,1],...}> => for each item group in row, where to apply zeros to coords]
                        output_as = 'textfile',  # [<'Damon','datadict','array','textfile','Damon_textfile','datadict_textfile','array_textfile','hd5'>]
                        outfile = 'coord_0_example.csv',    # [<None, name of the output file/path prefix when output_as includes 'textfile'>]
                        delimiter = None,    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '   '>]
                        bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]> ]
                        bankf1 = None,  # [<None => no bank,[<'All', list of F1 (Col) entities>]> ]
                        verbose = True, # [<None, True> => print useful information and messages]
                        )
    data_file = 'a_data_coord_0_example.csv'
    model_file = 'a_model_coord_0_example.csv'
else:
    data_file = data_source

# Load the data file
data = dmn.Damon(data = data_file,    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                  format_ = 'textfile',    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                  workformat = 'RCD_dicts_whole',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                  validchars = ['All', ['All'], 'Num'],   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                  nheaders4rows = 1,  # [number of columns to hold row labels]
                  key4rows = 0,   # [<None, nth column from left which holds row keys>]
                  rowkeytype = 'S60',     # [<None, type of row keys>]
                  nheaders4cols = 1,  # [number of rows to hold column labels]
                  key4cols = 0, # [<None, nth row from top which holds column keys>]
                  colkeytype = 'S60',     # [<None, type of column keys>]
                  check_dups = 'warn',   # [<None,'warn','stop'> => response to duplicate row/col keys]
                  dtype = [object, 3], # [[type of 'whole' matrix, <None, int number of decimals>], e.g. ['S60',8],[object,None] ]
                  nanval = -999,    # [Value to which non-numeric/invalid characters should be converted.]
                  missingchars = None,  # [<None, [list of elements to make missing]>]
                  miss4headers = None, # [<None, [[list of elements to make missing in headers]>]
                  recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
                  cols2left = None,    # [<None, [ordered list of col keys, to shift to left and use as rowlabels]>]
                  selectrange = None,   # [<None,[slice(StartRow,EndRow),slice(StartCol,EndCol)]>]
                  delimiter = ',',  # [<None, character to delimit input file columns (e.g. ',' for .csv and '  ' for .txt tab-delimited files)]
                  pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                  verbose = True,    # [<None, True> => report method calls]
                  )

# Load the model file
model = dmn.Damon(data = model_file,    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                  format_ = 'textfile',    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                  workformat = 'RCD_dicts_whole',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                  validchars = ['All', ['All'], 'Num'],   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                  nheaders4rows = 1,  # [number of columns to hold row labels]
                  key4rows = 0,   # [<None, nth column from left which holds row keys>]
                  rowkeytype = 'S60',     # [<None, type of row keys>]
                  nheaders4cols = 1,  # [number of rows to hold column labels]
                  key4cols = 0, # [<None, nth row from top which holds column keys>]
                  colkeytype = 'S60',     # [<None, type of column keys>]
                  check_dups = 'warn',   # [<None,'warn','stop'> => response to duplicate row/col keys]
                  dtype = [object, 3], # [[type of 'whole' matrix, <None, int number of decimals>], e.g. ['S60',8],[object,None] ]
                  nanval = -999,    # [Value to which non-numeric/invalid characters should be converted.]
                  missingchars = None,  # [<None, [list of elements to make missing]>]
                  miss4headers = None, # [<None, [[list of elements to make missing in headers]>]
                  recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
                  cols2left = None,    # [<None, [ordered list of col keys, to shift to left and use as rowlabels]>]
                  selectrange = None,   # [<None,[slice(StartRow,EndRow),slice(StartCol,EndCol)]>]
                  delimiter = ',',  # [<None, character to delimit input file columns (e.g. ',' for .csv and '  ' for .txt tab-delimited files)]
                  pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                  verbose = True,    # [<None, True> => report method calls]
                  )

# Display data
print 'data=\n', data

# Analyze, starting by standardizing to a 'PreLogit' metric
data.standardize(metric = 'PreLogit',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                    referto = 'Cols',   # [<None,'Whole','Cols'>]
                    rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                    std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                    add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                    )

# Get best dimensionality, then coordinates
data.coord(ndim = [range(1, 11)],      # [<None,[[dim list],'search','homogenize']> => set dimensionality or search range, possibly homogenized]
              runspecs = [0.0001,20],  # [<[StopWhenChange,MaxIteration]>]
              seed = 'Auto',  #[<None,int,'Auto',{'MinR':0.90,'MaxIt':<10,[3,10]>,'Facet':<0,1>,'Stats':[<'Stab','Acc','Obj','PsMsResid','NonDegen'>],'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[...]},'Group2':{'Get':'AllExcept','Labels':'index','Entities':[...]}}>]
              homogenize = None,    # [<None,{'ApplyAncs':<True,False>,'Facet':1,'Max':500,'Form':'Cov'} => homogenize params]
              anchors = None,    # [<None,{'Bank':<bank,pickle file>,'Facet':<0,1>,'Coord':<'ent_coord','ear_coord','se_coord'>,'Entities':<['All',list entities]>,'Freshen':<None,True>}> ]
              quickancs = None,  # [<None,[<0,1>,ent x ndim array]> => facet, anchor array]
              startercoord = None,    # [<None,[<0,1>,ent x ndim array]> => facet, starter array]
              pseudomiss = None,    # [<None,True> => make cells pseudo-missing for "official" run]
              miss_meth = 'IgnoreCells', # [<'ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells>]
              solve_meth = 'LstSq', # [<'LstSq','IRLS'> => method for solving equations]
              solve_meth_specs = None,    # [<None, spec dictionary> => specs for solve_meth (cf. solve2() docs), e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]
              condcoord_ = {'Fac0':'Orthonormal','Fac1':None},  # [<None,{'Fac0':<'a func',myfunc>,'Fac1':<'a func',myfunc>}> ]
              weightcoord = True,   # [<None,True> => downweight influential coordinates]
              jolt_ = None,  # [<None,[sigma,jolt_]> e.g., [20,1.5] => Apply 1.5 noise factor if sigma exceeds 20]
              feather = None,     # [<None,float> => add small amount of randomness to the data]
              )

# Get base estimates, residuals, expected absolute residuals, standard errors
# Just trusting defaults
data.base_est()
data.base_resid()
data.base_ear()
data.base_se()
data.base_fit()

# Convert estimates into logits
data.est2logit()

# Compute summary statistics
# Use base_est_out to get all statistics.  Use est2logit_out to get most, but in a true logit metric.
data.summstat(data = 'base_est_out', # [<'base_est_out','fin_est_out','est2logit_out','data_out',...> => data for calculating stats]
             getstats = ['All'], #['Mean','Resid','Corr'], # [Select stats from [<'Mean','SE','SD','Corr','Resid','RMSEAR','Sep','Rel','Fit_MeanSq','Fit_Perc>2','Count','Min','25Perc','Median','75Perc','Max','Coord'>] ]
             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}>]
             getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}>]
             itemdiff = None,   # [<None,True> => express item (column) mean as a probabilistically defined item "difficulty" ]
             outname = None    # [<None,'cluster_name'> => name to append to summstat outputs for multiple analyses]
             )

# Print summary statistics by person
print '\nPerson statistics=\n', dmnt.tabulate(data.row_ents_out.whole, 'firstrow')

# Print summary statistics by person
print '\nItem statistics=\n', dmnt.tabulate(data.col_ents_out.whole, 'firstrow')

data.export(outputs = ['row_ents_out', 'col_ents_out'],   # [['coord_out','base_est_out',...] => string list of desired datadict outputs]
           output_as = 'textfile',    # [<'textfile','hd5','pickle'> => type of output file]
           outprefix = 'coord_0',    # [string prefix to all file names, may be a path to a designated directory]
           outsuffix = '.csv',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
           delimiter = ',', # [<None,text delimiter, e.g. ',' or '      '>]
           format_ = '%.60s',    # [<None, format code of cell contents>  => See numpy.savetxt() docs]
           obj_params = None,    # [<None,True> => export pickle file of Damon non-data __init__ parameters]
           )

# Look at objectivity per dimension
dim = data.objperdim.core_col['Dim']
acc = data.objperdim.core_col['Acc']
stab = data.objperdim.core_col['Stab']
obj = data.objperdim.core_col['Obj']

plt.plot(dim, acc, 'r-', label='Accuracy')
plt.plot(dim, stab, 'b-', label='Stability')
plt.plot(dim, obj, 'k-', label = 'Objectivity')
plt.xlabel('Dimensionality')
plt.ylabel('Objectivity')
plt.legend()
plt.savefig('coord_0_obj_plot.png')
plt.clf()

if data_source == 'create_data':
    
    # Compare observations, estimates with model values
    est = data.base_est_out['coredata']
    true = model.coredata
    obs = data.coredata
    r_est_true = dmnt.correl(est, true)
    r_obs_true = dmnt.correl(obs, true)
    print '\nr_est_true=\n', round(r_est_true, 3)
    print 'r_obs_true=\n', round(r_obs_true, 3)
    
    # Plot observed vs true
    valid = [(obs != d.nanval) & (true != d.nanval)]
    plt.plot(obs[valid], est[valid], 'k.')
    plt.xlabel('true')
    plt.ylabel('observed')
    plt.savefig('coord_0_obs_v_true.png')
    plt.clf()
    
    # Compare estimates with model values
    valid = [(est != d.nanval) & (true != d.nanval)]
    plt.plot(true[valid], est[valid], 'k.')
    plt.xlabel('true')
    plt.ylabel('estimate')
    plt.savefig('coord_0_est_v_true.png')

print '\nfinished'























##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
##if __name__ == "__main__":
##    a = my_func(...)
##    print a



















