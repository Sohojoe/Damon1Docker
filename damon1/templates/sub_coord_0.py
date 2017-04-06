# -*- coding: utf-8 -*-
# opts.py
"""sub_coord_0.py
Template for Damon-analyzing data with multiple subspaces.

Copyright (c) 2009 - 2011, [Developer Name] for [Company Name].

Purpose:

Damon Version:
Python Version:
Numpy Version:

License
-------
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


How To Use
----------
You can run Damon from the prompt interactively on IDLE, or
from scripts, or you can run it online using a Damon
notebook on the Wakari cloud platform (see Pythias website).
This template is for writing Damon scripts.

In the damon1 package, there is a folder called "templates".
This is one of those templates.  Each template is for a
different style of analysis, the particulars of which are
described below.

Open the template from the Python shell menu by selecting
file/Open Module.  type_:  damon1.templates.blank

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

Sometimes, you will want to print tabular arrays, with
labels and columns properly aligned.  Use the tabulate()
function for this:

>>>  print tabulate(my_array, 'firstline')

For informaton on how to customize tabulate() further, type:

>>>  help(tabulate)

You will also want to create charts and graphs using matplotlib.
To get started, type:

>>>  help(plt.plot)

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

from tabulate import tabulate

import damon1 as damon1
import damon1.core as dmn
import damon1.tools as dmnt


# Create data as Damon object
nrows = 100
ncols = 100
extra_headers = {'sub0':0.50, 'sub1':0.50}
apply_zeros = [1, {'sub0':[0, 1, 1],
                   'sub1':[1, 0, 1]}]


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
                    nheaders4cols = 2,  # [Number of header row labels to put before each column]
                    extra_headers = extra_headers,  # [<0, int, {'0':0.25, '1':0.75}> => If headers > 1, range of ints for labels, randomly assigned or in blocks]
                    input_array = None,   # [<None, name of data array, {'fac0coord':EntxDim row coords,'fac1coord':EntxDim col coords}>]
                    apply_zeros = apply_zeros, # [<None, [row, {'sub1':[0,1,1],...}> => for each item group in row, where to apply zeros to coords]
                    output_as = 'Damon',  # [<'Damon','datadict','array','textfile','Damon_textfile','datadict_textfile','array_textfile','hd5'>]
                    outfile = None,    # [<None, name of the output file/path prefix when output_as includes 'textfile'>]
                    delimiter = None,    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '   '>]
                    bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]> ]
                    bankf1 = None,  # [<None => no bank,[<'All', list of F1 (Col) entities>]> ]
                    verbose = True, # [<None, True> => print useful information and messages]
                    )

# Get data
d = cd['data']
m = cd['model']

np.set_printoptions(precision=2, suppress=True)
print 'data=\n', d.collabels

# Analyze, starting by standardizing to a 'PreLogit' metric
d.standardize(metric = 'PreLogit',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                referto = 'Cols',   # [<None,'Whole','Cols'>]
                rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                )

# Get best dimensionality, then coordinates
d.sub_coord(subspaces = {'row':1},    # [<{'row':int row of subspace labels}, ['key', {'sub0':['i1', i2'],...}], ['index', {'sub0':[0, 1],...}]> => identify subspaces]
              coord_subs = {'All':{'ndim':[[2]]}},  # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for each subspace or for all subspaces]
              coord_resids = {'All':{'ndim':[[1]]}},    # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for analyzing residuals of each subspace]
              unique_weights = {'All':'Auto'},    # [<{'All':'Auto'} or {'sub0':<'Auto', p>, 'sub1':<'Auto', p>, where 0 < p < 1  > => how much to weight unique component for each subspace]
              share_if = {'targ_<':30, 'pred_>': 4},   # [<{'targ_<':int, 'pred_>':int}> => when to share info between subspaces]
              min_rel = 0.02,   # [< 0 < min_rel < 1  > => minimum reliability to use in unique weighting formula]
              rpt_optimal = None,    # [<None, True> => calculate and return optimal unique weight]
              )

# Get base estimates, residuals, expected absolute residuals, standard errors
# Just trusting defaults
d.base_est()
d.base_resid()
d.base_ear()
d.base_se()
d.base_fit()

# Convert estimates into logits
d.est2logit()

# Compute summary statistics
# Use base_est_out to get all statistics.  Use est2logit_out to get most, but in a true logit metric.
d.summstat(data = 'base_est_out', # [<'base_est_out','fin_est_out','est2logit_out','data_out',...> => data for calculating stats]
             getstats = ['All'], #['Mean', 'SE', 'SD', 'Corr', 'Resid', 'Rel', 'Fit_MeanSq', 'Fit_Perc>2'], # [Select stats from [<'Mean','SE','SD','Corr','Resid','RMSEAR','Sep','Rel','Fit_MeanSq','Fit_Perc>2','Count','Min','25Perc','Median','75Perc','Max','Coord'>] ]
             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}>]
             getcols = 'SummWhole', #{'Get':'AllExcept','Labels':'key','Cols':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}>]
             itemdiff = None,   # [<None,True> => express item (column) mean as a probabilistically defined item "difficulty" ]
             outname = None    # [<None,'cluster_name'> => name to append to summstat outputs for multiple analyses]
             )

# Print summary statistics by person
print 'Person statistics=\n', d.summstat_out['row_ents']

d.export(outputs = ['row_ents_out'],   # [['coord_out','base_est_out',...] => string list of desired datadict outputs]
               output_as = 'textfile',    # [<'textfile','hd5','pickle'> => type of output file]
               outprefix = 'aa',    # [string prefix to all file names, may be a path to a designated directory]
               outsuffix = '.csv',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
               delimiter = ',', # [<None,text delimiter, e.g. ',' or '      '>]
               format_ = '%.60s',    # [<None, format code of cell contents>  => See numpy.savetxt() docs]
               obj_params = None,    # [<None,True> => export pickle file of Damon non-data __init__ parameters]
               )

# Compare observations, estimates with model values
est = d.base_est_out['coredata']
true = m.coredata
obs = d.coredata
r_est_true = dmnt.correl(est, true)
r_obs_true = dmnt.correl(obs, true)
print 'r_est_true=\n', round(r_est_true, 3)
print 'r_obs_true=\n', round(r_obs_true, 3)

# Plot observed vs true
valid = [(obs != d.nanval) & (true != d.nanval)]
plt.plot(obs[valid], est[valid], 'k.')
plt.xlabel('true')
plt.ylabel('observed')
plt.show()
plt.clf()

# Compare estimates with model values
valid = [(est != d.nanval) & (true != d.nanval)]
plt.plot(true[valid], est[valid], 'k.')
plt.xlabel('true')
plt.ylabel('estimate')
plt.show()

print 'finished'




























##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
##if __name__ == "__main__":
##    a = my_func(...)
##    print a



















