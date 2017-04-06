# -*- coding: utf-8 -*-
# opts.py

# -*- coding: utf-8 -*-
# opts.py

"""rasch_1.py
Template for doing Rasch analyses on mixed dichotomous and
polytomous data.

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


"""
This template shows how to create a mixed model polytomous
dataset and analyze it with the Rasch model. Person and item,
parameters are calculated, with reliabilities, printed into
the shell and exported.

Review the rasch_0.py template first.
"""

# Prep the create_data() validchars parameter by making the first 40 items
#dichotomous, the rest polytomous.
# This requires Python's for loop, range() function, and dictionary {} object.
ncols = 80
vc = {}
for item in range(1, ncols + 1):
    if item <= ncols / 2.0:
        vc[item] = [0, 1]
    else:
        vc[item] = [0, 1, 2]
validchars = ['Cols', vc, 'Num']

# Prep the extra_headers parameter. This assigns column labels '0' and '1' to reflect
# the different response options. Note that nheaders4cols has to be 2 to hold the
# extra labels.
extra_headers = {'0':0.50, '1':0.50}
    

# Create the data file 'a_data_rasch_1_example.csv'
cd = dmn.create_data(nfac0 = 100,  # [Number of facet 0 elements -- rows/persons]
                    nfac1 = 80,  # [Number of facet 1 elements -- columns/items]
                    ndim = 1,   # [Number of dimensions to create]
                    seed = None,  # [<None => randomly pick starter coordinates; int => integer of "seed" random coordinates>]
                    facmetric = [4, 0],  # [[m,b] => rand() * m + b, to set range of facet coordinate values]
                    noise = 10, # [<None, noise, {'Rows':<noise,{1:noise1,4:noise4}>,'Cols':<noise,{2:noise2,5:noise5}> => add error to rows/cols]
                    validchars = validchars,   # [<None, ['All',[valid chars]]; or ['Cols', {1:['a','b','c','d'],2:['All'],3:['1.2 -- 3.5'],4:['0 -- '],...}]> ]
                    mean_sd = None, # [<None, ['All',[Mean,SD]], or ['Cols', {1:[Mean1,SD1],2:[Mean2,SD2],3:'Refer2VC',...}]> ]
                    p_nan = 0.10,  # [Proportion of cells to make missing at random]
                    nanval = -999.,  # [Numeric code for designating missing values]
                    condcoord_ = None,  # [< None, 'Orthonormal'>]
                    nheaders4rows = 1,  # [Number of header column labels to put before each row]
                    nheaders4cols = 2,  # [Number of header row labels to put before each column]
                    extra_headers = extra_headers,  # [<0, int, {'0':0.25, '1':0.75}> => If headers > 1, range of ints for labels, randomly assigned or in blocks]
                    input_array = None,   # [<None, name of data array, {'fac0coord':EntxDim row coords,'fac1coord':EntxDim col coords}>]
                    apply_zeros = None, # [<None, [row, {'sub1':[0,1,1],...}> => for each item group in row, where to apply zeros to coords]
                    output_as = 'textfile',  # [<'Damon','datadict','array','textfile','Damon_textfile','datadict_textfile','array_textfile','hd5'>]
                    outfile = 'rasch_1_example.csv',    # [<None, name of the output file/path prefix when output_as includes 'textfile'>]
                    delimiter = ',',    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '   '>]
                    bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]> ]
                    bankf1 = None,  # [<None => no bank,[<'All', list of F1 (Col) entities>]> ]
                    verbose = True, # [<None, True> => print useful information and messages]
                    )

# Prep the validchars parameter, just like before.
# Note that you shouldn't do ['All', [0, 1, 2], 'Num'] on the theory
# that this covers the dichotomous columns too.  The rasch() method
# looks to the validchars parameter to figure out the rating scale.
vc = {}
for item in range(1, ncols + 1):
    if item <= ncols / 2.0:
        vc[item] = [0, 1]
    else:
        vc[item] = [0, 1, 2]
validchars = ['Cols', vc, 'Num']

# Load dataset using Damon. The missing parameters go to their defaults.
data = dmn.Damon(data = 'a_data_rasch_1_example.csv',    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                  format_ = 'textfile',    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                  workformat = 'RCD_dicts_whole',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                  validchars = validchars,   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                  nheaders4rows = 1,  # [number of columns to hold row labels]
                  key4rows = 0,   # [<None, nth column from left which holds row keys>]
                  rowkeytype = int,     # [<None, type of row keys>]
                  nheaders4cols = 2,  # [number of rows to hold column labels]
                  key4cols = 0, # [<None, nth row from top which holds column keys>]
                  colkeytype = int
                  )

# Analyze with Rasch model. Note the groups parameter.
data.rasch(groups = {'row':1},    # [<None, {'row':int row of group labels}, ['key', {'group0':['i1', i2'],...}], ['index', {'group0':[0, 1],...}]> => identify groups]
              anchors = None,   # [<None, {'Bank':<pickle file>, 'row_ents':[<None,'All',row entity list>], 'col_ents':[<None,'All',col entity list>]}> ]
              runspecs = [0.0001,20],  # [<[stop_when_change, max_iteration]> => iteration stopping conditions ]
              minvar = 0.001,  # [<decimal> => minimum row/col variance allowed during iteration]
              maxchange = 10,  # [<+num> => maximum change allowed per iteration]
              )


# Get person meaures
person_measures = dmn.Damon(data = data.rasch_out['fac0coord'],
                           format_ = 'datadict',
                           workformat = 'RCD_dicts_whole',
                           verbose = None)

print 'person_measures=\n', person_measures

# Get item difficulties
item_diffs = dmn.Damon(data = data.rasch_out['fac1coord'],
                       format_ = 'datadict',
                       workformat = 'RCD_dicts_whole',
                       verbose = None)

print '\nitem difficulties=\n', item_diffs

# Get reliabilities
rel = dmn.Damon(data.rasch_out['reliability'],
                format_ = 'datadict',
                workformat = 'RCD_dicts_whole',
                verbose = None)

print 'reliabilities=\n', rel

# Export reliabilities
rel.export(outputs = ['data_out'],   # [['coord_out','base_est_out',...] => string list of desired datadict outputs]
           output_as = 'textfile',    # [<'textfile','hd5','pickle'> => type of output file]
           outprefix = 'aa_reliability',    # [string prefix to all file names, may be a path to a designated directory]
           outsuffix = '.csv',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
           delimiter = ',', # [<None,text delimiter, e.g. ',' or '      '>]
           format_ = '%.60s',    # [<None, format code of cell contents>  => See numpy.savetxt() docs]
           obj_params = None,    # [<None,True> => export pickle file of Damon non-data __init__ parameters]
           )
                



















##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
##if __name__ == "__main__":
##    a = my_func(...)
##    print a



















