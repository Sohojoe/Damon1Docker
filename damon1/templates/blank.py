# -*- coding: utf-8 -*-
# opts.py

"""blank.py
Generic template for importing modules needed to write Damon scripts.

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
import warnings
warnings.simplefilter('ignore')  # matplotlib produces annoying warnings

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.ma as npma
np.seterr(all='ignore')

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# If damon1 is not stored in your Python's site-packages folder, specify path
damon1_path = '/Users/markhmoulton/Google Drive/my_python'
if damon1_path:
    sys.path.append(damon1_path)

import damon1 as damon1
import damon1.core as dmn
import damon1.tools as dmnt


# Start programming here...


























##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
##if __name__ == "__main__":
##    a = my_func(...)
##    print a



















