
"""
Copyright (C) 2008 - 2016 Mark H. Moulton for Pythias Consulting, LLC.
9703 SE Butte Avenue
Vancouver, WA  98664
Email:  pythiasconsulting@gmail.com
Website:  www.pythiasconsulting.com

Additional Contributors (under terms of an Apache 2.0 based contributor
license agreement):

Mohit Khanna, Amritsar, India

Version
-------
Damon Version:  1.2.01
Damon Release Date: 3/21/2017

Damon is written in Python 2.7.13 and Numpy 1.11.3 as distributed by
by Continuum Analytics, Anaconda distribution. 

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


Changes
-------
This version included a lot of new work with standard error estimation.

Fixed bug in tools.median().

In Damon.__init__(), used new tool functions to avoid numpy deprecation
warnings.

In Damon.__init__(), improved handling of floats in validchars.

In tools.flag_invalid() extended formula to allow counts as proportions.

In Damon.coord(), expanded the condcoord_ parameter to allow specification
of the 'first' facet for which to calculate coordinates.

In Damon.equate(), fixed bugs that messed up rescaling in anchored analysis 
designs.

In Damon.__init__(), forced index to be integer to get around deprecation
waring raised by latest version of Numpy.

In Damon.extract_valid(), added capacity to check row and column entities
against a bank.

In Damon.fin_est(), changed conversion back to ratio scale to remove
requirement to have same original mean, sd.

In Damon.base_est(), added functionality to refit parameter to allow 
control of the degree of the fit. Removed functionality for standardizing
base estimates.

In Damon.base_ear(), changed how Damon is used to calculate standard
errors, opting for a 2-dimensional log solution.

In Damon.equate(), changed the SE and EAR aggregation formula to sum across
log coordinates.  This causes a slight negative error estimation bias 
in equate() relative to the RMSE type of aggregation used in summstat(). 
Added a "group_se" option. 

In Damon.summstat(), added a "group_se" option to control the aggregation
of standard errors for measures.  


Modules
-------
The damon1 package is found within the Python site-packages
directory, though the precise pathname varies by machine and
distribution.  It is composed of these modules:

    core.py         =>  defines the Damon class and its methods
                        as well as some important non-Damon
                        functions, such as create_data() and
                        predamon().  It also holds a lot of
                        documentation.

    tools.py        =>  defines a series of generic functions that are
                        not restricted to Damon objects, though they
                        are used heavily by Damon methods.

    utils.py        =>  contains all the heavy-duty code (and none of
                        the documentation) in the form of functions
                        called by the methods defined in core.  So
                        utils is the back-end of core.

    template.py     =>  provides a template for writing Damon apps,
                        with useful documentation and the necessary
                        import statements.

    doc             =>  a folder within damon1 that (will) contain
                        documentation and tutorials (under construction).

    tests           =>  a folder within damon1 that contains unit-tests
                        for important methods and functions.  It also
                        contains a ut_template.py file (unit-test template)
                        for constructing unit tests.  They allow you
                        to test any or all combinations of parameters
                        that may go into a function or method, as well
                        as a wide variety of setup scenarios.


Damon, Alternating Least Squares, NOUS, and Object-Oriented Statistics
----------------------------------------------------------------------
Damon, an implementation of NOUS Object-Oriented Statistics, locates row
objects (such as "persons") and column objects (such as "items")
in a space of N specified dimensions using a matrix decomposition method
worked out initially by aerospace engineer Howard A. Silsdorf in 2003
and expanded by Mark H. Moulton, a psychometrician and
author of Damon.  It turns out, in fact, to be a reinvention of
a matrix decomposition algorithm known as "Alternating Least Squares"
(ALS) that has been floating around in various forms since at least the
1970's, made famous by its use in the Netflix contest in 2009.  Suitably
applied, it specifies an "objective model", as opposed to a "descriptive
model".

"NOUS" refers to the particular flavor of objective models that Mark Moulton
has been working on.  "Damon" refers to software he has written to
implement these models. The term "Object-Oriented Statistics" (OOS)
refers to Danish mathematician Georg Rasch's foundational notion of
"special objectivity" (Rasch, 1960), developed in the context of
psychometrics.  OOS is intended to guide the generation
of statistics that are invariant across samples ("objective"), regardless of
the shape of the distribution or how it is selected.  In this way OOS differs
from regression, neural networks, clustering and Bayesian approaches, which
tend to produce results dependent to varying degrees on the data samples
they are applied to.  Rasch models, NOUS, and to a lesser degree established
matrix decomposition models such as Singular Value Decomposition, are examples
of Object-Oriented Statistical models -- when properly applied, they yield
results that generalize across samples.  In point of fact, all of these models
have strong computational similarities.  "Object-orientation" best describes
the GOAL of sample independence and a systematic MEANS to approach it.

Object-Oriented Statistical Models are used in knowledge discovery (data
mining), artificial intelligence, psychometrics, survey research, and other
fields to generate statistics that extend robustly across multiple
real and hypothetical data sets.  NOUS comes from the field of psychometrics
and may, with a little squinting, be thought of as a multidimensional
generalization of the Rasch model.  Alternatively, the Rasch model may
be thought of as a probabilistic, 1-dimensional, non-negative matrix
decomposition algorithm that is used to identify entities that may be
used interchangeably within a given domain.  NOUS initial applications
have been in the field of education -- measuring student abilities along
multiple dimensions simultaneously -- but it is of course applicable to
any field.

The field of psychometrics has produced a variety of important multidimensional
models under the rubric of Multidimensional Item Response Theory (MIRT)
(see Reckase's book of that name, 2009).  There are also multidimensional
Rasch models.  Damon employs an algorithm that is quite different from
those used to implement most MIRT models, though it has the same goal
as Rasch models (objectivity) and some important algorithmic similarities.
Damon's ALS algorithm is closer to the matrix decomposition techniques
used increasingly in data mining.  Damon is meant to be something of a
a bridge connecting the objectivity definition of Rasch psychometrics
to modern data mining.

The way Damon works is that it computes spatial coordinates for each object
(or "entity") using an iterative multidimensional least squares algorithm
that alternates between row and column coordinates, refining them until fit
is maximized between the observations and the model's estimates for a given
number of dimensions.  The optimal dimensionality is determined by making data
"pseudo-missing" for a range of possible dimensionalities, and picking the
number of dimensions that best predicts the pseudo-missing cells.  Damon
draw heavily on its ability to circumvent missing data -- ignore it
completely -- by analyzing each row and column entity independently of the
others with only the data it contains.  When iteration is complete, each row
and column entity has its own set of spatial coordinates.

These are used to:
    a) predict values for missing cells;
    b) replace observed values with more likely expected values;
    c) measure objects on specified scales;
    d) combine data sets through common entities.

As yet, Damon coordinates are not used for graphing or interpreting.
They are purely abstract.  While they do yield objective estimates
(when the data fit the model), the coordinate system itself -- its
origin and orientation -- is quite arbitrary (although it can be
conditioned to compute non-arbitrary cosines between entities).
However, it turns out that the arbitrariness of the coordinate
system is irrelevant to two very important functions of objective
methods:  1) forcing different datasets to share the same coordinate
structure through "anchoring"; 2) generating cell measures and
predictions that are sample-independent.

Thus, when the data are determined to "fit" the spatial model by accurately
predicting values that are made missing, Damon measurements and predictions
have the property of "objectivity," -- they are reproducible
across different data sets.

Further documentation on how this works can be found in the documentation
associated with each Damon method.  Mathematical proofs and analysis
procedures can be obtained from Pythias Consulting directly.


Importing Damon and Accessing Documentation
-------------------------------------------
A tutorial can be found on the Pythias website.  Most of the
documentation is found at the level of individual functions
and methods and can be accessed using help().

Top-level documentation (this doc) can be accessed using:

>>>  import damon1
>>>  help(damon1)

Documentation at the module (file) level is accessible by
either typing:

>>>  help(damon1.core)  which is a LOT of docs

For specific classes/functions/methods:

>>>  help(damon1.core.TopDamon)  or
>>>  help(damon1.core.Damon.__init__)  or
>>>  help(damon1.core.Damon.merge)

It is often easier to import core or tools first, under
another name:

>>>  import damon1.core as dmn
>>>  help(dmn.TopDamon)
>>>  help(dmn.Damon.__init__)
>>>  help(dmn.Damon.merge)

>>>  import damon1.tools as dmnt
>>>  help(dmnt.correl)

To get a top-level list of contents in core:

>>>  import damon1.core as core
>>>  dir(core)
['Damon', 'TopDamon', '__builtins__', '__doc__', '__file__', '__name__',
'__package__', 'cPickle', 'core', 'create_data', 'np', 'npla', 'npma',
'npr', 'os', 'sys', 'tools', 'utils']

To get a list of functions in tools:

>>> import damon1.tools as dmnt
>>> dir(dmnt)
['__builtins__', '__doc__', '__file__', '__name__', '__package__',
'addlabels', 'cPickle', 'condcoord', 'core', 'correl', 'cumnormprob',
'damon_dicts', 'dups', 'estimate', 'faccoord', 'fit', 'fitsd',
'getkeys', 'guess_validchars', 'h_stat', 'homogenize', 'invUTU',
'irls', 'jolt', 'log2prob', 'mergetool', 'metricprob', 'np', 'npla',
'npma', 'npr', 'objectivity', 'obsdeltatest', 'obspercell', 'os',
'percent25', 'percent75', 'ptbis', 'pytables', 'rand_chunk',
'reliability', 'rescale', 'residuals', 'resp_prob', 'rmsear',
'rmsr', 'separation', 'solve1', 'solve2', 'sterrpbc', 'subscale_filter',
'sys', 'test_damon', 'tools', 'triproject', 'tuple2table', 'unbiascoord',
'unbiasest', 'utils', 'valchars', 'weight_coord', 'zeros_chunk']

To get a list of Damon methods (functions belong to each Damon
object):

>>> import damon1.core as dmn
>>> dir(dmn.Damon)
['__class__', '__delattr__', '__dict__', '__doc__', '__format__',
'__getattribute__', '__getitem__', '__hash__', '__init__', '__module__',
'__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
'__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'bank',
'base_ear', 'base_est', 'base_fit', 'base_resid', 'base_se',
'combokeys', 'coord', 'equate', 'est2logit', 'export', 'extract',
'extract_valid', 'fillmiss', 'fin_ear', 'fin_est', 'fin_fit',
'fin_resid', 'fin_se', 'item_diff', 'merge', 'merge_info', 'objectify',
'parse', 'pseudomiss', 'rasch', 'restore_invalid', 'score_mc',
'standardize', 'subscale', 'summstat', 'table2tuple', 'transpose']


To get help for just the Damon initialization method (used A LOT):

>>>  import damon1.core as dmn
>>>  help(dmn.Damon.__init__)

To get help with the create_data() function (not a Damon method):

>>>  help(dmn.create_data)

To get help with TopDamon(), very useful as a way to preformat
messy data as a preliminary Damon object:

>>>  help(dmn.TopDamon)

To get help for the coord() Damon method:

>>>  help(dmn.Damon.coord)

To get help with the addlabels() function in tools:

>>>  import damon1.tools as dmnt
>>>  help(dmnt.addlabels)

Don't be alarmed when you get help and see a lot of text wrapping.
This is just the line documentation for each argument in the function/method.
Paste it into your python script and it will look fine.

Help with Tables and Graphs
---------------------------
Sometimes, you will want to print tabular arrays, with
labels and columns properly aligned.  Use the tabulate()
function for this.  tabulate() is automatically imported
by the templates, or you can import it manually:

>>>  from tabulate import tabulate
>>>  import numpy as np
>>>  my_array = np.random.randint(0, 9, (5, 3))
>>>  print tabulate(my_array, 'firstline')

This will return a nice table, treating the first line as
column headers.  For informaton on how to customize tabulate()
further, type:

>>>  help(tabulate)

You will also want to create charts and graphs using matplotlib.
After importing one of the templates, type:

>>>  help(plt.plot)

The plt.plot() function will serve most of your purposes and
introduce you to the wonderful matplotlib library.


Easiest Way to Get Started
--------------------------
In IDLE, under file/Open Module, type:

damon1.templates.coord_0

A pre-formatted Python script window will pop up.  Save it under a
different name where you want it.  When you hit F5 (run), all
the appropriate modules will be loaded and a sample Damon analysis
will be performed centered around the coord() method.  To run your
own analyses, change the filename and tweak the parameters as desired.

There are other templates for other types of analysis, as well as a
"blank" template that just does imports.


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
d.merge_summstat()              =>  Merge multiple summstat() runs
d.plot_two_vars()               =>  Plot two variables against each other
d.wright_map()                  =>  Plot person and item distributions
d.bank()                        =>  Save row/column coordinates in "bank" file
d.export()                      =>  Export specified outputs as files
d.to_dataframe()                =>  Convert datadict to a Pandas DataFrame
d.flag()                        =>  Flag persons and items using statistics
d.extract()                     =>  Extract a subset of a Damon object
d.transpose()                   =>  Transpose a Damon object

"""

##import damon1.helper as helper
##import damon1.validation as validation
##import damon1.unittests_generator as unittests_generator
##import damon1.models as models
##import damon1.code_generator as code_generator
##
##try:
##    import damon1.utils_R as utils_R
##    import damon1.core_R as core_R
##    import damon1.models_R as models_R
##    R_flag = True
##except ImportError:

R_flag = False

import damon1.core as core
import damon1.tools as tools
import damon1.utils as utils

__all__ = ['core','tools','utils','helper','validation','utils_R','core_R',
           'models_R','unittests_generator','models','code_generator']

__version__ = '1.2.01'
