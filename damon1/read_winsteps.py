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
to either the Gnu Affero General Public License or the Pythias
Commercial License, a copy of which is contained in the current
working directory.

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

import csv


def read_winsteps(data):
    """Convert Winsteps control file in Damon object

    Returns
    -------
        {'data':Damon object,
         'anskey':answer key
         }

    Comments
    --------
        This function was a quick and dirty effort to
        read a Winsteps control file for a particular case.
        It probably won't work on your files without some
        editing.  Save a copy and edit it to fit your situation.

    Arguments
    ---------
        "data" is a path name to a Winsteps control file that
        contains both specifications and data.

    """

    clean_lines = []

    # Get clean list of lines, capturing some variables
    with open(data, 'rb') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.replace('"',"").strip()
            clean_lines.append(line)

            if 'Item1' in line:
                start_resp = int(line[line.find('Item1') + 6:]) - 1

            if 'Name1' in line:
                start_name = int(line[line.find('Name1') + 6:]) - 1

            if 'Codes' in line:
                validchars_ = line[line.find('Codes') + 6:]
                
            if 'Key' in line:
                key = line[line.find('Key') + 4:]

            if '&END' in line:
                start_items = i + 1

            if 'END NAMES' in line:
                stop_items = i
                start_data = i + 1

    # Get varianbles
    items = clean_lines[start_items:stop_items]
    validchars = ['All', list(validchars_)]
    anskey = dict(zip(items, list(key)))

    data_lines = clean_lines[start_data:]

    persons = []
    person_resps = []
    nitems = len(items)

    # Read the data file, parse out persons
    for line in data_lines:
        x = line[start_name:start_resp].strip()
        person = x.replace(' ', '')    # Remove gaps in person ids (temp)
        persons.append(person)

        resps = list(line[start_resp:start_resp + nitems])
        person_resps.append(resps)

    # Convert into arrays
    persons.insert(0, 'id')
    items.insert(0, 'id')

    rowlabels = np.array(persons)[:, np.newaxis]
    collabels = np.array(items)[np.newaxis, :]
    coredata = np.array(person_resps)

    # Build datadict for Damon
    datadict = {'rowlabels':rowlabels,
                'collabels':collabels,
                'coredata':coredata,
                'nheaders4rows':1,
                'key4rows':0,
                'rowkeytype':'S60',
                'nheaders4cols':1,
                'key4cols':0,
                'colkeytype':'S60',
                'validchars':validchars,
                'nanval':'-999',
                }

    d = dmn.Damon(datadict, 'datadict', verbose=True)

    return {'data':d,
            'anskey':anskey}
                






##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
if __name__ == "__main__":
    
    workfile = 'Mark_Medical_Con.txt'
    a = read_winsteps(workfile)
    d = a['data']
    ak = a['anskey']

    d.score_mc(['Cols', ak])
    print 'd.score_mc_out=\n', d.score_mc_out['coredata']

    d.standardize()
    d.coord([[3]])
    d.base_est()
    d.base_resid()
    d.base_ear()
    d.base_se()
    d.base_fit()
    d.est2logit()
    d.summstat('base_est_out', ['Mean', 'SD', 'SE', 'Rel',
                                 'Fit_MeanSq', 'Fit_Perc>2',
                                 'Count', 'Min', 'Max'],
               getrows='SummWhole')

    d.export(['row_ents_out', 'col_ents_out'], outprefix='Medical')

##    dim = d.objperdim.core_col['Dim']
##    acc = d.objperdim.core_col['Acc']
##    stab = d.objperdim.core_col['Stab']
##    obj = d.objperdim.core_col['Obj']
##
##    plt.plot(dim, acc, 'r-', label='Accuracy')
##    plt.plot(dim, stab, 'b-', label='Stability')
##    plt.plot(dim, obj, 'k-', label='Objectivity')
##    plt.xlabel('Dimensionality')
##    plt.ylabel('Objectivity Stats')
##    plt.legend()
##    plt.savefig('read_winsteps.png')
##    plt.clf()

    
    

    #d.export(['data_out'], outsuffix = '.txt', delimiter='\t')


    
















