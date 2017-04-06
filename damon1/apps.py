# -*- coding: utf-8 -*-
"""
The apps.py module contains Damon "applications" built
to address specific use-cases.

It is a module in the damon1 package.

"""
# Import system modules
import os
import sys

# Import numpy and other python modules
#import cPickle
import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.ma as npma
import ast

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import damon1 as dmn
from damon1.tabulate import tabulate


def simple_nous(filename,
                delimiter = r'\t',
                n_headers_for_cols = 2,
                column_keys = 0,
                n_headers_for_rows = 1,
                row_keys = 0,
                valid_characters = ['All', ['All'], 'Num', 'Guess'],
                dimensionality = range(1, 11),
                report_as_logits = False,
                display = ['item_stats', 'person_stats', 'objectivity_curve', 'slice_plot'],
                slice_plot = {'x_name':3, 'y_name':4, 'cosine_correct':True},
                how_to_color_persons = 'rand',
                how_to_color_items = 'rand',
                output_prefix = 'simple_nous',
                verbose = True
                ):
    """Apply nous to a dataset.

    """

    # Load the data file
    d = dmn.Damon(data = filename,    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                  format_ = 'textfile',    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                  workformat = 'RCD_dicts_whole',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                  validchars = validid_characters,   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                  nheaders4rows = n_headers_for_rows,  # [number of columns to hold row labels]
                  key4rows = row_keys,   # [<None, nth column from left which holds row keys>]
                  rowkeytype = 'S60',     # [<None, type of row keys>]
                  nheaders4cols = n_headers_for_cols,  # [number of rows to hold column labels]
                  key4cols = column_keys, # [<None, nth row from top which holds column keys>]
                  delimiter = delimiter,  # [<None, character to delimit input file columns (e.g. ',' for .csv and '  ' for .txt tab-delimited files)]                  
                  verbose = verbose,    # [<None, True> => report method calls]
                  )

    if d.verbose is True:
        print '\nCheck the data\n'
        print d

    # Standardize data
    d.standardize()

    # Calculate coordinates, estimates, residuals, errors, fit
    d.coord([dimensionality])
    d.base_est()
    d.base_resid()
    d.base_ear()
    d.base_se()
    d.base_fit()

    # Convert to logits
    if report_as_logits is True:
        d.est2logit()

    # Calculate summary statistics
    d.summstat(data = 'est2logit_out' if report_as_logits is True else 'base_est_out', # [<'base_est_out','fin_est_out','est2logit_out','data_out',...> => data for calculating stats]
             getstats = ['Mean', 'SE', 'Corr', 'Resid', 'Rel', 'Outfit', 'Min', 'Median', 'Max'], # [Select stats from [<'Mean','SE','SD','Corr','Resid','RMSEAR','Sep','Rel','Outfit','Fit_Perc>2','Count','Min','25Perc','Median','75Perc','Max','Coord'>] ]
             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}>]
             getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}>]
             itemdiff = True if report_as_logits is True else None,   # [<None,True> => express item (column) mean as a probabilistically defined item "difficulty" ]
             outname = None,    # [<None,'cluster_name'> => name to append to summstat outputs for multiple analyses]
             labels = {'row_ents':'person', 'col_ents':'item'}   # [<None, {'row_ents':<None, 'person',...>, 'col_ents':<None, 'item',...>}> => to describe summarized entities]
             )

    # Export output statistics
    d.export(outputs = ['row_ents_out', 'col_ents_out'],
               outprefix = output_prefix,    # [string prefix to all file names, may be a path to a designated directory]
               outsuffix = '.txt',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
               delimiter = '\t', # [<None,text delimiter, e.g. ',' or '      '>]
               )

    if 'objectivity_curve' in display:
        print '\n\n'
        print damon1.tools.print_objperdim(d, output_prefix+'_objectivity.txt', 'show')
        print 'Best dimensionality: ', d.bestdim


    # Print item stats
    if 'item_stats' in display:
        print '\n\nItem Statistics\n'
        print tabulate(d.col_ents_out.whole,
                       headers = 'firstrow'
                       )
        
    # Print item stats
    if 'person_stats' in display:
        print '\n\nPerson Statistics\n'
        print tabulate(d.row_ents_out.whole,
                       headers = 'firstrow'
                       )

    # Plot two items
    if slice_plot is not None:
        d.plot_two_vars(xy_data = 'est2logit_out' if report_as_logits is True else 'base_est_out', # ['my_datadict_out', e.g., 'merge_summstat_out']
                        x_name = slice_plot['x_name'],    # [name of x variable to use for x-axis]
                        y_name = slice_plot['y_name'],    # [name of y variable to use for y-axis]
                        ent_axis = 'col',    # [<'row', 'col'>] => how variables are situated]
                        err_data = 'logit_se_out' if report_as_logits is True else 'base_se_out',  # [<None, 'my_datadict_out'> => e.g., 'merge_summstat_out']
                        x_err = slice_plot['x_name'],    # [<None, size, name of x variable to use for error statistic> => create bubbles]
                        y_err = slice_plot['y_name'],    # [<None, size, name of y variable to use for error statistic> => create bubbles]
                        color_by = how_to_color_persons,  # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                        cosine_correct = 'coord|corr' if cosine_correct is True else None,    # [<None, 'coord', 'corr', 'coord|corr'> => correct for cosine between variables]
                        max_cos = 0.99,    # <unsigned corr> => trigger exception when cos(x, y) > max_cos]
                        plot = {'se_size':se_size, 'xy_labels':xy_labels},  # [<None, {plotting parameters}> => see docs to customize]
                        savefig = output_prefix+'item_plot.png'  # [<None, 'show', 'filename.png'> => how to output chart]
                        )

    return d





#def rasch_w_banking
