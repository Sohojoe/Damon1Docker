# -*- coding: utf-8 -*-
"""
The tools.py module contains generic functions that
are not specific to the Damon class, but used by it.

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

##try:
##    import matplotlib.pyplot as plt
##except ImportError:
##    pass

from tabulate import tabulate

# Import Damon utilities
import damon1 as dmn
import tester as ut

# Define exception classes
class getkeys_Error(Exception): pass
class addlabels_Error(Exception): pass
class mergetool_Error(Exception): pass
class valchars_Error(Exception): pass
class rescale_Error(Exception): pass
class invUTU_Error(Exception): pass
class condcoord_Error(Exception): pass
class solve1_Error(Exception): pass
class solve2_Error(Exception): pass
class jolt_Error(Exception): pass
class faccoord_Error(Exception): pass
class get_unique_weight_Error(Exception): pass
class resp_prob_Error(Exception): pass
class residuals_Error(Exception): pass
class obspercell_Error(Exception): pass
class cumnormprob_Error(Exception): pass
class log2prob_Error(Exception): pass
class metricprob_Error(Exception): pass
class separation_Error(Exception): pass
class reliability_Error(Exception): pass
class stability_in_coord_Error(Exception): pass
class rmsr_Error(Exception): pass
class ptbis_Error(Exception): pass
class pytables_Error(Exception): pass
class test_damon_Error(Exception): pass
class frequencies_Error(Exception): pass
class print_objperdim_Error(Exception): pass
class cosine_Error(Exception): pass
class lookup_coords_Error(Exception): pass
class check_datadict_Error(Exception): pass
class lookup_group_colors_Error(Exception): pass




###########################################################################

def getkeys(datadict,   # [data parsed as a datadict]
            facet = 'Row',   # [<'Row','Col'> => row keys or col keys]
            range_ = 'All',  # [<'All','Core',slice> => 'Core' means keys for just data portion of labels]
            type_ = 'Auto',    # [<'Auto',type> => type to which to cast keys]
            strict = None,  # [<None,True> => return error if keys don't cast to specified type]
            ):
    """Pull key identifiers from a datadict.

    Returns
    -------
        key() returns a numpy array of key identifiers drawn
        from either the rowlabels or collabels array of
        a datadict, cast either to the type specified in the
        datadict or to the least general type possibe for
        the specified range_.

    Comments
    --------
        It is often necessary to build an array of keys (unique
        row or column identifiers).  The datadict format (rowlabels,
        collabels, coredata, other variables) provides information
        necessary to get these keys, but the syntax for reading
        this information is not always entirely obvious.

        For instance, the syntax

            colkeys = collabels[key4cols,:].astype(colkeytype)

        generally works, but not always.  If colkeytype is "int"
        and collabels contains other rows with floats and strings,
        the keys may appear in the collabels array as ['0.0','1.0',
        '2.0',...].  When numpy tries to cast these directly to
        ints, it returns an error.  The jump from string to float
        to int is too large.  The syntax that works is

        colkeys = collabels[key4cols,:].astype(float).astype(colkeytype)

        getkeys() is intended to standardize the extraction of
        keys by centralizing it in one function and to save the
        user from having to think through type-casting scenarios.
        The basic rule is this:

            Rule:   Damon will try to make each key in a list
                    of keys controlled by the range_ parameter be
                    the type specified by the datadict's rowkeytype or
                    colkeytype.  If a given key cannot be cast to
                    this type, it will be cast to the next most general
                    type (generally string of some kind).  getkeys()
                    will then return the elements in this list
                    as a Numpy array, whose elements will all be
                    assigned the least general type necessary to
                    accommodate all the key types in the list.

        Thus, if rowkeytype is int but the header of the key column
        is the string 'ID', and the range is 'All', getkeys() returns:

            ['ID','1','2','3','4']

        However, if the range is 'Core', it returns:

            [1,2,3,4]

        It may be wondered why datadicts and Damon objects don't just
        store the keys at the specified type.  The answer is that sometimes
        the user needs a little extra control over key extraction.
        For instance, the keys for the core portion of a rowlabels
        array may be integer, with rowkeytype = int, while one of
        the values in the non-core portion of the array is string
        (say, 'PersonID').  Attempts to cast the whole array of
        keys to int fail.  However, by specifying range_ = 'Core'
        we can extract just the keys corresponding to the data
        portion of the array and cast them to int without error.

        getkeys() does not check that the keys are unique.  This
        is handled in Damon.__init__ and can be done manually
        using tools.dup().

    Arguments
    ---------
        "datadict" is a dictionary of arrays consisting of 'coredata',
        'rowlabels', 'collabels' and additional descriptor variables
        such as 'nheaders4rows','key4rows','rowkeytype', and so on.
        They are a standard output of every Damon method.

        ------------
        "facet" <'Row','Col'> is the type of keys you want to
        extract -- row entity keys or column entities keys.

        ------------
        "range_" allows you to select a specific subset of keys
        prior to casting them to the desired type.

            range_ = 'All'      =>  Draw keys from the
                                    entire range of labels:

                                    rowlabels[:,key4rows]
                                    collabels[key4cols,:]

            range_ = 'Core'     =>  Draw keys from the
                                    that range of labels that
                                    corresponds to the core data:

                                    rowlabels[nheaders4cols:,key4rows]
                                    collabels[key4cols,nheaders4rows:]

            range_ = slice(2,10)=>  Draw keys using the
                                    indicated slice object, e.g.,
                                    from position 2 (counting from 0)
                                    up to but not including position
                                    10).  Equivalent:

                                    rowlabels[2,10]

            range_ = slice(2,None,2)
                                =>  Slice objects use "None" to mean
                                    "go to the end" and can include a
                                    step indicator.  In this case,
                                    we want alternate keys starting from
                                    2 and going to the end.  Equivalent:

                                    rowlabels[2::2]

        ------------
        "strict" <None,True> indicates whether to return an exception
        if the specified range of keys cannot be cast to the specified
        type.

            strict = None       =>  If the specified keys cannot be cast
                                    to the specified type (say, int), then
                                    getkeys() will automatically cast them
                                    to string.

            strict = True       =>  If the specified keys cannot be cast
                                    to the specified type (say, int), then
                                    getkeys() will return an exception.

    Examples
    --------

        [under construction]

    Paste function
    --------------
        getkeys(datadict,   # [data parsed as a datadict]
                facet = 'Row',   # [<'Row','Col'> => row keys or col keys]
                range_ = 'All',  # [<'All','Core',slice> => 'Core' means keys for just data portion of labels]
                type_ = 'Auto',    # [<'Auto',type> => type to which to cast keys]
                strict = None,  # [<None,True> => return error if keys don't cast to specified type]
                )

    """

    # Check datadict
    D = datadict
    if isinstance(D,dmn.core.Damon):
        D = D.data_out

    if 'nheaders4rows' not in D.keys():
        D['nheaders4rows'] = np.size(D['rowlabels'],axis=1)
    if 'nheaders4cols' not in D.keys():
        D['nheaders4cols'] = np.size(D['collabels'],axis=0)

    str_warn = False

    # Set slice
    if facet == 'Row':
        Labels = D['rowlabels'][:,D['key4rows']]
        if range_ == 'All':
            Slice = slice(None,None)
        elif range_ == 'Core':
            Slice = slice(D['nheaders4cols'],None)
        else:
            Slice = range_

        # Set type
        if type_ == 'Auto':
            Type_ = D['rowkeytype']

            # Work-around to circumvent Numpy bug: overwrite of alphas when casting string slice to int
            if (Type_ == int
                or Type_ == long
                or Type_ == np.int
                or Type_ == np.long
                ):
                Heads =  Labels[Slice][:D['nheaders4cols']]
                for i in range(np.size(Heads)):
                    try:
                        int(Heads[i])
                    except ValueError:
                        str_warn = True
                        break
        else:
            Type_ = type_

    elif facet == 'Col':
        Labels = D['collabels'][D['key4cols'],:]
        if range_ == 'All':
            Slice = slice(None,None)
        elif range_ == 'Core':
            Slice = slice(D['nheaders4rows'],None)
        else:
            Slice = range_

        # Set Type2
        if type_ == 'Auto':
            Type_ = D['colkeytype']

            # Work-around to circumvent Numpy bug: overwrite of alphas when casting string slice to int
            if (Type_ == int
                or Type_ == long
                or Type_ == np.int
                or Type_ == np.long
                ):
                Heads =  Labels[Slice][:D['nheaders4rows']]
                for i in range(np.size(Heads)):
                    try:
                        int(Heads[i])
                    except ValueError:
                        str_warn = True
                        break
        else:
            Type_ = type_
    else:
        exc = 'Unable to figure out facet parameter.\n'
        raise getkeys_Error(exc)

    # Define function to cast individual keys
    def keylist(slice_,type_,strict):
        keys = []
        for key in slice_:
            try:
                k = type_(key)
                keys.append(k)
            except:
                try:
                    k = float(key)
                    j = type_(k)
                    keys.append(j)
                except:
                    if strict is not True:
                        k = str(key)
                        keys.append(k)
                    else:
                        exc = 'Unable to cast key to specified type.\n'
                        raise getkeys_Error(exc)
        return keys

    # Get keys and cast to specified type
    if str_warn is True:
        Keys = keylist(Labels[Slice],Type_,strict)
    else:
        try:
            if Type_ == str:
                Type_1 = 'S60'
            else:
                Type_1 = Type_
            Keys = list(Labels[Slice].astype(Type_1))

        except ValueError:
            Keys = keylist(Labels[Slice],Type_,strict)

    # Convert to Numpy array -- automatically the least general possible type
    Keys_arr = np.array(Keys,dtype=None)

    # If string, convert to 'S60' to avoid truncation of ints
    try:
        if isinstance(Keys_arr[0],str):
            Keys_arr = np.array(Keys,dtype='S60')
    except IndexError:
        print 'Warning in getkeys():  the keys array is empty for some reason.\n'
        print 'Labels = \n',Labels
        print 'Slice = ',Slice
        print 'Labels = \n', Labels
        print 'Keys = \n', Keys
        print 'Keys_arr = \n', Keys_arr
        pass

    return Keys_arr




###########################################################################

def damon_dicts(coredata,   # [see Damon.__init__() docs]
                rowlabels,   # [...]
                nheaders4rows,   # [...]
                key4rows,   # [...]
                rowkeytype,   # [...]
                collabels,   # [...]
                nheaders4cols,   # [...]
                key4cols,   # [...]
                colkeytype,   # [...]
                range4labels = 'Core',  # [<'All','Core'> => range of label dicts]
                strip_labkeys = None,   # [<True,None> => strip keys from row/col label dicts]
                whole = None,   # [<None,whole array> => build whole array dict]
                ):
    """Build dictionaries to provide key access to Damon formatted arrays.

    Returns
    -------
        damon_dicts returns a dictionary of 8 possible
        Damon dictionaries, built to provide key access
        to the rows and columns of the rowlabels, collabels,
        coredata, and whole arrays.

        out = {'rl_row',
               'rl_col',
               'cl_row',
               'cl_col',
               'core_row',
               'core_col',
               'whole_row',
               'whole_col'
               }

    Comments
    --------
        damon_dicts() is used by Damon internally in the
        Damon.__init__() method and the merge() method to
        build dictionaries useful for accessing rows and
        columns of Damon's rowlabels, collabels, coredata,
        and whole arrays.

        For a description of each dictionary, consult:

            >>>  import damon1.core as dmn
            >>>  help(dmn.Damon.__init__)

        damon_dicts() provides some flexibility in how
        the labels dictionaries are built.  They can be
        built so that the appropriate label keys return
        only labels corresponding to the "core" section
        of the dataset (what Damon.__init__() does).
        Or the appropriate label keys can return all the
        labels in the corresponding row or column (what
        Damon.merge() does).

        In addition, rowlabel and collabel dictionaries
        can be returned with the keys removed from their
        values.

        There is also an option not to return dictionaries
        for the whole array.  This is because building the
        whole array can be computationally expensive and is
        often skipped.

    Arguments
    ---------
        The coredata, rowlabels, nheaders4rows, key4rows,
        rowkeytype, collabels, nheaders4cols, key4cols, and
        colkeytype parameters are equivalent to the corresponding
        values in any datadict.  See Damon.__init__() for more
        documentation.

        --------------
        "range4labels" is used to specify whether the labels
        accessed from the rl_row, rl_col, cl_row, and cl_col
        dictionaries should contain all possible labels in that
        row or column or just those corresponding to the "core"
        data section of the array.

            range4labels = 'Core'   =>  Capture only core labels.

            range4labels = 'All'    =>  Capture all available labels.

        The 'Core' option is more user-friendly for most purposes.

        --------------
        "strip_labkeys" <None,True> causes the row keys to be stripped
        from the rowlabel values when building the rl_row dictionary
        and the column keys to be stripped from the collabels values
        when building the cl_col dictionary.  All other dictionaries
        are left unchanged.

        --------------
        "whole" is the whole array -- combining rowlabels, collabels,
        and coredata in one array.  If whole = None, the resulting
        whole dictionary is None.


    Examples
    --------

        [under construction]

    Paste Function
    --------------
        damon_dicts(coredata,   # [see Damon.__init__() docs]
                    rowlabels,   # [...]
                    nheaders4rows,   # [...]
                    key4rows,   # [...]
                    rowkeytype,   # [...]
                    collabels,   # [...]
                    nheaders4cols,   # [...]
                    key4cols,   # [...]
                    colkeytype,   # [...]
                    range4labels = 'Core',  # [<'All','Core'> => range of label dicts]
                    strip_labkeys = None,   # [<True,None> => strip keys from row/col label dicts]
                    whole = None,   # [<None,whole array> => build whole array dict]
                    )

    """
    datadict = locals().copy()

    # Index stripped label keys
    if strip_labkeys is True:
        r_ind = np.arange(nheaders4rows)
        rl_val_ind = r_ind[r_ind != key4rows]
        c_ind = np.arange(nheaders4cols)
        cl_val_ind = c_ind[c_ind != key4cols]
    else:
        rl_val_ind = slice(None,None)
        cl_val_ind = slice(None,None)

    # Set paramaters for rl and cl dicts only
    if range4labels == 'Core':
        rl_val_slice = [slice(nheaders4cols,None),rl_val_ind]
        cl_val_slice = [cl_val_ind,slice(nheaders4rows,None)]

        rl_val_slice_unstripped = [slice(nheaders4cols,None),slice(None,None)]
        cl_val_slice_unstripped = [slice(None,None),slice(nheaders4rows,None)]
        key_range = 'Core'

    elif range4labels == 'All':
        rl_val_slice = [slice(None,None),rl_val_ind]
        cl_val_slice = [cl_val_ind,slice(None,None)]

        rl_val_slice_unstripped = [slice(None,None),slice(None,None)]
        cl_val_slice_unstripped = [slice(None,None),slice(None,None)]
        key_range = 'All'

    # Build rl_row dict
    rl_row_keys = getkeys(datadict,'Row',key_range,'Auto',None)
    rl_row_vals = rowlabels[:,:][rl_val_slice]
    rl_row = dict(zip(rl_row_keys,rl_row_vals))

    # Build rl_col dict
    rl_key_slice = slice(None,nheaders4rows)
    rl_col_keys = getkeys(datadict,'Col',rl_key_slice,'Auto',None)
    rl_col_vals = np.transpose(rowlabels[:,:][rl_val_slice_unstripped])
    rl_col = dict(zip(rl_col_keys,rl_col_vals))

    # Build cl_row dict
    cl_key_slice = slice(None,nheaders4cols)
    cl_row_keys = getkeys(datadict,'Row',cl_key_slice,'Auto',None)
    cl_row_vals = collabels[:,:][cl_val_slice_unstripped]
    cl_row = dict(zip(cl_row_keys,cl_row_vals))

    # Build cl_col dict
    cl_col_keys = getkeys(datadict,'Col',key_range,'Auto',None)
    cl_col_vals = np.transpose(collabels[:,:][cl_val_slice])
    cl_col = dict(zip(cl_col_keys,cl_col_vals))

    # Build core_row dict
    core_row_keys = getkeys(datadict,'Row','Core','Auto',None)
    core_row_vals = coredata[:,:]
    core_row = dict(zip(core_row_keys,core_row_vals))

    # Build core_col dict
    core_col_keys = getkeys(datadict,'Col','Core','Auto',None)
    core_col_vals = np.transpose(coredata[:,:])
    core_col = dict(zip(core_col_keys,core_col_vals))

    # Build "whole" dicts
    if whole is not None:

        # Build whole_row
        whole_row_keys = getkeys(datadict,'Row','All','Auto',None)
        whole_row_vals = whole[:,:]
        whole_row = dict(zip(whole_row_keys,whole_row_vals))

        # Build whole_col
        whole_col_keys = getkeys(datadict,'Col','All','Auto',None)
        whole_col_vals = np.transpose(whole[:,:])
        whole_col = dict(zip(whole_col_keys,whole_col_vals))

    else:
        whole_row = None
        whole_col = None

    out = {'rl_row':rl_row,
           'rl_col':rl_col,
           'cl_row':cl_row,
           'cl_col':cl_col,
           'core_row':core_row,
           'core_col':core_col,
           'whole_row':whole_row,
           'whole_col':whole_col
           }

    return out


###########################################################################

def index_val(arr, val):
    """Get bool index of a value in an array.
    
    Returns
    -------
        Boolean index of a value in an array
    
    Comments
    --------
        This flags nanval, np.nan, and other characters for numerical, 
        string, and object array dtypes.  This function was made 
        necessary due to recent deprecation warnings triggered when 
        the array dtype is different from value type.
    
    Arguments
    ---------
        "arr" is a numpy array.
        
        -------
        "val" is an int, float, or string.
    
    Paste Function
    --------------
        index_val(arr, val)

    """
    if arr.dtype.kind in ['i', 'f']:
        val = float(val)
        ix = (arr == val) | (np.isnan(arr))
    elif arr.dtype.kind == 'S':
        nv0, nv1 = str(val), 'nan'
        try:
            nv2 = str(float(val))
        except ValueError:
            nv2 = str(val)
        ix = (arr == nv0) | (arr == nv1) | (arr == nv2)
    elif arr.dtype.kind == 'O':
        ix = arr == val
    else:
        exc = 'Unable to index locations of value in array.'
        raise TypeError(exc)


    return ix
              

###########################################################################

def apply_val(arr, index, val):
    """Apply a value to an array.
    
    Returns
    -------
        An array with values applied according to the index.
    
    Comments
    --------
        Ths function was made necessary due to recent deprecation
        warnings triggered when the array dtype is different
        from the value type.
    
    Arguments
    ---------
        "arr" is a numpy array.
        
        ------
        "index" is a boolean or numerical index to which values
        should be applied.
        
        ------
        "val" is an int, float, or string.
    
    Paste Function
    --------------
        apply_val(arr, index, val)
    """
    if arr.dtype.kind in ['i', 'f']:
        val = float(val)
        arr[index] = val
    elif arr.dtype.kind == 'S':
        arr[index] = str(val)
    
    return arr




###########################################################################

def guess_validchars(coredata,  # [2-D array of core data]
                     keys,  # [1-D array of core column keys
                     samp = 100,    # [<None,int,prop> => number or proportion of rows to inspect]
                     nanval = -999  # [not-a-number value]
                     ):
    """Guess validchars specification by looking at the data.

    Returns
    -------
        guess_validchars() returns a validchars specification suitable for
        use as a Damon object parameter.  Examples:

            validchars = ['All',[0,1,2],'Num']
            validchars = ['All',['a','b','c']]
            validchars = ['Cols',{0:[0,1,2],1:['a','b','c'],...}]

    Comments
    --------
        While it is preferable to define Damon's validchars specification
        explicitly, sometimes this is not practical.  This function
        constructs a validchars specification automatically by inspecting
        a sample of the coredata array.

        If a column has no valid characters in it, it is interpreted
        as {'colx':['All']}.

    Arguments
    ---------
        "coredata" is the 2-D core data array of a Damon object.

        -------------
        "keys" is the array of keys used to identify each column in
        coredata, preferably an output of tools.getkeys().  They must
        be the "Core" keys, not "All" keys.

        -------------
        "samp" is used to get a sample of rows to examine for unique
        valid characters for each column.

            samp = None             =>  Do not sample.  Examine the whole
                                        array.

            samp = 0.50             =>  Examine only the first 50% of
                                        rows.

            samp = 100              =>  Examine only the first 100 rows.

        -------------
        "nanval" is the not-a-number value.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        guess_validchars(coredata,  # [2-D array of core data]
                         keys,  # [1-D array of core column keys
                         samp = 100,    # [<None,int,prop> => number or proportion of rows to inspect]
                         nanval = -999  # [not-a-number value]
                         )

    """
    # Get sample of data to check
    nrows = np.size(coredata,axis=0)
    if samp is None:
        samp_ = coredata
    elif isinstance(samp,int):
        if nrows > samp:
            samp_ = coredata[:samp,:]
        else:
            samp_ = coredata
    elif 0.0 < samp < 1.0:
        n_samp = int(nrows * samp)
        if nrows > n_samp:
            samp_ = coredata[:n_samp,:]
        else:
            samp_ = coredata

    if isinstance(coredata[0,0],str):
        nanval = str(nanval)

    # Get unique chars per column
    vcdict = {}
    numflag = {}

    for i,key in enumerate(keys):
        col = samp_[:,i]
        valloc = np.where(col != nanval)
        chars = np.unique(col[valloc])
        nval = np.size(col[valloc])

        try:
            p = np.size(chars) / float(nval)

            if p > 0.80:
                vcdict[key] = ['All']
                numflag[key] = True
            else:
                try:
                    vcdict[key] = list(chars.astype(float)) #.astype(int))
                    numflag[key] = True
                except ValueError:
                    vcdict[key] = list(chars)
                    numflag[key] = False
        except ZeroDivisionError:
                vcdict[key] = ['All']
                numflag[key] = True

    # Build validchars spec
    validchars = []

    # Are the columns all the same?
    same = True
    key0 = vcdict.keys()[0]
    chars0 = vcdict[key0]
    for key in keys:
        if vcdict[key] != chars0:
            same = False
            break

    if same is True:
        validchars.append('All')
        validchars.append(chars0)
    else:
        validchars.append('Cols')
        validchars.append(vcdict)

    # Are they all numerical?
    num = True
    for key in keys:
        if numflag[key] is not True:
            num = False
            break

    if num is True:
        validchars.append('Num')


    return validchars






###########################################################################

def addlabels(coredata, # [2-d array of coredata]
              rowlabels,    # [2-d array of rowlabels]
              collabels,    # [2-d array of collabels]
              fill_top = 0,   # [<0 => No need to fill in top row labels, 1 => Fill in row labels to top>]
              fill_left = 0, # [0 => No need to fill in left col labels, 1 => Fill in col labels all the way to left]
              filler = None,     # [<None, value with which to fill empty corner cells if two preceding args = 1>]
              dtype = [object, None, None],    # [[label type, <None, int decimals>, <None, new nanval>], e.g. ['S20', 8, ''],[object, None, None]
              filename = None,   # [<None, name of output text file or .hd5 file>]
              delimiter = None,  # [<None, field delimiter character if filename refers to a text file>]
              nanval = -999,    # [<-999,...> => not-a-number value]
              pytables = None,  # [<None,True> ]
              ):
    """Returns row/col labels appended to data as array or file.

    Returns
    -------
        Labeled string array or 'Output' file.

    Arguments
    ---------
        "Output" is the name of the output file, if there is one.

        --------------
        "coredata" is the 2-d array of core data to which
        labels should be added.

        --------------
        "rowlabels" is a 2-d nRowEntity x nRowHeaders array of
        row labels.

        --------------
        "collabels" is a 2-d nColHeaders x nColEntity array of
        column labels.

        --------------
        "fill_top" flags whether the row labels go
        all the way to the top of the table (i.e., fill in
        cells to left of col labels) or just up to the
        first entity to be analyzed.

            fill_top = 0
                            =>  No need to fill in the top of
                                the row labels.

            fill_top = 1
                            =>  Fill in the top of the row
                                labels with the values specified
                                in "filler"

        --------------
        "fill_left" flags whether the column labels go
        all the way to the left of the table (i.e., fill in
        cells at the top of the row labels) or just up to the
        first entity to be analyzed.

            fill_left = 0
                            =>  No need to fill in the top of
                                the row labels.

            fill_left = 1
                            =>  Fill in the left of the column
                                labels with the values specified
                                in "filler"


        When both of the above are 0, meaning they are complete,
        the column labels take precedence in filling the top-left
        corner, essentially overwriting the row labels in that area.

        When both are 1, meaning they are incomplete, the corner cells
        are filled with the "filler=" value.

        --------------
        "filler" is the value to assign as filler in the top
        left of the data matrix, if needed.

        ------------
        "dtype" specifies the type to which all cells should be
        cast when the core data is merged with the row and column
        labels, as well as the integer number of decimals according
        to which the core data should be formatted.  The default,
        and recommended, dtype for these purposes is "object", as
        it is the most general and accommodates all possibilities.
        Another common dtype is string, with a number of characters
        specified, such as.

            dtype = [<object,'S60',...>,<None,int decimals>]

        Examples:

            dtype = [object,None]
                                =>  Cast whole array to object,
                                    the most general dtype.  No
                                    specific number of decimals
                                    is specified.

            dtype = ['S60',8]   =>  format_ each number in core data so
                                    that it prints out with 8 decimal
                                    places.  format_ row/column labels
                                    so they are capped at 60 characters.
                                    The 'S60' is referred to whenever
                                    there is a need to cast the array
                                    to string.

        When casting the whole array as a string it is a good
        idea to specify an integer decimal parameter (e.g., 8).  This
        avoids the awkwardness of accidentally casting scientific notation
        (which can pop up unexpectedly when numbers are small) to string,
        which may cause an unfortunate truncation.  For instance, one
        might have a cell value of 3.1234567891234e-17 which is
        accidentally truncated to 3.1234567891234 when the array
        is converted to ('S15'), even though the true number is almost
        zero.  (In this case, a larger string size such as 'S20' would
        avoid the problem by leaving room for the e-17 notation, although
        this consumes more memory.)

        Bear in mind, however, that specifying the decimal parameter
        can create a speed bottleneck in tools.addlabels()
        for very large arrays, which is used internally by Damon in
        several places.  Specifying as "object" avoids these issues.

        --------------
        "filename" specifies the name/path of a file to which
        the final array should be saved.  It may be a textfile
        (e.g., with .csv or .txt extension) or a pytables HD5
        file (with a .hd5 extension).  If None, no file will
        be saved.

        --------------
        "delimiter" <None, char> is the character (e.g., ',' or '\t')
        used to delimit fields if filename refers to a text file.

        --------------
        "pytables" <None, True> specifies whether or not to build
        the new array in chunks and store it as a "PyTable" in h5
        format. The option is useful with very large arrays that
        may have trouble fitting into available memory.  It
        requires that the pytables software package be available
        for import.  The file will be stored

            pytables = None     =>  Do not use pytables.

            pytables = True
                                =>  Place the output in the pytables
                                    .hd5 file specified under filename.

        Give the addlabels() output its own .hd5 file name, as it
        will be overwritten every time addlabels() is run.

        NOTE:  As of version 1.1.13, pytables is being deprecated
        and should not be used.

        --------------
        "nanval" is the value used to identify cells in coredata
        that are not-a-number, i.e., missing, so that they can be
        replaced by the value indicated in dtype[2].  nanval should
        usually equal my_datadict['nanval'].

            nanval = -999       =>  Missing values are marked by -999.

    Examples
    --------

        [Under construction]


    Paste function
    --------------
        addlabels(coredata, # [2-d array of coredata]
                  rowlabels,    # [2-d array of rowlabels]
                  collabels,    # [2-d array of collabels]
                  fill_top = 0,   # [<0 => No need to fill in top row labels, 1 => Fill in row labels to top>]
                  fill_left = 0, # [0 => No need to fill in left col labels, 1 => Fill in col labels all the way to left]
                  filler = None,     # [<None, value with which to fill empty corner cells if two preceding args = 1>]
                  dtype = [object, None, None],    # [[label type, <None, int decimals>, <None, whole nanval>], e.g. ['S20', 8, ''],[object, None, None]
                  filename = None,   # [<None, name of output text file or .hd5 file>]
                  delimiter = None,  # [<None, field delimiter character if filename refers to a text file>]
                  nanval = -999,    # [<-999,...> => not-a-number value]
                  pytables = None,  # [<None,True> ]
                  )

    """

    # Define variables
    nheaders4rows = np.size(rowlabels,axis=1)
    nheaders4cols = np.size(collabels,axis=0)
    fileh = None
    if len(dtype) < 3:
        dtype.append(None)

    # Regular (non-pytables) function
    if pytables is None:

        # coredata to string
        if dtype[1] is not None:
            if dtype[2] is not None:
                coredata = np.copy(coredata)
                #coredata[coredata == nanval] = np.nan
                
            D = str(int(dtype[1]))
            format_ = '%.'+D+'f'
            try:
                coredata = np.reshape([format_ % (x) for x in np.ravel(coredata[:,:])],np.shape(coredata[:,:]))
            except TypeError:       # when coredata is string, no need to format
                pass
        else:
            coredata = coredata.astype(dtype[0])
            
        # MISERABLE HACK.  Recode nanvals, very type sticky.
        if dtype[2] is not None:
            ix = index_val(coredata, nanval)
            coredata[ix] = dtype[2]
#            coredata[(coredata == 'nan') | (coredata == np.nan) |
#                     (coredata == nanval) | (coredata == str(float(nanval)))] = dtype[2]

        # Join arrays
        try:
            # Elements = 'R' scenario
            if fill_top == 1 and fill_left == 0:
                LabelTable = np.append(rowlabels[:,:],coredata,axis=1)
                LabelTable = np.append(collabels,LabelTable,axis=0)

            # Elements = 'C' scenario
            elif fill_top == 0 and fill_left == 1:
                LabelTable = np.append(collabels[:,:],coredata,axis=0)
                LabelTable = np.append(rowlabels,LabelTable,axis=1)

            # Scenario where both sets of labels are missing their top/left corner
            elif fill_top == 1 and fill_left == 1:
                FillColHeaders = np.ones((nheaders4cols,nheaders4rows))
                FillColHeaders[:,:] = filler
                collabels = np.append(FillColHeaders,collabels,axis=1)
                LabelTable = np.append(rowlabels[:,:],coredata,axis=1)
                LabelTable = np.append(collabels,LabelTable,axis=0)

            # Elements = 'All' scenario
            elif fill_top == 0 and fill_left == 0:
                LabelTable = np.append(rowlabels[nheaders4cols:,:],coredata,axis=1)
                LabelTable = np.append(collabels,LabelTable,axis=0)

            else:
                exc = "None of the labeling scenarios apply.\n"
                raise addlabels_Error(exc)

        except ValueError:
            print ('rowlabels, collabels, coredata shapes: ',
                   np.shape(rowlabels), np.shape(collabels), np.shape(coredata))
            exc = 'Unable to fit label and data arrays together.\n'
            raise addlabels_Error(exc)

    # Append pytables arrays together
    else:
        print '\nWarning: The tools.addlabels() pytables option has been deprecated.\n'
        ncols = np.size(collabels,axis=1)
#        nDatCols = ncols - nheaders4rows
        nrows = np.size(rowlabels,axis=0)
#        nDatRows = nrows - nheaders4cols

        if dtype[0] == object:
            nChars = 60
        else:
            nChars = int(dtype[0][1:])

        # Prep rowlabels
        if fill_top == 1:
            TruncRowLabels = rowlabels
        elif fill_top == 0:
            TruncRowLabels = rowlabels[nheaders4cols:,:]

        # Prep collabels
        if fill_left == 0:
            FilledColLabels = collabels
        elif fill_left == 1:
            if fill_top == 0:
                filler = rowlabels[:nheaders4cols,:]
            elif fill_top == 1:
                Fill = np.ones((nheaders4cols,nheaders4rows),dtype=dtype[0])
                Fill[:,:] = filler
            FilledColLabels = np.append(filler,collabels,axis=1)

        # Define chunkfunc
        def loadrows(TruncRowLabels,FilledColLabels,coredata,dtype,chunksize,chunkstart=0):

            # Adjust chunkstart to account for added collabels
            nheaders4cols = np.size(FilledColLabels,axis=0)
            if chunkstart != 0:
                chunkstart = chunkstart - nheaders4cols

            # Chunk rowlabels and coredata
            RLChunk = TruncRowLabels[chunkstart:(chunkstart + chunksize),:]
            CoreChunk = coredata[chunkstart:(chunkstart + chunksize),:]
            Chunk = np.append(RLChunk,CoreChunk,axis=1)

            # Incorporate collabels into first chunk
            if chunkstart == 0:
                Chunk = np.append(FilledColLabels,Chunk,axis=0)

            return Chunk

        # Get chunksize and nchunks
        nrows = np.size(coredata,axis=0) + nheaders4cols
        ncols = np.size(coredata,axis=1) + nheaders4rows

        # Def chunkdict
        chunkdict = {'chunkfunc':loadrows,
                     'nchunks':'Auto',
                     'chunksize':'Auto',
                     'nrows':nrows,
                     'ncols':ncols,
                     }
        # Def ArgDict
        ArgDict = {'TruncRowLabels':TruncRowLabels,
                   'FilledColLabels':FilledColLabels,
                   'coredata':coredata,
                   'dtype':dtype,
                   'chunksize':chunkdict['chunksize'],
                   'chunkstart':0
                   }

        loadDict = {'chunkdict':chunkdict,'ArgDict':ArgDict}

        # Convert to pytable
        if __name__ == "__main__":
            import damon1.tests.play
            Path = damon1.tests.play.__path__[0]
        else:
            Path = os.getcwd() + '/'

        LabelTable_ = pytables_(loadDict,'chunkfunc',Path+'AddLabels.hd5','w','Main',['whole'],
                                    None,'string',nChars,None,None)
        LabelTable = LabelTable_['arrays']['whole']
        fileh = LabelTable_['fileh']

    ########################
    # Save as text file or array
    if filename is not None and pytables is None:
        S = LabelTable.dtype.itemsize
        Fmt = '%.'+str(S)+'s'
        np.savetxt(filename,LabelTable,fmt=Fmt,delimiter=delimiter)
        print filename,'has been saved.'

    else:
        return {'whole':LabelTable,'fileh':fileh}





##################################################################################################

def mergetool(source,  # [array or dictionary FROM which rows or columns are to be extracted]
              target,  # [None => no target data will be appended; array TO which source data is to be appended]
              axis = 0, # [0 => target_ids label rows, 1 => target_ids label cols]
              source_ids = 0,    # [None => source is a dictionary; integer that gives row or col containing lookup IDs in source]
              target_ids = 0,     # [None; array of target IDs; integer that gives row or col in containing lookup IDs in target]
              dtype = object,    # [data type for whole array]
              nanval = -999.    # [Value to assign to missing fields.]
              ):
    """Extracts rows or columns from source array to append to target array.

    Returns
    -------
        target array (or just target IDs) with source data appended.

    Comments
    --------
        mergetool() is used to merge two data sets in terms of common
        row or column identifiers (ID's).  One data set is called "source"
        and contains the data FROM which data will be extracted.  The
        other data set is called "target" and contains the ID's TO
        which source data will be appended.

        There are a lot of variations on this basic structure.
        For instance, source may be an array or a dictionary.
        The IDs in source or target may be keys in a dictionary
        or may reside in a specific row or column, so that the
        user describes which row or column contains the lookup IDs.

        source and target are not DataDicts but arrays or
        dictionaries.  (If source and target are DataDicts, use
        the core.Damon.merge() method.)  This provides some
        flexibility that Damon.merge() lacks.

        A major limitation of mergetool() is that it forces
        labels and data to have the same data type since
        they are combined in a single array.  Thus, IDs which are
        integers will become strings if the core data are strings.
        Damon.merge() does not have this limitation.

    Arguments
    ---------
        "source" is the 2D array or dictionary FROM which rows or columns
        are to be extracted.  (It is not a datadict or a file.)

        ------------
        "target" is the 2D array TO which rows or columns in source are
        to be appended.  When target is None, the function refers to
        target_ids to get a list of target IDs and the target data is left
        out of the final merged array.  Options:

            target = None   =>  Only the IDs in target are to be retained
                                in the merged file.

            target = np.array([[....]])
                            =>  target is a 2-dimensional array of values,
                                including row or column IDs.

        ------------
        "axis" tells whether the target IDs label rows or colums:

            axis = 0        =>  target IDs label rows
            axis = 1        =>  target IDs label columns

        ------------
        "source_ids" tells the function where to get the IDs in the
        source array or dictionary:

            source_ids = None
                            =>  source is a Python dictionary and the
                                IDs are the dictionary keys

            source_ids = 0 (or any integer)
                            =>  The IDs reside in Row 0 of the source
                                array if axis = 0, otherwise in Column 0.
                                Counting starts at 0.

        ------------
        "target_ids" tells the function where to get the IDs in the
        target array.  It is either an array of IDs or it is an
        integer describing where to find the IDs in the target array.

            target_ids = np.array([1,2,3,4,5])
                            =>  target_ids is an array containing IDs
                                1, 2, 3, 4, and 5.

            target_ids = 0 (or any integer)
                            =>  The IDs in the target array reside in
                                Row 0 if axis = 0, otherwise in Column 0.

        ------------
        "dtype" is the data type to be applied to the whole array.  It
        must accommodate both the target IDs and source data.

        ------------
        "nanval" is the Not-a-Number-Value to be assigned to missing
        cells or to cells corresponding to target IDs that lack representation
        in the source array.  nanval must be numerical.

        If a target ID is missing among the source IDs, NaNVals are
        assigned to that target ID.

    Examples
    --------


    Paste function
    --------------
        tools.mergetool(source,  # [array or dictionary FROM which rows or columns are to be extracted]
                      target,  # [None => no target data will be appended; array TO which source data is to be appended]
                      axis = 0, # [0 => target_ids label rows, 1 => target_ids label cols]
                      source_ids = 0,    # [None => source is a dictionary; integer that gives row or col containing lookup IDs in source]
                      target_ids = 0,     # [None; array of target IDs; integer that gives row or col in containing lookup IDs in target]
                      dtype = object,    # [data type for whole array]
                      nanval = -999.    # [Value to assign to missing fields.]
                      )
    """

    # Build or define source dictionary
    if isinstance(source,np.ndarray):
        if axis == 0:
            SourceIDArr = np.squeeze(source[:,source_ids])
            SourceArray = np.delete(source,source_ids,axis=1)
            nSourceDat = np.size(SourceArray,axis=1)
        elif axis == 1:
            SourceIDArr = np.squeeze(source[source_ids,:])
            SourceArray = np.transpose(np.delete(source,source_ids,axis=0))
            nSourceDat = np.size(SourceArray,axis=1)
        SourceDict = dict(zip(SourceIDArr,SourceArray))
    elif (isinstance(source,dict)
          and 'coredata' not in source.keys()
          ):
        SourceDict = source
        nSourceDat = len(SourceDict.values()[0])
    else:
        exc = 'Unable to interpret source argument.'
        raise mergetool_Error(exc)

    # Build or define target identifiers
    if isinstance(target_ids,int):
        if axis == 0:
            TargetIDArr = np.squeeze(target[:,target_ids])
        elif axis == 1:
            TargetIDArr = np.squeeze(target[target_ids,:])
    elif isinstance(target_ids,np.ndarray):
        TargetIDArr = np.squeeze(target_ids)
    else:
        exc = 'Unable to interpret target argument.'
        raise mergetool_Error(exc)


    # Get values from source using target ID. nanval if not in source.

    # If target IDs are row labels
    if axis == 0:
        CumGetSource = np.zeros((0,nSourceDat)).astype(dtype)
        for i in xrange(len(TargetIDArr)):
            if TargetIDArr[i] not in SourceDict:
                GetSource = np.zeros((1,nSourceDat))
                GetSource[:,:] = nanval
            else:
                GetSource = SourceDict[TargetIDArr[i]][np.newaxis,:]
            CumGetSource = np.append(CumGetSource,GetSource,axis=0)
        if target is None:
            TargetArray = np.append(TargetIDArr[:,np.newaxis],CumGetSource,axis=1)
        else:
            TargetArray = np.concatenate((target,CumGetSource),axis=1)

    # If target IDs are column labels
    if axis == 1:
        CumGetSource = np.zeros((nSourceDat,0)).astype(dtype)
        for i in xrange(len(TargetIDArr)):
            if TargetIDArr[i] not in SourceDict:
                GetSource = np.zeros((nSourceDat,1))
                GetSource[:,:] = nanval
            else:
                GetSource = SourceDict[TargetIDArr[i]][:,np.newaxis]
            CumGetSource = np.append(CumGetSource,GetSource,axis=1)
        if target is None:
            TargetArray = np.append(TargetIDArr[np.newaxis,:],CumGetSource,axis=0)
        else:
            TargetArray = np.concatenate((target,CumGetSource),axis=0)

    return TargetArray




###########################################################################

def tuple2table(tup_data,    # [file or array of tuples: [(Fac0Element,Fac1Element,Datum),...]]
                format_ = 'array',   # [<'array','textfile'>]
                labels = None,  # [[list of keys] => that do not label coredata, only other labels]
                delimiter = None,    # [<None, ',', ...> => character used to separate fields in input file, e.g., ',','\t']
                nheaders4rows = 2,  # [int number of headers for rows = number of facets]
                nheaders4cols = 1,  # [int number of headers for cols => tuple column labels]
                nanval = -999., # [not-a-number value to flag missing data]
                ):
    """Converts array of (Fac0,Fac1,Data) tuples to a tabular array.

    Returns
    -------
        Row Entity x Col Entity table, with appended row labels
        and col labels.

    Comments
    --------
        Datasets can be stored as tables, rows and columns representing
        facet elements, or they can be stored as data values attached to
        facet entity pairs (2-facets), triplets (3-facets), etc., called
        tuples.

        The tabular format is more intuitive and efficient in some ways, but
        inefficient with large sparse matrices and difficult to scale up to
        more than two facets.

        Damon analyzes data in tabular format.  tuple2table converts
        tuple data to tabular data for Damon analysis.

    Arguments
    ---------


    Examples
    --------



    Paste function
    --------------
          tuple2table(tup_data,    # [file or array of tuples: [(Fac0Element,Fac1Element,Datum),...]]
                    format_ = 'array',   # [<'array','textfile'>]
                    labels = None,  # [[list of keys] => that do not label coredata, only other labels]
                    delimiter = None,    # [<None, ',', ...> => character used to separate fields in input file, e.g., ',','\t']
                    nheaders4rows = 2,  # [int number of headers for rows = number of facets]
                    nheaders4cols = 1,  # [int number of headers for cols => tuple column labels]
                    nanval = -999., # [not-a-number value to flag missing data]
                    )
   """

    # Load and format data
    datadict = dmn.core.Damon(tup_data,format_,'RCD',['All',['All'],'SkipCheck'],
                              nheaders4rows=nheaders4rows,key4rows=0,rowkeytype=object,
                              nheaders4cols=nheaders4cols,key4cols=0,colkeytype=object,
                              check_dups=None,dtype=[object,None],nanval=nanval,verbose=None
                              )

    # Define variables
    Fac0Fac1 = datadict.rowlabels
    data = np.squeeze(datadict.coredata)
    Fac0Dat = Fac0Fac1[nheaders4cols:,0]
    Fac1Dat = Fac0Fac1[nheaders4cols:,1]

    # Extract list of unique row/column elements
    rl = np.unique(Fac0Dat)
    cl = np.unique(Fac1Dat)

    if labels is not None:
        TabRowLabels = np.array(list(set(rl) - set(labels)))
        TabColLabels = np.array(list(set(cl) - set(labels)))
    else:
        TabRowLabels = rl
        TabColLabels = cl

    nTabRowLabels = len(TabRowLabels)
    nTabColLabels = len(TabColLabels)

    # Make row/column numbers
    TabRows = xrange(nTabRowLabels)
    TabCols = xrange(nTabColLabels)

    # Make dictionaries to associate rows/columns with facet elements
    RowDict = dict(zip(TabRowLabels,TabRows))
    ColDict = dict(zip(TabColLabels,TabCols))

    # Initialize table and populate from tuples
    DatTab = np.ones((nTabRowLabels,nTabColLabels)) * nanval
    DatTab = DatTab.astype(object)

    # Function to assign data to table cell
    def assign(i):
        try:
            row = RowDict[Fac0Dat[i]]
            col = ColDict[Fac1Dat[i]]
            DatTab[row,col] = data[i]
        except KeyError:
            pass

        return None

    # Populate core data table
    ndata = np.size(data,axis=0)
    [assign(i) for i in xrange(ndata)]

    # Add row/column labels
    DatTab_str = np.append(TabRowLabels[:,np.newaxis],DatTab,axis=1)
    TabColLabels0 = np.append([0],TabColLabels,axis=1)[np.newaxis,:]
    DatTab_str = np.append(TabColLabels0,DatTab_str,axis=0)

    return DatTab_str


###########################################################################

def table2tuple(tab_data,    # [datadict or Damon object, tabular]
                del_nan = True, # [<True,False] => delete records that have nanval as data]
                columns = ['person','item','data'], # [<[row facet, col facet, data]> => column labels]
                range_ = 'Core',    # [<'All','Core'> => include labels or just coredata]
                output_as = 'whole',    # [<'dict', 'whole', 'textfile'>]
                outfile = None,    # [<None,filename> => tuple output file]
                delimiter = ',', # [<None,',',...] => delimiter of output file]
                ):
    """Converts tabular row/col data to a tuple-style [rowkey,colkey,data].

    Returns
    -------
        table2tuple() returns a tuple-formatted dictionary, whole
        array, or textfile as specified in output_as.

        dict:   {'row_facet':
                 'col_facet':
                 'data':
                }

    Comments
    --------
        table2tuple() converts tabular data into "tuple-style" data.
        Datasets can be stored as tables, rows and columns representing
        facet elements, or they can be stored as data values attached to
        facet entity pairs (2-facets), triplets (3-facets), etc., called
        tuples.

        The tabular format is more intuitive and efficient in some ways, but
        inefficient with large sparse matrices and difficult to scale up to
        more than two facets.

    Arguments
    ---------
        "tab_data" is the tabular data stored as a datadict.

        "del_nan" <True, False> specifies whether to delete records whose
        datum is equal to nanval (as recorded by the nanval argument in
        tab_data).

        "columns" provides the column labels for the new tuple array.

        "range_" <'All','Core'> specifies whether to convert the whole array
        into tuples, including the labels, or just the core data.

        "output_as" specifies whether to output a 'dict' (keys: 'row_facet',
        'col_facet','data'), a 'whole' array, or a 'textfile'.

        "outfile" is a specifed file name or file path, ignored if
        output_as is 'dict' or 'whole'.

        "delimiter is the delimiter of the output file, if applicable.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        table2tuple(tab_data,    # [datadict or Damon object, tabular]
                    del_nan = True, # [<True,False] => delete records that have nanval as data]
                    columns = ['person','item','data'], # [<[row facet, col facet, data]> => column labels]
                    range_ = 'Core',    # [<'All','Core'> => include labels or just coredata]
                    output_as = 'whole',    # [<'dict', 'whole', 'textfile'>]
                    outfile = None,    # [<None,filename> => tuple output file]
                    delimiter = ',', # [<None,',',...] => delimiter of output file]
                    )

    """
    if isinstance(tab_data,dmn.core.Damon):
        data_ = tab_data.data_out
    else:
        data_ = tab_data

    # Reformat as Damon object
    d = dmn.core.Damon(data_,'datadict','RCD_whole',verbose=None)
    data = d.data_out

    nanval = data['nanval']

    # Extract keys
    rowkeys = getkeys(data,'Row',range_,'Auto',None)
    colkeys = getkeys(data,'Col',range_,'Auto',None)

    # Counts
    nrowkeys = np.size(rowkeys)
    ncolkeys = np.size(colkeys)

    # Construct separate columns => RowKeys, ColKeys, Data
    rowkeyscol = np.repeat(rowkeys,ncolkeys,axis=0)[:,np.newaxis].astype(object)
    colkeyscol = np.tile(colkeys,nrowkeys)[:,np.newaxis].astype(object)

    if range_ == 'Core':
        coredatacol = np.ravel(data['coredata'])[:,np.newaxis]
    else:
        coredatacol = np.ravel(d.whole)[:,np.newaxis]

    # Remove NaNVals
    if del_nan is True:
        locval = [(rowkeyscol != nanval) & (colkeyscol != nanval) & (coredatacol != nanval)]
        rowkeyscol = rowkeyscol[locval][:,np.newaxis]
        colkeyscol = colkeyscol[locval][:,np.newaxis]
        coredatacol = coredatacol[locval][:,np.newaxis]

    # Output as array (=0)
    if output_as == 'dict':
        return {columns[0]:rowkeyscol,
                columns[1]:colkeyscol,
                columns[2]:coredatacol,
                }
    else:
        # Combine labels, keys and core data
        tuplabels = np.array([columns])
        body = np.concatenate((rowkeyscol,colkeyscol,coredatacol),axis=1).astype(object)
        whole_ = np.append(tuplabels,body,axis=0)

        # Return whole array
        if output_as == 'whole':
            return whole_

        # Output as file
        elif output_as == 'textfile':

            # Output file
            np.savetxt(outfile,whole_,fmt = '%20s',delimiter=delimiter)
            print outfile,'has been saved.\n'




###########################################################################

##def triproject(dmnestfile = 'estimates.csv',  # [Expected values, person x items]
##               itemkeyfile = 'ItemKey.csv', # [Item key file: [ItemID,Trait,Pro-key,Anti-key]]
##               nheaders4cols_key = 1,  # [nHeaders for columns in Item key file]
##               min_rating = 1  # [None => no minimum specified, else minimum rating]
##               ):
##    """Applies answer keys to damon output, returns trait values in data() format.
##
##    Returns
##    -------
##        Person x Trait array, with trait values
##        {'TP_OutData':TP_OutData,'TraitLabels':TraitLabels}, where TP_OutData =
##
##        {'rowlabels':rowlabels,'collabels':collabels,'coredata':coredata,
##        'nheaders4rows':nheaders4rows,'key4rows':key4rows,'nheaders4cols':nheaders4cols,
##        'key4cols':key4cols,'nanval':nanval}
##
##    Comments
##    --------
##        TriProject uses the "triangular projection" formula to locate where
##        each person falls on a trait defined in terms of a person who lacks
##        the trait and a person who embodies the trait, represented by the
##        Anti-key and Pro-key respectively.  Each trait is embodied by a
##        set of items:
##            Formula:  x|T = ( R^2 - S^2 + T^2 ) / 2T
##            where R, S, T are the sides of triangle A,B,C (A = left, B = right, C = apex)
##            (R = AC, S = BC, T = AB)
##            and x|T is the perpendicular projection of C on AB (aka T).
##            In this case, A = AntiKey, B = key, C = target Person
##
##        There is only one trait per item.
##
##        TriProject is applied only to Damon-created expected values.  There
##        can be no missing data in the data file or key file.
##
##        The data file is assumed to have one header for each row (person)
##        and one header for each column (item).
##
##        The item key file is assumed to have two headers for each row
##        (ItemID, TraitLabel), and two numbers as core data corresponding
##        to a Pro-key value and an Anti-key value.  There is one header
##        for each column.  The column headers are: ItemID, TraitID, ProAnsKey, AntiAnsKey.
##        Each row is an item.
##
##    Paste function
##    --------------
##        triproject(dmnestfile = 'estimates.csv',  # [Expected values, person x items]
##                   itemkeyfile = 'ItemKey.csv', # [Item key file: [ItemID,Trait,Pro-key,Anti-key]]
##                   nheaders4cols_key = 1,  # [nHeaders for columns in Item key file]
##                   min_rating = 1  # [None => no minimum specified, else minimum rating]
##                   )
##
##    """
##
##    print 'Error in triproject:  This function is no longer in use.\n'
##    sys.exit()
##
##    # Retrieve data
##    TP_Data = data.datautil(dmnestfile, # [Filename and path if not in Damon folder.  If file = 0, reference an array.]
##                          file = 1, # [-1 => data is in RCD format, 0 => data is an array, 1 => data is a file]
##                          report = 'RCD',   # ['R' => rowlabels,'C' => collabels,'D' => core data,'RCD' => [R,C,D,nheaders4rows,key4rows,nheaders4cols,key4cols],'whole' => RCD concatenated as string array, 'RCD_whole' => RCD plus whole array, 'Dicts' => RCD dictionaries, 'RCD_dicts' => all arrays plus dicts.]
##                          nheaders4rows = 1,  # [Integer number of columns set aside for row labels, must have same number of elements as RowLabels2Left list]
##                          key4rows = 0,   # [Integer number of columns from left (after doing RowLabels2Left) which has row entity labels corresponding to a key in an entity dictionary]
##                          rowkeytype = int,     # [data type of Row keys => str, float, or int]
##                          nheaders4cols = 1,  # [integer number of rows set aside for column labels]
##                          key4cols = 0, # [Integer number of rows from top which has column entity labels corresponding to a key in an entity dictionary]
##                          colkeytype = int,     # [data type of Col keys => str, float, or int]
##                          delimiter = ',',  # [Character used to delimit columns in the input file, e.g. ',', '\t']
##                          miss4headers = ['.'], # [None => no specific characters to make missing in labels; [list of characters to make missing in labels]]
##                          nanval = -999.,    # [Numeric or alpha value to which all missing or invalid values in the core portion of the data shall be converted.]
##                          prenanval = None,  # [None => no pre-existing float nanval's, otherwise, the float that describes nanval's pre-existing in the dataset.]
##                          alpha = 0, # [None => Accept data as is; 0 => force core data to be numeric, 1 => core data are non-numeric, e.g., alphabetical]
##                          )
##
##    # Retrieve key file
##    IKey = data.datautil(itemkeyfile, # [Filename and path if not in Damon folder.  If file = 0, reference an array.]
##                          file = 1, # [-1 => data is in RCD format, 0 => data is an array, 1 => data is a file]
##                          report = 'RCD',   # ['R' => rowlabels,'C' => collabels,'D' => core data,'RCD' => [R,C,D,nheaders4rows,key4rows,nheaders4cols,key4cols],'whole' => RCD concatenated as string array, 'RCD_whole' => RCD plus whole array, 'Dicts' => RCD dictionaries, 'RCD_dicts' => all arrays plus dicts.]
##                          nheaders4rows = 2,  # [Integer number of columns set aside for row labels, must have same number of elements as RowLabels2Left list]
##                          key4rows = 0,   # [Integer number of columns from left (after doing RowLabels2Left) which has row entity labels corresponding to a key in an entity dictionary]
##                          rowkeytype = int,     # [data type of Row keys => str, float, or int]
##                          nheaders4cols = nheaders4cols_key,  # [integer number of rows set aside for column labels]
##                          key4cols = 0, # [Integer number of rows from top which has column entity labels corresponding to a key in an entity dictionary]
##                          colkeytype = int,     # [data type of Col keys => str, float, or int]
##                          delimiter = ',',  # [Character used to delimit columns in the input file, e.g. ',', '\t']
##                          miss4headers = ['.'], # [None => no specific characters to make missing in labels; [list of characters to make missing in labels]]
##                          nanval = -999.,    # [Numeric or alpha value to which all missing or invalid values in the core portion of the data shall be converted.]
##                          prenanval = None,  # [None => no pre-existing float nanval's, otherwise, the float that describes nanval's pre-existing in the dataset.]
##                          alpha = 0, # [None => Accept data as is; 0 => force core data to be numeric, 1 => core data are non-numeric, e.g., alphabetical]
##                          )
##
##    # Some TP_Data variables
##    TP_Data_nHeaders4Cols = TP_Data['nheaders4cols']
##    TP_Data_nHeaders4Rows = TP_Data['nheaders4rows']
##    NPersons = len(TP_Data['rowlabels'][TP_Data_nHeaders4Cols:,0])
##    TP_CoreData = TP_Data['coredata']
##    TP_NaNVal = TP_Data['nanval']
##    ColKeyRow = TP_Data['key4cols']
##    Items = TP_Data['collabels'][ColKeyRow,TP_Data_nHeaders4Rows:]
##
##    # Convert TP_Data Item IDs to integers if possible
##    try:
##        Items = Items.astype(float).astype(int)
##    except:
##        Items = Items
##
##    NItems = len(Items)
##
##    # Some IKey variables
##    IKey_nHeaders4Cols = IKey['nheaders4cols']
##    IKeyLabels = IKey['rowlabels'][IKey_nHeaders4Cols:,:]
##    IKeys = IKeyLabels[:,0]
##    IKeyCore = IKey['coredata'].astype(float)
##
##    # Try to convert Item Keys to integers
##    try:
##        IKeys = IKeys.astype(float).astype(int)
##    except:
##        IKeys = IKeys
##
##    Traits = IKeyLabels[:,1]
##
##    # Try to convert Trait Keys to integers
##    try:
##        Traits = Traits.astype(float).astype(int)
##    except:
##        Traits = Traits
##
##    # Create {Item:Trait} dictionary
##    ITraitDict = dict(zip(IKeys,Traits))
##
##    # Create {Item:[ProAnsKey,AntiAnsKey]} dictionary
##    IAnsKeyDict = dict(zip(IKeys,IKeyCore))
##
##    # Pull trait information from ITraitDict to TP_Data in TP_items order
##    TP_ItemTraitInfo = np.zeros((NItems,1),dtype=type(Traits[0]))
##    for i in xrange(NItems):
##        TP_ItemTraitInfo[i,0] = ITraitDict[Items[i]]
##    TP_ItemTraitInfo = np.transpose(TP_ItemTraitInfo)
##
##    # Pull answer key information from IAnsKeyDict to TP_Data in TP_items order
##    TP_ItemAnsKeyInfo = np.zeros((NItems,2),dtype=type(IKeyCore[0,0]))
##    for i in xrange(NItems):
##        TP_ItemAnsKeyInfo[i] = IAnsKeyDict[Items[i]]
##    TP_ItemAnsKeyInfo = np.transpose(TP_ItemAnsKeyInfo)
##
##    # List unique item clusters or "traits"
##    ITraits = np.unique1d(TP_ItemTraitInfo[0])
##    NTraits = len(ITraits)
##
##
##    # Calculate trait values using triangular projection
##    #####################################################################################
##    ##  Formula:  x|T = ( R^2 - S^2 + T^2 ) / 2T
##    ##  where R, S, T are the sides of triangle A,B,C (A = left, B = right, C = apex)
##    ##  (R = AC, S = BC, T = AB)
##    ##  and x|T is the perpendicular projection of C on AB (aka T)
##    ##  In this case, A = AntiKey, B = key, C = target Person
##    #####################################################################################
##
##
##    # Create trait array
##    TraitScores = np.zeros((NPersons,NTraits))
##
##    # Select data by trait
##    for j in xrange(NTraits):
##        TraitLoc = np.where(TP_ItemTraitInfo[0,:] == ITraits[j])
##        PersonIScores_Trait = TP_CoreData[:,TraitLoc[0]]
##        ItemAnsKey_Trait = TP_ItemAnsKeyInfo[:,TraitLoc[0]]
##        ItemAnsKey_Trait_Pro = ItemAnsKey_Trait[0]
##        ItemAnsKey_Trait_Anti = ItemAnsKey_Trait[1]
##
##        # Pre-calc T_sq since it is the same for all persons for trait j
##        T_sq = np.average((ItemAnsKey_Trait_Pro - ItemAnsKey_Trait_Anti)**2)
##
##        # Alert to zero denominators
##        if T_sq == 0:
##            print 'ERROR: ProAnsKey = AntiAnsKey for Trait',ITraits[j],'.'
##
##        # Select person item scores
##        for i in xrange(NPersons):
##            R_sq = np.average((PersonIScores_Trait[i] - ItemAnsKey_Trait_Anti)**2)
##            S_sq = np.average((PersonIScores_Trait[i] - ItemAnsKey_Trait_Pro)**2)
##
##            # Insert NaNVals if T_sq == 0
##            if T_sq == 0:
##                PersonTraitScore = TP_NaNVal
##            else:
##                PersonTraitScore = (R_sq - S_sq + T_sq)/(2 * np.sqrt(T_sq))
##
##            # Adjust trait score up by minimum rating
##            if min_rating == None:
##                TraitScores[i,j] = PersonTraitScore
##            else:
##                if PersonTraitScore == TP_NaNVal:
##                    TraitScores[i,j] = TP_NaNVal
##                else:
##                    TraitScores[i,j] = PersonTraitScore + min_rating
##
##    coredata = TraitScores
##    rowlabels = TP_Data['rowlabels']
##    TopLeftCol = rowlabels[:1,:1]
##    collabels = np.append(TopLeftCol,np.array(ITraits,ndmin=2),axis=1)
##
##    # Convert to data() format
##    TP_OutRCD = {'rowlabels':rowlabels,'collabels':collabels,'coredata':coredata,
##                'nheaders4rows':1,'key4rows':0,'rowkeytype':int,
##                'nheaders4cols':1,'key4cols':0,'colkeytype':int,
##                'nanval':TP_NaNVal,'validchars':None
##                 }
##
##    return {'TP_OutData':TP_OutRCD,'TraitLabels':ITraits}




###########################################################################

def valchars(validchars,    # ['validchars' output of data() function]
             dash = ' -- ', # [Expression used to denote a range]
             defnone = 'interval',   # [How to interpret metric when validchars = None]
             retcols = None,    # [<None, [list of core col keys]>]
             ):
    """Converts validchars information into scaling and rounding codes.

    Returns
    -------
        {'metric',  => [<'All','Cols'>,[<metric, metric column dictionary>]] ]
        'round_'     => [<'All','Cols'>,[<round_, round_ column dictionary>]] ]
        'minmax'    => [<'All','Cols'>,[<[Min,Max], minmax column dictionary>]] ]
        }

        where metric is 'nominal', 'ordinal', 'sigmoid', 'interval',
        or 'ratio', where round_ is 1 or None, and where [Min,Max]
        are the minimum and maximum of the range, if applicable.


        Examples
        --------
        MyOutput['metric']
            ['All','interval']  => All columns are interval measures

            ['Cols',{'It1':'interval','It2':'sigmoid',...}]
                        => 'It1' col is interval, 'It2' col is sigmoid

        MyOutput['round_']
            ['All',1]   => All columns are interval measures

            ['Cols',{'It1':1,'It2':None,...}]
                        => 'It1' col is rounded, 'It2' col is not rounded

        MyOutput['minmax']
            ['All',[-3,4]]
                        => The minimum of whole array is -3, the maximum 4.

            ['Cols',{'It1':[0,np.inf],'It2':None,'It3':[-3,4]}]
                        =>  Item 1 is on a ratio scale from 0 to +infinity.
                            Item 2 is on an interval scale, no min or max.
                            Item 3 has a min of -3 and max of 4.

        NOTE:  ratio scales can only be defined using range
        notation, e.g., '0.0 -- '.  dash has to be correct
        (whitespace,hyphen,hyphen,whitespace).  The left has to
        be 0.0 or 0.  If the left-hand number is anything else,
        the range is treated as interval. For example, '0.01 -- '
        is treated as an interval scale from 0.01 on up.

        This notational trick makes it possible to define an
        interval scale for just numbers greater than zero
        using range notation.  Sometimes this comes in handy.

    Comments
    --------
        valchars() converts the validchars specification captured
        and output by the data() function into a similar metric
        and rounding specification for use by various Damon
        methods.

    Arguments
    ---------
        "validchars" is output by the data() function and can be
        accessed as MyData['validchars'].

        -------------
        "dash" is the convention chosen in the data() function for
        conveying a range of values, such as ' -- '.

        -------------
        "defnone" directs valchars() how to interpret the specification
        validchars = None, i.e., what metric it implies.

        -------------
        "retcols" provides the option of returning column metrics in
        the ['Cols',{ID:metric,...}] format even if validchars
        is in the ['All',metric] format.  If the metrics are in
        the 'Cols' format but the metrics are all the same, they
        are automatically switched to the 'All' format if
        retcols = None.  Options:

            None        =>  return_ metrics in the same format as
                            validchars, i.e., ['All',metric] or
                            ['Cols',{ID:metric,...}] unless all the
                            column metrics are the same, in which
                            case convert to 'All' format.

            [list of core column IDs]
                        =>  return_ metrics in the ['Cols',{ID:metric,...}]
                            format, regardless of the validchars metric.
                            To build the dictionary, provide a list of
                            unique column IDs that correspond to columns
                            of analyzable data (coredata).

    Paste function
    --------------
        valchars(validchars,    # ['validchars' output of data() function]
                 dash = ' -- ', # [Expression used to denote a range]
                 defnone = 'interval',   # [How to interpret metric when validchars = None]
                 retcols = None,    # [<None, [list of core col keys]>]
                 )

    """

    # Variables
    LenDash = len(dash)

    # Evaluate validchars string
    if (isinstance(validchars,list)
        and isinstance(validchars[1],str)
        ):
        validchars[1] = ast.literal_eval(validchars[1])

    if (isinstance(validchars,list)
        and isinstance(validchars[1],dict)
        ):
        CDict = validchars[1]
        for key in CDict.keys():
            if isinstance(CDict[key],str):
                CDict[key] = ast.literal_eval(CDict[key])
            else:
                pass
        validchars[1] = CDict


    ##################
    ##  All Cols    ##
    ##  the Same    ##
    ##################

    # validchars = None
    if (validchars is None
        and retcols is None
        ):
        metric = ['All',defnone]
        round_ = ['All',None]
        minmax = ['All',None]

    # validchars = None (breaking out by columns)
    elif (validchars is None
          and retcols is not None
          ):
        colkeys = np.array(retcols)
        MetDict = {}
        RndDict = {}
        MinMaxDict = {}

        for key in colkeys:
            MetDict[key] = defnone
            RndDict[key] = None
            MinMaxDict[key] = None

        metric = ['Cols',MetDict]
        round_ = ['Cols',RndDict]
        minmax = ['Cols',MinMaxDict]

    # Check whether metrics are same
    elif (validchars[0] == 'Cols'
          and retcols is None
          ):
        ColDict = validchars[1]
        Metrics = ColDict.values()
        Metrics1 = [str(metric_) for metric_ in Metrics]
        if len(np.unique(Metrics1)) == 1:
            validchars = ['All',ColDict[ColDict.keys()[0]]]

    # Whole array
    if (validchars is not None
        and validchars[0] == 'All'
        and retcols is None
        ):

        # Not a range 'm -- n'
        if (dash not in str(validchars[1])):

            if validchars[1][0] == 'All':
                metric = ['All','interval']
                round_ = ['All',None]
                minmax = ['All',None]

            elif isinstance(validchars[1][0],int):
                metric = ['All','ordinal']
                round_ = ['All',1]
                minmax = ['All',[min(validchars[1]),max(validchars[1])]]

            elif isinstance(validchars[1][0],float):
                metric = ['All','ordinal']
                round_ = ['All',None]
                minmax = ['All',[min(validchars[1]),max(validchars[1])]]

            elif isinstance(validchars[1][0],str):
                metric = ['All','nominal']
                round_ = ['All',None]
                minmax = ['All',[min(validchars[1]),max(validchars[1])]]

            else:
                exc = 'Unable to figure out validchars parameter.\n'
                raise valchars_Error(exc)

        # Is a range 'm -- ' or 'm -- n' or ' -- ' or '. -- .', float (continuous) or int (ordinal)
        elif (dash in str(validchars[1])):
            Ran = validchars[1][0]

            #  ' -- ' is technically ordinal, but treating it as interval because boundless
            if Ran == dash:
                metric = ['All','interval']
                round_ = ['All',1]
                minmax = ['All',[-np.inf,np.inf]]

            elif Ran == '.'+dash+'.':
                metric = ['All','interval']
                round_ = ['All',None]
                minmax = ['All',[-np.inf,np.inf]]

            elif Ran[-LenDash:] == dash and '.' in Ran and '0.' in Ran:
                metric = ['All','ratio']
                round_ = ['All',None]
                minmax = ['All',[0,np.inf]]

            # ratio requires 0 minimum, else (e.g., 0.01) treat as interval (under protest)
            elif Ran[-LenDash:] == dash and '.' in Ran and '0.' not in Ran:
                metric = ['All','interval']
                round_ = ['All',None]
                minmax = ['All',None]

            elif Ran[-LenDash:] == dash and '.' not in Ran:
                metric = ['All','ratio']
                round_ = ['All',1]
                minmax = ['All',[0,np.inf]]

            # Upper and lower bound, continuous
            elif dash in Ran and '.' in Ran:
                metric = ['All','sigmoid']
                round_ = ['All',None]
                minmax = ['All',[float(Ran[0:Ran.find(dash)]),float(Ran[Ran.find(dash) + len(dash):])]]

            elif dash in Ran and '.' not in Ran:
                metric = ['All','ordinal']
                round_ = ['All',1]
                minmax = ['All',[float(Ran[0:Ran.find(dash)]),float(Ran[Ran.find(dash) + len(dash):])]]

            else:
                exc = 'Unable to figure out validchars parameter.\n'
                raise valchars_Error(exc)


    ###############
    ##   Cols    ##
    ##   Differ  ##
    ###############

    # VC['Cols',*]
    # Individual Columns
    elif (validchars is not None
          and (validchars[0] == 'Cols'
               or retcols is not None)
          ):

        # Get or construct validchars col dict
        if (retcols is None
            or (retcols is not None
                and type(validchars[1]) is type({}))
            ):
            ColDict = validchars[1]
            colkeys = ColDict.keys()
        else:
            ColDict = {}
            colkeys = np.array(retcols)
            for key in colkeys:
                ColDict[key] = validchars[1]

        MetDict = {}
        RndDict = {}
        MinMaxDict = {}

        for key in colkeys:

            # Not a range 'm -- n'
            if (dash not in str(ColDict[key])):

                if (ColDict[key] == ['All']
                    or ColDict[key] == [None]
                    ):
                    MetDict[key] = 'interval'
                    RndDict[key] = None
                    MinMaxDict[key] = None

                elif isinstance(ColDict[key][0],int):
                    MetDict[key] = 'ordinal'
                    RndDict[key] = 1
                    MinMaxDict[key] = [min(ColDict[key]),max(ColDict[key])]

                elif isinstance(ColDict[key][0],float):
                    MetDict[key] = 'ordinal'
                    RndDict[key] = None
                    MinMaxDict[key] = [min(ColDict[key]),max(ColDict[key])]

                elif isinstance(ColDict[key][0],str):
                    MetDict[key] = 'nominal'
                    RndDict[key] = None
                    MinMaxDict[key] = None

                else:
                    exc = 'Unable to figure out validchars parameter.\n'
                    raise valchars_Error(exc)

            # Is a range 'm -- ' or 'm -- n' or ' -- ' or '. -- .', float (continuous) or int (ordinal)
            elif (dash in str(ColDict[key])):

                Ran = ColDict[key][0]

                #  ' -- ' is technically ordinal, but treating it as interval as it is boundless
                if Ran == dash:
                    MetDict[key] = 'interval'
                    RndDict[key] = 1
                    MinMaxDict[key] = [-np.inf,np.inf]

                elif Ran == '.'+dash+'.':
                    MetDict[key] = 'interval'
                    RndDict[key] = None
                    MinMaxDict[key] = [-np.inf,np.inf]

                elif Ran[-LenDash:] == dash and '.' in Ran and '0.' in Ran:
                    MetDict[key] = 'ratio'
                    RndDict[key] = None
                    MinMaxDict[key] = [0,np.inf]

                # ratio requires 0 minimum, else (e.g., 0.01) treat as interval (under protest)
                elif Ran[-LenDash:] == dash and '.' in Ran and '0.' not in Ran:
                    MetDict[key] = 'interval'
                    RndDict[key] = None
                    MinMaxDict[key] = None

                elif Ran[-LenDash:] == dash and '.' not in Ran:
                    MetDict[key] = 'ratio'
                    RndDict[key] = 1
                    MinMaxDict[key] = [0,np.inf]

                # Upper and lower bound, continuous
                elif dash in Ran and '.' in Ran:
                    MetDict[key] = 'sigmoid'
                    RndDict[key] = None
                    MinMaxDict[key] = [float(Ran[0:Ran.find(dash)]),float(Ran[Ran.find(dash) + len(dash):])]

                elif dash in Ran and '.' not in Ran:
                    MetDict[key] = 'ordinal'
                    RndDict[key] = 1
                    MinMaxDict[key] = [float(Ran[0:Ran.find(dash)]),float(Ran[Ran.find(dash) + len(dash):])]

                else:
                    exc = 'Unable to figure out validchars parameter.\n'
                    raise valchars_Error(exc)

        metric = ['Cols',MetDict]
        round_ = ['Cols',RndDict]
        minmax = ['Cols',MinMaxDict]

    # return_ metric and round_ specifications
    return {'metric':metric,'round_':round_,'minmax':minmax}


###########################################################################

def dups(array,    # [array of values possibly containing duplicates]
         ):
    """return_ duplicate values and their frequency.

    Returns
    -------
        Dictionary listing duplicate ID's and how many
        duplicates for each.

        {'x':count[x], 'y':count[y], ...}

    Comments
    --------
        dups() checks for duplicate values in a list
        or array and returns a dictionary of duplicates
        per value.

    Arguments
    ---------
        "array" is any array or list of values that may
        contain duplicates.

    Examples
    --------

        [under construction]

    Paste function
    --------------
        dups(array,    # [array of values possibly containing duplicates]
             )

    """
    
    # Get uniques, count in bins
    keys = np.ravel(np.array(array))
    uniques = np.unique(keys)
    bins = uniques.searchsorted(keys)
    bincounts = np.bincount(bins)

    # Remove single counts
    dups = bincounts > 1
    dups_dict = dict(zip(uniques[dups], bincounts[dups]))

    return dups_dict
    



###########################################################################

def rescale(score,  # [individual score or array of same-metric scores]
            straighten = None,  # [<None,True,'Percentile'> => intermediate rank-logit step]
            logits = None,  # [<None, {'ecut':0, 'ear':<ear array or float}>]
            reverse = False, # [<True, False> => reverse sign of measures]
            mean_sd = None, # [<None,[mean,sd]> => target mean, standard deviation]
            m_b = None, # [<None,[m,b]> => multiply by m, add b]
            clip = None, # [<None, [min, max]> => min and max scores]
            round_ = None, # [<None, int decimals> => round outputs]
            nanval = -999, # [not-a-number value in score array]
            ):
    """Rescale scores to a new metric.

    Returns
    -------
        rescale() returns an individual score or an array of scores
        in a specified output metric.  The input scores must all
        be in the same input metric.

    Comments
    --------
        rescale() takes a score or an array of scores and multiplies
        them by some number and adds another, or adjusts them to
        have a target mean and standard deviation.  It also offers
        the option to "straighten" them to remove non-linearities
        by converting them into percentile logits.

    Arguments
    ---------
        "score" is either a single numerical score (360.3) or a
        1- or 2_ dimensional array of scores [320,350,200,...], .
        This is the score we want to transform from one metric to
        another.  If score is 2-dimensional, the whole array will
        be rescaled to the desired mean/sd, not its individual
        rows or columns.
                
        -----------
        "straighten", if "True", is used to remove non-linearities that
        sometimes occur when dichotomous data is used to estimate continuous
        data.  It does this by converting the continuous values into
        percentile ranks, then logits, then adjusting them to have the
        original mean and standard deviation.  In order to return
        plain percentiles rather than percentile logits, use
        the 'Percentile' option.

        -----------
        "logits" converts construct/subscale measures into logits with
        the appropriate logit standard errors.  Logits are log(p/(1-p) 
        where p is the probability calculated using cumnormprob() of
        exceeding the specified 'ecut' given a specified variance.  That
        variance can be:
        
            a)  the variance of each individual cell.  This yields a more
                accurate probability/logit per cell at the expense of metric 
                comparability across cells or tests.
            
            b)  A specified overall variance.  This can be the mean variance
                of the construct for the current test or that of another test.
                This emphasizes metric comparability, so that cell logits
                are treated as measures rather than probabilistic predictions.
            
            logits = None           =>  Do not rescale as logits.
            
            logits = {'ecut':0.0, 'ear':np.array([...])}
                                    =>  Rescale as logits corresponding to
                                        the probability of exceeding 0.0 given
                                        the variance of each cell.
            
            logits = {'ecut':1.0, 'ear':0.26}
                                    =>  Rescale as logits corresponding to
                                        the probability of exceeding 1.0 given
                                        an overall variance of 0.26.
        
        The "logits" option can be used in conjunction with the "mean_sd"
        and "m_b" arguments.  Logits are calculated first, then rescaled
        accordingly.

        -----------
        "reverse" <True, False> reverses the sign of the measures.  This
        option is applied after "straighten" and "logits" but before
        "mean_sd" and "m_b".

        -----------
        "mean_sd" specifies a target mean and standard deviation for
        the array of scores (ignored if "score" is an individual
        score).  If straighten or logits is specified, it is applied to the
        straightened or logitized scores.  If mean_sd is specified, 
        m_b is ignored.                        
                       
            mean_sd = None          =>  Do not rescale 'score'
                                        to have a target mean
                                        and standard deviation.

            mean_sd = [300,60]      =>  Rescale 'score', or its
                                        straightened/logit equivalent, to
                                        have a mean of 300 and
                                        standard deviation of 60.

        -----------
        "m_b" specifies that "score", whether as an array or
        individually, be multiplied by m and added to b.
        It is applied to straightened or logitized scores if available.
        It is ignored if mean_sd is specified.

            m_b = [10,100]          =>  Multiply the score(s) by 10
                                        and add 100.

        -----------
        "clip" forces scores to fall within a specified range.  It is the
        last rescaling step.
        
            clip = None             =>  Do not clip scores.
            
            clip = [200, 800]       =>  All scores less than 200 become
                                        200.  All scores greater than 800
                                        become 800.

        -----------
        "round_" rounds the scores to the specified number of decimals.
        
            round = None            =>  Do not round scores.
            
            round = 2               =>  Round to two decimal points (3.14)
            
        -----------
        "nanval" is the not-a-number value which identifies non-valid
        scores.

    Examples
    --------

        [under construction]


    Paste Function
    --------------
        rescale(score,  # [individual score or array of same-metric scores]
                straighten = None,  # [<None,True,'Percentile'> => intermediate rank-logit step]
                logits = None,  # [<None, {'ecut':0, 'ear':<ear array or float}>]
                reverse = False, # [<True, False> => reverse sign of measures]
                mean_sd = None, # [<None,[mean,sd]> => target mean, standard deviation]
                m_b = None, # [<None,[m,b]> => multiply by m, add b]
                clip = None, # [<None, [min, max]> => min and max scores]
                round_ = None, # [<None, int decimals> => round outputs]
                nanval = -999, # [not-a-number value in score array]
                )

    """

    if not isinstance(score, np.ndarray):
        if score == nanval:
            return nanval
    else:
        score = np.copy(score)
        
    if np.shape(score)[0] == 1:
        anchored_design = True
    else:
        anchored_design = False

    # TODO: This workflow won't work for anchored designs. 
    if anchored_design:
        if mean_sd is not None:
            if len(mean_sd) == 4:
                orig_mean = mean_sd[2]
                orig_sd = mean_sd[3]
            elif len(mean_sd) == 2:
                exc = ('In an anchored design, mean_sd needs four values: '
                       '[targ_mean, targ_sd, orig_mean, orig_sd]')
                raise rescale_Error(exc)
        if straighten is not None:
            exc = ('In an anchored design, rescale() does not support the '
                   '"straighten" parameter.')
            raise rescale_Error(exc)
                
    # Reverse the sign
    if reverse:
        nix = score == nanval
        score = score * -1
        score[nix] = nanval

    # Get original mean, sd
    if isinstance(score, np.ndarray):
        valloc = score != nanval
        
        if not anchored_design:
            if (straighten is True
                or mean_sd is not None
                ):
                if np.size(score[valloc]) == 0:
                    exc = 'score array is empty.\n'
                    raise rescale_Error(exc)
    
            orig_mean = np.mean(score[valloc])
            orig_sd = np.std(score[valloc])
            if orig_sd == 0:
#                print 'score=\n', score, score.dtype
                exc = 'Divide by zero error.  Not enough variation in scores.\n'
                raise rescale_Error(exc)
    else:
        straighten = None
        mean_sd = None

    # Straighten scores
    if (straighten is True
        or straighten == 'Percentile'
        ):
        if isinstance(score,np.ndarray):
            if len(np.shape(score)) == 1:
                score_ = score[:,np.newaxis]
            elif len(np.shape(score)) == 2:
                score_ = score
            else:
                exc = 'score array needs to have 1 or 2 axis in this context.\n'
                raise rescale_Error(exc)

            arr = dmn.core.Damon(score_,'array',nanval=nanval,verbose=None)
            metric = 'Percentile' if straighten == 'Percentile' else 'PLogit'
            arr.standardize(metric)
            p = arr.standardize_out['coredata']
            p_valloc = p != nanval

            if metric == 'PLogit':
                p_mean = np.mean(p[p_valloc])
                p_sd = np.std(p[p_valloc])
                p_std = (p[p_valloc] - float(p_mean)) / float(p_sd)
                score_[p_valloc] = p_std * float(orig_sd) + float(orig_mean)
            else:
                score_[p_valloc] = p[p_valloc] * 100
            
            score = score_
        else:
            exc = "'straighten' must be None if 'score' is not an array.\n"
            raise rescale_Error(exc)

    # Convert to logits TODO: ecut may need more usecases
    elif logits is not None:
        
        # Use existing 'prelogit' score
        if logits['ear'] is None:
            pass
        
        # Calculate logits using ear
        else:
            ecut = ['All', logits['ecut']]
            score = cumnormprob(score, logits['ear'], None, ecut, True, 
                                nanval)['Logit']

    # Rescale to target mean, sd
    if mean_sd is not None:
        std_score = (score[valloc] - float(orig_mean)) / float(orig_sd)
        score[valloc] = std_score * float(mean_sd[1]) + float(mean_sd[0])
        
    # Or just multiply and add
    elif m_b is not None:
        if isinstance(score,np.ndarray):
            score[valloc] = score[valloc] * float(m_b[0]) + float(m_b[1])
        else:
            score = score * float(m_b[0]) + float(m_b[1])

    if round_ is not None or clip is not None:
        nix = score == nanval        

    # Round scores
    if round_ is not None:
        score = np.around(score, round_)
        score[nix] = nanval
        
    # Clip scores
    if clip is not None:
        score = np.clip(score, clip[0], clip[1])
        score[nix] = nanval

    return score


###########################################################################

def meansd_to_mb(arr, mean_sd, nanval=-999):
    """Convert rescale() mean_sd param to equivalent m_b param.
    
    Returns
    -------
        [m, b] slope/intercept parameters for rescaling.
    
    Comments
    --------
        This is used by equate() for anchored designs. The mean_sd
        parameter in rescale() assumes access to the original mean and 
        standard deviation, which are not available in anchored designs.
        To get around this, equate() automatically replaces the mean_sd
        parameter with its m_b equivalent, which suffers no such 
        awkwardness.  The bank.equate_params['rescale'] parameters are
        changed accordingly.
    
    Parameters
    ----------
        "arr" is a 1-dimensional array of scores.
        
        -----------
        "mean_sd" is [target mean, target SD], the desired mean
        and standard deviation to which we are rescaling.
        
        -----------
        "nanval" is the not-a-number value in arr.
    
    """
    targ_mean, targ_sd = mean_sd[0], mean_sd[1]
    orig_mean, orig_sd = mean(arr, None, nanval), std(arr, None, nanval)
    
    m = targ_sd / orig_sd
    b = targ_mean - m * orig_mean
    
    return [m, b]
    
    


###########################################################################

def invUTU(U, # [array U of row or col coordinates]
           facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
           weights, # [None; 2-D array of weights corresponding to elements in U]
           nanval = -999.,  # [Not a number value]
           ):
    """Returns a component of B[solution], h, and B - B[missing]

    Returns
    -------
        (UTU)^-1

    Comments
    --------
        invUTU() converts matrix U into a square matrix by
        multiplying by its inverse (UT*U), then performs
        a matrix inversion.

        This expression is used repeatedly in calculating
        least square solutions, hat-matrix diagonals, and
        changes in solutions resulting from deletion of a
        data value.  Calculating it once and reusing the
        result improves efficiency.

        If invUTU() encounters a linear algebra error
        (generally a singular matrix error), it returns
        all zeros, the idea being that a downstream function
        will return the corresponding coordinates as
        NaNVals if necessary.

        --------------
        "U" is the R x D or D x C array of row or column
        coordinates.

        --------------
        "facet" tells how the U array is situated:
            'R'     => U is arranged as Rows by Dims
            'C'     => U is arranged as Dims by Cols

        --------------
        "weights" is a 2-D array of weights used to adjust the relative
        influence of each element in U.  It is obtained using the
        weight_coord() function.

    Paste function
    --------------
        invUTU(U, # [array U of row or col coordinates]
               facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
               weights, # [None; 2-D array of weights corresponding to elements in U]
               nanval = -999.,  # [Not a number value]
               )

    """

    # Define variables
    if facet == 'R':
        U = U

    elif facet == 'C':
        U = np.transpose(U)

    nDims = np.size(U,axis=1)

    # Calculate UTU^-1 unweighted
    if weights is None:
        UT = np.transpose(U)

        try:
            UTU = np.dot(UT,U)
        except ValueError:
            print 'UT=\n',UT
            print 'U=\n',U
            exc = 'Unable to calculate np.dot(UT,U) for some reason.\n'
            raise invUTU_Error(exc)

        try:
            invUTU_ = npla.inv(UTU)
        except npla.LinAlgError:
            invUTU_ = np.zeros((nDims,nDims))

    # Calculate UTW*U^-1 weighted
    else:
        W = weights
        UTW = np.transpose(U * W)
        UTW_U = np.dot(UTW,U)
        try:
            invUTU_ = npla.inv(UTW_U)
        except npla.LinAlgError:
            invUTU_ = np.zeros((nDims,nDims))

    # Transpose if necessary
    if facet == 'C':
        invUTU_ = np.transpose(invUTU_)

    return invUTU_



###########################################################################

def h_stat(U,   # [array of coordinates]
           facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
           invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
           ):

    """Returns array of h statistics corresponding to a row or column.

    Returns
    -------
        array of h statistics

    Comments
    --------
        "h" is the influence or leverage that a row
        or column entity has on the least squares solution
        that it contributes to.

            h[i] = U[i] * UTU^-1 * UT[i]

    Paste function
    --------------
        h_stat(U,   # [array of coordinates]
               facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
               invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
               )

    """

    # Define variables
    if facet == 'R':
        U = U
        UT = np.transpose(U)
        invUTU_ = invUTU_
    elif facet == 'C':
        UT = U
        U = np.transpose(UT)
        invUTU_ = np.transpose(invUTU_)

    # Calculate row leverage as Hat matrix diagonals
    N = np.size(U,axis=0)
    h = np.zeros(N)
    for i in xrange(N):
        h[i] = np.dot(np.dot(U[i,:],invUTU_),UT[:,i])

    if facet == 'R':
        h = h[:,np.newaxis]
    elif facet == 'C':
        h = np.transpose(h)[np.newaxis,:]

    return h




###########################################################################

def weight_coord(U, # [array of row or col coordinates]
                 logn, # [None; array of log counts per entity]
                 facet, # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
                 nanval,    # [Not-a-number value]
                 ):
    """Returns array of weights for use in weighted least squares

    Returns
    -------
        2-dimensional array of weights with same number of elements
        as U, and oriented like U.

    Comments
    --------
        weight_coord() provides a mechanism for forcing each row or column
        in U to have commensurate influence or leverage in the calculation
        of a solution.  This is intended to prevent "influence traps"
        where some entities, due merely to the arbitrary choice of starting
        coordinates, receive coordinates that are unduly influential
        and end up dominating the least squares solution in which they
        participate.  When this happens, it may appear that the model
        fits the data but the predictions for missing cells will be wildly
        wrong.

        The influence trap arises because in Damon the selection of a
        coordinate system is arbitrary, and some coordinate systems are
        prone to creating high-influence coordinates.  Suppose a high-influence
        person coordinate is assigned to an entity with a large block
        of missing cells.  When the item coordinates are being calculated,
        the influential person coordinate is ignored for the columns with
        missing cells but dominates the columns with non-missing cells.  The
        resulting item coordinates incline the person coordinate to become
        even more influential in the next iteration.  This leads to columns
        corresponding to the missing block becoming disconnected from those
        with non-missing data, and the common space fractures into two
        spaces, which breaks the predictive link between them.

        The occurrence of high-influence coordinates does not reflect any
        underlying influence level of the underlying entities since the
        selection of coordinates is purely arbitrary and done at random.
        In principle, there is no reason why one entity should be more
        influential than another (except for the number of observations
        it contains).  Therefore, it makes sense to force the coordinates
        to have similar influences, and this is done by influence-weighting.

        For this function, influence is measured as the root squared distance
        of each entity from the origin of the coordinate system (the squared
        sum of its coordinates), and the corresponding weight is 1 / Distance.
        Other measures of influence have been explored, e.g., the h-statistic,
        but the inverse distance approach is faster and so far has proven more
        effective in preventing influence traps, though it is not infallible.

        It is important to note that while this methodology is effective in
        preventing influence traps, there is still a modest chance of one arising
        by chance, depending on several factors.  They are detected by comparing
        the estimates for missing cells with those from non-missing cells
        and can be addressed by running more iterations (using the change
        between iterations as the driver) or by simply running coord()
        again, which will implement a new set of starting coordinates.

        weight_coord() also includes an option for weighting
        by the log of the number of observations associated with each
        row or column in U, since coordinates with a smaller number of
        observations are intrinsically less reliable.  This is in addition
        to influence weighting.

        "U" is the array of row or column coordinates to be weighted.

        "logn" is a 2-D array of log counts corresponding to elements
        in U.  logn instructs the function to weight each coordinate by
        the log of the number of observations associated with that entity.
        This weight is not instead of, but in addition to, the weight
        calculated to equalize influence.

            'None'  =>  Do not weight by observations.

            array of counts
                    =>  A 2-dimensional array of counts (row or column)
                        that corresponds to U.

        "facet" tells how the U array is situated:
            'R'     => U is arranged as Rows by Dims
            'C'     => U is arranged as Dims by Cols

        "nanval" is the numerical code that signifies "not a number."


    Paste function
    --------------
        weight_coord(U, # [array of row or col coordinates]
                     logn, # [None; array of log counts per entity]
                     facet, # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
                     nanval,    # [Not-a-number value]
                     )

    """

    # Define variables
    if facet == 'R':
        U = U
    elif facet == 'C':
        U = np.transpose(U)

    # Calc weights as inverse squared distances, clip min distance at 1
#    ValLoc = np.where(U[:,0] == nanval)
    Dist = np.where(U[:,0] == nanval,nanval,np.clip(np.sum(U**2,axis=1),1,np.inf))[:,np.newaxis]
    W = np.where(Dist == nanval,0,1/Dist)

    if logn is not None:
        W = np.where(W == nanval,nanval,W * logn)

    return W




###########################################################################

def condcoord(fac0coord,  # [None, nRowEnts x nDims facet1 coordinates array]
              fac1coord,   # [None, nColEnts x nDims Facet2 coordinates array]
              cond_facet,    # [<'Fac0','Fac1'>]
              function = 'Orthonormal',    # [<None,'Std','Orthonormal','NonNeg_1D',function object>]
              nanval = -999.   # [Not-a-number value]
              ):
    """Returns conditioned coordinates.

    Returns
    -------
        {'F0Std,    => standardized row coordinates
        'F1Std'     =>  standardized column coordinates
        }

    Comments
    --------
        condcoord() conditions row or column coordinates
        to meet a pre-defined mathematical requirement.
        It supports several pre-defined conditioning functions
        and allows the user to define his own conditioning
        functions.  While Damon coordinates do not strictly
        require conditioning, there are many cases where
        conditioning is a good idea, for instance to convert
        one set of coordinates to orthonormal (see below) or
        to handle data that is known to be in some sense non-linear.

        A point of terminology:  the edges of an array are called
        "facets".  Facet 0 entities are row entities.  Facet 1
        entities are column entities.  Facet 3 entities would be
        something like "depth" entities.

    Arguments
    ---------
        "fac0coord" is the Facet 0 coordinates array, where
        rows are row entities and columns are dimensions.

        --------------
        "fac1coord" is the Facet 1 coordinates array, where
        rows are column entities and columns are dimensions.

        --------------
        "cond_facet" specifies whether the coordinates to be
        conditioned are those in the fac0coord parameter ('Fac0'
        or those in the fac1coord parameter ('Fac1').  If
        fac1coord = None, you must specify cond_facet as 'Fac0',
        and vice versa.  If neither fac0coord nor fac1coord are
        None, cond_facet specifies the primary facet to be
        conditioned and adjusts the opposing facet so that there
        is no change in the estimates.

        --------------
        "function" refers either to a Damon-defined conditioning
        function or to a function defined by the user for modifying,
        or conditioning, the coordinates array specified by cond_facet.

            The options are:

                None        =>  Do not condition coordinates in any way.

                A Python dictionary containing specified functions or
                a user-defined function for imposing conditions on an
                array, with an optional 'refit' key:

                {'Fac0':condition,'Fac1':condition,'refit':metric}

                If 'refit' is not included, it is skipped.  Otherwise,
                it is applied.

            The condition functions are as follows:

                'Std'       =>  Adjust the specified coordiantes array so
                                that each column (dimension) has a standard
                                deviation of 1.0.  This option actually adjusts
                                both the target facet and the opposing facet,
                                though the latter is not used by the coord()
                                function.

                'Orthonormal'
                            =>  The coordinates array is transformed
                                to be orthonormal, i.e., so that the
                                dimensions are orthogonal to each other
                                (their dot products are zero) and the
                                root sum of squares (distance) for each axis
                                is 1.0.  This is the default option and has
                                several important benefits, described below.

                                condcoord_ = {'Fac0':'Orthonormal','Fac1':None}

                                Facet 0 coordinates are constrained to be
                                orthornormal.  Facet 1 coordinates float to
                                maximize fit to the data and to Facet 0.
                                This can be reversed so that the facet 1
                                coordinates are othonormal instead.

                'NonNeg_1D' =>  Force coordinates to be positive for both
                                facets.  This only works if ndim = 1.  This
                                condition creates coordinates that yield
                                estimates in an exponential metric that is
                                different from that of the observed data.
                                To get it back into the observed metric,
                                use the 'refit' option.  If the data are
                                dichotomous, 'refit' can be either '0-1'
                                or 'LstSq':

                                condcoord_ = {'Fac0':'NonNeg_1D',
                                              'Fac1':'NonNeg_1D'
                                              }

                 'AllSame'  =>  THIS OPTION NOT USED.

                                 Force facet 1 coordinates to all be one and
                                facet 0 coordinates to all equal the mean of
                                the facet 0 coordinates.  This is used to
                                model random data.  The coordinates arrays
                                will have one column or row and randomness
                                represents a condition where all entities
                                can be treated as essentially equivalent.
                                This conditioning is triggered in coord()
                                automatically by ndim = 0 and not by the
                                condcoord_ argument.

                User-defined function
                            =>  The user can define a function outside of the
                                coord() method and refer to it here.  Example:

                                def my_func(Fac):
                                    return np.exp(Fac)

                                Then, to apply my_func() to both facets:

                                condcoord_ = {'Fac0':my_func,
                                             'Fac1':my_func,
                                             'refit':'LstSq'
                                             }

                                Here's a fancier function:

                                def my_func2(Fac):
                                    Fac1 = np.zeros(np.shape(Fac))
                                    Fac1[:,0] = Fac[:,0] + 5
                                    Fac1[:,1] = Fac1[:,0]**2
                                    Fac1[:,2] = 1.0
                                    return Fac1

                                condcoord_ = {'Fac0':my_func2,
                                             'Fac1':None,
                                             }

                                Note: To pass in your function object, just make
                                sure to drop the parentheses.

        --------------
        "nanval" signifies the 'Not-a-Number Value".  It is used to flag
        missing values.

        Benefits of Orthonormal
        -----------------------
        As mentioned, arrays converted to orthonormal are such
        that the dimensional coordinates are orthorgonal to each
        other (have dot products of 0.0) and the root sum of squares
        of each column equals 1.  Only one facet or the other can
        be specified as orthonormal.  If both facets are orthonormal
        the coord() function will lurch back and forth between
        two competing and irreconcilable coordinate spaces.  Some
        benefits of orthonormal (which is usually applied to the
        Facet 0 or row coordinates) are as follows:

            1)  It can be used to prevent the random starter
                coordinates matrix from accidentally being
                ill-conditioned (e.g., not linearly independent).

            2)  It helps prevent the coordinates from accidentally
                exploding out of computing range during the
                coord() iterative process.

            3)  It makes the Silsdorf decomposition matrices
                orthonormal in a way that is similar to those
                produced by Singular Value Decomposition.

            3)  When applied to the final Fac0 (row) coordinate system,
                the resulting column vectors of the opposite facet
                occupy a space having a non-arbitrary origin such
                that the coordinate vectors alone are "sufficient" to
                calculate the objective distance of each vector from
                the origin, as well as the vector's angular relation
                (cosine) relative to other column vectors, and other
                statistics.  This distance is the geometrical equivalent of
                the standard deviation of the estimates for that
                column, but freed from sampling considerations.  The
                cosine between two column vectors is the geometrical
                equivalent of their correlation, again freed from sampling
                considerations.  The same properties hold for the row
                coordinates if the orthonormal option is applied to
                the columns (Facet 1).

                Why this is important:

                    a)  The vector distances and cosines are "objective" --
                        independent of the sample of entities used to calculate
                        them.  That means statistical assumptions such
                        as representative samples are not required for calculating
                        these statistics.  Conventional distances and cosines
                        (a.k.a., standard deviations and correlations) are
                        extremely sample-dependent.

                    b)  They are convenient, in that they make it easy to
                        calculate means, standard deviations, and correlations
                        (actually, their more accurate and reliable objective
                        equivalents) without having to refer to the
                        observations or estimates matrices.

                    c)  This means that the coordinate vectors become,
                        in effect, "sufficient" statistics -- sufficient to
                        describe the properties of each row/column without
                        reference to the underlying data.  They are the
                        multidimensional equivalent of Fischer's definition
                        of sufficiency as used in Rasch models.

    Examples
    --------

        [under construction]


    Paste function
    --------------
        condcoord(fac0coord,  # [None, nRowEnts x nDims facet1 coordinates array]
                  fac1coord,   # [None, nColEnts x nDims Facet2 coordinates array]
                  cond_facet,    # [<'Fac0','Fac1'>]
                  function = 'Orthonormal',    # [<None,'Std','Orthonormal','NonNeg_1D',function object>]
                  nanval = -999.   # [Not-a-number value]
                  )

    """

    # Define variables
    R = np.copy(fac0coord)
    C = np.copy(fac1coord)

    ##################################
    # Apply no function, just pass through
    if function is None:
        RStd = R
        CStd = C

    ##################################
    # funcstep dictionary -- steps by which to modify the R and C matrices
    elif hasattr(function,'__call__'):
#        Steps = function
        if cond_facet == 'Fac0':
            Fac = R
            MissLoc = np.where(R == nanval)
        elif cond_facet == 'Fac1':
            Fac = C
            MissLoc = np.where(C == nanval)
    
        # Run the conditioning function
        try:
            Fac = function(Fac)
            Fac[MissLoc] = nanval
        except:
            exc = sys.exc_info()
            raise condcoord_Error(exc)

        # Assign to correct facet
        if cond_facet == 'Fac0':
            RStd = Fac
            CStd = None
        elif cond_facet == 'Fac1':
            RStd = None
            CStd = Fac

    ##################################
    # 'NonNeg_1D' method for both facets
    elif function == 'NonNeg_1D':
        if cond_facet == 'Fac0':
            Fac = R
            MissLoc = np.where(R == nanval)
        elif cond_facet == 'Fac1':
            Fac = C
            MissLoc = np.where(C == nanval)

        # Force coordinates to be positive
        try:
            Fac = np.where(Fac == nanval, nanval, np.exp(Fac))
        except:
            exc = 'condcoord() function argument failed.\n'
            raise condcoord_Error(exc)

        # Assign to correct facet
        if cond_facet == 'Fac0':
            RStd = Fac
            CStd = None
        elif cond_facet == 'Fac1':
            RStd = None
            CStd = Fac

    ##################################
    # 'Orthonormal' method for relevant facet F
    elif function == 'Orthonormal':
        if cond_facet == 'Fac0':
            Fac = R
        elif cond_facet == 'Fac1':
            Fac = C

        # Select valid coordinates
        FacLocVal = np.where(Fac[:,0] != nanval)
        FacVal = Fac[FacLocVal]

        # Make sure array is large enough
        shape = np.shape(FacVal)

        if shape[0] < shape[1]:            
            exc = 'Unable to perform QR factorization.  Check dimensionality.\n'
            raise condcoord_Error(exc)

        # Perform QR decomposition
        # The Q array is equivalent to the Gram-Schmidt orthogonalization
        try:
            FacValStd = npla.qr(FacVal)[0]
        except:
            exc = 'QR factorization failed.\n'
            raise condcoord_Error(exc)

        # Transfer FStd to F
        FacStd = np.copy(Fac)
        FacStd[FacLocVal] = FacValStd

        # Assign to correct facet
        if cond_facet == 'Fac0':
            RStd = FacStd
            CStd = None
        elif cond_facet == 'Fac1':
            CStd = FacStd
            RStd = None


    ##################################
    # 'Std' method
    elif function == 'Std':

        # Define R, C masked arrays
        R_ma = npma.masked_values(R,nanval)
        C_ma = npma.masked_values(C,nanval)

        # Select facet to standardize
        if cond_facet == 'Fac0':
            F = R_ma
            OppF = C_ma
        elif cond_facet == 'Fac1':
            F = C_ma
            OppF = R_ma

        # Get standard deviation of columns in R
        FDimSD = np.std(F,axis=0)
        if FDimSD.any() == 0:
            exc = 'A dimension in row coordinates has zero variation.\n'
            raise condcoord_Error(exc)

        # Divide by SD
        InvFDimSD = 1/FDimSD

        # Adjust R, C
        FStd_ma = F * InvFDimSD
        OppFStd_ma = OppF * FDimSD

        # Convert back to normal array
        if cond_facet == 'Fac0':
            RStd = npma.filled(FStd_ma,fill_value=nanval)
            CStd = npma.filled(OppFStd_ma,fill_value=nanval)
        elif cond_facet == 'Fac1':
            RStd = npma.filled(OppFStd_ma,fill_value=nanval)
            CStd = npma.filled(FStd_ma,fill_value=nanval)

    ##################################
    # 'AllSame' method
    elif function == 'AllSame':
        
        if cond_facet == 'Fac0':
            fac = R
        elif cond_facet == 'Fac1':
            fac = C
        
        # Select valid coordinates
        faclocval = np.where(fac[:, 0] != nanval)
        facval = fac[faclocval]

        # Make all elements equal 1.0 or mean element
        if cond_facet == 'Fac0':
            facmean = np.mean(facval)
        elif cond_facet == 'Fac1':
            facmean = 1.0
            
        fac_same_ = np.zeros(np.shape(facval)) + facmean

        # Apply to original
        fac_same = np.copy(fac)
        fac_same[faclocval] = fac_same_

        # Assign to correct facet
        if cond_facet == 'Fac0':
            RStd = fac_same
            CStd = None
        elif cond_facet == 'Fac1':
            RStd = None
            CStd = fac_same
            
    else:
        exc = 'Unable to figure out condcoord() arguments.\n'
        raise condcoord_Error(exc)
       

    return {'F0Std':RStd,'F1Std':CStd}




###########################################################################

def solve1(U,   # [array of row or col coordinates]
           x,  # [2-D vector of row or col observations]
           invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
           weights, # [None; array of weights corresponding to elements in U]
           facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
           nanval = -999.,  # [Not-a-number value, for outputs]
           ):
    """Returns least squares solution given specified matrix elements.

    Returns
    -------
        Least squares solution given specified matrix elements.

    Arguments
    ---------
        The output is the same as numpy's lstsq() and solve() functions,
        but solve1() makes it possible to compute the inverse of UT*U
        outside the function.  This saves computation time when invUTU_
        is the same across rows/columns, as occurs when the
        'imputation' method is used.  solve1() also allows the
        use of weights to compute weighted least squares solutions.

        ------------------
        "U" is the 2-D array of row or column coordinates, sized
        either as Rows x Dims or Dims x Cols (see 'facet' below).

        ------------------
        "x" is one 2-D row or column of observations, with no
        missing values.

        ------------------
        "invUTU_" is the matrix inversion of the transpose of U times
        U (UTU^-1) and is returned by the invUTU() function.  It is
        calculated outside the function to avoid redundancy.

        ------------------
        "weights" is a 2-dimensional array of weights for setting the
        relative influence of each element in U.  It is generally
        obtained using the weight_coord() function, whose weights
        adjust for the relative influence of each element's coordinates
        and the number of observations it has.  The weighted
        and unweighted least squares equations for solution V are:

            V[unweighted] = (UT*U)^-1 * UT*X
            V[weighted] = (UTW*U)^-1 * UTW*X

        where W is (in theory, but not in practice) a diagonal matrix
        of weights drawn from the 'weights' argument.

        IMPORTANT:  When the weights parameter is used, it is assumed that
        the 'invUTU_=' array used in this function has been calculated outside
        the function using the same set of weights.  This pre-weighting
        is done using the 'weights' argument in the invUTU() function.
        The U and Obs parameters, on the other hand, do not require any
        special weighting outside the function.

        ------------------
        "facet" tells how the U array is situated:
            'R'     => U is arranged as Rows by Dims
            'C'     => U is arranged as Dims by Cols

        ------------------
        "nanval" is the Not-a-Number value to assign to invalid
        outputs.

    Examples
    --------



    Paste function
    --------------
        solve1(U,   # [array of row or col coordinates]
               x,  # [2-D vector of row or col observations]
               invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
               weights, # [None; array of weights corresponding to elements in U]
               facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
               nanval = -999.,  # [Not-a-number value, for outputs]
               )

    """

    # Intercept bad invUTU_
    if not isinstance(invUTU_,np.ndarray):
        V = np.zeros(np.shape(x)) + nanval
        return V

    # Define variables
    if facet == 'R':
        X = x
        UT = np.transpose(U)
        
    elif facet == 'C':
        X = np.transpose(x)
        UT = U
        U = np.transpose(UT)
        invUTU_ = np.transpose(invUTU_)

    # Calculate unweighted solution
    if weights is None:
        UTX = np.dot(UT,X)
        V = np.dot(invUTU_,UTX)

    # Calculate weighted solution (invUTU_ comes in already weighted)
    else:
        W = weights
        UTW = np.transpose(U * W)
        UTW_X = np.dot(UTW,X)
        V = np.dot(invUTU_,UTW_X)

    # Catch Inf and NaN
    if np.inf in V:
        V[:] = nanval

        exc = 'solve1() inf'
        raise solve1_Error(exc)

    if np.nan in V:
        V[:] = nanval

        exc = 'solve1() nan'
        raise solve1_Error(exc)

    # Orient solution vector
    if facet == 'C':
        V = np.transpose(V)

    return V


###########################################################################

def solve2(R,   # [ents x dims array of row coordinates, no NaNVals]
           C,   # [ents x dims array of col coordinates, no NaNVals]
           x,   # [2-D vector of data, no NaNVals]
           targfacet,   # [<'R','C'>, type of coordinates to calculate]
           invUTU_,  # [None, Output of invUTU(), U = opposing facet of 'targfacet': (UT * U)^-1]
           weights, # [None, array of weights corresponding to elements in U array]
           method,  # [<'LstSq','IRLS'>]
           meth_specs,   # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]
           nanval = -999.,  # [Not-a-number value, for invalid outputs]
           pytables = None, # [None, 1 => R, C, x are pytables and solution should be calculated in "chunks"]
           ):
    """Returns vector solution for a specified method given matrix elements.


    Returns
    -------
        2-D ent x dims vector solution for a specified method
        given matrix elements.

    Arguments
    ---------
        solve2() is one level higher than solve1(), which only
        solves least squares problems.  It was built to contain
        a variety of solutions as they may prove necessary to
        be called on by coord() under one name.  At this time
        it contains only ordinary least squares and iteratively
        reweighted least squares.

        ------------------
        "R" is an ents x dims array of row coordinates, with
        no NaNVals allowed.

        ------------------
        "C" is an ents x dims array of column coordinates.

        ------------------
        "x" is a string of data corresponding to the elements
        in R or C, whichever is specified by targfacet.

        ------------------
        "invUTU_" is a matrix inversion (transpose(U) * U^-1),
        the output of invUTU(), where U is the facet opposite
        targfacet.

        ------------------
        "weights" is the array of weights corresponding to
        elements in the U array, obtained from weight_coord().

        ------------------
        "method" is the the type method used to calculate each
        set of coordinates, the options being 'LstSq' and 'IRLS'.

        ------------------
        "meth_specs" is a dictionary of specifications unique
        to each method and is designed to be a flexible container
        for use with unforeseen methods.  For method = 'IRLS',
        the specs are:

            {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]

        ------------------
        "nanval" is the Not-a-number value used to identify
        non-numeric outputs such as inf or nan.  (Inputs are assumed
        not to contain NaNVals.)

    Examples
    --------

        [under construction]

    Paste function
    --------------
        solve2(R,   # [ents x dims array of row coordinates, no NaNVals]
               C,   # [ents x dims array of col coordinates, no NaNVals]
               x,   # [2-D vector of data, no NaNVals]
               targfacet,   # [<'R','C'>, type of coordinates to calculate]
               invUTU_,  # [None, Output of invUTU(), U = opposing facet of 'targfacet': (UT * U)^-1]
               weights, # [None, array of weights corresponding to elements in U array]
               method,  # [<'LstSq','IRLS'>]
               meth_specs,   # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]
               nanval = -999.,  # [Not-a-number value, for invalid outputs]
               pytables = None, # [None, 1 => R, C, x, invUTU_, weights are pytables and solution should be calculated in "chunks"]
               )

    """

    # Get least squares solution
    if method == 'LstSq':
        if targfacet == 'R':
            U = C
        elif targfacet == 'C':
            U = R

        # Apply solve1()
        V = solve1(U = U,   # [array of row or col coordinates]
                   x = x,  # [2-D vector of row or col observations]
                   invUTU_ = invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
                   weights = weights, # [None; array of weights corresponding to elements in U]
                   facet = 'R',   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
                   nanval = nanval,  # [Not-a-number value, for outputs]
                   )

    # Get iteratively reweighted least squares solution
    elif method == 'IRLS':
        if targfacet == 'R':
            U = C
        elif targfacet == 'C':
            U = R

        # Apply irls()
        V = irls(U = U, # [2_D array of coordinates]
                 x = x, # [2-D array of observations, no NaNVals]
                 facet = 'R', # ['R' => U = rows x dims; 'C' => U = dims x cols]
                 runspecs = meth_specs['RunSpecsDichot'],    # [[StopWhenChange,MaxIteration]]
                 ecutmaxpos = meth_specs['ecutmaxpos'],    # [[ECut,MaxPos]]
                 pcut = meth_specs['pcut'], # [pcut corresponding to ECut]
                 nanval = nanval,    # [Not-a-number value]
                 )
    else:
        exc = 'Unable to figure out method.\n'
        raise solve2_Error(exc)

    return V


###########################################################################

def irls(U, # [2_D array of coordinates]
         x, # [2-D array of observations, no NaNVals]
         facet = 'R', # ['R' => U = rows x dims; 'C' => U = dims x cols]
         runspecs = [0.01,5],    # [[StopWhenChange,MaxIteration]]
         ecutmaxpos = [0.5,1.4],    # [[ECut,MaxPos]]
         pcut = 0.5, # [pcut corresponding to ECut]
         nanval = -999.,    # [Not-a-number value]
         ):
    """Applies Iteratively Reweighted Least Squares to dichotomous data.

    Returns
    -------
        Returns coordinates correcting for expected cell variance.

    Comments
    --------
        irls() is applied when the expected variance per cell
        is not the same for all possible cell values, called
        "heteroscedasticity".  Its opposite, homoscedasticity,
        is one of the key requirements of the Gauss-Markov theorem,
        which describes the situations under which linear least
        squares is most likely to be effective.  Heteroscedasticity
        is especially problematic with dichotomous data and causes
        damon-calculated coordinates to lose their objectivity
        property (though the estimates are often quite usable).

        One way to correct for heteroscedasticity is to use a
        variation called "Iteratively Reweighted Least Squares."
        Ordinary least squares (in the context of damon) is applied
        to dichotomous data to calculate an initial set of solution
        coordinates.  These are combined with the basis coordinates (U)
        to calculate cell probabilities (est2prob() forces a 0-1 range).
        The cell probabilities are converted into expected cell variances.
        The variances are inverted to produce weights.  The weights W
        are entered into a weighted least squares formula to calculate
        a new set of solution coordinates:

            V = (UTWU)^-1 UTWx

        Solution coordinates V are multiplied by U to calculate new
        cell probabilities, which are inverted to get new weights W,
        to get a new V, and so on.  When V stabilizes or the iteration
        is halted, V is returned.

        "U" is the basis array of coordinates, i.e., the predictor
        variables for X.

        "x" is the array of observations, all NaNVals removed.  Thus,
        this function is only used within the coord() function, where
        each data vector is already resized to ignore missing values.

        "facet" specifies whether x and U are coming in as rows ('R')
        or columns ('C').

        "runspecs" identifies the conditions under which the function
        should stop iterating.  The format is:

            runspecs = [StopWhenChange,MaxIteration], where:

            StopWhenChange is the size of the change in V at or below
            which to stop iterating.

            MaxIteration is the maximum number of iterations before
            stopping.

        The function stops when either of the two conditions is met,
        whichever comes first.

        "ecutmaxpos" contains two parameters in a list:

            [ECut,MaxPos]

        "ECut" is the estimate cut-point that should correspond to the
        probability given by "pcut" (pcut is generally 0.50).  When the
        data are dichotomous, ECut will be 0.50.  estimates above
        ECut are interpreted as "success".

        "MaxPos" is the theoretical maximum possible estimate,
        obtained by observing the ogive function relating estimates to
        probabilities.  Any estimate above MaxPos is considered an
        outlier and the resulting probabilities are clipped to be
        slightly below 1.0.

        "pcut" is the probability that defines the lower boundary of
        "success".  By convention it is set at 0.50.

        To calculate weights we need probabilities between 0 and 1. But
        the estimates may spill above or below.  To force them within
        range and to get them into a valid probability metric, we apply
        the following function, copied from est2prob(), where a proof
        is given.

                P = Ex + y, where

        	x = 1/m - (mp - e)/m(m - e)
        	y = (mp - e) / (m - e), and

        	m = the theoretical maximum estimate (MaxPos) = 1.4,
        	e = the estimate cutpoint (ECut) corresponding to pcut = 0.5,
        	p = the probability cutpoint (pcut) = 0.5.

        "nanval" is the numerical code that signifies "not a number."

    Paste function
    --------------
        irls(U, # [2_D array of coordinates]
             x, # [2-D array of observations, no NaNVals]
             facet = 'R', # ['R' => U = rows x dims; 'C' => U = dims x cols]
             runspecs = [0.01,5],    # [[StopWhenChange,MaxIteration]]
             ecutmaxpos = [0.5,1.4],    # [[ECut,MaxPos]]
             pcut = 0.5, # [pcut corresponding to ECut]
             nanval = -999.    # [Not-a-number value]
             )

    """
    #Mark_irls

    # Orient inputs to rows
    if facet == 'C':
        X = np.transpose(x).astype(float)
        U = np.transpose(U).astype(float)

    elif facet == 'R':
        X = x

    # some variables
    nrows = np.size(U,axis=0)
    nDims = np.size(U,axis=1)
    StopWhenChange = runspecs[0]
    MaxIteration = runspecs[1]

    # Iterate weights
    Change = StopWhenChange + 1
    it = 0
    W = np.ones((nrows,1))
    V_Prev = np.zeros((nDims,1))
    SD = np.std(X)

#    # Get influence weights
#    W_coord = weight_coord(U, # [array of row or col coordinates]
#                             logn = None, # [None; array of log counts per entity]
#                             facet = 'R', # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
#                             nanval = nanval,    # [Not-a-number value]
#                             )

    # Prep for calculating probs
    ecut = ecutmaxpos[0]
    Max = ecutmaxpos[1]

    while it < MaxIteration and Change > StopWhenChange:

        # Weight UT, X and solve for V
        UTW = np.transpose(U * W)
        UTW_U = np.dot(UTW,U)
        UTW_X = np.dot(UTW,X)
        V = npla.solve(UTW_U,UTW_X)

        # Calc estimates
        E = np.dot(U,V)

        # Convert to probabilities (see docs)
        x = 1/Max - (Max*pcut - ecut) / (Max*(Max - ecut))
        y = (Max*pcut - ecut) / (Max - ecut)
        P = np.clip(E*x + y,0.001,0.999)  # Clip just in case

        # Compute new weight
        Var = (P * (1 - P))
        W = 1/Var  #(W_coord/Var)

        # Increment iterations
        it += 1
        Change = np.sqrt(np.mean((V_Prev - V)**2)) / float(SD)
        V_Prev = np.copy(V)

    # Orient coordinates
    if facet == 'C':
        V = np.transpose(V)

    return V



###########################################################################

def jolt(U,     # [ent x dims coordinates array]
         sigma = 10., # [(Max - Mean)/SD above which jolting is indicated]
         jolt_ = 0.30,  # [jolt_*rand() amount of randomness to add to U]
         joltflag = False,  # [<True,False> => True if previous facet called for a jolt]
         condcoord_ = None,  # [<None,condcoord() args> => condition jolt noise]
         facet = None,  # [<None,'Fac0','Fac1'> => facet of U (used only by condcoord_)]
         nanval = -999. # [Not-a-number value]
         ):
    """Add randomness to U to escape degenerate solutions.

    Returns
    -------
        jolt() returns [NewU, joltflag].

        NewU is a copy of U if the standardized difference
        between the maximum coordinate distance and the mean coordinate
        distance is less than or equal to sigma.  If greater, it is
        assumed that coord() is in danger of returning a degenerate
        solution.  In this case, U is multiplied by a "jolt" of
        randomness to steer coord() toward another (and, one hopes,
        less degenerate) solution.  The modified U is returned and
        fed into the coord() sub-routines in lieu of U.

        joltflag reports whether a jolt was administered.

    Comments
    --------
        It is occasionally possible for Damon's coord() function
        to assign starter coordinates that iterate toward a
        "degenerate" solution, i.e., a solution that is not linearly
        independent or where one or more entities are assigned
        coordinate values that are way out of range of the others.
        The "orthonormal" condcoord() option heads of the problem
        of linearly independence.  The weighting function in coord()
        prevents very large coordinate values from infecting downstream
        calculations.  But there still remains the possibility with
        some types of datasets of one or more entities ending up
        with unduly large coordinate values (generally with internally
        opposing signs), which can cause an apparently fitting model
        to return extremely bad predictions of missing cells.

        The jolt() function is used to detect such outlier entity
        coordinates (which have nothing to do with the entity per se
        but only with the arbitrary choice of starter coordinates)
        and to "jolt" the entire coordinate array, plus the opposing
        coordinate array (controlled using joltflag) into a different
        spatial orientation by adding a specified amount of random
        error.  It is hoped that the new spatial orientation can
        escape the degenerate solution.  If it doesn't, jolt will
        be triggered again, with new random numbers.

    Arguments
    ---------
        "U" is an ent x dim array of coordinates which may include
        NaNVals.

        --------------
        "sigma" is the value of Z = (Max - Mean)/SD which triggers the
        jolt command.  Each entity has a distance from the origin
        (the root sum of squares of its coordinates).  "Max" is the
        maximum such distance in U.  "Mean" is the mean distance.
        "SD" would ordinarily be the standard deviation of the
        distances, but in this case is the standard deviation of the
        lower half of the distances (< median).  This underestimates
        the true standard deviation but has the virtue of omitting
        any outlier distances from inflating the standard deviation,
        masking them in small samples.

        Z is the "quasi" standardized distance of the maximum distance
        from the mean.

        If the distribution of distances were normal (it is not), and
        if the standard deviation were calculated from the whole
        sample (it is not), then Z = 2 would indicate that the
        maximum distance has a 0.05 probability of occurring.
        A sigma > than 6, say, probably indicates a maximum
        distance that has a very low probability of occurring, and
        might be a reasonable flag for a possible degenerate solution.
        The optimal flag can only be found by trial and error.

        --------------
        "jolt_" controls the amount of randomness that will be
        multiplied (not added) by U to create a new U.  jolt_ = 0.30
        generates random numbers from 0 to 0.30, which are added
        to and centered on U.  A similar jolt is added to the
        opposing facet in the next half-iteration.

        --------------
        "condcoord_" provides options for conditioning the noise
        in U and goes into the condcoord() function.  To learn
        the syntax, consult help(tools.condcoord()).

        --------------
        "facet" <None,'Fac0',Fac1'> describes the facet of U.  It
        is only used for conditioning the coordinates.

        --------------
        "nanval" is the "Not-a-Number Value" used in U to indicate
        invalid coordinate values.

    Examples
    --------


    Paste function
    --------------
        tools.jolt(U,     # [ent x dims coordinates array]
                 sigma = 10., # [(Max - Mean)/SD above which jolting is indicated]
                 jolt_ = 0.30,  # [jolt_*rand() amount of randomness to add to U]
                 joltflag = False,  # [<True,False> => True if previous facet called for a jolt]
                 condcoord_ = None,  # [<None,condcoord() args> => condition jolt noise]
                 facet = None,  # [<None,'Fac0','Fac1'> => facet of U (used only by condcoord_)]
                 nanval = -999. # [Not-a-number value]
                 )

    """

    # Invalid U
    ValU = U[U[:,0] != nanval]

    # Calc distances, mean, max, sd, Z
    Dist = np.sqrt(np.sum(ValU**2,axis=1))[:,np.newaxis]
    Mean = np.mean(Dist,axis=0)
    Max = np.max(Dist,axis=0)
    median_ = np.median(Dist)

    if (ValU == 0.0).all():
        SD = 1
    else:
        SD = max(np.std(Dist[Dist < median_ + (Max - median_)/2.0],axis=0),1)

    Z = (Max - Mean) / SD
    nrows = np.size(U,axis=0)
    ncols = np.size(U,axis=1)

    if Z > sigma or joltflag is True:
        R = npr.rand(nrows,ncols) * jolt_ - jolt_ / 2.0

        if condcoord_ is not None:
            if facet == 'Fac0':
                R = condcoord(R,None,'Fac0',condcoord_['Fac0'],nanval)['F0Std']
            elif facet == 'Fac1':
                R = condcoord(None,R,'Fac1',condcoord_['Fac1'],nanval)['F1Std']
            else:
                exc = 'Need to specify a facet if using condcoord_.\n'
                raise jolt_Error(exc)

        NewU = np.where(U == nanval,nanval,U + R)

        if joltflag is True:
            joltflag = False
        else:
            joltflag = True
    else:
        NewU = U
        joltflag = False

    return [NewU,joltflag]




###########################################################################

def faccoord(targfac, # [ [FacetNum,FacetArray,Anchored], e.g., [0,FacetArray0,True] => existing facet array to recalculate] ]
             targdatindex, # [None,targdatindex, e.g., TargDatInd0 => index of valid data per targ entity]
             data,   # [2-D targfac x oppfac array of data (rotates so that targfac is always rows)]
             oppfac, # [Ents x dims array of coordinates of opposite facet(s), or their product]
             oppweights,  # [array of weights corresponding to opposite facet array]
             solve_meth = 'LstSq', # [method for calculating coordinates <'LstSq','IRLS'>]
             solve_meth_specs = None,  # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],...}]
             condcoord_ = None,  # [None,'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...}>,} ]
             miss_meth = 'IgnoreCells', # ['ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells entirely (preferred)]
             nanval = -999., # [Not-a-Number value, to label non-numerical outputs]
             ):
    """Calculates coordinates for all entities in a specified facet.

    Returns
    -------
        Output dictionary:
            {'Fac':FacNum           => facet integer (0,1)
            'FacCoord':FacCoord,    => array of coordinates for target facet
            'Warn1':Warn1,          => Warning if there was an error calculating an entity's coordinates
            'Warn2':Warn2           => Warning if one of the facet coordinates is flagged as invalid (e.g., insufficient data or variation)
            }

    Comments
    --------
        faccoord() calculates coordinates for each entity corresponding
        to a specified facet based on parameters received from
        the Damon.coord() method.

        --------------
        "targfac" is the target facet, the facet whose elements
        are to be computed.  It's number and array are entered
        as a list along with a True/False "anchored" flag:

            [target facet number, target facet array, anchored]
            [0, 0Array, False]  
                        =>  The target facet is facet 0 (rows),
                            its array is 0Array, and it is not
                            anchored.

            The target array is always Ents x Dims, regardless
            of facet.
            
            The anchored flag helps control how nanvals are
            interpreted.  If anchored = False, whenever a coordinate
            ends up being assigned a nanval due to a calculation
            exception, that nanval is preserved and propagated
            through subsequent iterations.  However, anchored
            coordinates may be nanval simply because values
            haven't been calculated for them yet.  anchored = True
            allows the missing coordinates to be calculated
            and not ignored.
            
        --------------
        "targdatindex" is an index telling which elements in
        the opposing facet array are to go into the calculation.
        It is based on the locations of the non-missing cells
        for the target facet and on whether the opposing facet
        elements are valid.  It is formatted as a list of
        arrays containing the valid row or column indices for
        each entity in the target facet.  Example:

            targdatindex = [array([0, 1, 2, 3, 4, 5, 6, 7]),
                            array([0, 1, 2, 3, 4, 5]),
                            array([0, 1, 4, 5, 6, 7])]

            If the target facet is FacNum = 0 (rows), this
            index suggests there are three elements in the
            target facet and 8 elements in the opposing facet
            (columns), and that the second row element should
            ignore columns 6 and 7 while the third row element
            should ignore columns 2 and 3.

        --------------
        "data" is the 2-D targfacet x OppFacet data array.  If
        the target facet is rows, then data is entered as
        rows x columns, matching the source data matrix.  If
        the target facet is columns, then data must be transposed
        so that the columns can be handled as rows.  coord()
        handles all this in passing information to faccoord().

        data can be a PyTable; data rows and columns are
        accessed with the same slice notation.

        --------------
        "oppfac" is an Ents x Dims array of coordinates for
        the facet opposite the target facet.  When Damon evolves
        to more than two facets, the opposing facet will be
        a tensor product of multiple facet arrays.

        --------------
        "oppweights" is Ents x 1 array of weights corresponding to
        the opposing facet, used to avoid degenerate solutions.

        --------------
        "solve_meth" is the method used for calculating coordinates
        for an individual entities.  There are two options:
        'LstSq' (ordinary least squares), 'IRLS' (iteratively reweighted
        least squares).

        --------------
        "condcoord_" is a specification for conditioning the target
        facet array after it has been calculated.  See condcoord()
        docs.

        --------------
        "miss_meth" specifies whether to use the 'ImputeCells' method
        for dealing with missing data or the 'IgnoreCells' method.

        --------------
        "nanval" is the Not-a-Number value to assign to non-numerical
        outputs.


    Paste function
    --------------
        faccoord(targfac, # [ [FacetNum,FacetArray,Anchored], e.g., [0,FacetArray0,True] => existing facet array to recalculate] ]
                 targdatindex, # [None,targdatindex, e.g., TargDatInd0 => index of valid data per targ entity]
                 data,   # [2-D targfac x oppfac array of data (rotates so that targfac is always rows)]
                 oppfac, # [Ents x dims array of coordinates of opposite facet(s), or their product]
                 oppweights,  # [array of weights corresponding to opposite facet array]
                 solve_meth = 'LstSq', # [method for calculating coordinates <'LstSq','IRLS'>]
                 solve_meth_specs = None,  # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],...}]
                 condcoord_ = None,  # [None,'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...}>,} ]
                 miss_meth = 'IgnoreCells', # ['ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells entirely (preferred)]
                 nanval = -999., # [Not-a-Number value, to label non-numerical outputs]
                 )

    """

    FacNum = targfac[0]
    FacCoord = targfac[1]
    anchored = targfac[2]
    OppCoord = oppfac
    nEnts = np.size(FacCoord,0)
    Warn1 = None
    Warn2 = None


    #####################
    ##  'IgnoreCells'  ##
    ##     method      ##
    #####################

    # 'IgnoreCells' method
    if miss_meth == 'IgnoreCells':

        # Prepare weights for 'LstSq'
        if solve_meth == 'LstSq':
            W_All = oppweights
        else:
            W_All = None

        # For each entity
        fsolve2 = solve2
        for i in xrange(nEnts):
            if (FacCoord[i][0] == nanval and not anchored
                ):
                FacCoord[i] = nanval
                Warn2 = True
            else:
                DataV = data[i]
                if targdatindex is not None:
                    U = OppCoord[targdatindex[i]]
                    x = DataV[targdatindex[i]][:,np.newaxis]
                else:
                    U = OppCoord
                    x = DataV

                if (solve_meth == 'LstSq'
                    and targdatindex is not None
                    and W_All is not None
                    ):
                    W = W_All[targdatindex[i]]
                else:
                    W = None

                invUTU_ = invUTU(U,'R',weights=W,nanval=nanval)

                # Solving as if for rows, regardless of target facet (facet is controlled outside the function)
                try:
                    v = fsolve2(R = None,   # [ents x dims array of row coordinates, no NaNVals]
                                C = U,   # [ents x dims array of col coordinates, no NaNVals]
                                x = x,   # [2-D vector or row or col coordinates, no NaNVals]
                                targfacet = 'R',   # [<'R','C'>, type of coordinates to calculate]
                                invUTU_ = invUTU_,  # [None, Output of invUTU(), U = opposing facet of 'targfacet': (UT * U)^-1]
                                weights = W, # [None, array of weights corresponding to elements in U array]
                                method = solve_meth,  # [<'LstSq','IRLS','Rasch'>]
                                meth_specs = solve_meth_specs,   # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],...}]
                                nanval = nanval,  # [Not-a-number value, for invalid outputs]
                                )
                    FacCoord[i] = np.transpose(v)

                except:
                    FacCoord[i] = nanval
                    Warn1 = True
                    pass


        # Condition the target facet
        if condcoord_ is not None:
            if FacNum == 0:
                Fac = 'Fac0'
                fac0coord = FacCoord
                fac1coord = None
                FacOut = 'F0Std'
            elif FacNum == 1:
                Fac = 'Fac1'
                fac0coord = None
                fac1coord = FacCoord
                FacOut = 'F1Std'

            CondOut = condcoord(fac0coord = fac0coord,  # [None, nRowEnts x nDims facet1 coordinates array]
                                fac1coord = fac1coord,   # [None, nColEnts x nDims Facet2 coordinates array]
                                cond_facet = Fac,    # [<'Fac0','Fac1'>]
                                function = condcoord_[Fac],    # [<None,'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...} >]
                                nanval = nanval   # [Not-a-number value]
                                )

            FacCoord = CondOut[FacOut]

        return {'Fac':FacNum,
                'FacCoord':FacCoord,
                'Warn1':Warn1,
                'Warn2':Warn2
                }

    #####################
    ##  'ImputeCells'  ##
    ##     method      ##
    #####################

    # 'ImputeCells' method
    elif miss_meth == 'ImputeCells':

        # Prepare weights for 'LstSq'
        if solve_meth == 'LstSq':
            W = oppweights
        else:
            W = None

        # Invert OppCoord
        U = OppCoord
        invUTU_ = invUTU(U,'R',weights=W,nanval=nanval)
        if not isinstance(invUTU_,np.ndarray):
            exc = "Linear algebra error.\n"
            raise faccoord_Error(exc)

        # For each entity
        fsolve2 = solve2
        for i in xrange(nEnts):
            x = data[i]

            try:
                # Solving as if for rows, regardless of target facet (facet is controlled outside the function)
                v = fsolve2(R = None,   # [ents x dims array of row coordinates, no NaNVals]
                           C = U,   # [ents x dims array of col coordinates, no NaNVals]
                           x = x,   # [2-D vector or row or col coordinates, no NaNVals]
                           targfacet = 'R',   # [<'R','C'>, type of coordinates to calculate]
                           invUTU_ = invUTU_,  # [None, Output of invUTU(), U = opposing facet of 'targfacet': (UT * U)^-1]
                           weights = W, # [None, array of weights corresponding to elements in U array]
                           method = solve_meth,  # [<'LstSq','IRLS','Rasch'>]
                           meth_specs = solve_meth_specs,   # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],...}]
                           nanval = nanval,  # [Not-a-number value, for invalid outputs]
                           )

                FacCoord[i] = np.transpose(v)

            except:
                FacCoord[i] = nanval
                Warn1 = True
                pass

        # Condition the target facet
        if condcoord_ is not None:
            if FacNum == 0:
                Fac = 'Fac0'
                fac0coord = FacCoord
                fac1coord = None
                FacOut = 'F0Std'
            elif FacNum == 1:
                Fac = 'Fac1'
                fac0coord = None
                fac1coord = FacCoord
                FacOut = 'F1Std'

            CondOut = condcoord(fac0coord = fac0coord,  # [None, nRowEnts x nDims facet1 coordinates array]
                                       fac1coord = fac1coord,   # [None, nColEnts x nDims Facet2 coordinates array]
                                       cond_facet = Fac,    # [<'Fac0','Fac1'>]
                                       function = condcoord_[Fac],    # [<None,'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...} >]
                                       nanval = nanval   # [Not-a-number value]
                                       )

            FacCoord = CondOut[FacOut]

        return {'Fac':FacNum,
                'FacCoord':FacCoord,
                'Warn1':Warn1,
                'Warn2':None
                }




###########################################################################

def get_unique_weight(targ, # [<target subspace label>]
                      targ_obj, # [<target subspace Damon object>]
                      res_obj,  # [<target subspace residuals Damon object>]
                      unique_weights,   # [<unique_weights param>]
                      min_rel = 0.02,   # [<minimum reliability>]
                      rpt_optimal = None    # [<None, True> => report optimal weight]
                      ):
    """Estimate unique weight for sub_coord().

    Returns
    -------
        Output dictionary:
            {'unique_weight':_, =>  weight actually used by sub_coord() for unique dimensions
             'optimal_weight':_ =>  optimal weight, reported separately for simulations
             }

    Comments
    --------
        The get_unique_weight() function is used only to support
        the Damon.sub_coord() method.  sub_coord() partitions the
        variance in a target subspace into three categories:

            a)  p_comm: the target variance explained by the predictor
                subspace, i.e., the variance in common between the
                predictor and target data.

            b)  p_unique: the target variance explained by those
                dimensions that are unique to the target subspace.

            c)  noise: the remaining variance explained by random
                noise.

        sub_coord() calculates separate R coordinates for the
        p_comm and p_unique dimensions.  In the absence of noise,
        the two sets of R coordinates can simply be appended to
        each other to model the target data.  However, as noise
        is added, the unique R coordinates are progressively
        degraded and should therefore be downweighted.  get_unique_weight()
        is used to estimate the downweight.

        (Note that p_comm always receives full weight.  Only p_unique
        is downweighted.)

        The downweighting formula is:

            p_uni_comm = p_unique / (p_unique + p_comm)
            unique_weight = sqrt(reliability * p_uni_comm)

        Thus, the more reliable the analysis of the unique component
        of the data, and the more pronounced the effect of the unique
        component on the data, the higher the unique weight (maximum
        of 1.0, minimum of 0.0).  As either reliability goes to zero,
        or the role of p_unique goes to zero, the unique weight goes
        to zero.

        Reliability is based on Damon-analyzing the residuals array
        obtained by subtracting out the estimates obtained by applying
        the predictor R coordinates to the target data.  It is sensitive
        to the count of data points and the amount of noise.

    Arguments
    ---------
        "targ" is the target subspace label

        -----------
        "targ_obj" is the target portion of the dataset, formatted
        as a Damon object and passed in by sub_coord()

        -----------
        "res_obj" is derived from the target residuals obtained by
        subtracting the estimates computing using predictor R coordinates
        from the observed data.  The target residuals arrays is
        formatted as a Damon object and passed in by sub_coord()

        -----------
        "unique_weights" is the same as the unique_weights parameter
        in the sub_coord() method.  Example:

            unique_weights = {'sub0':'Auto', 'sub1':0.50}

        -----------
        "min_rel" is the same as the min_rel parameter in sub_coord().
        It is the minimum below which the reliability will not go.

        -----------
        "rpt_optimal" is the same as the rpt_optimal parameter in
        sub_coord() and specifies whether to report the optimal
        parameter even when unique weights are manually specified.

    Examples
    --------

        [under construction]
        
    Paste Function
    --------------
        get_unique_weight(targ, # [<target subspace label>]
                          targ_obj, # [<target subspace Damon object>]
                          res_obj,  # [<target subspace residuals Damon object>]
                          unique_weights,   # [<unique_weights param>]
                          min_rel = 0.02,   # [<minimum reliability>]
                          rpt_optimal = None    # [<None, True> => report optimal weight]
                          )
    """

    out = {}

    if (unique_weights[targ] != 'Auto' and rpt_optimal is not True):
        out = {'unique_weight':unique_weights[targ], 'optimal_weight':None}
        
    else:
        ndim = res_obj.coord_out['ndim']
        nitem = np.size(res_obj.coredata, axis=1)

        if nitem <= ndim:

            if unique_weights[targ] == 'Auto':
                exc = 'Too few items to estimate a unique weight for subspace'+targ+'. Specify a unique weight in sub_coord().\n'
                raise get_unique_weight_Error(exc)
            else:
                out = {'unique_weight':unique_weights[targ], 'optimal_weight':None}
                return out
        
        nanval = targ_obj.nanval
        
        # Get standard errors
        res_obj.base_est()
        res_obj.base_resid()
        res_obj.base_ear()

        try:
            res_obj.base_se(obspercellmeth='PickMinFac')
        except damon1.utils.base_se_Error:
            out = {'unique_weight':unique_weights[targ], 'optimal_weight':None}
            return out
            
        # Get reliability
        rmse = rmsr(None, None, res_obj.base_se_out['coredata'], nanval)
        rel = reliability(None,
                          estimates = res_obj.base_est_out['coredata'],
                          sterr = rmse,
                          nanval = nanval
                          )
        
        rel = max(min_rel, rel)

        # Variance explained by common dimension
        p_comm = correl(targ_obj.coredata,
                        targ_obj.base_est_out['coredata'],
                        nanval
                        )**2

        # Variance explained by unique dimension
        p_unique = correl(targ_obj.coredata,
                          res_obj.base_est_out['coredata'],
                          nanval
                          )**2

        # Unique variance as proportion of non-noise variance
        p_uni_comm = p_unique / (p_unique + p_comm)

        # Calculate optimal weight
        unique_weight_ = np.sqrt(p_uni_comm * rel)

        if unique_weights[targ] != 'Auto':
            out['unique_weight'] =  unique_weights[targ]
        else:
            out['unique_weight'] = unique_weight_

        out['optimal_weight'] = unique_weight_

    return out




###########################################################################

def unbiascoord(V,   # [2-D single row or col coordinate vector to unbias, facet opposite from U]
               invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
               U,   # [2-D U matrix]
               h,   # [2-D array of h-statistics corresponding to U]
               X,   # [2-D array of observations corresponding to U and B]
               i,   # [Index of observation to be made missing]
               facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
               ):
    """Returns coordinates as if observation i were deleted.

    Returns
    -------
        Returns coordinates as if observation i were deleted.

    Comments
    --------
        The formula for the change in the coordinate caused
        by the deletion of a missing observation is:

        V - V[unbiased] = (UTU)^-1 * UT_i * Res_i / (1 - h_i)

    Paste function
    --------------
        unbiascoord(V,   # [2-D single row or col coordinate vector to unbias, facet opposite from U]
                   invUTU_,   # [Output of invUTU_():  (UT * U)^-1]
                   U,   # [2-D U matrix]
                   h,   # [2-D array of h-statistics corresponding to U]
                   X,   # [2-D array of observations corresponding to U and B]
                   i,   # [Index of observation to be made missing]
                   facet,   # ['R' => U coordinates arranged as rows x dims; 'C' => U coordinates arranged as dims x cols]
                   )

    """
    # Define variables
    if facet == 'C':
        V = np.transpose(V)
        invUTU_ = np.transpose(invUTU_)
        U = np.transpose(U)
        h = np.transpose(h)
        X = np.transpose(X)

    # Compute residuals
    E = np.dot(U,V)
    Res_i = (X - E)[i,:]

    # Get U transpose element
    UT_i = np.transpose(U)[:,i][:,np.newaxis]

    # Get h element
    h_i = h[i,:]

    # Compute change in U resulting from deleting datum i
    DeltaV = (np.dot(invUTU_,UT_i) * Res_i) / (1 - h_i)
    V_Unbiased = V - DeltaV

    # Orient solution vector
    if facet == 'R':
        V_Unbiased = V_Unbiased
    elif facet == 'C':
        V_Unbiased = np.transpose(V_Unbiased)

    return V_Unbiased




###########################################################################
def estimate(fac0coord, # [N x D array of N row coordinates in D dimensions]
             fac1coord, # [I x D array of I column coordinates in D dimensions]
             nanval,    # [float Not-a-Number Value => marks missing coordinate values]
             ):
    """Returns cell estimates based on row and column coordinates.

    Returns
    -------
        array of estimates corresponding to the dot product of
        the coordinates for facet 1 (rows) and facet 2 (columns).

    Comments
    --------
        These estimates are biased in the direction of the
        observation for small datasets.  For estimates that
        are "objective", i.e., equivalent to the estimate
        achieved by making its observation missing, use
        unbiasest() or objectify().  However, estimate()
        is the official way to compute estimates from
        coordinates.

    Arguments
    ---------
        "fac0coord" is an N row entities x D dimensions
        array of coordinates.

        ----------
        "fac1coord" is an I column entities x D dimensions
        array of coordinates.

        ----------
        "nanval" is the Not-a-Number value.


    Paste function
    --------------
        estimate(fac0coord, # [N x D array of N row coordinates in D dimensions]
                 fac1coord, # [I x D array of I column coordinates in D dimensions]
                 nanval,    # [float Not-a-Number Value => marks missing coordinate values]
                 )

    """
    # Calculate dot product (includes NaNVals in calculation)
    Est = np.dot(fac0coord,np.transpose(fac1coord))

    # Get locations of missing coordinates
    F0NaN_Loc = np.where(fac0coord == nanval)[0]
    F1NaN_Loc = np.where(fac1coord == nanval)[0]

    # Apply NaNVals to rows with missing coordinates
    Est[F0NaN_Loc] = nanval

    # Apply NaNVals to cols with missing coordinates
    EstT = np.transpose(Est)
    EstT[F1NaN_Loc] = nanval
    Est = np.transpose(EstT)

    return Est



###########################################################################

def estimate_error(err, # [<datadict> => abs residual, ratio errors]
                   err_type, # [<'ear', 'se'> => type of coordinates to get]
                   anchors # [coord() anchor parameters to use]
                   ):
    """Use 2-d Damon to estimate ratio EAR and SE statistics.
    
    Returns
    -------
        estimate_error() returns a datadict of expected EAR
        or SE statistics that also contains EAR or SE 2-d log
        coordinates for use in anchored designs.
    
    Comments
    --------
        estimate_error() is used in base_ear() to convert
        absolute residuals into expected absolute residuals (EAR),
        with coordinates.  In base_se(), it is used to obtain
        SE coordinates, which take into account the number of
        observations associated with each cell, and slightly
        refined SE cell estimates.
        
        The estimation itself is done by applying Damon.coord()
        not to the array of observations but (for the EAR calculation) 
        to the array of squared residuals: (observations - estimates)**2.
        This in principle requires a 1-dimensional analysis of ratio
        scale data (since squared residuals have a hard floor of zero).
        However, Damon's ALS algorithm, which assumes interval data and
        homoscedastic errors, has trouble with this type of data.  
        
        Therefore, estimate_error() first takes the log of the data, 
        converting it to interval data with homescedastic errors. However,
        if E = R * C, then log(E) = log(R) + log(C). Each cell datum
        is now modeled as a SUM of coordinates rather than a product. To 
        analyze with ALS we need to introduce a second dimension, imagining
        for a given cell that:
            
            r = log(R), c = log(C)
            R_ = [r, 1]
            C_ = [1, c]
            E_ = dot(R_, C_) = r*1 + 1*c = log(R) + log(C)
           
        Since the cells are now modeled as a sum of products across 
        dimensions, Damon's ALS algorithm is now applicable.  And since
        we aren't interested in the values of r and c, but just the
        dot products, there is no need for any coordinates to be fixed at
        a value of 1.  Any two dimensional solution that fits the data
        will suffice.  Thus, we analyze the log(data) with a 2-dimensional
        model, and in the absence of noise will model it exactly.
        
        To convert the estimates from the log metric back to the original
        ratio metric, we take the exponent:
            
            E = exp(E_)
        
        However, as E will not fit optimally to the original residuals 
        array, it is "refit" to the squared residuals array using numpy's
        poly1fit() function, with degree 1 or 2.  Then the square root
        is taken.
         
        What this means for Damon is that: 
            
            1)  observed residuals must first be converted to logs
            2)  the resulting coordinates will be 2-dimensional
            3)  in equate() and summstat(), after summing coordinates,
                one takes the exponent of any dot products.
        
        So much for the theory.  In practice, when doing simulations,
        the "optimal" dimensionality is often 1, depending on how
        the noise is balanced across the two facets and how various
        the expected noise is per facet.  When the expected noise
        is equivalent per cell, the analysis is more or less equivalent
        to analyzing an array of random numbers and objectivity is
        very low, regardless of dimensionality.  When there is a
        reasonable amount of noise and the noise varies across entities,
        the 2-dimensional solution tends to be optimal.  If one of the 
        facets dominates in variation of expected noise, the 1-dimensional 
        solution tends to be optimal.
        
        For consistency of approach, Damon applies the 2-dimensional
        solution in all cases.
        
    Parameters
    ----------
        "err" is a datadict of either absolute residuals, EARs, or
        SEs -- any set of uniform ratio data with a theoretical 
        minimum of zero.
        
        -------
        "err_type" signifies the type of error we're estimating.
        
            err_type = 'ear'    =>  The "err" parameter is absolute
                                    residuals and we are estimating
                                    EAR statistics.
            
            err_type = 'se'     =>  The "err" parameter is standard
                                    errors and we are calculating
                                    SE coordinates and refining the
                                    SE estimates further.
         
        -------
        "anchors" is the dictionary of parameters to use for coord()'s
        anchor parameter in anchored designs.  When the primary analysis
        refers to an item bank to get anchored coordinates for one of
        the facets, coord() assigns the anchor parameter to the Damon
        object.  Otherwise it doesn't.  To obtain the anchors parameter,
        use something like:
            
            try:
                coord_anc = self.coord_out['anchors']
            except AttributeError:
                coord_anc = None
        
            out = tools.estimate_error(abs_resid, 'ear', coord_anc)
    
    Paste Function
    --------------
        estimate_error(err, # [<datadict> => abs residual, ratio errors]
                       err_type, # [<'ear', 'se'> => type of coordinates to get]
                       anchors # [coord() anchor parameters to use]
                       )
    
    """
    # square the err
    nanval = err['nanval']
    err2 = np.where(err['coredata'] == nanval, nanval, err['coredata']**2)
    err['coredata'] = err2
       
    lab = {'ear':'ear_coord', 'se':'se_coord'}
    err_type = lab[err_type]

    if anchors is not None:
        anc = {}
        for key in anchors.keys():
            anc[key] = anchors[key]
        anc['Coord'] = err_type
    else:
        anc = None
               
    # Estimate cell variance
    err_ = dmn.core.Damon(err, 'datadict', 'RCD', verbose=False)
    
    try:
        err_.standardize('LogDat')
    except:
        pass
        
    err_.coord([[2]], seed=1000, anchors=anc)
    err_.base_est()
    
    # Refit estimates
    nanval = err_.base_est_out['nanval']
    deg = 1  # deg = 2 can lead to neg values
    res = err['coredata']
    est = np.exp(err_.base_est_out['coredata'])
    ix = (res != nanval) & (est != nanval)    
    x = np.polyfit(est[ix], res[ix], deg)
    refit = np.poly1d(x)
    y = np.clip(refit(est), 0.0001, np.inf)**0.5

    y[(est == nanval)] = nanval
    y[np.isnan(y) | np.isinf(y)] = nanval

    out = {}
    for key in err_.base_est_out.keys():
        out[key] = err_.base_est_out[key]

    out['coredata'] = np.where(y == nanval, nanval, y)      
    out[err_type] = err_.coord_out
    out['refit'] = refit

    return out



###########################################################################

def resp_prob(entcore,    # [nrows x nRespCats-1 2D array of probabilities corresponding to a given column entity]
              resp_cats,   # [list of valid response integers in increasing order less the minimum possible integer, or list of alpha responses]
              return_ = ['Exp'],     # [<[None,'Extr','Exp','Pred','Probs','ExtrSD','ExpSD','PredSD','ProbsSD']> => list desired]
              resp2extr = None,  # [<None, ['a','c'] => responses whose probability to extract>]
              extr_est = None,    # [<None,True> => extract from entcore instead of probs]
              pred_key = None,   # [<None,{'a':0,'b':1,...}> => dict relating string responses to ints]
              pred_alpha = None, # [<None,True> => report predicted alpha character]
              metric = 'nominal',   # [<'ordinal','nominal'>]
              dropcol = None,  # [<None,True> => drop lead column of ordinal probabilities]
              nanval = -999., # [float Not-a-Number value]
              ):
    """
    Returns
    -------
        A dictionary of nRow x nResp arrays relating to
        the probabilities of a set of responses for a
        given entity under an ordinal and a nominal model:

        {'Extr',     =>  Probability of resp2extr response
        'Exp',      =>  Expected value
        'Pred',     =>  Predicted value
        'Probs',    =>  Probabilities of all responses
        'ExpSD',    =>  Standard deviation of expected value
        'pred_key'   =>  pred_key parameter {'a':0,'b':1,...}
        'PredProb'  =>  Tuples with 'Pred' response and probability:
                        [('a',0.52),('b',0.34),...]
        }

        Note:  To get a list of just the responses in their
        original format, or just the probabilities, use:

        Responses = [tup[0] for tup in Out['PredProb']]
        RespProbs = [tup[1] for tup in Out['PredProb']]

        where Out is the name given the resp_prob() outputs.


    Comments
    --------
        resp_prob(), used by the fin_est() method,
        calculates the probability of obtaining each possible
        response for a given person and column entity
        and, optionally, combines these to calculate an
        expected value for that person and column entity
        or to predict the most likely response.

        Note that resp_prob() is applied only to one column entity
        at a time.

        It is assumed that estimate() has been applied to
        a dichotomous matrix formatted by parse() to reflect an
        ordinal or nominal rating structure.  When data are ordinal
        ratings (e.g., 0,1,2,3), parse() assigns each its own
        column for a given entity and puts a 1 in each column that
        corresponds to an integer that is less than or equal to
        the actual observation.  Thus, a rating of 2 for an item
        with four possible ratings (0,1,2,3) would be parsed into
        four columns as follows (1,1,1,0).  Since the first column
        is always 1 (all ratings are at least as large as 0), that
        column is deleted before running through damon().

        When data are nominal ('a','b','c'), parse() assign each
        its own column and puts a 1 in only that column that
        corresponds to the actual response.

        The base_est() method converts the ones and zeros into
        probabilities.  The problem (in the ordinal case) is these
        are not the probability of obtaining a given rating but
        the probability of meeting or EXCEEDING a given
        rating.  To compute expected values, we need to convert
        these "at or above" probabilities into just the probability
        of obtaining the given rating.  That is what resp_prob()
        does.  If the return_ argument is 'Exp', the new probabilities are
        combined with the response integers to calculate an
        expected response.

        The formula is approximately as follows:

            P[x = k] = P[x >= k] - P[x > k]
                     = [base_est() output for k] - [base_est() output for k+1]

        where x is the response and k is an integer response value.
        Because the estimated probabilities are calculated independently
        of each other, the base_est() outputs are first normalized so that
        they exceed 1.0 (though they are allowed to sum to less than 1.0).
        In addition, it is possible that, due to measurement error, the
        probability of exceeding k + 1 might be greater than the probability
        of exceeding k, which would result in a negative response probability.
        All negative probabilities are automatically set to 0.0, which is
        equivalent to saying that the category was skipped over, not used.

        In this approach, the probability of getting a zero (the lowest
        category) is one minus the sum of the remaining response
        probabilities.  It is possible, even with normalization, for
        the sum of the remaining response probabilities to be too high,
        which will cause the probability of zero to be too low.  This is
        an inescapable effect of ordinal binning, where it is not
        possible to estimate the probability of the lowest category
        except by inference.

        Also, simply forcing negative response probabilities to be
        0.0 is almost certainly not the optimal way of dealing with
        the effects of random fluctuations.  However, a better way
        has not been worked out.  In general, the more accurate the
        base_est() estimates, the less problematic these issues are.

        resp_prob() offers the option of computing standard deviations
        for the various outputs.  The formula is based on the
        binomial distribution:

            SD = sqrt(sum[k -> K](k - Exp)^2 * p[k])

        where the expected value is the sum of the categories times
        their probabilities:

            Exp = sum[k -> K](k * p[K])


    Arguments
    ---------
        "entcore" is a damon()-calculated 2D array of columns
        of (usually) probabilities corresponding to a given entity, each
        column corresponding to a possible response.  However, it
        is possible for entcore to be standardized estimates in
        another metric if return_ = ['Extr'], in which case the
        appropriate estimates, along with an index, are returned.

        When metric = 'ordinal', the minimum possible response is
        missing, as it is removed by the parse() function.  Thus, if Entity 1
        has four possible ratings or response integers (0,1,2,3),
        entcore will be the three columns of Entity 1 for the
        1, 2, and 3 response integers, and the columns will consist
        of person response probabilities obtained by running the parsed
        dichotomous array through damon().

        When metric = 'nominal', all the responses and their respective
        probabilities will be present.

        ---------------
        "resp_cats" is a either a list of possible alpha responses (metric
        = 'nominal) for a given column entity or it is a list of possible
        integer esponses in increasing order, where the smallest
        response integer is assumed to be missing and to be one unit
        less than the minimum given in the list.  This is assumed
        because, as mentioned, the smallest response integer column
        is automatically deleted by the parse() function in order to
        avoid feeding damon() a column consisting of all ones.  There
        should be as many response integers as there are entity columns
        in entcore.

        ---------------
        "return_" specifies a list of desired outputs:

            return_ = [

                'Probs' =>  return_ the probability of each response.

                'Extr'  =>  Extract the probability of the response
                            specified in resp2extr.

                'Exp'   =>  return_ the expected value.

                'Pred'  =>  return_ the predicted value in the form
                            of an integer corresponding to the response
                            with the highest probability.  Separate
                            outputs incude the actual response and
                            its probability, as well as the response:
                            integer pred_key dictionary.

                'ProbsSD'=> return_ the standard deviation of the
                            probabilities

                'ExpSD' =>  return_ the standard deviation of the
                            expected value.

                'ExtrSD'=>  return_ the standard deviation of the
                            extracted value.

                'PredSD'=>  return_ the standard deviation of the
                            largest probability.

                        ]

            return_ = ['Exp','Pred','ExpSD']
                        =>  Returns a dictionary containing arrays
                            for 'Exp' and 'Pred', as well as for the
                            standard deviation of 'Exp', the remainder
                            being None.

        ---------------
        "resp2extr" tells the function which response probabilities
        to report if return_ = 'Extr'.

            resp2extr = None    =>  'Extr' not in return_

            resp2extr = ['a','b']
                                =>  'Extr' probabilities for responses
                                    'a' and 'b' for this entity.

        ---------------
        "extr_est" <None,True> tells the function to extract the
        estimates directly from entcore rather than their
        corresponding probabilities.  This option is only
        allowed when the data are nominal.

        ---------------
        "pred_key" assigns each response to an integer, so that
        when 'Pred' is specified and the responses are alpha
        they can be treated as numbers and share an array with
        other numerical outputs.  (Numpy requires all data to be
        of one type.)  Options:

            pred_key = None  =>  'Pred' is not specified in return_.

            pred_key = {'a':0,'b':1,'f':2}
                            =>  Response 'a' will be coded as 0,
                                'b' will be coded as 1, and 'f' will
                                be coded as 2.

        A pred_key dictionary has to be filled out if 'Pred' is
        specified in Returns.  resp_cat() returns 'pred_key'.

        In additon, resp_prob() returns 'PredProb', a list of
        (response,probability) tuples:  [('a',0.53),('b',0.34),...]

        ---------------
        "metric" specifies whether the response categories are
        ordinal, treated as increasing integers, or whether they
        are to be treated as nominal -- as unorderable responses.
        Integers can be treated as nominal if desired. Options:

            'ordinal'
            'nominal'

        Note:  'ordinal' assumes that the first (lowest) possible
        category is missing from entcore and that they are sorted
        from smallest to largest in entcore.  Both conditions are
        met automatically by parse() outputs.

        ---------------
        "dropcol" <None,True>, if True, means do not report the
        first column in a block of ordinal probabilities.  This
        is to match the block of estimates produced with parsed
        ordinal data, since parse() drops the lowest response
        integer.

        ---------------
        "nanval" is the value chosen to indicate "not a number".


    Examples
    --------



    Paste function
    --------------
        resp_prob(entcore,    # [nrows x nRespCats-1 2D array of probabilities corresponding to a given column entity]
                  resp_cats,   # [list of valid response integers in increasing order less the minimum possible integer, or list of alpha responses]
                  return_ = ['Exp']     # [<['Extr','Exp','Pred','Probs','ExpSD']> => list desired]
                  resp2extr = None,  # [<None, ['a','c'] => responses whose probability to extract>]
                  extr_est = None,    # [<None,True> => extract from entcore instead of probs]
                  pred_key = None,   # [<None,{'a':0,'b':1,...}> => dict relating string responses to ints]
                  metric = 'nominal',   # [<'ordinal','nominal'>]
                  dropcol = None,  # [<None,True> => drop lead column of ordinal probabilities]
                  nanval = -999., # [float Not-a-Number value]
                  )

    """

    # Initialize Results dictionary
    Results = {}
    nrows = np.size(entcore,axis=0)
    Results['pred_key'] = pred_key
    Results['PredProb'] = None
    NaNLoc = np.where(entcore == nanval)[0]
    resp_cats = np.array(resp_cats)
    entcore = np.where(np.logical_or(np.isnan(entcore),np.isinf(entcore)),nanval,entcore)

    # Cast resp_cats
    try:
        resp_cats = resp_cats.astype(float).astype(int)
    except ValueError:
        if metric == 'ordinal':
            print "Warning: tools.resp_prob() unable to cast response categories to integers.  Treating metric as 'nominal' and returning predictions.\n"
            metric = 'nominal'
            return_ = ['Pred' if x == 'Exp' else x for x in return_]
            return_ = ['PredSD' if x == 'ExpSD' else x for x in return_]
        else:
            pass

    # ordinal procedures
    if metric == 'ordinal':

        # Normalize entcore
        RespCats_ = list(resp_cats)
        ncols = len(resp_cats) + 1
        LeadCol = np.ones((nrows,1))
        TermCol = np.zeros((nrows,1))
        FillEntCols = np.concatenate((LeadCol,entcore,TermCol),axis=1)
        RespCats_.insert(0,RespCats_[0] - 1)
        RespCatsArr = np.array(RespCats_,ndmin=2)
        extr_est = None

        # Calculate probabilities
        Probx = np.zeros((nrows,ncols))
        for i in xrange(1,ncols + 1):
            Probx[:,-i] = FillEntCols[:,-i - 1] - FillEntCols[:,-i]

        # Apply NaNVals
        Probx = np.clip(Probx,0.0,1.0)
        Probx[NaNLoc] = nanval

    # nominal procedures
    elif metric == 'nominal':
        RespCatsArr = np.array(resp_cats,ndmin=2)
        Probx = entcore

        # Force probs to sum to 1.0
        Sums = np.sum(Probx,axis=1)[:,np.newaxis]
        Probx = Probx / Sums
        Probx[NaNLoc] = nanval

    # Get Probs
    if 'Probs' in return_:
        if dropcol is True:
            Results['Probs'] = Probx[:,1:]
        else:
            Results['Probs'] = Probx
    else:
        Results['Probs'] = None

    # Get Extr
    if 'Extr' in return_:
        RespLocs = []
        for Resp in resp2extr:
            if Resp not in resp_cats:
                exc = 'Unable to find resp2extr value among response categories.\n'
                raise resp_prob_Error(exc)
            else:
                Loc = np.where(RespCatsArr == Resp)[1][0]
                RespLocs.append(Loc)

        if extr_est is True:
            Results['Extr'] = entcore[:,RespLocs]
        else:
            Results['Extr'] = Probx[:,RespLocs]

        Results['ExtrIndex'] = RespLocs
    else:
        Results['Extr'] = None
        Results['ExtrIndex'] = None

    # Get Exp
    if 'Exp' in return_:
        if metric == 'nominal':
            print 'Warning: Expected values cannot be calculated from nominal data. Reporting predicted value instead.\n'
            Results['Exp'] = None
        else:
            Exp = np.sum(RespCatsArr * Probx,axis=1)[:,np.newaxis]
            Exp[NaNLoc] = nanval
            Results['Exp'] = Exp

    else:
        Results['Exp'] = None

    # Get Pred
    if ('Pred' in return_
        or (metric == 'nominal'
            and 'Exp' in return_
            )
        ):
        PredIndex = [np.where(Row == np.amax(Row))
                     for Row in Probx]

        Pred0 = [resp_cats[PredIndex[i]][0]
                           for i in range(nrows)]

        # Convert alpha resp to ints for reporting
        if pred_key is None:
                ints = range(len(resp_cats))
                pred_key = dict(zip(resp_cats,ints))

        try:
            Pred1 = np.array([pred_key[Pred0[i]] for i in range(len(Pred0))])[:,np.newaxis]
        except KeyError:
            Pred1 = np.array([pred_key[str(Pred0[i])] for i in range(len(Pred0))])[:,np.newaxis]

        Pred1[NaNLoc] = nanval
        Results['Pred'] = Pred1

        # Get prediction probabilities and store in tuples
        MaxProb = np.array([Probx[i][PredIndex[i]][0]
                   for i in range(nrows)])[:,np.newaxis]
        MaxProb[NaNLoc] = nanval
        PredProb = [(Pred0[Row],MaxProb[Row][0]) for Row in range(nrows)]
        Results['PredProb'] = PredProb

        for i in NaNLoc:
            PredIndex[i] = nanval

        Results['PredIndex'] = PredIndex

    else:
        Results['Pred'] = None
        Results['PredProb'] = None


    #####################
    # Define function to get SD from probabilities
    def getsd(p):
        q = 1.0 - p
        SD = np.sqrt(p * q)
        SD[NaNLoc] = nanval
        return SD

    # Get ProbsSD
    if 'ProbsSD' in return_:
        try:
            p = Results['Probs']
        except KeyError:
            exc = "Must specify 'Probs' if specifying 'ProbsSD'.\n"
            raise resp_prob_Error(exc)

        ProbsSD = getsd(p)

        if dropcol is True:
            Results['ProbsSD'] = ProbsSD[:,1:]
        else:
            Results['ProbsSD'] = ProbsSD
    else:
        Results['ProbsSD'] = None

    # Get ExtrSD
    if 'ExtrSD' in return_:
        try:
            p = Results['Extr']
        except KeyError:
            exc = "Must specify 'Extr' if specifying 'ExtrSD'.\n"
            raise resp_prob_Error(exc)

        ExtrSD = getsd(p)
        Results['ExtrSD'] = ExtrSD
    else:
        Results['ExtrSD'] = None

    # Get ExpSD (formula is fancier)
    if 'ExpSD' in return_:
        try:
            Exp = Results['Exp']
        except KeyError:
            exc = "Must specify 'Exp' if specifying 'ExpSD'.\n"
            raise resp_prob_Error(exc)
        ExpSD = np.sqrt(np.sum(Probx * (RespCatsArr - Exp)**2,axis=1))[:,np.newaxis]
        Results['ExpSD'] = ExpSD
    else:
        Results['ExpSD'] = None

    # Get PredSD
    if 'PredSD' in return_:
        try:
            p = MaxProb
        except KeyError:
            exc = "Must specify 'Pred' if specifying 'PredSD'.\n"
            raise resp_prob_Error(exc)

        PredSD = getsd(p)
        Results['PredSD'] = PredSD
    else:
        Results['PredSD'] = None

    return Results





###########################################################################

def residuals(observed, # [2D array of observed values]
              estimates,    # [2D array of cell estimates]
              psmsindex = None, # [<None, where()-style index of cells made pseudo-missing>]
              nearest_val = None,     # [<None,'Nearest','Match'> => first convert estimate to nearest valid observed value]
              ecut = None,  # [<None, [['All',ECut], ['Cols',[ECut1,ECut2,'Med',...]]> ]
              nanval = -999.,   # [Not-a-Number value for cells missing in observed]
              ):
    """Get residuals between observed values and cell estimates.

    Returns
    -------
        Numpy array of differences (observed - Estimated) between
        an array of observations and a corresponding array of
        cell estimates.  Missing values are assigned nanval.

    Arguments
    ---------
        "observed" is a 2D RowEnt x ColEnt array of observations.
        Missing values are marked by nanval.

        ---------------
        "estimates" is a 2D RowEnt x ColEnt array of cell estimates
        corresponding to the observed values, generally the output
        of running Damon's coord() and base_est methods.  Missing
        values (which should be rare), are marked by nanval.

        ---------------
        "psmsindex" is an index of cells made missing when
        calculating estimates but for which there exist valid values
        in the observed array.  The index is formatted like the output of
        Numpy's where() function and is created using the
        Damon.pseudomiss() method.

        ---------------
        "nearest_val" instructs the function to "round" or convert each
        estimate to the nearest valid value in observed before
        calculating differences.  When nearest_val = 'Nearest', the
        function first checks whether the observed values are continuous or
        discrete; if continuous, the column is left alone.  Options:

            nearest_val = None   =>  Do not adjust estimates; use
                                    them as they are.  Treated as
                                    None if ecut = None.

            nearest_val = 'Nearest'
                                =>  Adjust all estimates to their
                                    nearest observed value.

            nearest_val = 'Match'
                                =>  return_ 1 where observed and
                                    estimates are identical, 0
                                    otherwise.  Works on string
                                    alpha arrays.

        nearest_val can be used in place of the ecut argument, and
        may be a better idea when much of the data is not dichotomous.
        When the data are dichotomous, nearest_val is equivalent to
        ecut = ['All',0.5].  However, nearest_val is also much slower
        than ecut, and with large datasets ecut and nearest_val should,
        when the residuals are used to compute EARs, yield similar results,
        though this requires further investigation.

        ---------------
        "ecut", used when converting estimates to probabilities or
        logits, is the cut-point assigned to each column estimate (or the
        whole array) to distinguish "success" from "failure".  If
        an estimate is 3.6 and the ecut is 3.0, then the estimate
        signifies "success".  When ecut is signified, residuals are
        defined dichotomously regardless of the underlying metric.
        If a cell observation and its estimate fall on the same side
        of the cut-point, the residual is 0.  If they fall on opposite
        sides, the residual is 1.  Thus, the resulting residuals are
        only 0 and 1 -- no -1 -- and in this sense are not so much
        residuals as absolute residuals.

        Options:

            ecut = None         =>  Do not compute dichotomous
                                    residuals.  The nearest_val parameter
                                    will be used instead.

            ecut = ['All',0.50]
                                =>  Assign the same ecut to all
                                    columns.  That ecut is 0.50.

            ecut = ['All','Med']
                                =>  Assign the same ecut to all
                                    columns.  That ecut is the median
                                    of the whole array

            ecut = ['Cols',[0.50,2.5,'Med',...]]
                                =>  Assign the estimates in the first
                                    column an ecut of 0.50.  Assign
                                    the second column a 2.5.  Assign the
                                    third an ecut equal to the median
                                    estimate for that column.

            ecut = ['Cols','Med']
                                =>  Assign a different ecut to each
                                    column, where ecut is the median
                                    of that column.

        ---------------
        "nanval" is the float or integer value designated to mean
        a missing cell, or Not-a-Number value.



    Examples
    --------



    Paste function
    --------------
        residuals(observed, # [2D array of observed values]
                  estimates,    # [2D array of cell estimates]
                  psmsindex = None, # [<None, where()-style index of cells made pseudo-missing>]
                  nearest_val = None,     # [<None,'Nearest','Match'> => first convert estimate to nearest valid observed value]
                  ecut = None,  # [<None, [['All',ecut], ['Cols',[ECut1,ECut2,'Med',...]]> ]
                  nanval = -999.,   # [Not-a-Number value for cells missing in observed]
                  )

    """

    # Some variables
    shape = np.shape(observed)
    nrows = shape[0]
    ncols = shape[1]
    MsEst = np.where(estimates == nanval)
    observed = observed[:,:]
    estimates = estimates[:,:]

    # Check type of data
    try:
        observed[0,0] - estimates[0,0]
    except TypeError:
        try:
            observed = observed.astype(float)
            estimates = observed.astype(float)
        except ValueError:
            if nearest_val == 'Match':
                observed = observed.astype('S20')
                estimates = observed.astype('S20')
            else:
                exc = "Unable to get observed and estimates data types to match.\n"
                raise residuals_Error(exc)

    if (isinstance(observed[0,0],str)
        and isinstance(estimates[0,0],float)
        ):
        exc = 'type_ of data in observed and estimates does not match.\n'
        raise residuals_Error(exc)

    if (isinstance(observed[0,0],str)
        and isinstance(estimates[0,0],str)
        and nearest_val != 'Match'
        ):
        #print "Warning in tools.residuals:  If data is string, nearest_val must be 'Match'.  Making that change.\n"
        nearest_val = 'Match'

    # Get dichotomous residuals
    if ecut is not None:
        if ecut[0] == 'All':

            # Define ecut as median
            if ecut[1] == 'Med':
                ValEst = estimates[np.where(estimates != nanval)]
                ECut_ = np.median(ValEst)
            else:
                ECut_ = ecut[1]

            # Get dichotomous residuals
            Cond1 = observed == nanval
            Cond2 = estimates == nanval
            Cond3 = np.logical_and(observed < ECut_,estimates < ECut_)
            Cond4 = np.logical_and(observed >= ECut_,estimates >= ECut_)
            Resid = np.where(np.logical_or(Cond1,Cond2),nanval,
                             np.where(np.logical_or(Cond3,Cond4),0,1)
                             )

        elif ecut[0] == 'Cols':
            Resid = np.zeros((nrows,ncols))
            for i in range(ncols):

                # Define ecut as median
                if (ecut[1] == 'Med'
                    or (type(ecut[1]) is type([])
                        and ecut[1][i] == 'Med')
                    ):
                    ValEst = estimates[:,i][np.where(estimates[:,i] != nanval)]
                    ECut_ = np.median(ValEst)

                elif type(ecut[1]) is type([]):
                    ECut_ = ecut[1][i]
                else:
                    ECut_ = ecut[1]

                # Get dichotomous residuals
                Cond1 = observed[:,i] == nanval
                Cond2 = estimates[:,i] == nanval
                Cond3 = np.logical_and(observed[:,i] < ECut_,estimates[:,i] < ECut_)
                Cond4 = np.logical_and(observed[:,i] >= ECut_,estimates[:,i] >= ECut_)
                Resid[:,i] = np.where(np.logical_or(Cond1,Cond2),nanval,
                                      np.where(np.logical_or(Cond3,Cond4),0,1)
                                      )

    # Get residuals to "adjusted" estimates
    elif nearest_val == 'Nearest':

        # Define nearest(Vals,Est)
        def nearest(Vals,Est):
            HiLoc = np.searchsorted(Vals,Est)
            LoLoc = HiLoc - 1
            if LoLoc < 0:
                LoLoc = 0

            try:
                HiDiff = np.abs(Vals[HiLoc] - Est)
            except IndexError:
                HiDiff = np.inf

            LoDiff = np.abs(Vals[LoLoc] - Est)

            if HiDiff < LoDiff:
                Nearest = Vals[HiLoc]
            else:
                Nearest = Vals[LoLoc]

            return Nearest

        # Draw random sample for testing discreteness
        if nrows <= 20:
            SampLoc = None
        else:
            p = 20/float(nrows)
            SampLoc = np.where(npr.rand(nrows) < p)

        # Find nearest valid observed value per estimate
        Est1 = np.zeros((nrows,ncols))
        for i in range(ncols):
            if SampLoc is not None:
                nSamp = len(SampLoc[0])
                nVals = len(np.unique(observed[SampLoc][:,i][np.where(observed[SampLoc][:,i] != nanval)]))
                pUnique = nVals / float(nSamp)

                # Skip cols where more than 10% of data is unique
                if pUnique > 0.10:
                    Est1[:,i] = estimates[:,i]
                else:
                    Vals = np.unique(observed[:,i][np.where(observed[:,i] != nanval)])
                    try:
                        Est1[:,i] = [nearest(Vals,Est) for Est in estimates[:,i]]
                    except IndexError:
                        Est1[:,i] = nanval

            elif SampLoc is None:
                Vals = np.unique(observed[:,i][np.where(observed[:,i] != nanval)])
                try:
                    Est1[:,i] = [nearest(Vals,Est) for Est in estimates[:,i]]
                except IndexError:
                    Est1[:,i] = nanval

        Est1[MsEst] = nanval

        # Calculate residuals
        Cond1 = observed == nanval
        Cond2 = Est1 == nanval
        Resid = np.where(np.logical_or(Cond1,Cond2),nanval,
                         np.subtract(observed,Est1))

    # Look for exact match
    elif nearest_val == 'Match':
        Cond1 = observed == nanval
        Cond2 = estimates == nanval
        Resid = np.where(np.logical_or(Cond1,Cond2),nanval,
                         np.where(observed == estimates,0,1)).astype(float)

    # Get residuals for unadjusted estimates
    elif nearest_val is None:
        Cond1 = observed == nanval
        Cond2 = estimates == nanval
        Resid = np.where(np.logical_or(Cond1,Cond2),nanval,
                         np.subtract(observed,estimates)
                         )
    else:
        exc = 'Unable to figure out nearest_val parameter.\n'
        raise residuals_Error(exc)

    ########
    # Select only non-pseudo-missing residuals
    # Pseudo-missing residuals
    if psmsindex is not None:
        Temp = np.zeros((nrows,ncols)) + nanval
        Temp[psmsindex] = Resid[psmsindex]
        Resid = Temp

    return Resid



###########################################################################

def obspercell(obs, # [<None, obs array> ]
               by_rows = 'obs',   # [<'obs', ncols> => for obs counts by row]
               by_cols = 'obs',  # [<'obs', nrows>] => for obs counts by col]
               out_as = 'arr', # [<'arr', 'row', 'col', 'num'> => output shape]
               ndim = 2,    # [<int> dimensionality of estimates]
               facs_per_ent = [1, 2],    # [<[<0,1>,<1,2,[1,1,2,...]]> => facet and number of unanchored facets per entity]
               count_chars = True,   # [<bool, int, [ints]> => count n unique characters per cell]
               max_chars = 2, # [<int>, maximum chars per cell]
               p_items = None, # [<None, float] => percent items independent]
               meth = 'CombineFacs', # [<'PickMinFac','CombineFacs'>]
               nanval = -999,   # [Not-a-Number value]
               ):
    """Calculate obspercell_factor for standard errors.

    Returns
    -------
        2-D array (corresponding to "shape") with the obspercell_factor
        calculated for each cell.  This is used in Damon's standard
        error formula.

    Comments
    --------
        The core standard error formula in Damon is based on the
        classic statistical formula:

        Formula 1a (regular formula):

            n = number of observations
            SE = SD/sqrt(n)

        In Damon the 1/sqrt(n) portion of the formula is refined to take 
        into account the dimensionality of the estimates and the number of
        facets.  The number of observations is by cell -- for each cell, how
        many observations reside in its row and column.

        Formula 1b (Damon version):

            Basic Idea:  SE[cell] = nfacs * SD / (n - ndims)/ndims
                                  = nfacs * SD / (n/ndims - 1)

        This gets translated into one of two formulae (as expressed in 
        the dichotomous case):

        Formula 1b.1 ('PickMinFac' formula)

            SE[cell] = (nfacs * EAR) / minimum(sqrt(nrows / ndims - 1),
                                               sqrt(ncols / ndims - 1))

        Formula 1b.2 ('CombineFacs' formula)

            SE[cell] = (nfacs * EAR) / sqrt(sqrt(nrows / ndims - 1) * 
                                            sqrt(ncols / ndims - 1))

        The obspercell() method calculates the factor that, when multiplied
        by the EAR (or p * (1 - p) in the case of logits), returns the
        standard error, e.g.:
        
            obspercell_factor = nfacs / sqrt(sqrt(nrows / ndims - 1) * 
                                            sqrt(ncols / ndims - 1))
            
            SE = obspercell_factor * EAR
        
        With polytomous data, the nrows and ncols variables are replaced by 
        variables that take into account the number of rating scale "steps" 
        per cell (each additional step is like an additional observation):
        
            nrows becomes nrows * steps_per_cell across each column
            ncols becomes ncols * steps_per_cell across each row
        
        ncols has the additional complication that there may be multiple
        possible steps per column if items have different rating scales.
        
        Counts for continuous scales are indeterminate.  In principle, they
        contain an infinite number of steps; in practice, due to measurement
        error, this number may be quite small and currently requires human 
        judgement.  The max_chars character is used to set an upper bound 
        reflecting the maximum number of "statistically distinct values" 
        that a given response scale can support.
        
        Thus Damon replaces the familiar (1 / sqrt(n)) factor used in
        traditional standard error formulas (multiplied by SD) with an 
        elaborated factor that takes into account the number of facets, 
        dimensions, rows, columns, and rating scale steps.  This
        'obspercell_factor' is specific to NOUS.  
        
        'PickMinFac' or 'CombineFacs'
        -----------------------------
        'PickMinFac' tends to be more accurate than 'CombineFacs' with
        smaller datasets where the ratio of observations to dimensions
        is less than, say, 10.  However, with large datasets it will
        tend to overestimate the standard error since it ignores
        information contributed by one of the facets.

        'CombineFacs' is more accurate with large data sets.  However, as the
        ratio of observations to dimensions drops below, say, 10, it will tend
        to a standard error that is too small.  The reason is that
        in this situation (as the ratio of observations to dimensions
        becomes small) Damon estimates become biased toward the
        observations, resulting in an artificially low EAR and SE.
        'CombineFacs' is probably the most mathematically correct, but
        'PickMinFac' may be safer if you want to avoid false
        positive significance tests with smaller datasets.
        
        Type of Output
        --------------
        If the observed data array contains missing values, the 
        obspercell_factor and resulting standard error will vary across
        cells.  In this case, the obspercell_factor is ideally an array
        of values which gets multiplied elementwise by the EAR.  However, if
        there are no missing values, obspercell_factor reduces to a single
        number since each cell is associated with the same number of data
        values.  Sometimes, it is convenient to calculate just the single
        number, even if there are some missing values, e.g., when the relevant
        observed array is not available.  This behavior is controlled by the
        "out_as" parameter, which also supports scenarios where you know
        the number of missing cells for one facet but need to assume a
        complete array for the opposing facet.
        
        Parsed Data
        -----------
        When using Damon's parse() function (you probably shouldn't),
        the "obs" parameter can refer to self.parse_out['coredata'].
        Since, this (probably) causes the number of independent items to be 
        overestimated, you may want to adjust the "p_items" parameter
        downward to equal (ncols_orig_obs / ncols_parsed).  In general,
        the statistical implications of parsing data are poorly understood,
        so use parse() with caution.
               
    Arguments
    ---------
        "obs" <None, array>, is the pre-estimate data used to calculate counts 
        of observations and to determine unique valid characters in each 
        row and/or column, subject to the maximum given in max_chars.  It is
        also used to determine which cells are missing and how many valid
        data values are brought to bear on any given cell.
        In anchored designs, it is the data array for the current Damon
        object. The size of the historical data array used to calculate the 
        anchors must also be taken into account, and this is handled through
        the by_rows and by_cols parameters.  
             
            obs = observations      =>  An array of observations
            
            obs = None              =>  No observations are available. Specify
                                        array dimensions in by_rows and by_cols.
            
        ----------
        "by_rows" specifies how to get the number of columns per row and can
        either be set explicitly (e.g., for anchored designs) or by referring
        to the "obs" array (if it exists).
        
            by_rows = 'obs'         =>  Get the number of columns from the
                                        observed array.
            
            by_rows = 100           =>  Each row is associated with 100
                                        columns.

        ----------
        "by_cols" specifies how to get the number of rows per column and can
        either be set explicitly (e.g., for anchored designs) or by referring
        to the "obs" array (if it exists).
        
            by_cols = 'obs'         =>  Get the number of rows from the
                                        observed array.
            
            by_cols = 1000          =>  Each column is associated with 1000
                                        rows.

        ----------
        "out_as" specifies how to output the obspercell_factor, whether as
        a single number or as a type of array.  The selection is based on
        the availability of the observed data array and the prevalence of
        missing data.
        
            out_as = 'num'          =>  Return obspercell_factor as single
                                        number (which can then be multiplied
                                        by the EAR array to obtain the SE
                                        array). This yields valid results to
                                        the degree the observed array is
                                        complete but can be calculated 
                                        regardless.
        
            out_as = 'arr'          =>  Return as a 2-D array whose shape is
                                        controlled using by_rows and by_cols and
                                        which presumably matches the shape of
                                        the EAR array.
            
            out_as = 'row'          =>  Return a 1-D array whose length
                                        corresponds to the row length (i.e.,
                                        number of columns).  This assumes
                                        variation in missing per column but
                                        not by row.  This option is not likely
                                        to be used much.
            
            out_as = 'col'          =>  Return a 1-D array whose length 
                                        corresponds to the column length (i.e,
                                        number of rows). This assumes variation
                                        in missing per row but not by column.
                                        This is the default in equate()'s
                                        construct measures.
        
        ----------
        "ndim" is an integer describing the dimensionality used
        to compute the Damon estimates.  It can be found at:

            my_obj.coord_out['ndim']

        ----------
        "facs_per_ent" is used to specify the number of "unanchored"
        facets used in the calculation of either row entities or column
        entities, for use in the standard error formula.

        Syntax:  [facet, n unanchored facets per entity]

            facs_per_ent = [1,2]
                            =>  For facet 1 (columns), all items
                                were calculated with both the row
                                and column entities unanchored.

            facs_per_ent = [1,1]
                            =>  For facet 1 (columns), all items
                                were calculated with row coordinates
                                anchored.  Only one set of coordinates
                                (columns) was unanchored.

            facs_per_ent = [0,1]
                            =>  For facet 0 (rows), all persons were
                                calculated with column coordinates
                                anchored.  Only one set of coordinates
                                (rows) was unanchored.

            facs_per_ent = [1,[2,2,2,1]]
                            =>  For facet 1 (columns), there are four
                                items.  The first three were calculated
                                with two sets of unanchored coordinates
                                (rows and columns).  The fourth item
                                was calculated with only one set
                                of unanchored coordinates (columns).


        ----------
        "count_chars" <bool, int, [ints]> tells how to count the
        number of observed values that are possible per cell, up to 
        the number given by max_chars.
        
            count_chars = False     =>  The data are assumed dichotomous (two
                                        characters per cell)
            
            count_chars = True      =>  Get the character counts by looking at 
                                        a sample of rows (the first 100) in
                                        the "obs" array.  This assumes obs
                                        is an array.
            
            count_chars = 3         =>  All cells (for each column) have three
                                        characters per cell.
            
            count_chars = [4, 2, 3] =>  In a three column array, the first 
                                        column has four characters per cell,
                                        the second two, the third three.
                                        If data are continuous, specify a high
                                        integer; "max_chars" will clip it down
                                        to size.
                                       
        ----------
        "max_chars" <int> is the maximum number of valid characters per
        cell and sets an upper boundary on count_chars.  It was created
        to deal with continuous data for which the number of valid characters 
        is in principle infinite.  In practice, though, it seems like
        max_chars = 2 is what works best for all data types.
        
            max_chars = 2           =>  The number of valid characters is 
                                        capped at 2 for all item types.
            
        ----------
        "p_items <None, float> makes it possible to override the
        count of columns * steps in estimating the number of "independent 
        items".  Damon's standard error statistic assumes that each item is 
        statistically independent, but sometimes there are dependencies 
        between items that violate this assumption, causing the standard 
        errors to be too low. "p_items" offers a manual override.
        
            p_items = None (default)
                            =>  Get the item count from the number of
                                columns in "obs" * the number of steps
            
            p_items = 0.75
                            =>  Force the item count to be 
                                0.75 * n columns * n steps
            
        ----------
        "meth" specifies which procedure to use, 'PickMinFac' or
        'CombineFacs'.  See comments above.

        ----------
        "nanval" is the not-a-number value.
        
    Examples
    --------


    Paste function
    --------------
        obspercell(obs, # [<None, obs array> ]
                   by_rows = 'obs',   # [<'obs', ncols> => for obs counts by row]
                   by_cols = 'obs',  # [<'obs', nrows>] => for obs counts by col]
                   out_as = 'arr', # [<'arr', 'row', 'col', 'num'> => output shape]
                   ndim = 2,    # [<int> dimensionality of estimates]
                   facs_per_ent = [1, 2],    # [<[<0,1>,<1,2,[1,1,2,...]]> => facet and number of unanchored facets per entity]
                   count_chars = True,   # [<bool, int, [ints]> => count n unique characters per cell]
                   max_chars = 2, # [<int>, maximum chars per cell]
                   p_items = None, # [<None, float] => percent items independent]
                   meth = 'CombineFacs', # [<'PickMinFac','CombineFacs'>]
                   nanval = -999,   # [Not-a-Number value]
                   )

    """

    def col_steps(obs, max_rows=100, nanval=-999):
        "Get array of possible steps by col."
        
        samp_rows = min(np.size(obs, axis=0), max_rows)
        samp = obs[:samp_rows, :]
        ncols = np.size(obs, axis=1)
        n_steps = []
        
        for i in range(ncols):
            col = samp[:, i]
            col = col[col != nanval]
            n_unique = min(max(2, len(np.unique(col))), max_chars)
            n_steps.append(n_unique - 1)
        
        return np.array(n_steps)
    
    nanvalf = float(nanval)

    # Figure out if facets are anchored.  TODO: Not all cases thought through.
    anc = None
    if facs_per_ent == [0, 1]:
        anc = 1
    elif facs_per_ent == [1, 1]:
        anc = 0
    
    # Sort out number of cols
    if by_rows == 'obs':
        if obs is not None:
            ncols = np.size(obs, axis=1)
        else:
            exc = 'obs cannot be None if by_rows is "obs".'
            raise obspercell_Error(exc)
    else:
        ncols = by_rows
    
    # Sort out number of rows
    if by_cols == 'obs':
        if obs is not None:
            nrows = np.size(obs, axis=0)
        else:
            exc = 'obs cannot be None if by_cols is "obs".'
            raise obspercell_Error(exc)
    else:
        nrows = by_cols
        
    # Percentage of item/steps to treat as independent
    if p_items is None:
        p_items = 1.0

    # count_chars to array
    if isinstance(count_chars, list):
        count_chars = np.array(count_chars)
        
    # Define facs_per_ent 
    # TODO: the interaction of facets with error is still not clear, e.g.,
    # should cells in an anchored design be 1 or 2, or something in between?
    # sub_coord() seems to need entity level facets, but that's unclear.
    # Here, if there is anchoring, nfacs is 1, otherwise 2.
    
    if isinstance(facs_per_ent[1], np.ndarray):
        if 2 in facs_per_ent[1]:
            nfacs = 2
        else:
            nfacs = 1
    else:
        nfacs = facs_per_ent[1]
            
    # Get row_count, col_count as integers, assumes common rating scale
    # row_count => based on n rows; col_count => based on n cols
    if out_as == 'num':
        if count_chars is True:
            if obs is not None:
                c_steps = col_steps(obs, 500, nanval)
                col_count = np.sum(c_steps)
                row_count = np.mean(c_steps) * np.size(obs, axis=0)
            else:
                exc = 'obs cannot be None if count_chars is True.'
                raise obspercell_Error(exc)
        
        # Assume dichotomous -- one step
        elif count_chars is False:
            col_count = ncols  # * 1  
            row_count = nrows # * 1
        
        # Use specified int character count
        elif isinstance(count_chars, int):
            col_count = ncols * min(count_chars - 1, max_chars - 1)
            row_count = nrows * min(count_chars - 1, max_chars - 1)
        
        # Use specified array of character counts
        elif isinstance(count_chars, np.ndarray):
            col_count = np.sum(count_chars - 1)
            row_count = nrows * np.mean(count_chars - 1)
            
    # Get row_count and col_count as arrays
    elif out_as in ['row', 'col', 'arr']:
        
        # Build counts array assuming no missing
        counts_ = np.zeros(np.shape(obs))
        if count_chars is True:
            c_steps = col_steps(obs, 500, nanval)
            counts = counts_ + c_steps
        elif count_chars is False:
            counts = counts_ + 1
        elif isinstance(count_chars, int):
            counts = counts_ + min(count_chars, max_chars)
        elif isinstance(count_chars, np.ndarray):
            counts = counts_ + count_chars
            counts = np.clip(counts, 0, max_chars)

        # Deal with missing where desired
        counts_no_nan = counts
        counts_ma = npma.masked_array(counts, mask=(obs == nanval), 
                                      fill_value=nanvalf)       
        
        if out_as == 'arr':
            col_count = np.sum(counts_ma, axis=1).filled(nanvalf)
            row_count = np.sum(counts_ma, axis=0).filled(nanvalf)            
        elif out_as == 'col':
            col_count = np.sum(counts_ma, axis=1).filled(nanvalf)
            row_count = np.sum(counts_no_nan, axis=0)
        elif out_as == 'row':
            col_count = np.sum(counts_no_nan, axis=1)
            row_count = np.sum(counts_ma, axis=0).filled(nanvalf)
            
    # Calculate obspercell_factor as int    
    if out_as == 'num':
        if meth == 'PickMinFac':
            if nrows > ncols:
                denom = np.sqrt(col_count * p_items / ndim - 1)
            else:
                denom = np.sqrt(row_count / ndim - 1) if anc != 1 else 1        
            opc_fact = nfacs[0] / denom
    
        elif meth == 'CombineFacs':
            dnm_0 = np.sqrt(row_count / ndim - 1) if anc != 1 else 1
            dnm_1 = np.sqrt(col_count * p_items / ndim - 1) if anc != 0 else 1
            opc_fact = nfacs[0] / np.sqrt(dnm_0 * dnm_1)

    # Calculate obspercell_factor as array                  
    elif out_as == 'arr':
        one_0 = np.repeat(1.0, ncols)
        dnm_0 = np.sqrt((row_count / ndim) - 1) if anc != 1 else one_0
        one_1 = np.repeat(1.0, nrows)
        dnm_1 = np.sqrt((col_count * p_items / ndim) - 1) if anc != 0 else one_1
            
        row_fact = (1 / dnm_0)[np.newaxis, :]
        col_fact = (1 / dnm_1)[:, np.newaxis]

        if meth == 'CombineFacs':
            opc_fact = nfacs * np.sqrt(row_fact * col_fact)
                
        elif meth == 'PickMinFac':
            if nrows > ncols:
                row_fact = np.ones((1, ncols))
                opc_fact = nfacs * (row_fact * col_fact)
            else:
                col_fact = np.ones((nrows, 1))
                opc_fact = nfacs * (row_fact * col_fact)

        # Clean up bad cells
        opc_fact[:, row_count == nanvalf] = nanvalf
        opc_fact[col_count == nanvalf, :] = nanvalf
    
    elif out_as == 'col':
        one_1 = np.repeat(1.0, nrows)
        col_fact = 1 / np.sqrt(col_count * p_items / ndim - 1) if anc != 0 else one_1
        
        row_count_mean = np.mean(row_count)
        row_fact = 1 / np.sqrt(row_count_mean / ndim - 1) if anc != 1 else 1
        
        # facet count:  index error triggers choice.  Hacky and wrong.
        if meth == 'CombineFacs':
            opc_fact = nfacs * np.sqrt(row_fact * col_fact)
        elif meth == 'PickMinFac':
            if nrows > ncols:
                try:
                    opc_fact = nfacs * col_fact
                except ValueError:
                    opc_fact = nfacs[0] * col_fact
            else:
                try:
                    opc_fact = nfacs * row_fact 
                except ValueError:
                    opc_fact = nfacs[0] * row_fact
        
        # Clean
        opc_fact[col_count == nanvalf] = nanvalf
        
    elif out_as == 'row':
        one_0 = np.repeat(1.0, ncols)
        row_fact = 1 / np.sqrt(row_count / ndim - 1) if anc != 1 else one_0
        
        col_count_mean = np.mean(col_count)
        col_fact = 1 / np.sqrt(col_count_mean / ndim - 1) if anc != 0 else 1
        
        if meth == 'CombineFacs':
            opc_fact = nfacs * np.sqrt(row_fact * col_fact)
        elif meth == 'PickMinFac':
            if nrows > ncols:
                opc_fact = nfacs * col_fact
            else:
                opc_fact = nfacs * row_fact
        
        # Clean
        opc_fact[row_count == nanvalf] = nanvalf
    
    # Final cleaning
    opc_fact = np.where(np.isnan(opc_fact) | np.isinf(opc_fact),
                        nanvalf,
                        opc_fact)
    
    return opc_fact
    



###########################################################################

def pq_resid(est,     # [array of prelogit-based estimates]
             resid,   # [None, raw residuals]
             colkeys = None,    # [<None, 1-D array of column keys>]
             ecut = ['All', 0.0], # [<['All', ecut], ['Cols',{'ID1':ecut, ...}]
             ear = None, #[<None, float, ear array]
             new_logits = False, #[<True, False>]
             validchars = ['All', [0, 1]],
             nanval = -999. # [not-a-number value]
             ):
    """
    Adjust size of residuals/EARs for dichotomous or polytomous data.
    
    Returns
    -------
        A dict:
            {'new_resid':__, 
            'new_logit':__}
            
        where 'new_resid' is an array of residuals corrected for
        distortions caused by dichotomous or polytomous responses
        and 'new_logit' is an optional replacement of the prelogit-based
        estimates with logits that are compatible with the specified
        ear array.
        
    Comments
    --------
        When analyzing interval data, one expects the residuals between
        observations and true values to be the same on average across 
        the scale.  This assumption does not hold when the data are
        dichotomous or polytomous (less of a problem).  This is because,
        in the dichotomous case, when the "true" (eg "model") values are
        near the center of the scale (say 0.50), their residuals relative
        to the observed values (0 or 1) will be too large.  The reverse
        is true when the "true" values are at the extremes.  Since the
        estimates track the true values (more or less), the distortion
        ripples through to the residuals, expected absolute residuals,
        standard errors, and all derivative statistics.
                
        To correct the distortion, this function adjusts the residuals by 
        dividing by their binomial variance:
        
            new_resid = resid / sqrt(pq)
        
        where p is calculated from the prelogit-based estimates and
        q = (1 - p).
        
        pq_resid() is also applied in base_se() to complete the calculation
        of standard errors in this situation, but here the numerator
        is not the raw residuals but 1.0.  Optionally, the numerator can
        be an array of ears or any constant, controlled by the ear parameter.
        
        While the formula is targeted at dichotomous data, it seems to work
        acceptably for polytomous data as well.  A proper polytomous
        formula still needs to be derived.
        
        Mixed data designs (a mix of ordinal, interval, ratio data) are
        supported.  The binomial correction is only applied to columns
        that are found to be ordinal, based on the validchars parameter.
        All other residuals are passed through unchanged.
    
    Arguments
    ---------
        "est" is a 2-d array of estimates.  They must be in some form of
        logit metric, presumably "prelogits".
        
        --------------
        "resid" is a 2-d array of residuals between raw prelogit values
        and their corresponding cell estimates.
            
            resid = None    =>  Do not adjust residuals. The function will
                                refer to the "ear" parameter.

        --------------
        "colkeys" is a 1-dimensional array of column keys, the unique
        identifiers for each column in estimates and ear.  It only needs
        to be specified when ecut contains a column dictionary.
        
        --------------        
        "ecut" is specified in, and output as part of, the base_est()
        method as part of the ecutmaxpos parameter.  (The "maxpos" half of
        the parameter is not used.  Documentation on ecutmaxpos is available 
        in the base_est() docs.)  In short, ecut is the estimates cut-point
        above which an estimate is named a "success".

        Here are the options:

            ecut = ['All', ecut]

        means that the values given for ecut apply to the whole array, 
        across all columns.

            ecut = ['All', 'Med']

        means that ecut should be the median ('Med') of the whole array.

            ecut = ['Cols', 'Med']

        means that the ECut and MaxPos for each column should be the
        median and maximum value of that column, calculated separately
        for each column.

            ecut = ['Cols',{'ID1':'Med','ID2':25, ...}]

        means that for the 'ID1' column ecut should be the column median,
        for the 'ID2' column, ecut should be 25, and so on, for all the 
        columns.

        -------------- 
        "ear" is None, a float, or an array of expected absolute residuals.
        
            ear = None      =>  (default). Use "resid" in the numerator (see
                                formula in comments).  This adjusts the
                                residuals for the dichotomous distortion.
            
            ear = 1.0       =>  Use 1.0 in the numerator. This returns the
                                logit binomial variance.
            
            ear = ear_array =>  Use the array of expected absolute residuals
                                in the numerator.

        -------------- 
        "new_logits" (bool, default is False) controls how to calculate p 
        in the formula above.
        
            new_logits = True
                            =>  Calculate p and corresponding logits using
                                the "ear" parameter. 
                                
                                Advantage: the resulting logits and 
                                probabilities are rescaled from the
                                somewhat arbitrary "prelogit" metric to take
                                into account the noisiness of the data.  The
                                new logits will have a linear relationship to
                                the "prelogits".
                                
                                Disadvantage: the new logits, if used as cell
                                estimates, will no longer be in sync with the 
                                corresponding array coordinates, so will no
                                longer support equating.
            
            new_logits = False
                            =>  Preserve the prelogit metric.
                            
                                Since equating is of paramount importance and
                                the logit metric less so, this is the default.

        -------------- 
        "validchars" is the validchars parameter of the Damon object,
        presumably ['All', [0, 1], 'Num'] or something similar.  It
        is used to determine whether the data are ordinal.
        
        -------------- 
        "nanval" is the not-a-number value

    Paste Function
    --------------
        pq_resid(est,     # [array of prelogit-based estimates]
                 resid,   # [None, raw residuals]
                 colkeys = None,    # [<None, 1-D array of column keys>]
                 ecut = ['All', 0.0], # [<['All', ecut], ['Cols',{'ID1':ecut, ...}]
                 ear = None, #[<None, float, ear array]
                 new_logits = False, #[<True, False>]
                 validchars = ['All', [0, 1]],
                 nanval = -999. # [not-a-number value]
                 )

    """
    # Calc probabilities
    ear_ = ear if new_logits else None
    cum = cumnormprob(est, ear_, None, ecut, False, nanval)
    p = cum['Prob']
    vc = valchars(validchars)['metric']
    
    # Standardize by whole array
    if vc[0] == 'All':
        if vc[1] != 'ordinal':
            new_resid = resid
        else:
            if ear is None:
                res = resid
                if res is None:
                    exc = 'resid and ear params cannot both be None.'
                    raise ValueError(exc)
            else:
                res = ear
            
            if resid is not None:
                ix = resid == nanval
            elif isinstance(res, np.ndarray):
                ix = res == nanval
            else:
                ix = None
            new_resid = res / np.sqrt(p * (1 - p))
            
            if ix is not None:
                new_resid[ix] = nanval
    
    # Standardize by col
    elif vc[0] == 'Cols':
        new_resid = np.zeros(np.shape(resid))
        for i, col in enumerate(colkeys):
            
            # col not in vc[1] assumes parsed id and thus ordinal data
            if col not in vc[1] or vc[1][col] != 'ordinal':
                new_resid[:, i] = resid[:, i]
            else:
                if resid is not None:
                    res_ = resid[:, i]
                else:
                    res_ = None
                    
                if ear is None:
                    res = res_
                    if res is None:
                        exc = 'resid and ear params cannot both be None.'
                        raise ValueError(exc)
                else:
                    if isinstance(ear, np.ndarray):
                        res = ear[:, i]
                    else:
                        res = ear
                
                if resid is not None:
                    ix = res_ == nanval
                elif isinstance(res, np.ndarray):
                    ix = res == nanval
                else:
                    ix = None
            
                p_ = p[:, i]
                new_resid[:, i] = res / np.sqrt(p_ * (1 - p_))
                
                if ix is not None:
                    new_resid[ix, i] = nanval            
            
    return {'new_resid':new_resid, 'new_logit':cum['Logit']}




###########################################################################

def cumnormprob(estimates,  # [array of estimates for which we want a cumulative probability]
                ear,    # [<None, float, array of Expected Absolute residuals>]
                colkeys = None,    # [<None, 1-D array of column keys>]
                ecut = ['All', 0.0], # [<['All', ecut], ['Cols',{'ID1':ecut, ...}]> ]
                logits = None,   # [<None, True> => return logits with probabilities]
                nanval = -999., # [Not-a-Number Value]
                ):
    """Calculate a cell's cumulative normal probability.

    Returns
    -------
        {'Prob':__,
        'Logit':__
        }

        When ear is in fact the Expected Absolute Residual (EAR),
        the function returns an array of probabilities that each cell
        estimate will exceed a given value conditional on its EAR,
        the "noise" or "residuals" associated with that row and column
        regardless of the number of data points.  Each cell is the probability
        that a cell observation will be greater than the specified cut-point
        (ECut) given the cell estimate and its standard deviation.

        If logits = True, logit equivalents are also returned.

    Comments
    --------
        Note that we define the distribution in terms of the expected
        standard deviation of the estimate (a function of the size of
        the residuals in that cell's row and column), not its
        standard error (a function of the standard deviation
        and the number of observations that go into calculating the
        estimate).  Thus, the resulting probability is not sensitive
        to the size of the data set.  It is driven not by the precision
        and reliability of the estimate, which can be high even if
        it poorly predicts the observed values, but by the precision
        and reliability of the observation -- the observation error.

        One could, however, let EAR equal the cell standard error,
        which gets smaller as the number of observations increases.
        The resulting probability would reflect how certain we are
        that the estimate is indeed accurate, which has its own uses.

        There are other ways to define the probability.  The metricprob()
        function does a simple linear transformation to force Damon
        estimates to have the same sigmoid metric as probabilities.
        This function is suitable when there is confidence in the
        metric properties of the estimates and our goal is not
        so much to calculate a cell probability as a stable, reproducible
        set of measures.

        The cumnormprob() probabilities are defined more in accordance
        with statistical definitions of probability.  Probability is
        a function of "how certain" we are of meeting some condition.
        Thus, cumnormprob() is a function of the standard deviation (EAR)
        of the estimates -- their noisiness or uncertainty.  This gives
        a more accurate prediction of the probability of a response
        (since it takes into account the effect of error on probability)
        but at the expense of metric reproducibility.

        To calculate the cumulative probability, it is useful to
        consider that the cell estimate can be thought of as the mean
        of a large number of observations for that cell.  In reality,
        we only have one observation, but Damon leverages the data in
        the remainder of the dataset so that its cell estimates are
        mathematically comparable to taking the mean of a lot of
        observations for a given cell.  Similarly, the Expected Absolute
        Residual (EAR) is equivalent to its standard deviation.

        We are after the probability that the observed value is higher
        than the cut-point ("ecut") given the cell estimate and EAR, or one
        less the probability that the observed value is lower than the
        cut-point.  Consider a Gaussian distribution whose mean is
        the estimate and whose standard deviation is the EAR.  The ecut
        falls somewhere on this distribution.  The probability of
        failure (the observation being below ecut) is the area of the
        curve below ecut.  The probability of success is 1 - P[Failure].

        To calculate the error of the curve below ecut, we need to
        calculate the integral of the normal distribution curve, which
        has no explicit closed form solution.  However, it can be
        approximated by a number of methods.  One commonly used method,
        and the one used here, comes from Abramowitz and Stegun (1964)
        (see Wikipedia, Normal Distribution).

    Arguments
    ---------
        "estimates" is an array of cell estimates for which we want a probability.
        It will generally be the output of the base_est() method.

        --------------
        "ear" is the array of "expected absolute residuals" of the
        estimate, found by running the coord() and base_est() methods on
        an array of absolute residuals.  It can also be a single number,
        such as a mean EAR.  If ear is None, the estimates (which should be
        in a prelogit metric) will be returned unchanged, along with 
        probabilities.

        --------------
        "colkeys" is a 1-dimensional array of column keys, the unique
        identifiers for each column in estimates and ear.  It only needs
        to be specified when ecut contains a column dictionary.

        ---------------
        "ecut" is specified in, and output as part of, the base_est()
        method as part of the ecutmaxpos parameter.  (The "maxpos" half of
        the parameter is not used.  Documentation on ecutmaxpos is available 
        in the base_est() docs.)  In short, ecut is the estimates cut-point
        above which an estimate is named a "success".

        Here are the options:

            ecut = ['All', ecut]

        means that the values given for ecut apply to the whole array, 
        across all columns.

            ecut = ['All', 'Med']

        means that ecut should be the median ('Med') of the whole array.

            ecut = ['Cols', 'Med']

        means that the ECut and MaxPos for each column should be the
        median and maximum value of that column, calculated separately
        for each column.

            ecut = ['Cols',{'ID1':'Med','ID2':25, ...}]

        means that for the 'ID1' column ecut should be the column median,
        for the 'ID2' column, ecut should be 25, and so on, for all the 
        columns.

        --------------
        "logits" = True instructs the function to return a logit
        in addition to a probability, where

            Logit = log( p / (1 - p) )

        --------------
        "nanval" is the Not-a-Number value used to signify an invalid
        estimate.

    Examples
    --------



    Paste function
    --------------
    cumnormprob(estimates,  # [array of estimates for which we want a cumulative probability]
                ear,    # [<None, float, array of Expected Absolute residuals>]
                colkeys = None,    # [<None, 1-D array of column keys>]
                ecut = ['All', 0.0], # [<['All', ecut], ['Cols',{'ID1':ecut, ...}]> ]
                logits = None,   # [<None, True> => return logits with probabilities]
                nanval = -999., # [Not-a-Number Value]
                )
    """
    if len(np.shape(estimates)) < 2:
        Est = estimates[:, np.newaxis]
    else:
        Est = estimates

    if isinstance(ear, np.ndarray):
        if len(np.shape(ear)) < 2:
            ear = ear[:, np.newaxis]
    
    if ear is None:
        Log = Est
        P = np.where(Log == nanval, nanval, np.exp(Log) / (1 + np.exp(Log)))
        
        return {'Prob':P,'Logit':Log}
    
    ncols = np.size(Est, axis=1)

    # Define ECut_
    Check = ['All','Cols']
    if ('All' not in Check
        or 'Cols' not in Check
        ):
        exc = 'Unable to figure out ecut.\n'
        raise cumnormprob_Error(exc)

    ECut = ecut
    ECut_ = np.zeros((ncols))
    if ECut[0] == 'All':
        if ECut[1] == 'Med':
            ValEst = Est[Est != nanval]
            if np.sum(ValEst) == 0:
                exc = 'Could find no valid data.\n'
                raise cumnormprob_Error(exc)
            else:
                ECut_[:] = np.median(ValEst)
        else:
            ECut_[:] = ECut[1]

    elif ECut[0] == 'Cols' and not isinstance(ECut[1], dict):
        if ECut[1] == 'Med':
            for i in xrange(ncols):
                ValEst = Est[:,i][np.where(Est[:,i] != nanval)]
                if np.sum(ValEst) == 0:
                    ECut_[i] = nanval
                else:
                    ECut_[i] = np.median(ValEst)
        else:
            for i in xrange(ncols):
                ECut_[i] = ECut[1]

    elif ECut[0] == 'Cols' and isinstance(ECut[1], dict):
        col_dict = ECut[1]
        
        if colkeys is None:
            colkeys = col_dict.keys()
#            exc = ('column keys need to be specified if ecut contains a '
#                   'column dictionary.')
#            raise cumnormprob_Error(exc)

#        for k in col_dict.keys():
#            if col_dict[k] == 'Med':
#                val_est = Est[:]
            
        for i, k in enumerate(colkeys):
            if col_dict[k] == 'Med':
                val_est = Est[:]
                if np.sum(val_est) == 0:
                    ECut_[i] = nanval
                else:
                    ECut_[i] = np.median(val_est)
            else:
                ECut_[i] = col_dict[k]
            
#        for i in xrange(ncols):
#            if ECut[colkeys[i]][0] == 'Med':
#                ValEst = Est[:,i][Est[:,i] != nanval]
#                if np.sum(ValEst) == 0:
#                    ECut_[i] = nanval
#                else:
#                    ECut_[i] = np.median(ValEst)
#            else:
#                ECut_[i] = ECut[colkeys[i]][0]

    # Define z -- clip at z = -6.0 and 6.0
    z = np.where(Est == nanval, nanval,
                 np.clip((ECut_ - Est) / ear, -6.0, 6.0))
#    PiSqrt3 = 1.81379936423422

    # Magic constants
    b1 = 0.31938153
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419
    c2 = 0.3989423

    # array Calculation
    a = abs(z)
    t = 1.0 / (1.0 + (a * p))
    b = c2 * np.exp((-z) * (z / 2.0))
    Q = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    Q = 1.0 - (b * Q)

    Q = np.where(z == nanval, nanval,
                 np.where(z < 0.0, 1.0 - Q, Q))

    # Probability of success
    P = np.where(Q == nanval, nanval, 1 - Q)

    # return_ prob or logit
    if logits is True:
        Log = np.where(P == nanval, nanval, np.log( P / (1.0 - P)))
    else:
        Log = None

    return {'Prob':P,'Logit':Log}


###########################################################################

def log2prob(logits,    # [2-D array of logit values to convert to probabilities]
             colkeys,   # [1-D array of column keys]
             validchars,    # [Damon validchars specification]
             sigma_thresh = 2,    # [Number of categories below which ordinal scale is treated as ordinal]
             nanval = -999, # [Not-a-number value]
             ):
    """Convert logits to probabilities.

    Returns
    -------
        2-D array of probabilities calculated from base_est()
        outputs, where the inputs are PreLogits or PLogits.

    Comments
    --------
        "log2prob" is used in fin_est() and est2Prob()
        to convert logits into probabilities.  This is ordinarily
        a simple conversion according to the cell formula:

            Prob = exp(Logit) / (1 + exp(Logit))

        However, when data are ordinal and have only two or three
        categories, the logit formula produces distortions,
        in which case it is better to do a simple linear conversion:

            Prob = Logit * (Max_p - Min_p) / (MaxLog - MinLog)

        where Max_p and Min_p are the target maximum and minimum
        probabilities (calculated by the function) and MaxLog and
        MinLog are the maximum and minimum logit estimates.

        log2prob() handles these situations.

    Arguments
    ---------
        "logits" is a 2-D array of logit values to convert
        to probabilities

        ---------------
        "colkeys" is a 1-D array of column keys corresponding to
        columns with analyzable data in them.

        ---------------
        "validchars" is the validchars attribute of the Damon object.
        It's column IDs correspond to the original array, not to
        parsed or standardized arrays.  validchars may be either
        in 'All' or 'Cols' format.

        ---------------
        "sigma_thresh" is the integer number of response categories
        below which it is assumed that an ordinal scale should be
        treated as interval.  This is generally 2; dichotomous data
        have too few categories to discern its underlying sigmoid
        structure.

        ---------------
        "nanval" is the Not-a-Number value.

    Examples
    --------


    Paste function
    --------------
        log2prob(logits,    # [2-D array of logit values to convert to probabilities]
                 colkeys,   # [1-D array of column keys]
                 validchars,    # [Damon validchars specification]
                 sigma_thresh = 2,    # [Number of categories below which ordinal scale is treated as ordinal]
                 nanval = -999, # [Not-a-number value]
                 )

    """

    exc = 'Sorry, log2prob() is unavailable.\n'
    raise log2prob_Error(exc)

##    # Get metrics, min/max
##    VCOut = tools.valchars(validchars,    # ['validchars' output of data() function]
##                         dash = ' -- ', # [Expression used to denote a range]
##                         defnone = 'interval',   # [How to interpret metric when validchars = None]
##                         retcols = colkeys,    # [<None, [list of core col keys]>]
##                         )
##
##    OrigMetric = VCOut['metric']
##    minmax = VCOut['minmax']
##
##    # Get col ents
##    Ents = colkeys
##
##    # Get
##
##
##    # Case1:  convert to 0 - 1 without using the probability formula (for 2, 3 category data)
##    if OrigMetric[1][Ent] == 'ordinal':
##        Min = minmax[1][Ent][0]
##        Max = minmax[1][Ent][1]
##        nCats = Max - Min + 1
##        SigThresh = 2   # Magic number of categories below which the ordinal scale is treated like an interval scale
##
##        if nCats <= SigThresh:
##            MinEst = np.amin(AdjEntCore[np.where(AdjEntCore != nanval)])
##            MaxEst = np.amax(AdjEntCore[np.where(AdjEntCore != nanval)])
##
##            # PLogits are forced to extend from 0.0 to 1.0 (otherwise, they tend to be too narrow)
##            if stdmetric == 'PLogit':
##                Min_p = 0.0
##                Max_p = 1.0
##
##            # For other metrics, get min and max p from the estimates
##            else:
##                Min_p = np.exp(MinEst) / (1.0 + np.exp(MinEst))
##                Max_p = np.exp(MaxEst) / (1.0 + np.exp(MaxEst))
##
##            p_ = np.where(AdjEntCore == nanval,nanval,(AdjEntCore - MinEst) / (MaxEst - MinEst))
##            p_EntCore = np.where(p_ == nanval,nanval,p_ * (Max_p - Min_p) + Min_p)
##
##        else:
##            # Same as Case2
##            p_EntCore = np.where(AdjEntCore == nanval,nanval,np.exp(AdjEntCore) / (1.0 + np.exp(AdjEntCore)))
##
##            if GetSE is True:
##                p_EntCoreSE = np.where(AdjEntCore == nanval,nanval,(np.exp(AdjEntCore) * AdjEntCoreSE) / ((1.0 + np.exp(AdjEntCore))**2))
##
##    # Case2: convert to 0 - 1 using the p = exp(x) / (1 + exp(x)) formula
##    else:
##        p_EntCore = np.where(AdjEntCore == nanval,nanval,np.exp(AdjEntCore) / (1.0 + np.exp(AdjEntCore)))
##
##        if GetSE is True:
##            p_EntCoreSE = np.where(AdjEntCore == nanval,nanval,(np.exp(AdjEntCore) * AdjEntCoreSE) / ((1.0 + np.exp(AdjEntCore))**2))
##
##



###########################################################################

def metricprob(estimates,  # [array of estimates for which we want a cumulative probability]
               colkeys,    # [1-D array of column keys]
               ecutmaxpos, # [<['All',[ECut,MaxPos]], ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
               pcut = 0.50, # [Probability separating "success" from "failure"]
               logits = True,    # [<None, True> => return logits instead of probabilities]
               nanval = -999., # [Not-a-Number Value]
               ):
    """Calculate a cell's "metric" probability.

    Returns
    -------
        An array of probabilities representing a one-to-one metric
        conversion of Damon estimates to probabilities.  Each cell
        is the probability that a cell observation will be greater
        than the specified cut-point (ECut), assuming constant
        cell variance.

        cumnormprob() can output logits instead of probabilities.

    Comments
    --------
        metricprob() is not so much a statistical probability as
        a metric conversion from a fuzzy sigmoid estimates metric
        to a stricter sigmoid metric that falls between 0 and 1.

        The statistical probability is obtained used the cumnormprob()
        function.  It is more accurate as a "probability" because it
        takes into account the effect of differing cell errors on
        probability.   However, it does so at the cost of sacrificing
        any metric reproducibility across datasets.  Its measures
        are unique to the testing occasion.

        The metricprob() function, being a straightforward metric
        transformation from estimates to probabilities, produces
        probabilities and logits that are more likely to reproduce
        across datasets.  However, this only holds when we can
        have faith in the metric properties of the Damon estimates.
        With dichotomous data, these metric properties can suffer
        distortions at the extremes, in which case cumnormprob() is
        better.

        Why do a conversion at all?  Why not use Damon estimates as
        they are?  The reason is that Damon's coord() function is
        based on ordinary least squares, which assumes that the data
        are continuous and unbounded.  When the data are sigmoid, or
        polytomous, or dichotomous, some requirements of ordinary
        least squares are violated.  Most commonly, these issues are
        dealt with by not using least squares but some form of maximum
        likelihood, as is done with logistic regression and Rasch
        models.  However, in the context of Damon, maximum likelihood
        is computationally prohibitive.  Therefore, Damon uses a variation
        of what is called a Linear Probability model (LPM) -- basically,
        linear least squares applied to dichotomous data.  This approach
        is computationally less demanding and makes it possible to use the
        same algorithm for dichotomous, polytomous, and continuous data.
        The problem with LPM is that it applies a technique valid for
        continuous interval data to discrete and ordinal data, where it
        is not valid, and this can distort the estimates in several
        ways.  One of those ways is to produce estimates that look
        like probabilities (in the case of dichotomous data) but in
        fact spill above 1 and below 0 (or above the maximum and below
        the minimum of the ordinal range).  metricprob() forces the
        data into the proper range and to behave metrically like
        probabilities and logits.

        When Damon inputs are ordinal or, in the most extreme case,
        dichotomous (0,1), the resulting estimates, though continuous, are
        nonlinear.  They are influenced by a somewhat fuzzy floor and
        ceiling, spilling above the ceiling and below the floor.  They
        are too fuzzy to be used as probabilities (though they are
        nonlinear, like probabilities) and too nonlinear to be used as
        measures.  metricprob() is the solution to the problem of estimates
        that spill out of range.

        metricprob() is built for the fuzzily bounded sigmoid output
        produced by Damon when fed ordinal data.  It matches a fuzzy
        estimates sigmoid function against an idealized probability
        sigmoid function and converts one to the other.

    Derivation
    ----------
        Probability/Bounding Formula:
            m = theoretical max possible estimate
            e = estimate cut-point for success (ECut)
            p = target probability cut-point for success (pcut)
            E = nonlinear estimate
            P = target probability
            x = coefficient
            y = intersept

        It has been determined that damon estimates have the
        same sigmoid relationship to the underlying variable as
        the probability of "success" in exceeding a given
        threshold on the variable.  Because the estimates and
        their associated probabilities have the same sigmoidal
        relationship to a third variable, they have a linear
        relationship to each other, which can be expressed as
        follows:

        1	P = Ex + y

                Let:
        2	mx + y = 1
        3	ex + y = p

                Then,
        4	mx + y = 1
        5	mx = (1 - y)
        6	x = 1/m - y/m = (1 - y)/m

                Solve for y:
        7	ex + y = p
        8	e(1 - y)/m + y = p
        9	e/m - ye/m + y = p
        10	e/m + y(1 - e/m) = p
        11	y(1 - e/m) = p - e/m
        12	y = (p - e/m) / (1 - e/m)
        13	y = (mp - e) / (m - e)

                Solve for x:
        14	x = 1/m - y/m
        15	x = 1/m - (mp - e)/m(m - e)

        This formula does not work if m (theoretical max possible estimate)
        is less than or equal e (estimate cut-point for success), and it
        rarely makes sense for it to be less than 1.0.  Therefore, metricprob()
        sets 1.0 as the smallest allowable maximum value.  It can get m
        automatically by looking for maximum values in the array, or
        separately for each column, or it can be entered manually at any
        value.  These options are controlled with the 'MaxPos=' argument.

        Technically, this formula could yield values above 1.0 and below 0.0
        though only in a strictly linear way.  It could happen, for instance,
        if an estimate exceeds the maximum possible value, or is smaller than
        the smallest possible value as determined by the formula.  To avoid
        these cases (which cause an error in the logit formula), metricprob()
        clips probability values above 0.999999 or below 0.000001, forcing them to
        fall within range.

    Arguments
    ---------
        "estimates" is a 2-dimensional array of estimates which we want to
        convert to probabilities.

        ---------------
        "colkeys" is a 1-dimensional array of column keys which label each
        data column.

        ---------------
        "ecutmaxpos" is specified in and output as part of the base_est()
        method.  It specifies the ECut and MaxPos parameters for each
        item.  Documentation on ecutmaxpos is available in the base_est() docs.

        Here are the options:

            ecutmaxpos = ['All',[ECut,MaxPos]]

        means that the values given for ECut and MaxPos apply to the
        whole matrix, across all columns.

            ecutmaxpos = ['All',['Med','Max']]

        means that ECut should be the median ('Med') of the whole array
        and MaxPos should be the maximum ('Max') of the whole array.

            ecutmaxpos = ['Cols',['Med','Max']]

        means that the ECut and MaxPos for each column should be the
        median and maximum value of that column, calculated separately
        for each column.

            ecutmaxpos = ['Cols',{'ID1':['Med','Max'],'ID2':[10,25],...}]

        means that for the 'ID1' column ECut should be the column median
        and MaxPos should be the maximum value in the column.  For the
        'ID2' column, ECut should be 10 and MaxPos should be 25.  And so
        on, for all the columns.

        ---------------
        "pcut" is the probability value that corresponds to "ECut", almost
        always set at 0.50 by convention.  Say the estimates range from
        0 to 3 and the ECut is 2.0.  What this means is that an ECut of 2.0
        is defined to imply a probability of 0.50.  But you could define
        it to imply a probability of, say, 0.75.  pcut is how you do that.

        ---------------
        "logits" = True instructs the function to return a logit
        in addition to a probability, where

            logit = log( p / (1 - p) )

        ---------------
        "nanval" is the Not-a-Number value used to flag invalid values
        in the estimates array, if they exist.

    Examples
    --------



    Paste function
    --------------
        metricprob(estimates,  # [array of estimates for which we want a cumulative probability]
                   colkeys,    # [1-D array of column keys]
                   ecutmaxpos, # [<['All',[ECut,MaxPos]], ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                   pcut = 0.50, # [Probability separating "success" from "failure"]
                   logits = True,    # [<None, True> => return logits instead of probabilities]
                   nanval = -999., # [Not-a-Number Value]
                   )

    """

    # Get variables
    Est = estimates
    ncols = np.size(Est,axis=1)
    pcut = float(pcut)

    # Parse ecutmaxpos argument
    ECut = np.zeros((ncols)).astype(float)
    Max = np.zeros((ncols)).astype(float)

    if ecutmaxpos is None:
        print "Warning in metricprob(): ecutmaxpos = None.  Changing it to ['Cols',['Med','Max']].\n"
        ecutmaxpos = ['Cols',['Med','Max']]

    # Get ECut and MaxPos for whole array
    elif ecutmaxpos[0] == 'All':
        ValEst = Est[np.where(Est != nanval)]
        if np.sum(ValEst) == 0:
            exc = 'Could not find valid data.\n'
            raise metricprob_Error(exc)

        # Get ECut
        if ecutmaxpos[1][0] == 'Med':
            ECut[:] = np.median(ValEst)
        else:
            ECut[:] = ecutmaxpos[1][0]

        # Get MaxPos
        if ecutmaxpos[1][1] == 'Max':
            Max[:] = np.max(ValEst)
        else:
            Max[:] = ecutmaxpos[1][1]

    # Get ECut and MaxPos for individual columns
    elif ecutmaxpos[0] == 'Cols':

        # Get ECut
        if type(ecutmaxpos[1]) is not type({}):
            if ecutmaxpos[1][0] == 'Med':
                for i in xrange(ncols):
                    ValEst = Est[:,i][np.where(Est[:,i] != nanval)]
                    if np.sum(ValEst) == 0:
                        ECut[i] = nanval
                    else:
                        ECut[i] = np.median(ValEst)
            else:
                for i in xrange(ncols):
                    ECut[i] = ecutmaxpos[1][0]


        elif type(ecutmaxpos[1]) is type({}):
            for i in xrange(ncols):
                ECut[i] = ecutmaxpos[1][colkeys[i]][0]

        # Get MaxPos
        if type(ecutmaxpos[1]) is not type({}):
            if ecutmaxpos[1][1] == 'Max':
                for i in xrange(ncols):
                    ValEst = Est[:,i][np.where(Est[:,i] != nanval)]
                    if np.sum(ValEst) == 0:
                        Max[i] = nanval
                    else:
                        Max[i] = np.amax(ValEst)
            else:
                for i in xrange(ncols):
                    Max[i] = ecutmaxpos[1][1]

        elif isinstance(ecutmaxpos[1],dict):
            for i in xrange(ncols):
                Max[i] = ecutmaxpos[1][colkeys[i]][1]

    # Clip Max so that it is at least 1.0
    Max = np.clip(Max,1.0,max(np.max(Max),1.0))

    # Define x and y for each column
    x = np.zeros((ncols)).astype(float)
    y = np.zeros((ncols)).astype(float)
    for i in xrange(ncols):
        x[i] = 1.0/Max[i] - (Max[i]*pcut - ECut[i]) / (Max[i]*(Max[i] - ECut[i]))
        y[i] = (Max[i]*pcut - ECut[i]) / (Max[i] - ECut[i])

    # Multiply estimates in each column by their corresponding x, add y
    CellProb = Est * x
    CellProb = CellProb + y
    CellProb = np.clip(CellProb,0.000001,0.999999)
    CellProb = np.where(Est == nanval, nanval, CellProb)

    # Convert to logits
    if logits is True:
        Log = np.where(CellProb == nanval, 
                       nanval, 
                       np.log(CellProb / (1. - CellProb)))
    else:
        Log = None

    return {'Prob':CellProb,'Logit':Log}




###########################################################################

def unbiasest(E,    # [Estimate array to be adjusted]
              Res,  # [Residual array corresponding to E and to U coordinates]
              h,    # [h-statistics calculated using same U as Res, E]
              nanval,   # [float Not-a-Number Value]
              ):
    """Returns estimates as if corresponding observations were deleted.

    Returns
    -------
        estimates as if corresponding observations were deleted

    Comments
    --------
        'Res' (residual) and 'h' (leverage statistic) need to
        be computed from the same U coordinate system (R or C).
        The formula is:

            E1 - E[missing] = Res * h / (1 - h)
            E[missing] = E1 - (E1 - E[missing])

    Paste function
    --------------
        unbiasest(E,    # [Estimate array to be adjusted]
                  Res,  # [Residual array corresponding to E and to U coordinates]
                  h,    # [h-statistics calculated using same U as Res, E]
                  nanval,   # [float Not-a-Number Value]
                  )


    """
    UnbiasedEst = E - Res * h / (1 - h)
    UnbiasedEst = np.where(Res == nanval,E,UnbiasedEst)

    return UnbiasedEst


###########################################################################

def sterrpbc(st_err,  # [vector or array of cell standard errors]
             rmse_reg = 1,   # [None, Compute regular root mean square error (row_ptbis and col_ptbis = None)]
             row_ptbis = None,     # [None, Vector of row point biserials, data row by row of col means]
             col_ptbis = None,     # [None, Vector of col point biserials, data col by col or row means]
             nanval = -999.
             ):
    """Aggregates standard errors weighted by point biserial correlations.

    Returns
    -------
        Scalar representing the aggregated standard error
        taking into account the correlations between
        entities.

    Comments
    --------
        This is a method of aggregating cell standard errors
        that is specific to Damon and has not been studied
        in depth.  It is based on the premise that while an
        ordinary root mean squared error is sufficient for
        unidimensional arrays where the model correlations
        between entities is 1.0, it may understate the aggregated
        standard error for multidimensional arrays whose
        model correlations between entities may be less
        than 1.0.

        The method is currently not in use due to the
        formula not having been sufficiently studied and
        the computational complexity of implementing it.
        It requires calculating what is called the Root Point-
        Biserial Weighted Mean Square Error, which was calculated
        in the summstat() function but is no longer.  Perhaps it
        will be reintroduced.  Anyway, here is how it works.

        When aggregating a range of cells, the rows and columns are
        done separately, then combined.  The resulting formula for
        the RPBWMSE (Root Point-Biserial "Weighted" Mean Squared Error)
        is:

        RPBWMSE^2 = sum(s[ij]^2) / ( sum(R[iI]^2) * sum(R[jJ]^2) )
        RPBWMSE = sqrt(RPBWMSE^2)

        where s[ij] is the standard error of the estimate in the cell
        for row i and column j, and R[iI] is the pt.bis. correlation
        between row i and the row of column means I, and R[jJ] is
        the pt.bis. correlation between column j and the column
        of row means J.  When the data are 1-dimensional, all rows and
        columns are perfectly correlated with each other, causing
        the correlations to be 1.0, at which point RPBWMSE = RMSE across
        the range.

        "Weighted" is in quotes since the formula is not a true
        weighting; the numerator does not include multiplication
        by point biserials.  However, simulation studies suggest
        that this is the formula that properly bounds the situation
        where all the items are perfectly correlated (divided by
        sum of items) and the situation where the items are perfectly
        uncorrelated with themselves and partially correlated with
        the mean, in which case the denominator will tend toward 1.0
        and the variance of the mean will be the sum of its component
        variances.

    Arguments
    ---------
        "st_err" is an array of standard errors (SE).  The cell SE
        statistic is not to be confused with the cell EAR (expected
        absolute residuals) statistic.

        --------------
        "rmse_reg" specifies that a regular root mean squared
        error should be calculated, in which case set the
        next two parameters -- row_ptbis and col_ptbis --
        to None.  rmse_reg will yield a smaller standard error,
        but the proper conditions for using one statistic or
        the other are not fully understood.  RMSE is the industry
        standard.

        ---------------
        "row_ptbis" and "row_ptbis" are vectors
        of point-biserial correlations calculated outside
        the function.  (It used to be calculated internally
        by summstat(), but is no longer.)

        When the error vector is a "row" of cell errors,
        you should have calculated the col_ptbis, and
        row_ptbis should be set at 1.0.  When the error
        vector is a "column" of cell errors, you should have
        calculated the row_ptbis, and the col_ptbis
        should be set at 1.0.  When the error vector is a
        "range" of cell errors, you should have calculated
        both a row_ptbis and col_ptbis and entered
        them.

    Paste function
    --------------
        sterrpbc(st_err,  # [vector or array of cell standard errors]
                 rmse_reg = 1,   # [None, Compute regular root mean square error (row_ptbis and col_ptbis = None)]
                 row_ptbis = None,     # [None, Vector of row point biserials, data row by row of col means]
                 col_ptbis = None,     # [None, Vector of col point biserials, data col by col or row means]
                 nanval = -999.
                 )

    """
    #Mark_sterrpbc

    # Calculate standard error
    ValErr = st_err[np.where(st_err != nanval)]

    if (rmse_reg == 1
        or (row_ptbis is None
            and col_ptbis is None)
        ):
        sterr = np.sqrt(np.sum(ValErr**2) / np.size(ValErr))

    elif row_ptbis is not None and col_ptbis is not None:
        sterr = np.sqrt(np.sum(ValErr**2) / (row_ptbis * col_ptbis) )

    else:
        print 'Error in sterrpbc():  Unable to reconcile input parameters.\n'

    return sterr










###########################################################################

def separation(stdev,   # [<None, standard deviation of the cell values>]
               sterr,   # [<standard error, output of rmsr() or sterrpbc()>]
               estimates = None, # [<None, array of estimates>]
               nanval = -999 # [<not-a-number value]
               ):
    """Returns the "separation" of a set of measurements, their
    spread relative to their noise.

    Returns
    -------
        Scalar representing the ratio of the spread of a
        set of measures relative to their average error of
        measurement.  It is also described as the ratio of
        true to observed variance.  Converted to a 0.0 - 1.0
        metric with reliability(), it is equivalent to a
        Cronbach-alpha statistic.

    Comments
    --------
        "separation" provides some flexibility in the
        the range of input parameters.  If you have already
        aggregated standard errors and have a standard
        deviation, it is only necessary to enter values
        for stdev and sterr.

        If these statistics are not available, separation()
        will calculate them based on estimates and st_err.

        If you want to take into account correlations between
        entities in aggregating standard errors, you can enter
        non-1.0 point-biserial correlation vectors.  (This can
        also be done by entering the output of sterrpbc()
        for sterr.)

        Aggregating Standard Errors
        ---------------------------
        Ordinarily, standard errors are aggregated by taking
        the root of their mean squares.  Damon currently
        employs this method of aggregation.  However, it
        has been surmised that in multidimensional measurement
        account should be taken of the degree to which each
        contributing variable is correlated with the measure
        of interest.  This statistic can be obtained
        using Damon's sterrpbc() function.  However, it is
        not known yet whether it is worth the computational
        effort.        

    Arguments
    ---------
        "stdev" is the standard deviation of the cell estimates.
        If None, it will be calculated from estimates.  stdev can
        also be an array when calculating a separation statistic
        for each column.

        --------------
        "sterr" is the aggregated standard error, either the
        root mean squared error (which can be obtained using rmsr())
        or the point biserial correlation weighted standard error
        (obtained using sterrpbc()).  sterr can also be an array
        when calculating a separation statistic for each column.

        --------------
        "estimates" is an array of estimates, set to None if
        the stdev and sterr paramters are not None. 

        --------------
        "nanval" is the not-a-number value.

        NOTE: separation() supports array calculation when
        stdev and sterr are arrays.

    Paste function
    --------------
        separation(stdev,   # [<None, standard deviation of the cell values>]
                   sterr,   # [<standard error, output of rmsr() or sterrpbc()>]
                   estimates = None, # [<None, array of estimates>]
                   nanval = -999 # [<not-a-number value]
                   )
    """    
    if isinstance(stdev, np.ndarray) and isinstance(sterr, np.ndarray):
        diff = np.clip(stdev**2 - sterr**2, 0, np.inf)
        sep = np.sqrt(diff) / sterr
#        sep[np.isnan(sep)] = 0.0
        nix = (stdev == nanval) | (sterr == nanval)
        sep[nix] = nanval
        return sep
    else:
        if stdev is None:
            try:
                stdev = np.std(estimates[estimates != nanval])
            except TypeError:
                exc = "If stdev is None, estimates cannot be None"
                raise separation_Error(exc)
    
        # Calculate separation
        if (stdev**2 - sterr**2) <= 0.:
            sep = 0.
        else:
            sep = np.sqrt(stdev**2 - sterr**2) / sterr
    
        return sep


###########################################################################

def reliability(sepstat,   # [None, output of separation()]
                estimates = None, # [None, array of estimates]
                stdev = None, # [<None, stdev of estimates]
                sterr = None,   # [None, root mean square error, output of rmsr() or sterrpbc()]
                nanval = -999,
                ):
    """Returns the "reliability" of a set of measurements as a Cronbach-alpha.

    Returns
    -------
        Scalar between zero and 1.0 representing the
        reliability of a set of measures, the degree
        to which they are disentangled from error.

    Comments
    --------
        reliability is a conversion of the Separation
        ratio into a number between zero and one.  It answers
        the question, "How orderly are these measures?"  If
        the measures are representative of what can be expected
        of these entities in other testing situations, the
        number indicates the overall likelihood of obtaining
        the same measures from different data.

        It is equivalent to the Cronbach-alpha statistic.

    Arguments
    ---------
        "sepstat" is the Separation statistic output by
        separation().  If this is available, all the remaining
        parameters may be set to None

        --------------
        "estimates" is an array of estimates, set to None if
        the stdev and sterr paramters are not None. Set to None 
        if sepstat is not None.

        --------------
        "stdev" is the standard deviation of the cell estimates.
        They cannot contain NaNVals.  Set to None if sepstat
        is not None.

        --------------
        "sterr" is the aggregated standard error, either the
        root mean squared error (which can be obtained using rmsr())
        or the point biserial correlation weighted standard error
        (obtained using sterrpbc()).  Set to None if sepstat
        is not None.

        --------------
        "nanval" is the not-a-number value.

        NOTE:  reliability() supports array calculation if sepstat
        is an array.

    Paste function
    --------------
        reliability(sepstat,   # [None, output of separation()]
                    estimates = None, # [None, array of estimates]
                    stdev = None, # [<None, stdev of estimates]
                    sterr = None,   # [None, root mean square error, output of rmsr() or sterrpbc()]
                    nanval = -999,
                    )
    """
    if isinstance(sepstat, np.ndarray):
        rel = sepstat**2 / (1 + sepstat**2)
        rel[sepstat == nanval] = nanval
        return rel
    else:
        if sepstat is None:
            if stdev is not None and sterr is not None:
                sepstat = separation(stdev, sterr)
            elif estimates is not None and sterr is not None:
                stdev = np.std(estimates[estimates != nanval])
                sepstat = separation(stdev, sterr)
            else:
                exc = "If sepstat is None, estimates and sterr cannot be None."
                raise reliability_Error(exc)
    
        # Calculate reliability
        rel = sepstat**2 / (1 + sepstat**2)
    
        return rel



###########################################################################

def rmsear(ear,     # [array of cell Expected Absolute residuals (EAR), same size as estimates]
           nanval = -999.    # [Not-a-Number value]
           ):
    """Returns root mean expected absolute residual.

    Returns
    ------
        The square root of the mean of the squared expected absolute residuals
        (EAR) across the range of the array.  When aggregated, these
        are very similar to the root mean squared residuals (RMSR).

    Comments
    --------
        rmsear() is a way to measure fit between observations and estimates
        in an absolute sense (as opposed to a relational sense, as with
        correlations).  It may be thought of in the same way as the
        root mean squared residual, but the root mean square expected absolute
        residual differs in two important respects:

            1) the RMSEAR is based on the EAR which is an expected residual
               rather than an observed residual;

            2) the EAR is calculated for every cell in the matrix, including
               missing cells, so the RMSEAR is based on the whole range;

            3) the individual EAR's for each cell are more precise and
               reliable than their raw counterparts, since they draw information
               from the rest of the array.

        While missing values are not expected in the ear, a nanval argument
        is included in case one pops up.


    Paste function
    --------------
        rmsear(ear,     # [array of cell Expected Absolute residuals (EAR), same size as estimates]
               nanval = -999.    # [Not-a-Number value]
               )

    """
    # Compute RMSEAR
    ValEAR = ear[np.where(ear != nanval)]
    nEAR = float(np.size(ValEAR))
    RMSEAR = np.sqrt(np.sum(ValEAR**2) / nEAR)

    return RMSEAR




###########################################################################

def correl(observed,    # [2-D array of "observed" responses, may include NaNVals]
           estimates,   # [2-D array of estimates, same size as observed, may include NaNVals]
           nanval = -999.,       # [Not-a-Number value]
           form = 'Corr',   # [<'Corr','Cov',,'SumProd','MeanProd'> => formula to apply]
           count = None,   # [<None,True> => report count of valid pairs]
           ):
    """Correlates observations to estimates.

    Returns
    -------
        When form = 'Corr', correl() returns a scalar between -1.0
        and 1.0 representing the correlation between an array of
        observations and an array of estimates.  It can also return
        covariance and mean product.

        When Counts is True, correl() returns a two-element list
        instead of a scalar, where the second element is the count
        of valid pairs that went into the statistic reported as
        the first element:

            [Stat,count]

    Comments
    --------
        correl() is an alternative way to measure fit between observations
        and estimates.  It does not look at how closely they match but
        whether they tend to move up or down similarly.  It is possible
        to have a high correlation even if the observations and estimates
        are very different from each other in absolute terms.

        correl() handles missing data by doing pair-wise deletion of
        corresponding values in the two matrices.  When the percentage
        of missing data is not "a lot", this is reasonable for most
        purposes.  But it is worth noting that correlations computed
        with missing values may not be positive semi-definite, a feature
        of correlation matrices with complete data.  This can cause
        errors with Generalized Least Squares eigenvalue decomposition,
        and other methodologies.  However a Damon decomposition,
        like Singular Value Decomposition, does not require its data
        matrices to be positive semi-definite matrices; it just calculates
        the best-fitting estimates matrix for a given dimensionality.
        This makes it possible for the _bestdim() function to quickly
        determine the dimensionality of a rectangular matrix by analyzing
        its (smaller) square correlation/covariance matrix for a specified
        range of dimensionalities.

        Incidentally, although the correlation matrix of an observed
        array may not (due to missing data) be positive semi-definite,
        the correlation matrix of an estimates array will be.  So Damon
        is a simple way to find the "nearest" positive semi-definite
        matrix to a given non-positive semi-definite matrix.

        Although, correl() calls for "observed" values and "estimates",
        it can be used generically to compute the correlation between
        any two same-sized arrays.

        correl() allows NaNVals in either the observed or estimates arrays.

        correl() uses Numpy's corrcoef(), which is a Pearson correlation,
        or if form = 'Cov' the cov() function.  Covariance is similar to
        correlation, except that it is not standardized.  'MeanProd' is
        an average of products and isn't mean-referenced like covariance
        and correlation.

        To calculate the standard error of a correlation, use
        tools.correl_se().

    Arguments
    ---------
        "observed" is a 2-dimensional array of observed responses (or
        any generic array), and may include NaNVals.

        ------------
        "estimates" is a 2-dimensional array of estimates (or any
        generic array) that is the same size as observed.  It also
        may include NaNVals.

        ------------
        "nanval" is the Not-a-Number value, castable to int.

        ------------
        "form" is the formula to apply:
            form = 'Corr'       =>  Pearson correlation
            form = 'Cov'        =>  Covariance
            form = 'SumProd'    =>  Sum of products, like
                                    covariance but not norm
                                    centered.
            form = 'MeanProd'   =>  Mean of the products.


        ------------
        "count" <None,True>, when True, tells the function to report
        the count of valid pairs used to calculate the statistic
        specified by "form".  (Correlations, covariances, and mean
        products all involve sums of products between pairs of data
        values.  However, if either value in a pair is missing (nanval),
        that product is made missing from the summation.)

        When count = True, correl() returns a two-element list, where
        the first element is the statistic and the second is the count.

    Examples
    --------


    Paste function
    --------------
        correl(observed,    # [2-D array of "observed" responses, may include NaNVals]
               estimates,   # [2-D array of estimates, same size as observed, may include NaNVals]
               nanval = -999.,       # [Not-a-Number value]
               form = 'Corr',   # [<'Corr','Cov','SumProd','MeanProd'> => formula to apply]
               count = None,   # [<None,True> => report count of valid pairs]
               )

    """
    # Convert into 1-dimensional arrays
    obs = np.ravel(observed)
    est = np.ravel(estimates)

    # Filter out cases where either array has a nanval
    nix_obs, nix_est = index_val(obs, nanval), index_val(est, nanval)
    nix = nix_obs | nix_est
    ValObs, ValEst = obs[~nix], est[~nix]

    # Check for div/0 errors
    calc = True
    ValObs_SD = np.std(ValObs)
    ValEst_SD = np.std(ValEst)

    if (ValObs_SD == 0
        or ValEst_SD == 0
        ):
        ObsEst = nanval
        calc = False

    if count is True:
        N = np.size(ValObs)

    # Calc relationship
    if calc is True:
        try:
            if form == 'Corr':
                ObsEst = np.corrcoef(ValObs,ValEst)[0,1]

            if form == 'Cov':
                ObsEst = np.cov(ValObs,ValEst)[0,1]

            if form == 'SumProd':
                ObsEst = np.sum(np.multiply(ValObs,ValEst))

            if form == 'MeanProd':
                ObsEst = np.mean(np.multiply(ValObs,ValEst))

            if np.isnan(ObsEst):
                ObsEst = nanval

        except IndexError:
            ObsEst = nanval

    # return_ list if counting
    if count is True:
        return [ObsEst,N]
    else:
        return ObsEst




###########################################################################

def correl_se(correl,   # [Correlation coefficient]
              count    # [Count of observations]
              ):
    """Compute the standard error of a correlation.

    Returns
    -------
        correl_se() returns a scale equal to the standard error
        of the correlation.

    Comments
    --------
        The (approximate) formula for the standard error of correlation r
        given sample size n is:

            SE[r] = sqrt( (1-r^2)/(n-2) )

        Hotelling, H. (1953). New light on the correlation coefficient and its
        transforms. Journal of the Royal Statistical Society, Series B, 15(2),
        193-232.

        Ghosh, B. K. (1966). Asymptotic expansions for the moments of the
        distribution of correlation coefficient. Biometrika, 53(1/2), 258-262.

    Arguments
    ---------
        "correl" is the correlation coefficient for which the
        standard error is desired.

        "count" is the number of valid data points used to calculate
        the correlation.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        correl_se(correl,   # [Correlation coefficient]
                  count    # [Count of observations]
                  )

    """
    se = np.sqrt((1 - correl**2) / float(count - 2))

    return se




###########################################################################

def stats_per_dim(obj,  # [Damon object]
                  stats,    # [<'Acc','Err','Stab','Speed','NonDegen'>] => stats to calculate]
                  dim,  # [int dimensionality, overwrites coord_args['ndim']]
                  coord_args,   # [dict of coord() arguments for internal use]
                  seed_stat = None,    # [<None,'Acc','Stab','Obj'> => stat used by _bestseed() to decide best seed]
                  nanval = -999,    # [not-a-number value]
                  ):
    """Calculate specified stats per dimension for _bestdim()

    Returns
    -------
        stats_per_dim returns the following statistics as
        a dictionary:

            {'Acc',
            'Err',
            'Stab',
            'Speed',
            'NonDegen',
            'Obj'
            }

        If the function is unable for some reason to compute
        a statistic, it reports it as numpy's nan value.

    Comments
    --------
        stats_per_dim() is primarily a private function used by the
        _bestdim() function within the coord() method.  It calculates
        statistics useful for assessing the "objectivity" of the solution
        for a given dimensionality.

        "Acc" (accuracy), ranging from 0.0 to 1.0, refers to Damon's accuracy in
        predicting pseudo-missing observation at the specified dimensionality.
        (See tools.accuracy() docs.)

        "Err" (psmsresid), ranging from 0.0 to +inf, is similar to accuracy except
        that it averages the residuals between the estimates and pseudo-missing
        observations.

        "Stab" (stability), ranging from 0.0 to 1.0, refers to the stability of
        Damon's coordinate structure at the specified dimensionality.
        (See tools.stability() docs.)

        "Speed", ranging from 0.0 to 1.0, refers to how quickly Damon converges.
        This is sometimes a useful indicator of correct dimensionality.

        "NonDegen" (nondegeneracy), ranging from 0.0 to +inf, is used to flag
        "degenerate" solutions where pseudo-missing estimates tend to be
        significantly different from the observed values in the remainder
        of the observations.

        "Obj" (objectivity) combines accuracy and stability in one statistic.

        stats_per_dim() looks first in the _bestseed() outputs for each statistic,
        and calculates it from scratch if it doesn't find it.

    Arguments
    ---------

        "obj" is a Damon object.  This will include an ndim parameter which
        is automatically overwritten by the "dim" parameter below.

        ------------
        "stats" is the desired list of stats, where the options are:

            ['Acc','Err','Stab','Speed','NonDegen']

            stats = ['Acc','Obj']
                            =>  Calculate and report statistics only
                                for Accuracy and Objectivity.

        ------------
        "dim" (an int) is the dimensionality at which to run coord() internally.
        It overwrites the ndim parameter in coord_args.

        ------------
        "coord_args" are the arguments needed to run the coord() function
        within stats_per_dim() in the event that coord() has not yet been run.

            coord_args = None
                            =>  coord() has already been run for the
                                Damon object and there's no need to run
                                it again.

            coord_args = {'ndim':[[3]],'runspecs'=[0.0001,10],'seed'=1}
                            =>  coord() will be run at dimensionality 3
                                and seed 1 with the specified runspecs.

                                All other args will be the coord() defaults.

        ------------
        "seed_stat" is the statistic used to decide the best seed while finding
        best dimensionality and is assigned by the _bestdim() function.  It is
        not essential but saves _bestdim() the trouble of recalculating
        accuracy or stability if they were already calculated in _bestseed().

        The options are:

            seed_stat = <None,'Stab','Acc','Obj'>

        ------------
        "nanval" is the not-a-number value.  This will generally be the
        same as obj.nanval, but if the data was string it may need to be
        converted to int or float.

    Examples
    --------

        [Under construction]


    Paste Function
    --------------
        stats_per_dim(obj,  # [Damon object]
                      stats,    # [<'Acc','Err','Stab','Speed','NonDegen'>] => stats to calculate]
                      dim,  # [int dimensionality, overwrites coord_args['ndim']]
                      coord_args,   # [dict of coord() arguments for internal use]
                      seed_stat = None,    # [<None,'Acc','Stab','Obj'> => stat used by _bestseed() to decide best seed]
                      nanval = -999,    # [not-a-number value]
                      )

    """
#    floor = 0.0000001
#    ceiling = 0.9999999
    coord_args = coord_args.copy()
    coord_args['ndim'] = [[dim]]

    # Decide whether to calc accuracy in stability()
    if ('Obj' in stats
        or ('Stab' in stats
            and ('Acc' in stats
                 or 'PsMsResid' in stats
                 or 'NonDegen' in stats
                 or 'Speed' in stats
                 )
             )
        ):
        acc_in_stab = ['Accuracy','Err','Speed']

        if 'NonDegen' in stats:
            acc_in_stab.append('NonDegen')
    else:
        acc_in_stab = None

    # Try to get stats from _bestseed() outputs
    try:
        bestseed = obj.seed['BestSeed']
        stats_per_seed = obj.seed['StatsPerSeed']
        loc = np.where(stats_per_seed.core_col['Seed'] == bestseed)[0][0]
    except AttributeError:
        stats_per_seed = None

    # Get or calculate stability
    if 'Stab' in stats or 'Obj' in stats:
        try:
            stability_ = stats_per_seed.core_col['Stab'][loc]
            stab_out = None
        except (AttributeError,KeyError):
            try:
                stab_out = stability(obj,coord_args,acc_in_stab,1,'Auto','Auto',nanval,True)

                try:
                    stability_ = stab_out['Stability']
                except TypeError:
                    stability_ = np.nan

            except stability_in_coord_Error:
                stability_ = np.nan

    else:
        stab_out = None
        stability_ = np.nan

    # Get or calculate accuracy and related stats
    if ('Acc' in stats
        or 'Obj' in stats
        or 'Err' in stats
        or 'NonDegen' in stats
        ):

        # Accuracy
        if 'Acc' in stats:
            try:
                accuracy_ = stats_per_seed.core_col['Acc'][loc]
            except (AttributeError,KeyError):
                try:
                    accuracy_ = stab_out['Accuracy']
                except (TypeError, UnboundLocalError):
                    nondegen = True if 'NonDegen' in stats else None
                    acc_out = accuracy(obj,coord_args,nondegen,nanval)
                    accuracy_ = acc_out['Accuracy']
        else:
            nondegen = True if 'NonDegen' in stats else None
            acc_out = accuracy(obj,coord_args,nondegen,nanval)
            accuracy_ = np.nan

        # Err (pseudo-missing root mean squared resid)
        if 'Err' in stats:
            try:
                psmsresid = stats_per_seed.core_col['Err'][loc]
            except (AttributeError,KeyError):
                try:
                    psmsresid = stab_out['Err']
                except (TypeError,KeyError,UnboundLocalError):
                    try:
                        psmsresid = acc_out['Err']
                    except UnboundLocalError:
                        acc_out = accuracy(obj,coord_args,nondegen,nanval)
                        psmsresid = acc_out['Err']
        else:
            psmsresid = np.nan

        # Nondegeneracy
        if 'NonDegen' in stats:
            try:
                nondegen_ = stats_per_seed.core_col['NonDegen'][loc]
            except (AttributeError,KeyError):
                try:
                    nondegen_ = stab_out['NonDegen']
                except (TypeError,UnboundLocalError):
                    try:
                        nondegen_ = acc_out['NonDegen']
                    except (UnboundLocalError,KeyError):
                        acc_out = accuracy(obj,coord_args,nondegen,nanval)
                        nondegen_ = acc_out['NonDegen']

        else:
            nondegen_ = np.nan

    # Calc convergence speed
    if 'Speed' in stats:
        try:
            speed = stats_per_seed.core_col['Speed'][loc]
        except (AttributeError,TypeError):
            try:
                speed = stab_out['Speed']
            except (TypeError,KeyError,UnboundLocalError):
                try:
                    changelog = obj.coord_out['changelog']
                    speed = 1.0 - np.sum(changelog[3:,1]) / np.sum(changelog[:,1])
                except AttributeError:
                    speed = np.nan
    else:
        speed = np.nan

    # Calc objectivity
    if 'Obj' in stats:
        if (accuracy_ is not np.nan
            and stability_ is not np.nan
            ):
            acc_stab = np.array([np.log(accuracy_),np.log(stability_)])
            objectivity = round(np.exp(np.average(acc_stab)), 3)
        else:
            if accuracy_ is not np.nan:
                objectivity = accuracy_
            else:
                objectivity = np.nan
    else:
        objectivity = np.nan

    return {'Acc':accuracy_,
            'Err':psmsresid,
            'Stab':stability_,
            'Speed':speed,
            'NonDegen':nondegen_,
            'Obj':objectivity
            }



###########################################################################

def accuracy(obj,   # [Damon object]
             coord_args,   # [<None,{'ndim':[[3]],'seed':1,'...}> => coord() args dict for calibrating groups]
             nondegen,  # [<True,None> => calculate non-degeneracy statistic]
             nanval,    # [not-a-number value]
             ):
    """Calculate pseudo-missing predictive accuracy for _bestseed() and _bestdim().

    Returns
    -------
        accuracy() returns a dictionary with three important diagnostic
        statistics computed for pseudo-missing cells:

        {'Accuracy',    =>  correl(est,obs) in 0-1 metric
         'Err',         =>  rmsr(est,obs)
         'NonDegen'     =>  (est_missing - est_nonmissing) / sd_nonmissing,
                             in 0-1 metric
         'Speed'        =>  convergence speed in 0-1 metric
         }

    Comments
    --------
        accuracy() compares estimates and observations for pseudo-missing
        cells (for which cells have been made missing) and generates an
        accuracy statistic (0.0 - 1.0, where 1.0 means accurate), a
        root mean square statistic (0.0 - inf, where 0.0 means accurate),
        and a "nondegeneracy" statistic (0.0 - 1.0, where 1.0 means there
        is no evidence of a degenerate solution.

        Degenerate solutions occur when coordinates produce estimates that
        match the observed values closely but diverge wildly from reality
        when cells are missing.  They are diagnosed by comparing the estimates
        of the missing and non-missing cells.  If they are significantly
        different on average, there is a possibility of a degenerate solution
        and the nondegeneracy statistic will be close to 0.0.

        accuracy() is intended to be primarily a private function, serving
        _bestseed() and _bestdim().

    Arguments
    ---------
        "obj" is a Damon object.

        ------------
        "coord_args" are the arguments needed to run the coord() function
        within accuracy() in the event that coord() has not yet been run.
        The most important are ndim, which must be an integer (e.g., [[3]]),
        and seed.  coord_args must be in dictionary form.

            coord_args = None
                            =>  coord() has already been run for the
                                Damon object and there's no need to run
                                it again.

            coord_args = {'ndim':[[3]],'runspecs'=[0.0001,10],'seed'=1}
                            =>  coord() will be run at dimensionality 3
                                and seed 1 with the specified runspecs.
                                All other args will be the coord() defaults.

        ------------
        "nondegen", if True, tells the function to compute a nondegeneracy
        statistic.  Otherwise, NonDegen will be returned as None.

        ------------
        "nanval" is the not-a-number value.  This will generally be the
        same as obj.nanval, but if the data was string it may need to be
        converted to int or float.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        accuracy(obj,   # [Damon object]
                 coord_args,   # [<None,{'ndim':[[3]],'seed':1,'...}> => coord() args dict for calibrating groups]
                 nondegen,  # [<True,None> => calculate non-degeneracy statistic]
                 nanval,    # [not-a-number value]
                 )

    """

    # Control verbosity
    if obj.verbose is True:
        reset_verbose = True
        obj.verbose = None
    else:
        reset_verbose = False

    # Get or build pseudo-missing index
    try:
        if obj.pseudomiss_out['parsed_psmsindex'] is not None:
            msindex = obj.pseudomiss_out['parsed_psmsindex']
        else:
            msindex = obj.pseudomiss_out['psmsindex']
    except AttributeError:
        size = np.size(obj.coredata)
        if size > 10000:
            rand_nan = 1000. / float(size)
        else:
            rand_nan = 0.10

        obj.pseudomiss('All',rand_nan,None,None,1)
        msindex = obj.pseudomiss_out['psmsindex']

    # Get pseudo-missing observations
    obs = obj.coredata[msindex][:,np.newaxis]

    # Get or calc coordinates
    def getcoords(obj, coord_args):
        if coord_args['seed'] == 'Auto4BestDim':
            coord_args['seed'] = 1

        obj.coord(**coord_args)
        R = obj.coord_out['fac0coord']['coredata']
        C = obj.coord_out['fac1coord']['coredata']

        return {'R':R,'C':C}


    if coord_args is None:
        
        try:
            R = obj.coord_out['fac0coord']['coredata']
            C = obj.coord_out['fac1coord']['coredata']
            rc_out = {'R':R,'C':C}
        except AttributeError:
            rc_out = getcoords(obj, coord_args)
    else:
        rc_out = getcoords(obj, coord_args)
        R = rc_out['R']
        C = rc_out['C']

    # Reset verbose
    if reset_verbose is True:
        obj.verbose = True

    # Get pseudomiss estimates
    range_msindex = xrange(np.size(msindex[0]))
    est = np.array([nanval if ((nanval in R[msindex[0][h]]) | (nanval in C[msindex[1][h]]))
                    else np.dot(R[msindex[0][h]],C[msindex[1][h]])
                    for h in range_msindex]
                   )[:,np.newaxis]

    # Get psms residuals
    resid = residuals(obs,est,None,None,None,nanval)
    valloc = np.where(resid[:,:] != nanval)
    valresid = resid[:,:][valloc]
    psms_resid = round(np.sqrt(np.mean(valresid**2.)),3)

    # Get accuracy
    accuracy = np.clip(correl(obs[valloc],est[valloc],nanval),0.000001,0.999999)

    # Get NonDegeneracy
    if nondegen is True:
        est_whole = np.dot(R,C.T)
        est_whole[np.where(R[:,0] == nanval)] = nanval
        np.transpose(est_whole)[np.where(C[:,0] == nanval)] = nanval
        est_miss = est_whole[msindex]
        est_miss_msq = np.mean(est_miss[est_miss != nanval]**2)

        # Nonmissing observations
        non_miss = obj.coredata[obj.coredata != nanval]
        non_miss_msq = np.mean(non_miss**2)
        non_miss_SD = np.std(non_miss**2)

        # Are missing estimates significantly different from non-missing?
        contrast = abs(est_miss_msq - non_miss_msq) / non_miss_SD
        nondegeneracy = 1 - (contrast / float(1 + contrast))
    else:
        nondegeneracy = None

    # Get convergence speed
    try:
        changelog = obj.coord_out['changelog']
        speed = 1.0 - np.sum(changelog[3:,1]) / np.sum(changelog[:,1])
    except AttributeError:
        speed = None

    return {'Accuracy':accuracy,'Err':psms_resid,'NonDegen':nondegeneracy,'Speed':speed}






###########################################################################

def stability(obj,  # [Damon object]
              coord_args,   # [{'ndim':[[3]],'seed':1,'...} => coord() args dict for calibrating groups]
              stats = ['Accuracy','Err','NonDegen','Speed'],   # [<None,[<'Accuracy','Err','NonDegen','Speed'>]> => calculate accuracy, predictive error, and nondegeneracy]
              facet = 1,    # [<0,1> => facet to split into groups 1 and 2]
              group1 = 'Auto', # [<'Auto',{'Get':'NoneExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}> => extract statement for 'Entities' ('NoneExcept')]
              group2 = 'Auto', # [<'Auto',{'Get':'AllExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}> => complement extract statement for 'Entities' ('AllExcept')]
              nanval = -999,     # [not-a-number value]
              verbose = True,   # [<True,None> => print error message]
              ):
    """Calculate coordinate stability for _bestseed() and _bestdim().

    Returns
    -------
        stability() returns a "Stability" statistics, the correlation
        between the row coordinates calculated from one group of columns
        and those calculated from another group of columns.  For efficiency,
        it is also possible to use stability() to calculate 'Accuracy',
        'Err' (pseudo-missing root mean square residuals), and 'NonDegen'
        statistics:

            {'Stability',
            'Accuracy',
            'Err',
            'NonDegen'
            }


    Comments
    --------
        The "stability" statistic measures the degree to which different
        subgroups of entities yield the same coordinates -- an essential property
        of object-oriented statistics and one of the two criteria (the
        other is "accuracy") used to build Damon's "objectivity" statistic.

        Let us assume that the entity subgroups are columns (facet = 1) and that
        we have we have a 100 row x 80 column array.  The procedure is
        somewhat as follows:

            1.  Divide the array in half by row, the first 50 rows and
                the second 50 rows (actually Damon uses alternating
                rows and columns).

            2.  Apply coord() to just the first 50 rows and calculate
                column coordinates.

            3.  Now divide the array in half by columns, the first
                40 columns and the second 40 columns (alternating
                in real life).

            4.  Calculate row coordinates anchoring on the coordinates
                calculated in Step 2 using data in just the FIRST 40
                columns and the second 50 rows.

            5.  Calculate row coordinates anchoring on the coordinates
                calculated in Step 2 using data in just the SECOND 40
                columns and the second 50 rows.

            6.  Calculate the correlation between the row coordinates
                arrays calculated in Steps 4 and 5.  A correlation of
                1.0 is a strong indicator that the dataset is "objective".
                In other words, a correlation R of 1.0 means that we got
                the same person coordinates regardless of which sample
                of items we used, which is what Damon means by "stable"
                and is an essential condition of objectivity.

        The stability statistic is called by Damon's private _bestseed()
        function to help calculate the best seed and, indirectly, the
        best dimensionality.

        The seed signifies a set of random starter coordinates and is
        passed to stability() through the coord_args 'seed' parameter.
        The _bestseed() function tries out a set of seeds to see which
        yields the highest stability.

        Important
        ---------
        You will run into problems if the groups get too small, i.e., have
        too few observations relative to the number of dimensions.  stability()
        will return None if it determines this has occurred, in which case
        _bestseed() ignores stability when calculating objectivity.

        stability() can also be used to calculate Accuracy and related
        statistics.  Calculation cycles are reduced by calculating Accuracy
        and related statistics from the same coord() run used to compute
        initial coordinates (Step 2 above).  Because only half the dataset
        is used in this case, the results will differ somewhat from
        calculating Accuracy separately.

    Arguments
    ---------
        "obj" is a Damon object, generally passed by the _bestseed() function.

        ------------
        "coord_args" is a dictionary of parameters to feed into stability()'s
        internal coord() call (see Step 2 above).  The most important parameters
        are "ndim" and "seed", both of which must be simple integers:

            coord_args = {'ndim':[[3]],'seed':1}
                            =>  coord() will be run with a dimensionality of
                                3 and a seed of 1.  All other parameters will
                                be the coord() defaults.

                                This parameter won't work if ndim and seed
                                do not refer to single integers:

            coord_args = {'ndim':[range(1,6)],'seed':'Auto'}
                            =>  These parameters won't work and will throw
                                an exception.
        ------------
        "stats" specifies additional statistics that may be calculated using
        the same coord() runs used to calculate stability.  Options are:

            stats = None    =>  Do not calculate additional statistics.

            stats = ['Accuracy','Err','NonDegen','Speed']
                            =>  Calculate Accuracy, Error, Nondegeneracy,
                                and Speed statistics.  You can specify any
                                subset of these.

                                Accuracy is the correlation between observed
                                values for pseudo-missing cells and their
                                estimates.  Error is the root mean squared
                                residuals for the same cells.  Nondegeneracy
                                is a comparison of the estimates for missing
                                and non-missing cells to see if their ranges
                                are comparable.  Speed is the speed of
                                convergence.

        ------------
        "facet" is the facet in terms of which to split the dataset into
        groups.  Generally, you will set facet = 1 (columns), but you may
        want to calculate it in terms of rows sometime.

            facet = 1       =>  Divide the dataset into two groups of columns.
                                The row coordinates calculated from Group 1
                                will be correlated with those calculated from
                                Group 2 to calculate stability.

            facet = 0       =>  Divide the dataset into two groups of rows.
                                The column coordinates calculated from Group 1
                                will be correlated with those calculated from
                                Group 2 to calculate stability.

        ------------
        "group1" uses a variation of the extract() syntax to allow you to
        specify all the columns (or rows) that should go into Group 1.  You
        can do this a lot of ways.  The two groups may be math and language
        items, for instance, in which case you would use the appropriate
        syntax to extract out, say, the math items for group1 and the language
        items for group2.  (Refer to help(dmn.Damon.extract).)  The default,
        however, (when 'Auto' is specified) is to use the "index" method to
        specify which columns belong to a group:

        group1 = 'Auto', is the same as:
        group1 = {'Get':'NoneExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}

                        =>  This means:  Get NO columns EXCEPT those starting
                            after the row headers, and then extract only alternating
                            columns.  Here, cols is a numbered list of all
                            column indices and obj.nheaders4rows is the number
                            of row headers.

                            This uses Numpy's "slice" syntax, which supports
                            specifying a "stride" -- a number of columns to skip.
                            The number 2 means, "select every 2nd column".
                            Consult Numpy's slice documentation for more info.

        Because facet = 1, the "Entities" in this case are columns.  Otherwise,
        they would be rows.

        ------------
        The "group2" syntax is like "group1", except that you want all the entities
        that are NOT group1 entities.  In the example above, this is handled easily
        using the 'Get':'AllExcept' command:

        group2 = 'Auto', is the same as:
        group2 = {'Get':'AllExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}

                        =>  This means:  Get ALL columns EXCEPT those specified
                            by those above in group1 (excluding row headers).

                            Thus, the group2 syntax is just like the group1 syntax
                            except that we use 'AllExcept' instead of 'NoneExcept'.

                            However, just because these are the defaults does not
                            mean you can't extract the two groups any way you please.
                            Nor do they have to comprise all the columns in the array,
                            just enough to compute the necessary coordinates.

        ------------
        "nanval" is the not-a-number value.  This will generally be the
        same as obj.nanval, but if the data was string it may need to be
        converted to int or float.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        stability(obj,  # [Damon object]
                  coord_args,   # [{'ndim':[[3]],'seed':1,'...} => coord() args dict for calibrating groups]
                  stats = ['Accuracy','Err','NonDegen','Speed'],   # [<None,[<'Accuracy','Err','NonDegen','Speed'>]> => calculate accuracy, predictive error, and nondegeneracy]
                  facet = 1,    # [<0,1> => facet to split into groups 1 and 2]
                  group1 = 'Auto', # [<'Auto',{'Get':'NoneExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}> => extract statement for 'Entities' ('NoneExcept')]
                  group2 = 'Auto', # [<'Auto',{'Get':'AllExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}> => complement extract statement for 'Entities' ('AllExcept')]
                  nanval = -999,     # [not-a-number value]
                  verbose = True,   # [<True,None> => print error message]
                  )


    """
    
    warn1 = "Warning in coord()/seed(): Dataset is too small relative to the number of dimensions to calculate a seed stability stat.  Setting seed = 1.\n"
    nrows = np.size(obj.coredata,axis=0) + obj.nheaders4cols
    ncols = np.size(obj.coredata,axis=1) + obj.nheaders4rows
    rows = np.arange(nrows)
    cols = np.arange(ncols)

    # Test ndim
    if len(coord_args['ndim'][0]) > 1:
        exc = 'Unable to use the coord_args ndim parameter.  Must refer to a single integer dimensionality.\n'
        raise stability_in_coord_Error(exc)
    else:
        dim = coord_args['ndim'][0][0]

    # Test seed
    if not isinstance(coord_args['seed'],int):
        #print "Error: coord_args['seed'] = ",coord_args['seed']
        exc = 'Unable to use the coord_args seed parameter.  Must be a single integer.'
        raise stability_in_coord_Error(exc)

    # Interpret groups
    if facet == 1:
        if group1 == 'Auto':
            group1 = {'Get':'NoneExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}

        if group2 == 'Auto':
            group2 = {'Get':'AllExcept','Labels':'index','Entities':cols[obj.nheaders4rows::2]}

    elif facet == 0:
        if group1 == 'Auto':
            group1 = {'Get':'NoneExcept','Labels':'index','Entities':rows[obj.nheaders4cols::2]}

        if group2 == 'Auto':
            group2 = {'Get':'AllExcept','Labels':'index','Entities':rows[obj.nheaders4cols::2]}


    ###################
    ##  Precalibrate ##
    ##   Half Data   ##
    ##    (group0)   ##
    ###################

    if facet == 1:
        row_ind = np.array(range(nrows))[obj.nheaders4cols::2]
        other_row_ind = np.array(range(nrows))[obj.nheaders4cols + 1::2]

        group0 = obj.extract(obj.data_out,
                             getrows = {'Get':'NoneExcept','Labels':'index','Rows':row_ind},
                             getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]},
                             labels_only = None
                             )
    elif facet == 0:
        col_ind = np.array(range(ncols))[obj.nheaders4rows::2]
        other_col_ind = np.array(range(ncols))[obj.nheaders4rows + 1::2]

        group0 = obj.extract(obj.data_out,
                             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                             getcols = {'Get':'NoneExcept','Labels':'index','Cols':col_ind},
                             labels_only = None
                             )

    else:
        exc = 'Unable to figure out seed facet.\n'
        raise stability_in_coord_Error(exc)

    # Handle hd5 format
    if obj.pytables is None:
        format0 = format1 = format2 = 'datadict'
        pytab0 = pytab1 = pytab2 = None
    else:
        format0 = format1 = format2 = 'hd5'
        pytab0 = '_seed0.hd5'
        pytab1 = '_seed1.hd5'
        pytab2 = '_seed2.hd5'


    ############################################################
    # Build group0 Damon object and compute accuracy, if desired
    group0 = dmn.core.Damon(group0,format0,pytables=pytab0,verbose=None)

    if stats is not None:
        size = np.size(group0.coredata)
        if size > 10000:
            rand_nan = 1000. / float(size)
        else:
            rand_nan = 0.10

        group0.pseudomiss('All',rand_nan,None,None,1)

    # Precalibrate group0 col coordinates
    group0.coord(**coord_args)

    # Calc accuracy and pseudomissing residuals
    if stats is not None:
        nondegen = True if 'NonDegen' in stats else False
        acc_out = accuracy(group0,None,nondegen,nanval)
        acc_ = acc_out['Accuracy']
        psms_resid = acc_out['Err']
        nondegen = acc_out['NonDegen']
    else:
        acc_ = None
        psms_resid = None
        nondegen = None

    # Get convergence speed
    try:
        changelog = group0.coord_out['changelog']
        speed = 1.0 - np.sum(changelog[3:,1]) / np.sum(changelog[:,1])
    except AttributeError:
        speed = None


    ############################################################

    # Add all coordinates to bank
    group0.bank(None,
                bankf0 = {'Remove':[None],'Add':['All']},
                bankf1 = {'Remove':[None],'Add':['All']}
                )


    ##############
    ##  group1  ##
    ##  columns ##
    ##############

    # Extract a group of columns
    if facet == 1:
        group1 = obj.extract(obj.data_out,
                             getrows = {'Get':'NoneExcept','Labels':'index','Rows':other_row_ind},
                             getcols = {'Get':group1['Get'],'Labels':group1['Labels'],'Cols':group1['Entities']},
                             labels_only = None
                             )
        # Update key indices (rowlabels shrank to row keys)
        group1['key4rows'] = 0

        # Check size of group relative to ndim
        if np.size(group1['coredata'],axis=1) <= dim:
            if verbose is True:
                print warn1

            # Close hd5 files
            try:
                obj.fileh.close()
                group0.fileh.close()
            except AttributeError:
                pass

            return None

    # Or extract a group of rows
    elif facet == 0:
        group1 = obj.extract(obj.data_out,
                             getrows = {'Get':group1['Get'],'Labels':group1['Labels'],'Rows':group1['Entities']},
                             getcols = {'Get':'NoneExcept','Labels':'index','Cols':other_col_ind},
                             labels_only = None
                             )

        # Update key indices (collabels shrank to col keys)
        group1['key4cols'] = 0

        # Check size of group relative to ndim
        if np.size(group1['coredata'],axis=0) <= dim:
            if verbose is True:
                print warn1

            # Close hd5 file
            try:
                obj.fileh.close()
                group0.fileh.close()
            except AttributeError:
                pass

            return None

    # Convert group1 to Damon object
    group1 = dmn.core.Damon(group1,format1,pytables=pytab1,verbose=None)

    # Calculate target entity coordinates
    group1.coord(ndim = coord_args['ndim'],
                 seed = 1,
                 anchors = {'Bank':group0.bank_out,
                            'Facet':facet,
                            'Coord':'ent_coord',
                            'Entities':['All'],
                            'Freshen':None,
                            },
                 feather = 0.00001
                 )

    ##############
    ##  group2  ##
    ##  columns ##
    ##############

    # Extract a group of columns
    if facet == 1:
        group2 = obj.extract(obj.data_out,
                             getrows = {'Get':'NoneExcept','Labels':'index','Rows':other_row_ind},
                             getcols = {'Get':group2['Get'],'Labels':group2['Labels'],'Cols':group2['Entities']},
                             labels_only = None
                             )
        # Update key indices (rowlabels shrank to row keys)
        group2['key4rows'] = 0

        # Check size of group relative to ndim
        if np.size(group2['coredata'],axis=1) <= dim:
            if verbose is True:
                print warn1

            # Close hd5 files
            try:
                obj.fileh.close()
                group0.fileh.close()
                group1.fileh.close()
            except AttributeError:
                pass

            return None

    # Or extract a group of rows
    elif facet == 0:
        group2 = obj.extract(obj.data_out,
                             getrows = {'Get':group2['Get'],'Labels':group2['Labels'],'Rows':group2['Entities']},
                             getcols = {'Get':'NoneExcept','Labels':'index','Cols':other_col_ind},
                             labels_only = None
                             )

        # Update key indices (collabels shrank to col keys)
        group2['key4cols'] = 0

        # Check size of group relative to ndim
        if np.size(group2['coredata'],axis=0) <= dim:
            if verbose is True:
                print warn1

            # Close hd5 file
            try:
                obj.fileh.close()
                group0.fileh.close()
                group1.fileh.close()
            except AttributeError:
                pass

            return None

    # Convert group2 to Damon object
    group2 = dmn.core.Damon(group2,format2,pytables=pytab2,verbose=None)

    # Calculate target entity coordinates
    group2.coord(ndim = coord_args['ndim'],
                 seed = 1,
                 anchors = {'Bank':group0.bank_out,
                            'Facet':facet,
                            'Coord':'ent_coord',
                            'Entities':['All'],
                            'Freshen':None,
                            },
                 feather = 0.00001
                 )

    ################
    ##  Correlate ##
    ##   Groups   ##
    ################

    if facet == 1:
        g1coord = group1.fac0coord['coredata']
        g2coord = group2.fac0coord['coredata']
    elif facet == 0:
        g1coord = group1.fac1coord['coredata']
        g2coord = group2.fac1coord['coredata']

    # Correlate coordinates of two groups
    corr = round(correl(g1coord,g2coord,nanval), 3)

    # Close hd5 files
    try:
        group0.fileh.close()
        group1.fileh.close()
        group2.fileh.close()
    except AttributeError:
        pass

    return {'Stability':corr,
            'Accuracy':acc_,
            'Err':psms_resid,
            'NonDegen':nondegen,
            'Speed':speed
            }





###########################################################################

def homogenize(arr, # [2-D array to homogenize]
               facet = 1,   # [<0,1> => do by rows or columns]
               form = 'SumProd',   # [<'Corr','Cov','SumProd','MeanProd'> => formula to apply]
               max_ = None, # [<None,int> => maximum number of entities to sample in opposing facet]
               nanval = -999,   # [not-a-number value]
               ):
    """Cross-multiply rows or cols to increase dimensional homogeneity.

    Returns
    -------
        homogenize() returns an array which is generally some
        form of square covariance matrix.

    Comments
    --------
        To address "between-item" multidimensionality or "ragged"
        dimensionalities, it is sometimes helpful to "homogenize" the
        data by replacing it with a form of covariance matrix.  This
        function computes that matrix.

        Although it is possible to compute an actual covariance
        matrix or a correlation matrix (which reduces the dimensionality
        of the data array by 1), the SumProd method (sum of vector
        products) is simpler and just as useful.

        To speed up the calculation, it is possible to constrain the
        amount of data involved in each vector multiplication using
        the max_ option.

    Arguments
    ---------
        "arr" is a 2-dimensional data array.

        ------------
        "facet" specifies whether to cross-multiply by rows (the
        0th array dimension) or columns (the 1th array dimension).
        It generally makes sense to choose the facet with the smallest
        number of elements, simply for performance reasons.  This
        is generally columns (facet = 1).

        ------------
        "form" is the formula to apply:
            form = 'Corr'       =>  Pearson correlation
            form = 'Cov'        =>  Covariance
            form = 'SumProd'    =>  Sum of products, like
                                    covariance but not norm
                                    centered.
            form = 'MeanProd'   =>  Mean of the products.

        ------------
        "max_" sets a maximum to number of opposing entities
        to use in each vector multiplication.

            max_ = None         =>  If facet = 1, use all row
                                    entities when doing the vector
                                    multiplication.

            max_ = 500          =>  If facet = 1, the number of
                                    row entities to use in each
                                    vector multiplication should
                                    not exceed 500.  Sampling
                                    is done by using alternating
                                    rows, as necessary.

        ------------
        "nanval" is the Not-a-Number value, castable to int.


    Examples
    --------

        [under construction]


    Paste Function
    --------------
        homogenize(arr, # [2-D array to homogenize]
                   facet = 1,   # [<0,1> => do by rows or columns]
                   form = 'SumProd',   # [<'Corr','Cov','SumProd','MeanProd'> => formula to apply]
                   max_ = None, # [<None,int> => maximum number of entities to sample in opposing facet]
                   nanval = -999,   # [not-a-number value]
                   )


    """
    # Transpose if necessary
    if facet == 0:
        arr = np.ascontiguousarray(np.transpose(arr))

    # Shorten array
    nrows = np.size(arr,axis=0)
    if max_ is not None:
        if nrows > max_:
            step = nrows / float(max_)
            arr = arr[0:nrows:step,:]

    # Cross-multiply columns
    f_prod = correl
    ncols = np.size(arr,axis=1)
    ran = range(ncols)
    sumprod = [f_prod(arr[:,i],arr[:,j],nanval,form,None)
               for i in ran for j in ran]

    homodata = np.resize(sumprod,(ncols,ncols))

    return homodata



###########################################################################

def rmsr(observed = None,    # [<None,array of "observed" responses>, may include NaNVals]
         estimates = None,   # [<None,array of estimates>, may include NaNVals]
         residuals = None,  # [<None,array of residuals> => used in place of observed, estimates]
         nanval = -999.,       # [Not-a-Number value]
         sqrt_n = False     # [<False, True> => divide by sqrt(n)]
         ):
    """Aggregate residuals-style arrays using root mean squares.

    Returns
    -------
        Scalar representing the root mean squared residual between
        an array of observations and an array of estimates.

        The function can be adapted to any array that is aggregated
        the same way.  Thus, it can return the root mean squared EAR
        and the root mean squared SE.

    Comments
    --------
        rmsr() is a way to measure fit between observations and
        estimates -- as the root mean squared residual, or Euclidean
        norm.  Unlike correlations which measure only the relative
        relationship between two variables, rmsr() looks at how closely
        they match in an absolute sense.

        rmsr() allows NaNVals in either the observed or estimates arrays.

        "observed" and "estimates" can be any two comparable arrays.

        "residuals" can be any array that is appropriately summarized
        by a root mean square statistic.

    Arguments
    ---------
        "observed" is an array of "observed" responses and may include
        NaNVals.  If the residuals argument is used, observed should
        be None.

        -------------
        "estimates" is an array of estimates and may include
        NaNVals.  If the residuals argument is used, estimates
        should be None.

        -------------
        "residuals" is an array of residuals between observed and
        estimated values and may include NaNVals.  If the observed
        and estimates arguments are used, residuals should be None.

        -------------
        "nanval" is the value to indicate Not-a-Number.

        -------------
        "sqrt_n", if True, divides the root mean squared residual by
        the square root of the number of values.  
        
        Note: Keep default as False -- this is assumed throughout the code
        unless explcitly set at True.

    Examples
    --------


    Paste function
    --------------
        rmsr(observed = None,    # [<None,array of "observed" responses>, may include NaNVals]
             estimates = None,   # [<None,array of estimates>, may include NaNVals]
             residuals = None,  # [<None,array of residuals> => used in place of observed, estimates]
             nanval = -999.,       # [Not-a-Number value]
             sqrt_n = False     # [<False, True> => divide by sqrt(n)]
             )

    """

    # Use residuals array, if available
    if residuals is not None:
        ValResid = residuals[np.where(residuals != nanval)]
        sqrt_n = np.sqrt(len(ValResid)) if sqrt_n else 1.0
        RMSRObsEst = np.sqrt(np.mean(ValResid**2)) / sqrt_n

    elif observed is not None and estimates is not None:

        # Convert into 1-dimensional arrays
        RavObs = np.ravel(observed)
        RavEst = np.ravel(estimates)

        # Filter out cases where either array has a nanval
        Valid = np.where(np.logical_and(RavObs != nanval,RavEst != nanval),1,0)
        ValObs = np.compress(Valid == 1,RavObs)
        ValEst = np.compress(Valid == 1,RavEst)

        # Correlate observed and Expected
        sqrt_n = np.sqrt(len(ValObs)) if sqrt_n else 1.0
        RMSRObsEst = np.sqrt(np.mean((ValObs - ValEst)**2)) / sqrt_n

    else:
        exc = 'Unable to figure out arguments.\n'
        raise rmsr_Error(exc)

    return RMSRObsEst


###########################################################################

def group_se(rmse, # [<float> => root mean square error]
             group_se, # [[<str, float, func>, <str, float, func>]] => grouping functions] 
             ndim,  # [<int> => number of dimensions] 
             axis,  # [<0, 1> => axis of entities being aggregated] 
             nrows=None,  # [<None, int> => number of rows behind rmse]
             ncols=None,   # [<None, int> => number of cols behind rmse]
             nanval=-999, # [Not-a-number value]
             ):
    """Apply grouping denominator to root mean square error for aggregation.
    
    Returns
    -------
        A float equal to the root mean square error multiplied by
        a factor calculated with a user-specified formula.
    
    Comments
    --------
        Damon does a somewhat reasonable job estimating standard errors
        for each cell.  However, what one generally wants is the standard
        error for a construct, which is an average of estimates across
        (generally) a group of columns.  We want the standard error of
        each average, of the construct measure.
        
        Ordinarily one obtains such standard errors using:
            
            RMSE / sqrt(n)
        
       the root mean square error across the row divided by the square
       root of the number of columns.  This assumes that the standard
       error for each cell is statistically independent of the other
       cells.  However, that is not the case with Damon's estimate
       and standard error arrays.  Both are calculated using row
       and column coordinates, which means each cell in the row has
       been calculated in part using a common set of row coordinates,
       which means the errors are not statistically independent.  Therefore,
       sqrt(n) is too large.
       
       Alternatively, we could use:
           
           RMSE / sqrt(1)
       
       which assumes that the errors are statistically dependent on each
       other, but this yields a denominator that is too small.
       
       Trial and error suggests that a reasonable denominator is based
       on the number of coordinate dimensions used to model each cell:
           
           RMSE / sqrt(2 * ndim)
       
       While a proof has yet to be derived, it does make sense intuitively.
       Each cell basically has (2 * ndim) degrees of freedom.
       
       However, experiment also suggests there are more factors at play:
           
           *  whether coord()'s final iteration ends up on row coordinates
              or column coordinates
           
           *  what function was used to condition the coordinates
           
           *  the relative number of rows and columns, e.g., whether there
              are many more rows than columns
           
           *  the degree to which the underlying noise fluctuates across
              rows and columns and between facets
           
           *  whether the generating coordinates are all positive
           
           *  whether the dimensionality is 3 or less
           
       These factors distort the standard errors in various ways, and
       failure to account for them can cause the reliability statistics,
       for instance, to be too large or too small.
       
       This function offers a way to specify formulas for calculating
       the denominator and to nudge it appropriately.  The recommended
       formulas are tuned to the following assumptions:
           
           *  coord()'s condcoord['first'] parameter = 1 (default), which
              means that row-based measures and standard errors will tend 
              to be a more accurate than column-based measures.  Persons
              measures will be more accurate than item measures.
           
           *  there will tend to be more, possibly many more, rows than 
              items in a typical dataset, and items will typically be less
              than 200.  A 1000 x 100 array would be typical.
           
           *  the row measures will tend to be noisier than column
              measures, but both will be quite noisy, and the range of 
              noise across columns or rows will tend to be fairly small.
           
           *  dimensionality will tend to be small, less than 5.
       
       Obviously, many datasets will be different from this -- and Damon's
       error estimates will work reasonably well in the great majority
       of scenarios, meaning they will be in roughly the correct range.
       But when it is extremely important to get the errors right, it
       may be necessary to simulate data that looks like the target
       dataset and tweak the formulas to yield optimal errors.
       
       In the common use-case just described, analyzed using default 
       settings, you can expect that the person aggregated standard
       errors will fall in a clump centered on the identity line with 
       plenty of outliers to the right and well below the identity line.
       You can expect the item aggregated standard errors to form a
       noisier clump and a more pronounced band to the right below
       the identity line.  In some cases, it will form a horizonal
       band bisected by the identity line.  This means that given
       these conditions, and a low dimensionality, Damon's errors will
       tend to vary around a single value whereas the "true" errors
       will vary across a wider range. The horizontal band will tend
       to adhere more closely to the identity line as dimensionality
       increases.
   
   Parameters
   ----------
       "rmse" is the root mean square error across a row or column,
       e.g., the output of tools.rmsr().
       
       ----------
       "group_se" is the parameter as input in summstat() or equate(),
       a list specifying an aggregation formula for each axis. Note
       that group_se() will only use one of the formulas, as specified by
       the "axis" parameter below:
           
           [axis 0 (row) formula, axis 1 (col) formula)]
        
       Each formula can be a string, a float, or a function.  The string
       options are:
           
           {'1/sqrt(n)': rmse / np.sqrt(n),
            '1/sqrt(2d)': rmse / np.sqrt(2 * ndim)
            '1/sqrt(2d)*??': rmse / (np.sqrt(2 * ndim) * ??, some number
            '1/sqrt(2d)n**??': rmse / (np.sqrt(2 * ndim) * nrows**??, some number
            }
       
           group_se = None  =>  calculate SE = RMSE / sqrt(1) = RMSE
           
           group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75']
           
                            =>  Multiply rmse by the formula indicated
                                in the string expression. If axis=0, the
                                formula corresponds to: '1/sqrt(2d)n**0.1429'.
                                If axis=1: '1/sqrt(2d)*0.75'
                                
                                Note that the string expression is not 
                                evaluated; it is treated as the string
                                name of a function.
                                
                                However, you CAN adjust the numerical
                                values after the '**' and '*' characters,
                                e.g., '1/sqrt(2d)*1.5. The function will 
                                adapt accordingly.
                                
                                Note that when axis = 0 (calculated person 
                                errors),  a small adjustment needs to be made 
                                for the  number of rows.  This is not needed 
                                when axis = 1.
           
           group_se = [0.05, '1/sqrt(2d)']
           
                            =>  Multiply rmse by 0.05 if axis=0, otherwise
                                by 1 / np.sqrt(2 * ndim)
           
           group_se = [0.05, my_func]
                            =>  Here we pass a function object called 
                                my_func. The result will be:
                                    
                                    se = my_func(rmse)
                                
                                You will need to write other function
                                inputs into the body of the function,
                                but that shouldn't be a problem.
                               
       ----------
       "ndim" is an integer giving the number of dimensions used for
       the analysis.
      
       ----------
       "axis" <0, 1> is a bit tricky.  It refers to the type of entities
       (row or column) being aggregated.  
       
           axis=0            =>  means you are aggregating across rows to 
                                 get column entity standard errors.

           axis=1            =>  you are aggregating across columns to 
                                 get row entity standard errors.  
        
       ----------
       "nrows" <None, int> is the number of rows of the standard errors 
       array, only needed for one of the formulas.

       ----------
       "ncols" <None, int> is the number of columns of the standard errors 
       array.     

       ----------
       "nanval" is the not-a-number value.
       
   Paste Function
   --------------
       group_se(rmse, # [<float> => root mean square error]
                group_se, # [[<str, float, func>, <str, float, func>]] => grouping functions] 
                ndim,  # [<int> => number of dimensions] 
                axis,  # [<0, 1> => axis of entities being aggregated] 
                nrows=None,  # [<None, int> => number of rows behind rmse]
                ncols=None,   # [<None, int> => number of cols behind rmse]
                nanval=-999, # [Not-a-number value]                
                )
        
    """
    
    if group_se is None:
        return rmse
    
    if isinstance(rmse, np.ndarray):
        nix = rmse == nanval
    
    # canned funcs
    def f0(rmse, num):
        return rmse * float(num)
    
    def f1(rmse, n):
        return rmse / np.sqrt(n)
    
    def f2(rmse, ndim):
        return rmse / np.sqrt(2 * ndim)
    
    def f3(rmse, ndim, k):
        return rmse / (np.sqrt(2 * ndim) * k)
    
    def f4(rmse, ndim, n, k):
        nk = n**k
        return rmse / (np.sqrt(2 * ndim) * nk)

    # Apply func to get se
    grp = group_se[axis]
    if isinstance(grp, (int, float)):
        se = f0(rmse, group_se[axis])
            
    elif grp == '1/sqrt(n)':
        if ncols is None:
            exc = ('equate() does not support the "1/sqrt(n)" formula '
                   'for subscales.')
            raise NotImplementedError(exc)
        n = ncols if axis == 0 else nrows
        se = f1(rmse, n)
    
    elif grp == '1/sqrt(2d)':
        se = f2(rmse, ndim)
    
    elif '1/sqrt(2d)*' in grp:
        s = group_se[axis]
        k = float(s[s.find('*') + 1:])
        se = f3(rmse, ndim, k)
     
    elif '1/sqrt(2d)n**' in grp:
        s = group_se[axis]
        k = float(s[s.find('**') + 2:])
        n = nrows if axis == 0 else ncols
        se = f4(rmse, ndim, n, k)
    
    elif callable(grp):
        se = grp(rmse)
    
    else:
        exc = ('Unable to figure out group_se parameter: '
               'group_se={0}, axis={1}'.format(group_se, axis))
        raise ValueError(exc)
    
    if isinstance(rmse, np.ndarray):
        se[nix] = nanval
    
    return se
            

###########################################################################

def fit(observed,    # [None if 'residuals' exists, array of "observed" responses, may include NaNVals]
        estimates,   # [None if 'residuals' exists, array of estimates, same size as observed]
        ear,    # [None if 'fit' is used, array of cell Expected Absolute residuals (EAR), same size as estimates]
        residuals,   # [None if Obs and Est exist, array of signed residuals]
        fit,    # [None, array of fit statistics to be summarized]
        summfit = None, # [<None,'MeanAbs','MeanSq','Perc>2.0'> => calculate summary fit statistic]
        nanval = -999.       # [Not-a-Number value]
        ):
    """Returns cell fit and a summary fit statistics.

    Returns
    -------
        A dictionary with the following:
        'cellfit'       =>  array of fit statistics for
                            individual cells

        'summfit'       =>  an average absolute fit
                            statistic for the whole
                            array

    Comments
    --------
        The formula for fit for a given cell is:

            fit = (observed - Estimate) / EAR

        where the elements in the formula pertain to the values
        for that individual cell and where EAR is the Expected
        Absolute Residual.

        fit is interpreted as the degree to which the observed
        value and the Estimate are "different" from each other
        to a statistically significant degree, i.e., relative
        to the expected residual for that cell.  If the observed
        residual equals the expected residual, fit = 1.0.

        fit > 2.0 implies that the observed value and estimate
        are significantly different at the 95% confidence level.

        By chance, one would expect approximately 5% of the data
        to have fit statistics greater than 2.0 and less than -2.0.

        The summary fit is a summary fit statistic used to
        describe the whole array.  Options:

            None        =>  Do not return a summfit stat.

            'MeanAbs'   =>  report mean absolute fit statistic.
                            This should approximate 1.0.

            'MeanSq'    =>  report mean squared fit statistic.
                            This should approximate 1.0.

            'Perc>2.0'  =>  report the percentage (proportion)
                            of cells with fits greater than 2.0
                            or less than -2.0.  By chance, this
                            should approximate 0.05.

        When fit statistics deviate greater than is expected
        by chance, the data do not fit the Damon model and it
        may be necessary to investigate and possibly make missing
        the misfitting cells.

    Paste function
    --------------
        tools.fit(observed,    # [None if 'residuals' exists, array of "observed" responses, may include NaNVals]
                estimates,   # [None if 'residuals' exists, array of estimates, same size as observed]
                ear,    # [None if 'fit' is used, array of cell Expected Absolute residuals (EAR), same size as estimates]
                residuals,   # [None if Obs and Est exist, array of signed residuals]
                fit,    # [None, array of fit statistics to be summarized]
                summfit = None, # [<None,'MeanAbs','MeanSq','Perc>2.0'> => calculate summary fit statistic]
                nanval = -999.       # [Not-a-Number value]
                )

    """
    # Calculate residuals
    if fit is not None:
        cellfit = fit
    else:
        if residuals is None:
            residuals = np.where(np.logical_or(observed == nanval,estimates == nanval),
                                 nanval, (observed - estimates))

        # Clarify ear
        if isinstance(ear, (int, float)):
            ear = np.zeros(np.shape(residuals)) + float(ear)
        ear = np.where(np.logical_or(ear == nanval, ear < 0.001), nanval, ear)

        # Calculate fit
        cellfit = np.where(np.logical_or(residuals == nanval,ear == nanval),
                           nanval, residuals / ear)

    # Calculate summary fit
    if summfit is not None:
        ValidFit = cellfit[np.where(cellfit != nanval)]
        nFit = np.size(ValidFit)

        if nFit == 0:
            SummFit_ = nanval

        elif summfit == 'MeanAbs':
            SummFit_ = np.mean(np.absolute(ValidFit))

        elif summfit == 'MeanSq':
            SummFit_ = np.sqrt(np.mean(ValidFit**2))

        elif summfit == 'Perc>2.0':
            SummFit_ = np.sum(np.logical_or(ValidFit > 2.0,ValidFit < -2.0)) / float(nFit)

        else:
            print 'Error in tools.fit():  Unable to figure out summfit.\n'
    else:
        SummFit_ = None

    return {'cellfit':cellfit,'summfit':SummFit_}


###########################################################################
## FOR TESTING ONLY -- NOT YET ADOPTED

def fitsd(observed,    # [array of "observed" responses, may include NaNVals]
          estimates,   # [array of estimates, same size as observed]
          nanval = -999.  # [Not-a-Number value]
          ):
    """Returns 1 - standardized residuals.

    Returns
    --------------
        1 - summmary standardized residual

    Comments
    --------------
        FitSD = 1 - [(X - E)/SD(E)]^2

        1.0 means perfect fit.  0.0 means very poor fit.

    Paste function
    --------------
        fitsd(observed,    # [array of "observed" responses, may include NaNVals]
              estimates,   # [array of estimates, same size as observed]
              nanval = -999.  # [Not-a-Number value]
              )

    """


    # Get SD and NaN index
    NaNIndex = np.where(np.logical_or(observed == nanval,estimates == nanval))
    SDEst = np.std(estimates[np.where(estimates != nanval)])

    # Get cell residuals
    cellfit = (observed - estimates)
    cellfit[NaNIndex] = nanval

    #print 'cellfit=\n',cellfit

    # Summarize
    ValidFit = cellfit[np.where(cellfit != nanval)]
    nFit = np.size(ValidFit)
    summfit = np.sqrt(sum(ValidFit**2) / float(nFit))
    SummFitSD = 1 - (summfit/SDEst)**2
    if SummFitSD > 1.0:
        SummFitSD = 1.0
    elif SummFitSD < 0.0:
        SummFitSD = 0.0

    return SummFitSD


###########################################################################

def ptbis(datadict,  # [e.g., self.base_est_out, self.data_out]
          cols2sum = 'All', # [<'All', col attribute name>]
          att_row = 1,   # [<None, row in collabels containing attributes>]
          targ_in_sum = None, # [<None, True> => include target col in sum]
          ):
    """return_ point biserial correlation of each column to a sum of columns.

    Returns
    -------
        ptbis() returns a datadict whose coredata is a 1 x nItems
        array of point biserial correlations -- the correlation
        between an individual column and the sum across
        columns.

    Comments
    --------
        The ptbis() function allows the computation of
        point biserial correlations between each column
        of a data array and a "sum of columns".  It answers
        the statistical question, how closely does this
        item correlate with the test as a whole.  A point
        biserial correlation near zero suggests that the
        item forms its own dimension and does not fit into
        the dimensional subspace of the test as a whole.
        In one-dimensional datasets, this usually indicates
        a "bad" item.  In multidimensional datasets,
        low point-biserial correlations are par for the
        course and have no negative connotation.  They are
        used more to assess the degree to which an item
        is "duplicated" by the rest of the dataset.

        The function supports computing point biserials
        relative to the test as a whole, or for a specified
        subset of the test where the items all share the
        same attribute.  The attribute must appear in an
        extra attribute row in the collabels array.

        To compute point biserials between row entities,
        use the Damon.transpose() method on the datadict
        first.

    Arguments
    ---------
        "datadict" is the input data in Damon's datadict
        format.  It is assumed that columns are the entities
        being correlated.

        -------------
        "cols2sum" is a way to identify which columns to
        sum to create a column of scores, each row of which
        is the sum of observations across the row for the
        specified columns.  The options are:

            'All'       => sum all the observations in the row,
                           across all the columns.

            'Attrib'    => sum only those observations in columns
                           labeled with the specifed 'AttributeName'.

        -------------
        "att_row" is the row in the collabels array that
        contains the column attribute labels.  (This will not
        be the row of entity unique IDs.)  AttRows takes
        an integer, starting from 0, counting from the top.
        Otherwise, it is None.

        -------------
        "targ_in_sum" specifies whether the target col (for which
        we are calculating the correlation) is to be included
        in the sum.  Generally, targ_in_sum should be None, but there
        are situations (when dealing with small clusters of items,
        for example), when including the target might be helpful.
        It is important to realize, however, that this biases the
        correlation upward toward 1.00.

    Examples
    --------


    Paste function
    --------------
        tools.ptbis(datadict,  # [e.g., self.base_est_out, self.data_out]
                  cols2sum = 'All', # ['All' or col attribute name]
                  att_row = 1,   # [None, row in collabels containing attributes]
                  targ_in_sum = None, # [None, True => include target col in sum]
                  )

    """

    # Define variables
    coredata = datadict['coredata']

    rowlabels = datadict['rowlabels']
    nheaders4rows = datadict['nheaders4rows']
    key4rows = datadict['key4rows']
#    rowkeytype = datadict['rowkeytype']

    collabels = datadict['collabels']
    nheaders4cols = datadict['nheaders4cols']
    key4cols = datadict['key4cols']
    colkeytype = datadict['colkeytype']

    validchars = datadict['validchars']

    nanval = datadict['nanval']
    try:
        nanval = float(nanval)
    except:
        exc = 'nanval needs to be castable to float.\n'
        raise ptbis_Error(exc)

    nrows = np.size(coredata,axis=0)
    ncols = np.size(coredata,axis=1)


    # TODO:  This really needs to be optimized
    # Get sum across all columns
    if cols2sum == 'All':
        AllSum = np.zeros((nrows,1))
        PtBis = np.zeros((1,ncols))

        #Mark_ptbis

        # Get sum for each row
        for i in xrange(nrows):
            ValCoreData = coredata[i,:][np.where(coredata[i,:] != nanval)]
            Size = np.size(ValCoreData)
            if Size == 0:
                AllSum[i,:] = nanval
            else:
                AllSum[i,:] = np.sum(ValCoreData,axis=None)

        # Get PtBis for each col
        for j in xrange(ncols):
            TargCol = coredata[:,j][:,np.newaxis]

            if targ_in_sum is None:
                AdjSum = np.where(np.logical_or(TargCol == nanval,AllSum == nanval),nanval,AllSum - TargCol)
            else:
                AdjSum = AllSum

            # Correlate column with sum
            PtBis[0,j] = correl(TargCol,AdjSum,nanval,'Corr',None)

        # Prepare output
        PtBis = np.where(np.isnan(PtBis),nanval,PtBis)
        corner = collabels[:nheaders4cols,key4rows][:,np.newaxis]
        rowlabels = np.append(corner, np.array([['PtBis']]),axis=0)
        collabels = np.append(corner, collabels[:, nheaders4rows:], axis=1)
        
        # NOTE: Had to fix collabels, only did for cols2sum == 'All'
        
        PtBisRCD = {'rowlabels':rowlabels, 'collabels':collabels,
                    'coredata':PtBis,
                    'nheaders4rows':1,'key4rows':0,'rowkeytype':'S60',
                    'nheaders4cols':nheaders4cols, 'key4cols':key4cols,
                    'colkeytype':colkeytype,
                    'nanval':nanval,'validchars':validchars
                    }

    ####################################################
    # Get sum across all items with a specific attribute
    elif cols2sum != 'All':
        Atts = collabels[att_row,nheaders4rows:]
        ColLoc = np.where(Atts == cols2sum)
        AttData = coredata[:,ColLoc[0]]
        nAttCols = np.size(AttData,axis=1)
        PtBis = np.zeros((1,nAttCols))

        # Sum across attribute cols
        AttSum = np.zeros((nrows,1))
        for i in xrange(nrows):
            ValAttData = AttData[i,:][np.where(AttData[i,:] != nanval)]
            Size = np.size(ValAttData)
            if Size == 0:
                AttSum[i,:] = nanval
            else:
                AttSum[i,:] = np.sum(ValAttData,axis=None)

        # Get PtBis for each attcol
        for j in xrange(nAttCols):
            TargCol = AttData[:,j][:,np.newaxis]
            if targ_in_sum is None:
                AdjAttSum = np.where(np.logical_or(TargCol == nanval,AttSum == nanval),nanval,AttSum - TargCol)
            else:
                AdjAttSum = AttSum

            # Correlate column with attribute sum
            PtBis[0,j] = correl(TargCol,AdjAttSum,nanval,'Corr',None)

        # Prepare output
        PtBis = np.where(np.isnan(PtBis),nanval,PtBis)
        rowlabels = np.append(collabels[:nheaders4cols,key4rows][:,np.newaxis],np.array([['PtBis']]),axis=0)
        AttColLabels = np.append(collabels[:nheaders4cols,key4rows][:,np.newaxis],collabels[:,nheaders4rows:][:,ColLoc[0]],axis=1)

        PtBisRCD = {'rowlabels':rowlabels,'collabels':AttColLabels,'coredata':PtBis,
                    'nheaders4rows':1,'key4rows':0,'rowkeytype':str,
                    'nheaders4cols':nheaders4cols,'key4cols':key4cols,'colkeytype':colkeytype,
                    'nanval':nanval,'validchars':validchars
                    }

    # return_ 'All' or 'Attribute' correlations
    return PtBisRCD





###########################################################################

def print_objperdim(data,    # [<Damon object with objperdim attribute>]
                    savetab = 'obj_rpt.txt',
                    savefig = 'obj_rpt.png',  # [<'show', 'filename.png'>]
                    ):
    """Chart and table of objectivity per dimension

    Returns
    -------
        print_objperdim() returns a pretty-printed table of
        accuracy, stability, and objectivity per dimension.  It
        also exports the table and corresponding chart as files.
        To view the chart during runtime, specify show = True.

    Comments
    --------
        The objperdim table can also be exported using the export()
        method applied to my_obj.objperdim(), and this is good for
        working in Excel.

        print_objperdim() is good for getting a stand-alone table
        and chart.

    Arguments
    ---------
        "data" is a Damon object.  You must have run my_obj.coord(...)
        with multiple dimensions, thus creating objperdim output, or
        the function will return an error.

        ------------
        "savetab" is the name of the file or path name to which you
        wish to save the formatted table, e.g., 'obj_rpt.txt'.

        ------------
        "savefig" is the name of the file or path name to which
        you wish to save the chart, e.g., 'obj_rpt.png' or 'obj_rpt.pdf'.

        ------------
        "show" <True, False> allows you to display the chart when you
        do the run.  If working in IDLE, this will cause the program to
        pause and show the chart, then resume when you close the chart.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        print_objperdim(data,    # [<Damon object with objperdim attribute>]
                        savetab = 'obj_rpt.txt',
                        savefig = 'obj_rpt.png',  # [<'show', 'filename.png'>]
                        )

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        exc = 'Unable to find matplotlib.pyplot.'
        raise ImportError(exc)

    d = data
    
    try:
        d.objperdim
    except AttributeError:
        exc = 'Could not find my_obj.objperdim.  You need to specify multiple dimensions in coord().'
        raise print_objperdim_Error(exc)

    # Build chart
    dim = d.objperdim.core_col['Dim']
    acc = d.objperdim.core_col['Acc']
    stab = d.objperdim.core_col['Stab']
    obj = d.objperdim.core_col['Obj']

    plt.clf()
    plt.plot(dim, acc, 'k-.', label='Accuracy')
    plt.plot(dim, stab, 'k--', label='Stability')
    plt.plot(dim, obj, 'k-', label = 'Objectivity')
    plt.legend(loc='best')
    plt.xlabel('Dimensionality')
    plt.ylabel('Objectivity')
    plt.ylim(0, 1.1)
    plt.xticks(dim)

    if savefig == 'show':
        plt.show()
    else:
        plt.savefig(savefig)

    plt.clf()

    # Format table
    tab = tabulate(d.objperdim.whole[:, 1:],
                   headers = 'firstrow',
                   stralign = 'center',
                   numalign = 'center')

    with open(savetab, 'w+') as f:
        f.write(tab)
        
    return tab



###########################################################################
# The following are a batch of color related mini-functions

def check_colors(colors     # [list of matplotlib color codes]
                 ):
    """Check that list of colors is valid

    Returns
    -------
        check_colors() returns True if each of a list of
        colors is valid input to matplotlib, otherwise
        False.

    Comments
    --------
        For purposes of Damon, matplotlib colors can either be one of
        matplotlib's supported colors, a string decimal to indicate
        gray shading, or a (Red, Green, Blue) tuple of proportions to
        create a customized color.  check_colors assures that a list
        of colors fits this format.

        See Damon.plot_two_vars() for more information.

    Arguments
    ---------
        "colors" is a list or array of colors or color codes.

    Example
    -------

        [Under construction]

    Paste Function
    --------------
        check_colors(colors     # [list of matplotlib color codes]
                     )

    """
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    pass_ = False
    
    for val in colors:
        if val not in color_list:
            try:
                float(val)
                pass_ = True
            except (ValueError, TypeError):
                if (isinstance(val, (tuple, list, np.ndarray))
                    and len(val) == 3
                    and np.all((val < 1) & (val > 0))
                    ):
                    pass_ = True
                else:
                    pass_ = False
        else:
            pass_ = True
            
    return pass_


# Functions to look up colors for groups
def try_(lookup, val):
    "Deal with failed lookup"
    try:
        color = lookup[val]
    except KeyError:
        color = '0.30'

    return color
            
def get_group_vals(d, var, ent_axis):
    "Get group values"

    if ent_axis == 'row':
        try:
            vals = d.rl_col[var]
        except KeyError:
            try:
                vals = d.core_col[var]
            except KeyError:
                exc = "Could not figure out groups in 'color_by' parameter"
                raise lookup_group_colors_Error(exc)

    elif ent_axis == 'col':       
        try:
            vals = d.cl_row[var]
        except KeyError:
            try:
                vals = d.core_row[var]
            except KeyError:
                exc = "Could not figure out groups in 'color_by' parameter"
                raise lookup_group_colors_Error(exc)
    return vals
    
def assign_color(c, val):
    "Assign color value to tuple"
    if c == 'r':
        tup = (val, 0.10, 0.10)
    elif c == 'g':
        tup = (0.10, val, 0.10)
    elif c == 'b':
        tup = (0.10, 0.10, val)
    return tup

def lookup_group_colors(d, color_by, ent_axis):
    "Get list of colors coded by group"

    if isinstance(color_by, list):
        var = color_by[0]
        if isinstance(color_by[1], dict):
            lookup = color_by[1]
            vals = get_group_vals(d, var, ent_axis)
            colors = [try_(lookup, val) for val in vals]

        elif tools.check_colors(color_by[1]) is True:
            vals = get_group_vals(d, var, ent_axis)
            try:
                vals = vals.astype(np.float)
                vmin = np.amin(vals)
                vmax = np.amax(vals)
                vals = vals / (vmax - vmin)
            except ValueError:
                exc = "Could not treat group variable as numerical in 'color_by' parameter."
                raise lookup_group_colors_Error(exc)

            c = 'g' if color_by[1] not in ['r', 'g', 'b'] else color_by[1]

            colors = [assign_color(c, val) for val in vals]

        else:
            exc = "Could not figure out 'color_by' parameter."
            raise lookup_group_colors_Error(exc)
    else:
        x = {'row':0, 'col':1}
        n = np.size(d.coredata, axis=x[ent_axis])
        
        if color_by is None:
            colors = ['w' for i in range(n)]
        elif color_by is 'rand':
            colors = [npr.rand(3) for i in range(n)]
        else:
            colors = [color_by for i in range(n)]

        if check_colors(colors) is not True:
            exc = "Could not figure out colors in 'color_by'."
            raise lookup_group_colors_Error(exc)
            
    return colors


###########################################################################

def plot_identity(x, # [array of x values]
                  y, # [array of y values]
                  xy_labels = None, # [array of point labels]
                  title = None, # [None, chart title]
                  xlabel = None, # [None, x-axis label]
                  ylabel = None, # [None, y-axis label]
                  out_as = None, # [None, path]
                  nanval = -999 # Not-a-number value
                  ):
    """Plot two variables together to check for degree of similarity.
    
    Returns
    -------
        plot_identity() returns a dict containing:
            - 'plot': scatterplot showing the similarity between two variables
            - 'corr': their Pearson correlation
            - 'rmse': the root mean square difference from the identity line
            - 'contrast': (mean_x - mean_y) / rmse
    
    Comments
    --------
        This is a convenience function for comparing two variables.  Nanvals
        will be removed.
    
    Arguments
    ---------
        "x" is an array of values to be assigned to the x-axis.  
        
        ----------
        "y" is an array of values to be assigned to the y-axis.

        ----------
        "xy_labels" <None, array> is an array of labels to assign
        to each point.
        
        ----------
        "title" <None, str> is the title to put at the top of the chart. 
        
        ----------
        "xlabel" <None, str> is the label for the x-axis.
        
        ----------
        "ylabel" <None, str> is the label for the y-axis.

        ----------
        "out_as" <str> is the path or filename of the plot figure. No
        need to include the .png extension.
        
        ----------
        "nanval" is the not-a-number value.
    
    Paste Function
    --------------
        plot_identity(x, # [array of x values]
                      y, # [array of y values]
                      xy_labels = None, # [array of point labels]
                      title = None, # [None, chart title]
                      xlabel = None, # [None, x-axis label]
                      ylabel = None, # [None, y-axis label]
                      out_as = None, # [None, path]
                      nanval = -999 # Not-a-number value
                      )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        exc = 'Unable to find matplotlib.pyplot.'
        raise ImportError(exc)

    # Remove nanval, define vars
    valix = (x != nanval) & (y != nanval)
    x, y = x[valix], y[valix]
    min_ = min(np.amin(x), np.amin(y))
    max_ = max(np.amax(x), np.amax(y))

    # Get stats.  rmse is relative to identity line
    corr = round(correl(x, y), 3)
    sin_45 = 0.70710678118
    diffs = sin_45 * np.abs(x - y)
    rmse_ = np.sqrt(np.mean(diffs**2))
    rmse = round(rmse_, 3)
    contrast = (0.0 if rmse == 0 else 
                round((np.mean(x) - np.mean(y)) / rmse_, 3))

    # Plot points with identity line
    plt.clf()
    plt.plot(x, y, 'k.')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.plot([min_, max_], [min_, max_], 'k-')

    if xy_labels is not None:
        for i in range(len(xy_labels)):
            plt.annotate(xy_labels[i], (x[i], y[i]), xytext=(x[i], y[i])) 

    # Add labels
    if title:
        plt.title(title + '\ncorr = ' + str(corr) + 
                  ', rmse = ' + str(rmse) + 
                  ', contrast = ' + str(contrast))
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    # Output plot
    if out_as:
        plt.savefig(out_as)
    else:
        plt.show()
    
    return {'corr':corr, 'rmse':rmse, 'contrast':contrast}



###########################################################################

def plot(A_ids,   # [array of point A identifiers]
         Ax,    # [x-coordinates of each point A]
         Ay,    # [y-coordinates of each point A]
         B_ids = None,     # [None, array of point B identifiers]
         Bx = None, # [<None, x-coordinates of points on line B]
         By = None, # [<None, y-coordinates of points on line B]
         a_err = None,  # [<None, array of SEs for each point A on line a(=x)>]
         b_err = None,  # [<None, array of SEs for each point A on line b>]
         x_name = 'x',  # [label of line a]
         y_name = 'y',  # [label of line b]
         theta = None,  # [<None, angle between line a(=x) and line b>]
         colors = False,    # [<None, array of colors for each point A>]
         plot_params = None     # [dictionary of plot parameters]
         ):
    """Bubble plot with cosine correction

    Returns
    -------
        plot() returns a bubble chart as a figure that other methods
        can show or save.

    Comments
    --------
        plot() supports the Damon.plot_two_vars() method but is
        extracted for possible help in other plotting methods. To
        understand the parameters, the following is assumed:

        "A" is a point in a 2-dimensional x, y space.
        "a" is a line with a human defined meaning defining x.
        "b" is another line which lies in the space, but is not an axis.
        "B" is a point on line b, which lies in the space.

        "Ax" is where A projects on x (= line a)
        "Ay" is where A projects on y.
        "Bx" is where B projects on x (= line a).
        "By" is where B projects on y.

        "a" and "x" are treated as synonymous.
        "y" is the component of b that is orthorgonal to a.

        "a_err" is the standard error of a point A as it projects on a.
        "b_err" is the standard error of a point A as it projects on b.
        (They need not be standard errors.  They can be any measure
        of dispersion around a center.)

        "theta" is the angle (in radians) between lines a and b.

        Note: There cannot be any nanvals in any of the coordinate
        arrays.  Data must be pre-cleaned and the arrays must align.
        The only exception is that the B arrays will have a different
        number of elements than the A arrays.

        plot() allows you to treat a and b as x and y axes of
        a scatterplot, where each point A has a value on those
        axes.  This is the usual way of plotting.  The problem with
        this approach is that if a and b are correlated (as they
        almost certainly are), the resulting scatterplot does not
        place points in their true spatial locations in x, y space.
        They will be crunched together according to the degree of
        correlation.  This is fine for evaluating the strength of
        the a, b relationship, but it is not useful for looking at
        patterns in how points are distributed through space.

        To address this limitation of scatterplots, plot_two_vars()
        calculates a true orthogonal y-axis which does not require
        the points to be crunched in space. The points preserve their
        true spatial locations (within a 2-dimensional slice) and
        line b is shown at the correct angle relative to a.  plot()
        supports this type of graph.

        In addition, plot() supports representing standard errors
        as ellipses around points and can apply different colors
        to each point/ellipse according to its group membership and
        other factors.

        See the Damon.plot_two_vars() method for more information.

    Arguments
    ---------
        "A_ids" is an array of identifiers for each point A.

        ----------
        "Ax" is an array of x-coordinates for each point A.  (No
        nanvals are allowed.)

        ----------
        "Ay is an array of y-coordinates for each point A.

        All the "A" parameters are required.  The following "B"
        parameters and "err" parameters are optional.

        ----------
        "B_ids" is an array of identifiers describing each point
        B on line b.  Can be None.

        ----------
        "Bx" are the x-coordinates for each point B.  Can be None.

        ----------
        "By" are the y-coordinates for each point B.  Can be None.

        ----------
        "a_err" is an array of standard errors for each A's projection
        on line a (= x-axis).  Can be None.

        ----------
        "b_err" is an array of standard errors for each B's projection
        on line b (the line traversing the x, y space).  Can be None.

        ----------
        "x_name" is the string label associated with line a.

        ----------
        "y_name" is the string label associated with line b.

        ----------
        "theta" is the angle, in radians, between lines a and b.

        ----------
        "colors" is an array of colors to apply to each point A.

        ----------
        "plot_params" is a dictionary of plotting parameters that
        can be used to overwrite individual default parameters.


    Paste function
    --------------
        plot(A_ids,   # [array of point A identifiers]
             Ax,    # [x-coordinates of each point A]
             Ay,    # [y-coordinates of each point A]
             B_ids = None,     # [None, array of point B identifiers]
             Bx = None, # [<None, x-coordinates of points on line B]
             By = None, # [<None, y-coordinates of points on line B]
             a_err = None,  # [<None, array of SEs for each point A on line a(=x)>]
             b_err = None,  # [<None, array of SEs for each point A on line b>]
             x_name = 'x',  # [label of line a]
             y_name = 'y',  # [label of line b]
             theta = None,  # [<None, angle between line a(=x) and line b>]
             colors = False,    # [<None, array of colors for each point A>]
             plot_params = None     # [dictionary of plot parameters]
             )

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        exc = 'Unable to find matplotlib.pyplot.'
        raise ImportError(exc)

    # Plot defaults
    plot = {'title':'Auto', # ['Auto', 'My Title']
            'se_size':0.25,     # [0.25 => figure radius = 0.25 * SE]
            'xlabel':x_name,    # [label x variable (=a)]
            'ylabel':y_name,    # [label y variable (=b)]
            'xy_labels':False,   # [<True, False>]
            'label_offset':0,  # [<0.02,...> diagonal offset in points]
            'xy_nchars':5,  # [max number of characters in label]
            'y_line_tick':'ko', # ['ko' => black bullet ticks on y-line (=b)]
            'y_line':'k:',  # [color of y-line (=b)]
            'y_line_ncuts':4,   # [number of ticks on y-line (=b)]
            'xy_marker':'k.',   # [shape and color of xy marker, color overwritten by 'marker_color']
            'markersize':5,     # [size of marker in points]
            'marker_color':'w', # [color of marker]
            'ellipse_color':'rand',     # [color of ellipse or rectangle]
            'transparency':0.50,    # [<0.73> => proportion of transparency]
            'legend':'best',    # [<'upper right', 'lower left', 'best', ...] => location of legend]
            'xlim':'match_xy',  # [<'match_xy', 'min_max', (-3.0, 4.5)> => limit values of x-axis]
            'ylim':'match_xy',  # [See xlim]
            'savefig':'plot_xy.png',    # [<None, 'show', 'my_file.png'> => overwritten by savefig parameter]
            'x_buffer':0.05,    # [Add whitespace on x-axis so points don't hit frame. buff = x_buffer * (max_x - min_x)]
            'y_buffer':0.05,    # [See x-buffer]
            'subplot':111,  # [111 => plot proportions are equal (first 11), for subplot 1 (many possible)]
            'aspect':'equal',   # [<'auto', 'equal'> 'equal' forces square plots]
            'cosine_corrected':False,   # [<True, False>] => two-var cosine corrected plot?]
            'wright_map':False,     # [<True, False> => wright map?]
            'shape':'ellipse'   # [<'ellipse', 'circle', 'rectangle'> => shape of error shading]
            }

    # Overwrite defaults
    if plot_params is not None:
        for key in plot_params.keys():
            plot[key] = plot_params[key]

    # Apply colors, etc.
    plot['ellipse_color'] = colors

    if plot['ellipse_color'] == 'rand':
        plot['ellipse_color'] = npr.rand(3)
        
    # Import matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        exc = 'Unable to import matplotlib.pyplot.  Make sure you have it.'
        raise plot_two_vars_Error(exc)

    from matplotlib.patches import Ellipse, Circle, Rectangle

    # Axes limits
    min_ = {}
    max_ = {}
    for i in ['x', 'y']:
        for j in [Ax, Ay]:
            min_[i] = np.amin(j)
            max_[i] = np.amax(j)
            
    buff_x = plot['x_buffer'] * (max_['x'] - min_['x'])
    buff_y = plot['y_buffer'] * (max_['y'] - min_['y'])              

    # Make x, y match
    if plot['xlim'] == 'match_xy' or plot['ylim'] == 'match_xy':
        all_min = min(min_['x'], min_['y']) 
        all_max = max(max_['x'], max_['y'])
        all_buff = max(buff_x, buff_y)

    # Add buffers and customize
    lim = {}
    for i in ['xlim', 'ylim']:
        if 'match_xy' in [plot['xlim'], plot['ylim']]:
            lim[i] = (all_min - all_buff, all_max + all_buff)
            
        elif plot[i] == 'min_max':
            if 'x' in i:
                lim[i] = (min_['x'] - buff_x, max_['x'] + buff_x)
            elif 'y' in i:
                lim[i] = (min_['y'] - buff_y, max_['y'] + buff_y)
        else:
            if 'x' in i:
                lim[i] = (plot[i][0] - buff_x, plot[i][1] + buff_x)
            elif 'y' in i:
                lim[i] = (plot[i][0] - buff_y, plot[i][1] + buff_y)

    # Build figure
    plt.clf()
    fig = plt.figure(1, figsize=(1, 1), dpi=80)
    ax = fig.add_subplot(plot['subplot'],
                         aspect = plot['aspect'],
                         xlim = lim['xlim'],
                         ylim = lim['ylim']
                         )
    
    #ax.spines['left'].set_position(plot['spines'])
    #ax.spines['right'].set_visible(False)
    
    # Create error-based ellipses
    sin_theta = 1.0 if theta is None else np.sin(theta)
    width = buff_x * 0.10 #np.std(Ax[Ax < 0]) * 0.25

    if plot['shape'] == 'ellipse':
        shapes = [Ellipse(xy = (Ax[i], Ay[i]),
                        width = 2 * a_err[i] * plot['se_size'], 
                        height = 2 * b_err[i] * sin_theta * plot['se_size'],
                        edgecolor = 'none')
                for i in range(len(Ax))]

    # TEMPORARY:  Assumes error is in b_err
    elif plot['shape'] == 'circle':
        shapes = [Circle((Ax[i], Ay[i]),
                         b_err[i] * plot['se_size'],
                         edgecolor = 'none')
                for i in range(len(Ax))]

    # TEMPORARY:  Assumes error is in b_err
    elif plot['shape'] == 'rectangle':
        shapes = [Rectangle((Ax[i] - width/2., Ay[i] - (b_err[i] * plot['se_size'])/2.),
                            width,
                            b_err[i] * plot['se_size'],
                            edgecolor = 'none')
                for i in range(len(Ax))]
        
        
    # Shade ellipes
    for i, e in enumerate(shapes):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(plot['transparency'])
        e.set_facecolor(plot['ellipse_color'][i])

    # Plot entity points
    ax.plot(Ax, Ay, plot['xy_marker'],
            markersize = plot['markersize'],
            markerfacecolor = plot['marker_color'],
            markeredgecolor = plot['marker_color']
            )

    # Label points
    off = plot['label_offset']
    if plot['xy_labels'] is True:
        for i in range(len(A_ids)):
            plt.annotate(A_ids[i], (Ax[i], Ay[i]), xytext=(Ax[i] + off, Ay[i] + off)) 


    # Plot Bx, By if available
    if Bx is not None and By is not None:
        ax.plot(Bx, By, plot['y_line_tick'])
        ax.plot(Bx, By, plot['y_line'], label=y_name)

        for i in range(len(B_ids)):
            ax.annotate(B_ids[i], (Bx[i], By[i]), xytext=(Bx[i] + 0.005, By[i] + 0.005),
                        weight = 'bold')

    # Plot origin point
    ax.plot(0, 0, 'k+', markersize=30)

    # Title parameters
    err_text = '\nShaded area = SE*'+str(plot['se_size']) if a_err is not None else ''
    cos_text = ', Cosine-Corrected' if Bx is not None else ''
    
    if plot['title'] == 'Auto':
        plot['title'] = str(x_name)+' vs '+str(y_name)+cos_text+err_text

    if plot['cosine_corrected'] is True:
        plot['ylabel'] = str(y_name) if Bx is None else 'Orthogonal Component of '+str(y_name)

    # Plot
    plt.title(plot['title'])
    plt.xlabel(str(plot['xlabel']))
    plt.ylabel(str(plot['ylabel']))
    ax.legend(loc=plot['legend'])

    return fig
    


def length(var,    # [1-d array of data or column coordinates]
           coord = True,   # [<True, False> => are x and y col coordinates or data]
           nanval = -999    # [not-a-number value]
           ):
    """Compute the "length" of a vector or variable.
    
    Returns
    -------
        Float representing either the vector distance or
        the statistical standard deviation of a variable.
    
    Comments
    --------
        length() is used in the cosine() function but also
        stands on its own as a way to calculate the length
        of a coordinate vector.  When coord = True:
        
            length[var] = sqrt(sumsq(var))

        However, for the coordinates in this case to be valid, the 
        coordinates for the opposing facet must be orthonormal.  By default,
        Damon.coord() makes its row coordinates othonormal, which
        means that the length() result is valid only when the
        variable corresponds to a column entity.  The Damon.coord()
        default can be changed using the condcoord_ parameter.
        
        When coord is False, length is interpreted as the statistical
        standard deviation of the variable.

    Arguments
    ---------
        "var" is a 1-dimensional array of either coordinates 
        or data.  If coordinates, they cannot contain nanvals.

        ----------
        "coord = True" flags the variables as vector coordinates,
        resulting in a different length formula.

        ----------
        "nanval" is the not-a-number value.

    Examples
    --------

        None
    
    Paste Function
    --------------
        length(var,    # [1-d array of data or column coordinates]
               coord = True,   # [<True, False> => are x and y col coordinates or data]
               nanval = -999    # [not-a-number value]
               )

    """
    if coord is True:
        if nanval in var1 or nanval in var2:
            exc = 'Coordinates not allowed to contain nanvals'
            raise cosine_Error(exc)
        
        length = np.sqrt(np.sum(var**2))

    else:
        length = np.stdev(var[var != nanval])

    return length


###########################################################################

def cosine(var1,    # [1-d array of data or column coordinates]
           var2,    # [1-d array of data or column coordinates]
           coord = True,   # [<True, False> => are x and y col coordinates or data]
           nanval = -999    # [not-a-number value]
           ):
    """Compute cosine between two variables.

    Returns
    -------
        Float representing either the vector cosine
        or the statistical correlation between
        two variables.

    Comments
    --------
        cosine() mainly supports the plot_two_vars() function,
        which has the option of correcting for the cosine
        between two variables when plotting them against each
        other, thus giving a more accurate representation of
        their "true" spatial locations.  When coord = True,
        the formula is:
        
            cosine[1, 2] = dot(var1, var2) / (length(var1) * length(var2))
        
        where var1 and var2 are spatial coordinates.

        However, for the coordinates to be valid, the coordinates
        for the opposing facet must be orthonormal.  By default,
        Damon.coord() makes its row coordinates othonormal, which
        means that the cosine() result is valid only when the
        variables correspond to column entities.  The Damon.coord()
        default can be changed using the condcoord_ parameter.

        cosine() has the option of computing the cosine as
        simply the statistical (Pearson) correlation between
        two variables, but this is only valid to the degree
        that the distribution of entities in space is multivariate
        normal.  When computed using the coordinates between
        two variables, the cosine is no longer (as) sample
        dependent -- though this property is still under
        investigation.


    Arguments
    ---------
        "var1" and "var2" are 1-dimensional arrays of
        either vector coordinates (columns usually) or data.  
        If coordinates, they cannot contain nanvals.  

        ----------
        "coord = True" flags the variables as vector coordinates,
        resulting in a different cosine formula.

        ----------
        "nanval" is the not-a-number value.

    Examples
    --------

        None

    Paste Function
    --------------
        cosine(var1,    # [1-d array of data or column coordinates]
               var2,    # [1-d array of data or column coordinates]
               coord = True,   # [<True, False> => are x and y col coordinates or data]
               nanval = -999    # [not-a-number value]
               )

    """

    if coord is True:
        if nanval in var1 or nanval in var2:
            exc = 'Coordinates not allowed to contain nanvals'
            raise cosine_Error(exc)
        
#        d_var1 = np.sqrt(np.sum(var1**2))
#        d_var2 = np.sqrt(np.sum(var2**2))
        
        d_var1 = length(var1, True, nanval)
        d_var2 = length(var2, True, nanval)
        cos = np.dot(var1, var2) / (d_var1 * d_var2)

    else:
        cos = correl(var1, var2, nanval, 'Corr', None)

    return cos
        


###########################################################################

def lookup_coords(obj,     # [Damon object]
                  x_name,
                  y_name,
                  ent_type = 'col_ents'
                  ):
    """Get component items for specified stats.

    Returns
    -------
        An dictionary of x-coordinate and y-coordinate
        arrays:

            {'x':_,
             'y':_
             }

        The arrays may either be coordinates for individual
        entities or averaged across multiple entities

    Comments
    --------
        lookup_coords() supports the plot_two_vars() method.  It's
        purpose is to return coordinates for two specified entities
        or groups of entities, as with a subscale.

        In general, lookup_coords() is used to get the coordinates
        of column entities such as items for an important mathematical
        reason: row coordinates, by default, are orthonormal which
        makes it possible to calculate a valid cosine from the column
        coordinates.  It is possible to use lookup_coords() to get
        row entity coordinates, but this is only valid if the column
        coordinates have been made orthonormal using coord()'s
        cond_coord_ argument.

    Arguments
    ---------
        "obj" is a Damon object on which the coord() method has
        been run.

        -----------
        "x_name" is a string name of an entity or subscale for which
        identifiers exist in the coord() outputs.  x_name may also
        be a subscale with the prefix 'Mean_' as output by the
        merge_summstat() method.  Generally, x_name should refer to
        a column entity, such as items.

            x_name = 'Item_1'       =>  Get coordinates for 'Item_1'

            x_name = 'Math'         =>  Get the mean of coordinates
                                        across all "Math" items

            x_name = 'Mean_Math'    =>  Same as above

        -----------
        "y_name" defines a second variable for purposes of comparison
        with x_name and is otherwise the same.

        -----------
        "ent_type" gives the type of entity for which coordinates
        are to be obtained, row entities or column entities.  ent_type
        should almost always refer to column entities.

            ent_type = 'row_ents'   =>  x_name and y_name refer to row
                                        entities, such as persons.

            ent_type = 'col_ents'   =>  x_name and y_name refer to column
                                        entitites, such as items.

    Examples
    --------

        [Under construction]

    Paste Function
    --------------
        lookup_coords(obj,     # [Damon object]
                      x_name,
                      y_name,
                      ent_type = 'col_ents'
                      )

    """

    # Check that coordinates exist
    fac = {'row_ents':'fac0coord', 'col_ents':'fac1coord'}
    try:
        coord = obj.coord_out[fac[ent_type]]
    except AttributeError:
        exc = 'Unable to find coord() outputs.'
        raise lookup_coords_Error(exc)

    # Convert coord to Damon object
    d = dmn.core.Damon(coord, 'datadict', 'RCD_dicts', verbose=None)

    # Define function to extract coordinates
    def extract_coords(d, name):
        try:
            z = d.core_row[name]
        except KeyError:
            if 'Mean_' in name:
                sub = name[name.find('Mean_') + 5:]
            else:
                sub = name

            ind = np.where(d.rowlabels[1:, :] == sub)[1][0]
            d_z = d.extract(d,
                            getrows = {'Get':'NoneExcept', 'Labels':ind, 'Rows':[sub]},
                            getcols = {'Get':'AllExcept', 'Labels':'key', 'Cols':[None]})
            
            if d.nanval in d_z['coredata']:
                exc = 'Found nanval in coordinates'
                raise lookup_coords_Error(exc)

            z = np.mean(d_z['coredata'], axis=0)

        return z
            
    x = extract_coords(d, x_name)
    y = extract_coords(d, y_name)

    return {'x':x, 'y':y}



###########################################################################

def check_datadict(d, data_, names, axis, range_):
    "Format a datadict as a Damon object and check for variables"

    # Convert data to Damon object
    try:
        data = d.__dict__[data_]
    except TypeError:
        data = d

    if isinstance(data, dmn.core.Damon):
        if data.core_col is None:
            dat = dmn.core.Damon(data.data_out, 'datadict', 'RCD_dicts', verbose=None)
        else:
            dat = data
    else:
        dat = dmn.core.Damon(data, 'datadict', 'RCD_dicts', verbose=None)

    # Check that names are in keys
    x = {'row':'Row', 'col':'Col'}
    keys = getkeys(dat, x[axis], range_, 'Auto', None)

    for name in names:
        if name not in keys:
            exc = 'Unable to find variable names in data.  Check key type.'
            raise check_datadict_Error(exc)
            
    return dat




###########################################################################

def frequencies(data, # [<array, datadict, Damon object>]
                axis = 'Col', # [<'All', 'Row', 'Col'> => axis for which to get frequencies]
                report = 'p_valid', # [<'count', 'p_valid', 'p_all'> => reporting metric]
                cell_delimiter = None, # [<None, ',' ...> => string character used to delimit values within a cell]
                missing = None,     # [<None, ['*','',...]> => list of characters that signify missing]
                round_p = 3  # [<None, n decimal places> => round proportions]
                ):
    """Obtain row-wise or column-wise response frequencies

    Returns
    -------
        frequencies() returns a nested dictionary of frequencies
        giving the count or proportion of unique values associated
        with a given row or column entity.

        To pretty print the responses frequencies for a given item,
        use the tabulate() function:

        >>>  from tabulate import tabulate
        >>>  freqs = dmnt.frequencies(...)
        >>>  print tabulate(sorted(freqs['Item1'].items()),
                            headers = ['Item1', 'p'])
        
    Comments
    --------
        This tool is useful for getting frequencies of values
        in an array or Damon object.  Frequencies can be obtained
        for each row entity, for each column entity, or for the
        array as a whole.  They may be reported as counts, as
        proportions of all values within the specified range, or
        as proportions only of valid values (ignoring missing).

        frequencies() has another very useful feature.  It occasionally
        happens that, as in a survey, multiple responses are recorded
        for a given item or column.  frequencies() looks inside each
        cell and parses multiple responses into separate values to
        allow them to be counted.  This saves you from the trouble of
        building an array with a separate column for every possible
        response.

    Arguments
    ---------
        "data" may be an array, datadict, or Damon object.

        ----------
        "axis" specifies whether to get frequencies for row entities,
        column entities, or the whole array:

            axis = 'Row'        =>  Get frequencies for row entities.

            axis = 'Col'        =>  Get frequencies for column entities.

            axis = 'All'        =>  Get one set of frequencies for the
                                    whole array.
        ----------
        "report" specifies the output metric of the frequencies:

            report = 'count'    =>  Return counts of values.

            report = 'p_valid'  =>  Return proportions where the counts
                                    are divided by the number of valid
                                    values, ignoring missing.  With this
                                    option, the proportion of missing cells
                                    is not reported.

            report = 'p_all'    =>  Return proportions where the counts
                                    are divided by all values, including
                                    missing.

        ----------
        "cell_delimiter" tells how to split up the values in a cell if
        the cell contains multiple values.

            cell_delimiter = None
                                =>  There are no multiple values in any
                                    cells.

            cell_delimiter = ','
                                =>  Where there are multiple values in a
                                    cell, they are separated by commas.

        ----------
        "missing" specifies a list of characters to treat as missing.  All
        such occurrences will be changed to the nanval assigned to the
        "data" Damon object, or to '-999' if data is an array.

            missing = None      =>  No characters are to be specified as
                                    missing other than those that are already
                                    coded that way.

            missing = ['', '*'] =>  Anytime a blank cell occurs, or an asterisk,
                                    convert it to nanval.  NOTE: When there are
                                    multiple values in a cell, it is assumed that
                                    all are to be treated as non-missing.
                                    
        ----------
        "round_p" gives the option of rounding proportions to a desired
        number of decimal places.

            round_p = None      =>  Do not round proportions.

            round_p = 3         =>  Round proportions to three decimal places


    Examples
    --------

        [Under construction]
                

    Paste Function
    --------------
        frequencies(data, # [<array, datadict, Damon object>]
                    axis = 'Col', # [<'All', 'Row', 'Col'> => axis for which to get frequencies]
                    report = 'p_valid', # [<'count', 'p_valid', 'p_all'> => reporting metric]
                    cell_delimiter = None, # [<None, ',' ...> => string character used to delimit values within a cell]
                    missing = None,     # [<None, ['*','',...]> => list of characters that signify missing]
                    round_p = 3,  # [<None, n decimal places> => round proportions]
                    )
    """

    # Define frequencies function
    def get_freqs(arr, report, round_p, cell_delimiter, nanval):
        "Get counts of uniques in array"

        if len(np.shape(arr)) == 2:
            arr = np.ravel(arr)

        # Merge cells with multiple values all into the same array
        if cell_delimiter is not None:
            has_delim = [cell_delimiter in arr[i] for i, j in enumerate(arr)]
            if any(has_delim):
                vals_ = [val.strip('"').split(cell_delimiter) for val in list(arr)]
                vals = []
                [vals.extend(val_list) for val_list in vals_]
                arr = np.array(vals)
            
        # Get uniques
        uniques = np.unique(arr)

        # Get denominator
        if report == 'count':
            denom = 1.0
            round_p = 0
        elif report == 'p_valid':
            denom = np.sum(~index_val(arr, nanval))
        elif report == 'p_all':
            denom = np.size(arr)
        else:
            exc = 'Unable to figure out report parameter.'
            raise frequencies_Error(exc)

        freqs = {}
        for unique in uniques:
            freqs[unique] = round(np.sum(arr == unique) / np.float(denom), round_p)

        if report == 'p_valid':
            try:
                del freqs[nanval]
            except KeyError:
                pass

        return freqs


    def compile_freqs(data, axis, report, round_p, cell_delimiter):
        "Build dictionary of frequencies for each row, col entity"

        if axis in ['Row', 'Col']:
            keys = getkeys(data, axis, 'Core', 'Auto', None)
            if axis == 'Row':
                lookup = data.core_row
            elif axis == 'Col':
                lookup = data.core_col

            dicts = [get_freqs(lookup[key], report, round_p, cell_delimiter, data.nanval)
                     for key in keys]
            freqs = dict(zip(keys, dicts))

        elif axis == 'All':
            freqs = {}
            freqs['All'] = get_freqs(data.coredata, report, round_p, cell_delimiter, data.nanval)

        return freqs


    # Format data
    if isinstance(data, np.ndarray):
        if len(np.shape(data)) == 1:
            data = data[:, np.newaxis]
        data = dmn.core.Damon(data, 'array', 'RCD_dicts',
                              nheaders4rows = 0,
                              nheaders4cols = 0,
                              nanval = '-999',
                              missingchars = missing,
                              verbose = None)

    elif isinstance(data, dict) and 'coredata' in data.keys():
        data = dmn.core.Damon(data, 'datadict', 'RCD_dicts', missingchars=missing, verbose=None)
        
    elif isinstance(data, dmn.core.Damon):
        data = dmn.core.Damon(data.data_out, 'datadict', 'RCD_dicts', missingchars=missing, verbose=None)
            
    else:
        exc = 'Unable to figure out data format.'
        raise frequencies_Error(exc)

    # Get the frequencies for each row, col entity, or for the whole array    
    freqs = compile_freqs(data, axis, report, round_p, cell_delimiter)

    return freqs


###########################################################################

def count(arr, 
          axis=None, 
          nanval=-999
          ):
    """Get count of valid values in array.
    
    Returns
    -------
        count() returns counts of valid values along a specified 
        array axis.  If axis is None, count() returns the count
        of valid values in the whole array as a scalar.
    
    Comments
    --------
        This wraps numpy's masked array count() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the mean should
        be calculated.
            
            axis = 0        =>  calculate the count of each column.
            
            axis = 1        =>  calculate the count of each row.
            
            axis = None     =>  calculate count of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    --------------
        count(arr, 
              axis=None, 
              nanval=-999
              )
    """
    a = npma.MaskedArray(arr, arr == nanval)
    
    return npma.count(a, axis)


###########################################################################

def amin(arr, 
         axis=None, 
         nanval=-999
         ):
    """Get minimum of array.
    
    Returns
    -------
        amin() returns a 1-dimensional array of minima for 
        a specified axis.  If axis is None, amin() returns
        the minimum of the unraveled array as a scalar.  Nanvals
        are omitted.
    
    Comments
    --------
        This wraps numpy's masked array min() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the mean should
        be calculated.
            
            axis = 0        =>  get the min of each column.
            
            axis = 1        =>  get the min of each row.
            
            axis = None     =>  get the min of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    -------------
        amin(arr, 
             axis=None, 
             nanval=-999
             )
    
    """
    a = npma.MaskedArray(arr, arr == nanval)
    
    if axis is None:
        return float(npma.min(a, axis))
    else:
        return npma.min(a, axis).filled(nanval)


###########################################################################

def amax(arr, 
         axis=None, 
         nanval=-999
         ):
    """Get maximum of array.
    
    Returns
    -------
        amax() returns a 1-dimensional array of maxima for 
        a specified axis.  If axis is None, amax() returns
        the maximum of the unraveled array as a scalar.  Nanvals
        are omitted.
    
    Comments
    --------
        This wraps numpy's masked array max() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the mean should
        be calculated.
            
            axis = 0        =>  get the max of each column.
            
            axis = 1        =>  get the max of each row.
            
            axis = None     =>  get the max of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    -------------
        amax(arr, 
             axis=None, 
             nanval=-999
             )
    
    """
    a = npma.MaskedArray(arr, arr == nanval)
    
    if axis is None:
        return float(npma.max(a, axis))
    else:
        return npma.max(a, axis).filled(nanval)
        


###########################################################################

def diff(a,
         b,
         nanval=-999
         ):
    """Get difference between two arrays.
    
    Returns
    -------
        diff() returns a 2-dimensional array of differences with
        nanvals.
    
    Comments
    --------
        
    
    Arguments
    ---------
        "a" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "b" is a 1- or 2-dimensional array that may contain nanvals.
        It must be the same shape as "a".
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    -------------
        diff(a,
             b,
             nanval=-999
             )
    
    """
    a = npma.MaskedArray(a, a == nanval)
    b = npma.MaskedArray(b, b == nanval)

    diff = a - b
    
    return diff.filled(nanval)
    

    
###########################################################################

def mean(arr, 
         axis=None, 
         nanval=-999
         ):
    """Get mean of array.
    
    Returns
    -------
        mean() returns a 1-dimensional array of means calculated
        along a specified axis.  If axis is None, mean() returns
        the scalar mean of the unraveled array.  Nanvals are
        omitted from the calculation.
    
    Comments
    --------
        This wraps numpy's masked array mean() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the mean should
        be calculated.
            
            axis = 0        =>  calculate the mean of each column.
            
            axis = 1        =>  calculate the mean of each row.
            
            axis = None     =>  calculate mean of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    -------------
        mean(arr, 
             axis=None, 
             nanval=-999
             )
    
    """
    a = npma.MaskedArray(arr, arr == nanval)
    
    if axis is None:
        return float(npma.mean(a, axis))
    else:
        return npma.mean(a, axis).filled(nanval)


###########################################################################

def median(arr, 
           axis=None, 
           nanval=-999
           ):
    """Get median of array.
    
    Returns
    -------
        median() returns a 1-dimensional array of medians calculated
        along a specified axis.  If axis is None, median() returns
        the scalar median of the unraveled array.  Nanvals are
        omitted from the calculation.
    
    Comments
    --------
        This wraps numpy's masked array median() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the median should
        be calculated.
            
            axis = 0        =>  calculate the median of each column.
            
            axis = 1        =>  calculate the median of each row.
            
            axis = None     =>  calculate median of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    -------------
        median(arr, 
               axis=None, 
               nanval=-999
               )
    
    """
    a = npma.MaskedArray(arr, arr == nanval)
    
    if axis is None:
        out = float(npma.median(a, axis))
    else:
        out = npma.median(a, axis).filled(nanval)

    return out


###########################################################################

def std(arr, 
        axis=None, 
        nanval=-999
        ):
    """Get std of array.
    
    Returns
    -------
        std() returns a 1-dimensional array of standard deviations 
        calculated along a specified axis.  If axis is None, std() returns
        the scalar mean of the unraveled array.  Nanvals are
        omitted from the calculation.

        In order to support functions that look for variation in
        strings, std() automatically converts strings to ints and
        takes the standard deviations of those.  They don't mean
        much mathematically, but any standard deviation greater than
        0.0 means the strings are not all of the same value, which
        is sometimes useful to know.
        
    Comments
    --------
        This wraps numpy's masked array std() function.
    
    Arguments
    ---------
        "arr" is a 1- or 2-dimensional array that may contain nanvals.
        
        --------
        "axis" <0, 1, None> is the axis along which the standard deviation 
        should be calculated.
            
            axis = 0        =>  calculate the standard deviation of each column.
            
            axis = 1        =>  calculate the standard deviation of each row.
            
            axis = None     =>  calculate standard deviation of the whole array.
        
        --------
        "nanval" is the not-a-number value.        
            
            
    Paste Function
    --------------
        std(arr, 
            axis=None, 
            nanval=-999
            )
    
    """
    # Do a procedure for strings
    if arr.dtype.char == 'S':
        uniques = np.unique(arr)
        d = {}
        for i, char in enumerate(uniques):
            d[char] = i
        d[nanval] = nanval
        
        e = np.zeros(np.shape(arr))
        for char in uniques:
            e[arr == char] = d[char]
        arr = e
        nanval = np.float(nanval)

    # create masked array
    a = npma.MaskedArray(arr, arr == nanval)
    
    if axis is None:
        return float(npma.std(a, axis))
    else:
        return npma.std(a, axis).filled(nanval)



###########################################################################

def percent25(arr,   # [array]
              median_=None,   # [<None, median of array>]
              nanval=-999,  # [not-a-number value]
              ):
    """Returns the value of an entity at the 25th percentile.

    Returns
    -------
        Value of an entity at the 25th percentile given
        a range of entity values.

    Comments
    --------
        Defined in terms of the median.
    
    Arguments
    ---------
        "arr" is a 1-dimensional array of values.
        
        --------
        "median" is the median of the array.  If None, the median
        is calculated.
        
        --------
        "nanval" is the not-a-number value.

    Paste function
    --------------
        percent25(arr,   # [array]
                  median_=None,   # [<None, median of array>]
                  nanval=-999,  # [not-a-number value]
                  )

    """
    med = median_
    
    # Calculate median if necessary
    if med is None:
        med = float(median(arr, None, nanval))

    # Calculate 25th percentile
    P25 = median(arr[arr <= med], None, nanval)
    
    return P25


###########################################################################
# TODO:  Weirdness: percent25() and percent75() have virtually the same
# code, but the first prints the median in brackets and the second
# prints what looks like a float, but is an array.


def percent75(arr,   # [array]
              median_=None,   # [<None, median of array>]
              nanval=-999,  # [not-a-number value]
              ):
    """Returns the value of an entity at the 25th percentile.

    Returns
    -------
        Value of an entity at the 75th percentile from
        a range of entity values.

    Comments
    --------
        Defined in terms of the median.
    
    Arguments
    ---------
        "arr" is a 1-dimensional array of values.
        
        --------
        "median" is the median of the array.  If None, the median
        is calculated.
        
        --------
        "nanval" is the not-a-number value.

    Paste function
    --------------
        percent75(arr,   # [array]
                  median_=None,   # [<None, median of array>]
                  nanval=-999,  # [not-a-number value]
                  )

    """
    med = median_
    
    # Calculate median if necessary
    if med is None:
        med = float(median(arr, None, nanval))

    # Calculate 75th percentile
    P75 = median(arr[arr >= med], None, nanval)
    
    return P75



###########################################################################

def obsdeltatest(datadict,   # [array of observations, may include NaNVals]
                objcoordfunc,   # [String representation of ObjCoord() parameters]
                estimates = None,  # [None => calc initial estimates; array of estimates]
                delta = 0.1,    # [Size of delta in std.devs. => amount of noise to add]
                nanval = -999.  # [Not-a-Number Value]
                ):
    """Returns 0 < R^2 < 1 of change in observations vs. resulting change in estimates.

    Returns
    -------
        The the correlation^2 between change in observation
        and resulting change in estimates.

    Comments
    --------
        This is an experimental function for testing overfit
        for any of a range of matrix modeling techniques, even
        those that do not allow missing data.  It is related
        to objectivity, but compares the change in the estimate
        that results from a specified (small) change in the
        observed values.

        This function makes use of the eval() function, which
        has not been safely filtered, that is, it allows the
        user to evaluate any Python expression, even bad ones.
        Only allow safe expressions to be entered.

    Paste function
    --------------
        obsdeltatest(datadict,   # [array of observations, may include NaNVals]
                    objcoordfunc,   # [String representation of ObjCoord() parameters]
                    estimates = None,  # [None => calc initial estimates; array of estimates]
                    delta = 0.1,    # [Size of delta in std.devs. => amount of noise to add]
                    nanval = -999.  # [Not-a-Number Value]
                    )

    """
    # Get observed data
    observed = datadict['coredata']

    # Calc initial estimates matrix, if needed
    if estimates is None:
        Est1Coord = eval(objcoordfunc,{'__builtins__':None},{})
        if 'fac0coord' not in Est1Coord:
            Est1 = Est1Coord['obj_est']['coredata']
        else:
            estimate(Est1Coord['fac0coord'], # [N x D array of N row coordinates in D dimensions]
                     Est1Coord['fac1coord'], # [I x D array of I column coordinates in D dimensions]
                     nanval,    # [float Not-a-Number Value => marks missing coordinate values]
                     )
    else:
        Est1 = estimates

    # Calculate standard deviation of observed and noise
    stdev = np.std(observed[np.where(observed != nanval)])
    DeltaSD = delta * stdev
    nrows = np.size(observed,axis=0)
    ncols = np.size(observed,axis=1)
    noise = npr.rand(nrows,ncols) * DeltaSD - (DeltaSD / 2.0)

    # Add noise to observed data, fold back into datadict
    ObsNoise = np.where(observed == nanval,nanval,observed + noise)
    datadict['coredata'] = ObsNoise

    # Run modified datadict through ObjCoord() to get modified estimates
    Est2Coord = eval(objcoordfunc,{'__builtins__':None},{})

    if 'fac0coord' not in Est1Coord:
        Est2 = Est2Coord['obj_est']['coredata']
    else:
        estimate(Est2Coord['fac0coord'], # [N x D array of N row coordinates in D dimensions]
                 Est2Coord['fac1coord'], # [I x D array of I column coordinates in D dimensions]
                 nanval,    # [float Not-a-Number Value => marks missing coordinate values]
                 )

    # Calculate delta -- resultant change in estimates
    DeltaEst = Est2 - Est1

    # Calculate objectivity statistic for whole array
    DeltaStat = 1 - np.corrcoef(np.ravel(DeltaEst),np.ravel(noise))[0][1]**2

    return DeltaStat


###########################################################################

def objectivity(observed,   # [array of observations]
                biased_est,  # [array of ObjCoord() 'Descriptive' estimates]
                obj_est,   # [array of ObjCoord() 'Objective' estimates]
                nanval = -999., # [Not-a-Number value]
                ):
    """Returns 0 < objectivity < 1 for each cell, and 0 < Leverage < inf.

    Returns
    -------
        array of cell objectivity statistics between 0 and 1

    Comments
    --------
        This function supports the cellse() function.

        'biased_est' is the array of estimates returned when
        objcoord() is in 'Descriptive' mode.  When objcoord()
        computes starter coordinates for the 'Objective' analysis,
        it does so in 'Descriptive' mode automatically, so those
        estimates can be used for 'biased_est'.

        'obj_est' is the output of whatever routine is used to
        create 'objective' statistics, whether ObjEnt is 'Cell',
        'Row', or 'Col'.  But the best stats are found using
        'Row' or 'Col'.  'Cell' is only of theoretical interest.

        'obj_est' also depends on whatever 'ObjFinish' procedure
        is used.

        The formula is:

            objectivity = (Xni - E[Biased]) / (Xni - E[Objective])


        Unfortunately, the calculation of objective estimates is
        computationally expensive, so this function will generally
        be used sparingly.  An alternative method for computing
        unbiased standard errors is to specify
        residuals = ['psmiss',msindex] in the report() function.
        This results in unbiased residuals for a random sample
        of estimates whose observations have been made
        pseudo-missing, resulting in unbiased standard errors.

    Paste function
    --------------
        objectivity(observed,   # [array of observations]
                    biased_est,  # [array of ObjCoord() 'Descriptive' estimates]
                    obj_est,   # [array of ObjCoord() 'Objective' estimates]
                    nanval = -999., # [Not-a-Number value]
                    )


    """
    # Define biased and objective residuals
    BiasRes = np.where(np.logical_or(observed == nanval,biased_est == nanval),1,observed - biased_est)
    ObjRes = np.where(np.logical_or(observed == nanval,obj_est == nanval),1,observed - obj_est)

    # objectivity formula
    CellObj = np.where(np.logical_and(BiasRes != nanval,ObjRes != nanval),BiasRes / float(ObjRes),1)

    return CellObj




###########################################################################

def triproject(A,  # [<None,coordinates of ref point A>]
               B,  # [<None,coordinates of ref point B>]
               C,  # [<None,coordinates of target point C>]
               AB2, # [<None,AB^2 distance> => overrides A,B]
               AC2, # [<None,AC^2 distance> => overrides A,C]
               BC2, # [<None,BC^2 distance> => overrides B,C]
               nanval = -999   # [not-a-number value]
               ):
    """Project point C onto line AB.

    Returns
    -------
        triproject() returns he projection of point C onto line AB,
        i.e., where on AB point C falls, treating point A as the origin.

    Comments
    --------
        triproject() implements a simple but extraordinarily useful
        formula for locating points floating in n-space onto a defined
        scale.

        Given:
            A = a reference point indicating "less" of a trait
            B = a reference point indicating "more" of a trait

            A and B differ in no other way.

            Let C be a person with some unknown amount of a trait,
            who may vary from A and B in any number of additional
            unknown ways:

            Let:
            R = dist(C,A)
            S = dist(C,B)
            T = dist(A,B)

        Then, applying the Pythagorean theorem it can be shown that:

            C|AB = (R^2 - S^2 + T^2) / 2T

        where C|AB is where point C projects onto line AB.  It is interpreted
        as the amount of the trait that C possesses.

        It is also the central equation in the moth algorithm (see moth.py)
        for finding the coordinate location of a point in space.

    Arguments
    ---------
        A, B, and C, are 1-D arrays of spatial coordinates (outputs of
        Damon's coord() method), where A and B are idealized to represent
        points that differ only in terms of the trait in question and
        in no other respect.  If you have distances instead of point
        coordinates, set A, B, and C to None

        ----------
        "AB2" is the squared distance between A and B.  When AB2 is not
        None, the A and B arguments are ignored.  Similarly, AC2 and
        BC2 are the squared distances between A and C and B and C.

        ----------
        "nanval" is the not-a-number value.

    Examples
    --------

        [under construction]

    Paste Function
    --------------
        triproject(A,  # [<None,coordinates of ref point A>]
                   B,  # [<None,coordinates of ref point B>]
                   C,  # [<None,coordinates of target point C>]
                   AB2, # [<None,AB^2 distance> => overrides A,B]
                   AC2, # [<None,AC^2 distance> => overrides A,C]
                   BC2, # [<None,BC^2 distance> => overrides B,C]
                   nanval = -999   # [not-a-number value]
                   )

    """
    if (A is not None
        and B is not None
        and C is not None
        ):
        if (nanval in A
            or nanval in B
            or nanval in C
            ):
            return nanval

    # Interpoint distances
    R2 = np.sum((C - A)**2) if AC2 is None else AC2
    S2 = np.sum((C - B)**2) if BC2 is None else BC2
    T2 = np.sum((B - A)**2) if AB2 is None else AB2

    # Projection
    if T2 == 0:
        proj = nanval
    else:
        proj = (R2 - S2 + T2) / np.float(2 * np.sqrt(T2))

    return proj




###########################################################################

def subscale_filter(base_est,   # [estimates datadict, e.g., base_est_out]
                    cols,   # [[index] of subscale columns]
                    coords,  # [None,self.coord_out datadicts]
                    method = 'UseEst',  # [<'UseAllEst','UseSubEst','UseCoord'> => how to apply filter]
                    lo_hi = [-3,3], # [<'Auto',[lo,hi]> => high/low data values on estimates scale]
                    ):
    """Filters out contamination from other subscales.

    Returns
    -------
        subscale_filter() returns a 1-D array of subscale measures.
        It supports Damon's subscale() method, but only when it is used
        after coordinates and estimates have been calculated.

        Workflow:
            d = dmn.Damon()
            d.coord(...)
            d.base_est(...)
            d.subscale(...)     # Uses subscale_filter()

    Comments
    --------
        When computing measures for individual subscales using items
        from other subscales -- particularly when the subscales do not
        share a common space ("between-item multidimensionality") --
        there is a degree to which student scores on other subscales
        bias the target subscale away from the latent trait of the
        target subscale.  subscale_filter() was developed as one approach
        to correct for this bias.  Unfortunately, it does not appear to
        succeed in this goal, though it remains an intriguing approach
        to defining a subscale.

        It does so by creating two hypothetical reference persons, A and B,
        whose "responses" differ a lot on the subscale of interest but
        not at all on all other subscales.  Using the distances between
        each real person C and A and B, C is projected onto the dimension
        defined by A and B.  The formula is given in the documentation
        for tools.triproject.

        It is important that the target subscale really be mathematically
        distinct from the other subscales.  To the degree it is not,
        the distance between A and B will shrink toward zero and the
        error of C's projection on line AB will increase to infinity.

        Similarly, the fewer the number of items in the subscale, the
        smaller AB will be, which will increase subscale error substantially.

    Arguments
    ---------
        "base_est" is the output of Damon's base_est() method: base_est_out.
        It is in datadict format.

        --------------
        "cols" is a list containing the positions of each item
        in the target subscale.

        --------------
        "coords" is the output of Damon's coord() method: coord_out.
        It contains both row coordinates (fac0coord) and column coordinates
        (fac1coord).

        --------------
        "method" specifies the method to use in performing the projection.
        Three are supported:

            method = 'UseAllEst'    =>  Referents A and B consist of
                                        hypothetical arrays of estimates
                                        across all items where A and B
                                        differ only for the subscale items.

            method = 'UseSubEst'    =>  Referents A and B consist only of
                                        the subscale estimates, which differ.
                                        The other items are ignored.

            method = 'UseCoord'     =>  Referents A and B are defined in
                                        terms of the coordinates obtained
                                        by applying the existing anchored
                                        column coordinates to arrays of
                                        hypothetical A, B estimates for
                                        all items.

        --------------
        "lo_hi" specifies artificial "responses" to assign to reference
        point A (low values on the target subscale) and reference point
        B (high values on the target subscale).  The values should be in
        the same range as the estimates array as a whole, but defining
        high and low points in the range.  They don't need to be extreme,
        just very different.

            lo_hi = [-3,3]          =>  All the artificial "responses" for
                                        hypothetical reference person A
                                        will be -3 for the subscale items.
                                        For reference person B, they will be
                                        3.
    Examples
    --------

        [under construction]

    Paste Function
    --------------
        subscale_filter(base_est,   # [estimates datadict, e.g., base_est_out]
                        cols,   # [[index] of subscale columns]
                        coords,  # [<None,self.coord_out> => coord_out datadicts]
                        method = 'UseEst',  # [<'UseAllEst','UseSubEst','UseCoord'> => how to apply filter]
                        lo_hi = [-3,3], # [<'Auto',[lo,hi]> => high/low data values on estimates scale]
                        )


    """
    nanval = base_est['nanval']

    ################
    ##  UseSubEst ##
    ################

    if method == 'UseSubEst':
        est = base_est['coredata'][:,cols]
        ncols = len(cols)
        nrows = np.size(est,axis=0)

        if lo_hi == 'Auto':
            est_ma = npma.masked_values(est,nanval)
            A = npma.amin(est_ma,axis=0)
            B = npma.amax(est_ma,axis=0)
        else:
            A = np.zeros((ncols)) + lo_hi[0]
            B = np.zeros((ncols)) + lo_hi[1]

        T2 = np.sum((B - A)**2)

        # Get filtered subscale values
        subscale = np.zeros((nrows)) + nanval
        for r in xrange(nrows):
            C = est[r,:]
            subscale[r] = triproject(A,B,C,T2,nanval)


    ################
    ##  UseAllEst ##
    ################

    elif method == 'UseAllEst':
        est = base_est['coredata']
        ncols = np.size(est,axis=1)
        nrows = np.size(est,axis=0)

        est_ma = npma.masked_values(est,nanval)
        col_means = npma.mean(est_ma,axis=0)
        A = np.copy(col_means)
        B = np.copy(col_means)

        if lo_hi == 'Auto':
            A[cols] = npma.amin(est_ma[:,cols],axis=0)
            B[cols] = npma.amax(est_ma[:,cols],axis=0)
        else:
            A[cols] = lo_hi[0]
            B[cols] = lo_hi[1]

        AB2 = np.sum((B - A)**2)

        # Get filtered subscale values
        subscale = np.zeros((nrows)) + nanval
        for r in xrange(nrows):
            C = est[r,:]
            subscale[r] = triproject(A,B,C,AB2,None,None,nanval)


    ################
    ##  UseCoord  ##
    ################

    elif method == 'UseCoord':

        # Ref values, non-subscale => column means
        est_ma = npma.masked_values(base_est['coredata'],nanval)
        col_means = npma.mean(est_ma,axis=0)

        # Ref values, low
        ref_A = np.copy(col_means)
        ref_A[cols] = lo_hi[0]

        # Ref values, high
        ref_B = np.copy(col_means)
        ref_B[cols] = lo_hi[1]

        # Get ref coordinates
        ref_ = np.vstack((ref_A,ref_B))
        ref = dmn.core.Damon(ref_,'array',validchars=None,verbose=None)
        ref.coord(quickancs = [1,coords['fac1coord']['coredata']])

        # Consider adding coordinate refinement here to ensure that ref data fit model
        #

        # Coordinates of A, B reference points
        A = ref.coord_out['fac0coord']['coredata'][0,:]
        B = ref.coord_out['fac0coord']['coredata'][1,:]
        AB2 = np.sum((B - A)**2)

        # Get filtered subscale values
        fac0coord = coords['fac0coord']['coredata']
        nrows = np.size(fac0coord,axis=0)
        subscale = np.zeros((nrows)) + nanval
        for r in xrange(nrows):
            C = fac0coord[r,:]
            subscale[r] = triproject(A,B,C,AB2,None,None,nanval)

    return subscale




###########################################################################

def pytables_(data,   # [<None, array, file,[files],datadict, hd5 file, or data generating function {'chunkdict':{...},'ArgDict':{...}}> ]
             format_ = 'array',  # [<'array','textfile',['textfiles'],'datadict','hd5','init_earray','chunkfunc'>]
             putinfile = 'data.hd5',    # [<None,'MyFileName.hd5',MyPyTable['fileh']>  => name of file in which to put groups and arrays> ]
             filemode = 'w',    # [<None,'w','r','a','r+'>  => mode in which to open file (write, read, append, read+) ]
             ingroup = 'Group1',   # [<None,GroupName>  => e.g., 'Group1' ]
             array_names = ['array'],   # [ ['ArrayName0','ArrayName1']>  => list of one or more arrays to be created and/or read]
             arraytype = None,   # [IGNORE, NOT YET SUPPORTED. <None,'Table','array','EArray','VLArray'> => type_ of array to create. ]
             atomtype = None,   # [<None, <'string','int','float',object> >  => type of data to create (see dtype arg in np.array)]
             atomsize = None,   # [<None, int size of atom (generally 4)>.  atomtype='string'=> atomsize=nChars; 'int'=> [1,2,4,8]; 'float'=> [4,8]]
             shape = None,  # [<None, (nrows,ncols)>  e.g. (10,0) => cols extendable.  Only supports format_ = 'init_earray'.]
             delimiter = None,  # [<None, input file field delimiter>  => e.g. <',','\t'>]
             ):
    """Converts data entity into a PyTable array on disk.

    Returns
    -------
        pytables() returns one or more arrays that can be read
        and manipulated like regular numpy arrays, except that
        they reside on disk and not in memory.  It also returns
        a file object that can be used to apply file methods
        such as close().

            {'arrays'   =>  dictionary of specified arrays
             'fileh'    =>  pytables file object
             }

        Output arrays are created or loaded in one of several ways,
        as specified by 'format_':

            1.  The array already resides in memory.

            2.  data exists in one or more textfiles.

            3.  An array needs to be generated from scratch
                according to a data generating function.

            4.  data resides in one or more arrays in a data
                dictionary.

            5.  data has already been stored on disk using
                pytables(), and it just needs to be read.

        The output array(s) are generally stored in a
        dictionary accessed using one of the arrays listed
        in array_names:

        MyLabels = pytables(...,array_names=['MyLabels'])['arrays']['MyLabels']

    Comments
    --------
        pytables() is a wrapper function for the pytables
        package (www.pytables.org).  It is used mainly to handle data
        arrays that are too large to fit comfortably in memory.
        It does this by storing and manipulating large arrays
        directly on disk rather than in memory.  It provides
        a nice speed boost even with moderately sized files.

        Like Numpy, pytables is not provided automatically as
        part of Damon.  You have to install it separately.  You
        can install it yourself by going to pytables.org, but this
        entains installing yet more dependencies such as numexpr.
        Because the different versions of numpy, numexpr, pytables,
        and so on may not be completely compatible, the installation
        procedure can be crazy-making.  I recommend purchasing the
        Enthought distribution (www.enthought.com), which combines
        these and many other scientifically-oriented Python packages
        into one tidy package, with all the dependencies ironed out.

        "format_" must accurately describe the format of the data
        in data.  Otherwise, an error is thrown.

        "putinfile" is the name of a new pytables file to
        create (with .hd5 extension) or open (if it was
        already created), and "ingroup" is the name of a "directory"
        within which the data specifed by data = is to be stored.
        Your computer sees all this as just one big file.  However,
        you will store them and read them as if they are directories
        within a root directory, where the root directory is
        the file specified by "putinfile".  This pytables() wrapper
        only supports one layer of directories, called "groups", within
        the root directory, though the pytables package can handle
        nested directory designs.

        When format_ = 'hd5', the file already exists under the file
        name given in data and putinfile is ignored and can be
        set to None.

        The last four parameters -- arraytype, atomtype, atomsize,
        and delimiter -- only apply when format_ is 'textfile',
        ['textfiles'], or 'chunkfunc'.  Otherwise, they can be set
        to None.

        When a Damon object is initialized, a PyTable is (optionally)
        assigned to hold all outputs created by Damon methods,
        outputs that would otherwise reside entirely in memory.
        This is a lot of data.  The Damon report() method
        provides a way to extract just those outputs that you want
        to save and move them to another PyTable HD5 file.  (Saving
        them under the existing file unfortunately does not free
        up space.)

        Once a group or array is assigned to a file, it cannot be
        overwritten.  Therefore, all groups and arrays created and
        assigned to the Damon .hd5 file are only created once with
        filemode = 'r+'.  Though they can't be overwritten, they can
        be modified.  Functions or methods that may be used multiple
        times are assigned their own pytables .hd5 files, which are
        overwritten by specifying filemode = 'w'.

        Warning:  Be careful to debug your 'chunkfunc' function
        very carefully.  pytables has a habit of not reporting
        an error and returning what look like normal outputs.
        One sign that an error has occurred is that you will
        get a message (if the function is called again) that
        the file has not been closed -- even though the pytables()
        function code says that the file should have been closed.
        A common bug is treating counts as floats rather than ints.

        How To Append/Accumulate arrays into a PyTable
        ----------------------------------------------
        There is trick for building a PyTable array using
        the pytables append() function that you will probably
        need at some point.  It uses format_ = 'init_earray'.

        Say you have a function that iteratively adds columns
        (or rows) to a Numpy array, but you want that array to
        be a PyTable.  In theory you could use the pytables()
        'chunkfunc' option and run the whole chunk function inside
        of pytables().  But this can get unwieldy if the
        chunk function is large.  Alternatively, you can
        apply regular pytables commands at different stages of
        your module:

            a)  Initializing a PyTable EArray (extendable array)
                before the chunk is calculated.

            b)  Use PyTable's append method: My_hdfarray.append(chunk).

            c)  After the array is fully built, make it readable
                and close the .hd5 file.

        Here is an example:

        # Initialize EArray
        PyTabCore = tools.pytables_(None,'init_earray',pytables,'r+','parse_out',
                                   ['coredata'],None,'float',4,(nDatRows,0),None)
        coredata = PyTabCore['arrays']['coredata']

        # Calculate chunk and append to EArray
        for i in range(nchunks):
            |
            |   your code for a given chunk
            V
            Chunk = blah, blah

            coredata.append(Chunk)

        # Read and close the file
        coredata = PyTabCore['GetArr'].read()
        PyTabCore['fileh'].close()

        For more information about how to use pytables, go to
        pytables.org.

    Arguments
    ---------
        "data" is one of several types of data containers: a file,
        a set of files, a numpy array, a data dictionary, an hd5 file
        (the format in which Pytables files are stored on disk),
        or a function for generating rows from scratch rather than
        reading them in.  The type of container is specified in
        format_.  Options:

            data = None             =>  This only applies when format_ =
                                        'init_earray', i.e., an EArray is
                                        being initialized for use outside
                                        the function.

            data = array            =>  data is a numpy array residing
                                        in memory that we want to store
                                        on disk.

            data = 'MyFile.csv'     =>  pytables reads in a text
                                        file, whose delimiter is
                                        specified in the delimiter
                                        argument below.  It can also
                                        be a file path.

            data = ['MyFile0.csv','MyFile1.csv',...]
                                    =>  data are stored across multiple
                                        text files, all formatted the
                                        same.

            data = datadict         =>  data has already been parsed
                                        into a set of arrays, generally
                                        the rowlabels, collabels, and
                                        coredata arrays extracted by the
                                        Damon.__init__ method or one
                                        of the other Damon() methods.
                                        However, it can be any dictionary of
                                        arrays. Dictionary elements that aren't
                                        worth storing as a PyTable can be
                                        ignored.

            data = hd5 file         =>  This is a file that has already been
                                        created by pytables() and is stored
                                        with a .hd5 extension.  'hd5' refers to
                                        the Hierarchical data format_ used for
                                        storing very large files that can
                                        hold nested data objects directory
                                        style.

                                        If 'hd5' is specified in format_, it
                                        means you do not need to convert data
                                        from some other format into a pytables
                                        format.  You just want to read files
                                        that have already been stored in
                                        the hd5 format by pytables.  (All pytables
                                        files are stored in 'hd5' format.)

                                        In this case, the putinfile, arraytype,
                                        atomtype, atomsize, and delimiter parameters
                                        are ignored (set them to None or just
                                        ignore them).

            data =  chunk-wise data generating function = {'chunkdict':{...},'ArgDict':{...}},
                    where chunkdict = {'chunkfunc':_,'nchunks':_,'chunksize':_,'ncols':_,'nrows':_}
                    and ArgDict = {0:ChunkFuncArg0,1:ChunkFuncArg1,...,N:chunkstart=0}

                                    =>  You want to create or calculate new
                                        data with a user-defined function applied
                                        to successive 'chunks' of rows.  This
                                        function, called 'chunkfunc', may
                                        create random data or find a dot
                                        product or standardize a range, whatever
                                        you need.

                                        If the array you want to create
                                        or calculate is small, you can do
                                        it all in one chunk of rows.  But if
                                        the data you want to create won't
                                        fit into memory, chunkwise calculation
                                        provides a way to build your array
                                        without running up against
                                        memory limits.  Because this is
                                        machine-specific, you have the option
                                        to tweak the chunksize and the number
                                        of chunks yourself.  Rule of thumb,
                                        most machines can handle 10,000
                                        rows at a time pretty well (depending
                                        on the size of each datum and number
                                        of columns).  Or you can let the function
                                        set 'chunksize' and 'nchunks' automatically.

                                        The 'chunkfunc' function is defined
                                        by you, the user, outside the pytables function.
                                        It can take as many arguments as you like.
                                        Its arguments are stored in an 'ArgDict'
                                        Python dictionary.  This, plus another
                                        dictionary called 'chunkdict', is entered
                                        under the data argument.

                                        Bear in mind that chunks are always "chunks
                                        of rows".  You can build an array by adding
                                        chunks of rows, but not chunks of columns.
                                        This is a pytables constraint.

                                        These two dictionaries, 'chunkdict' and
                                        'ArgDict', are used by pytables() to
                                        build a new dataset in row chunks and store
                                        this new dataset in the specified hd5 file
                                        and group.

                                        The chunkfunc function and the two dictionaries,
                                        while providing great flexibility, require a
                                        specific syntax that needs a bit of study.

            The data argument is set to equal a python dictionary, and this
            dictionary includes two dictionaries, one called 'chunkdict' and the
            other called 'ArgDict'.  'chunkdict' captures a fixed set of chunk-
            related variables, including the name of the user-defined chunk-generating
            function called 'chunkfunc'.  'ArgDict' captures the arguments required
            for running 'chunkfunc'.  It may refer to variables stored in 'chunkdict'.

                data = {'chunkdict':chunkdict,'ArgDict':ArgDict}

            Within 'chunkdict', we have the chunkdict dictionary which contains
            the following required variables:

                chunkdict =
                    'chunkfunc':MyFunc  =>  Name of the function created by you,
                                            the user, for creating a chunk of
                                            data.

                    'nchunks':<int,'Auto'>
                                        =>  Integer number of chunks needed to build
                                            the final array (rounded up, if fractional),
                                            or you can specify 'Auto' and let pytables()
                                            assign a number.

                    'chunksize':<int rows per chunk, 'Auto'>
                                        =>  Integer number of rows to be assigned to
                                            each chunk.  It is okay for the last chunk
                                            to have smaller number of rows than chunksize;
                                            don't worry about getting the chunksize to
                                            divide evenly into the number of rows.
                                            You can also specify 'Auto' and let pytables()
                                            assign a number.

                    'nrows':int         =>  Integer number of rows in final array that
                                            you plan to end up with -- after all the
                                            chunks are put together.

                    'ncols':int         =>  Integer number of columns in the final
                                            array.

            Within 'ArgDict' we have the ArgDict dictionary which contains
            the parameters needed to run your chunkfunc data-generating function.
            The key names for these parameters are specified by the user, with
            the exception of the 'chunkstart' argument, which must always be
            defined and initialized to 0.  pytables() updates 'chunkstart'
            automatically in the course of appending new chunks, so you don't need
            to worry about it.

                ArgDict =

                    {'KeyWord0': Arg0   =>  The first argument in your chunk
                                            generating function, the function whose
                                            name is assigned to 'chunkfunc' in the
                                            chunkdict.

                    'KeyWord1': Arg1    =>  The second argument in chunkfunc.

                    |
                    |
                    V

                    'chunkstart': 0     =>  The chunkstart parameter used to give
                                            the row number of the leading row in
                                            the chunk under consideration
                                            and initialized to 0 (the first row
                                            of the first chunk).

            Example1 (oversimplified):  Create an array of random integers.

                    Outside the pytables() function, define your chunkfunction
                    and define the chunkdict and ArgDict dictionaries. In
                    pytables(), set data = {'chunkdict':MyChunkDict,'ArgDict':MyArgDict}.

                    Here is what it looks like to create a 500,000 x 500 array of random
                    integers.

                    def rand_int(chunksize, ncols, chunkstart=0): return npr.randint(0,9,(chunksize,ncols))

                    data = {'chunkdict':{'chunkfunc':rand_int,
                                         'nchunks':50,          # or 'nchunks':'Auto'
                                         'chunksize':10000,     # or 'chunksize':'Auto'
                                         'nrows':500000,
                                         'ncols':500
                                         },
                            'ArgDict':   {'chunksize':chunkdict['chunksize'],
                                          'ncols':chunkdict['ncols'],
                                          'chunkstart':0                   # the default starting row of the first chunk
                                          }
                            }

            Note:  You may well need to include 'chunksize' or 'nchunks'
            in ArgDict.  If so, make sure that you call the relevant
            ArgDict arguments 'chunksize' or 'nchunks.  pytables() looks
            inside 'ArgDict' for arguments with these names and pulls
            the relevant values from 'chunkdict'.  This saves a lot of
            programming headaches when trying to define the chunkfunc.

            Example2:  Calculate the dot product of two matrices R and C.

                    def my_dot(R,C,chunksize,chunkstart=0):
                        return np.dot(R[chunkstart:(chunkstart + chunksize),:],C)

                    data = {'chunkdict':{'chunkfunc':my_dot,
                                         'nchunks':50,          # or 'nchunks':'Auto'
                                         'chunksize':10000,     # or 'chunksize':'Auto'
                                         'nrows':500000,
                                         'ncols':500
                                         },
                            'ArgDict':   {'R':R,
                                          'C':C,
                                          'chunksize':chunkdict['chunksize'],
                                          'chunkstart':0                   # the default starting row of the first chunk
                                          }
                            }

            Some commonly used 'ChunkFuncs' are defined and saved
            in the tools module.


        Dealing with arrays Larger than Memory
        --------------------------------------
        Numpy only works with arrays in memory.  If Numpy is trying to
        create or analyze an array larger than what your computer's
        RAM can handle, it will return a memory error.  Using
        the pytables 'array' or 'datadict' formats doesn't quite get
        around the problem, because the memory error pops up before
        pytables() can do anything about it.

        In these situations, you either need to have the data stored
        across one or more text files, or stored as an 'hd5' file,
        or you need to generate the dataset yourself in chunks
        using the 'chunkfunc' option.  Then you can load the data
        chunk-wise into an 'hd5' file (if it isn't in one already)
        without bumping against the memory limits of your machine.

        When, then, is it useful to use the 'array' and 'datadict'
        formats?  If you have already converted your data to 'hd5'
        format and you create an array or datadict that refers
        directly or indirectly to arrays in your 'hd5' format,
        then you can use the 'array' and 'datadict' formats even
        with memory-busting arrays.  You are in effect making a copy
        of one hd5 array and saving it under another name within
        the 'hd5' file's directory structure.  This uses up space
        on the hard-drive but keeps your RAM free.

        If your Numpy arrays do happen to fit in memory -- as they
        generally do -- the pytables() 'array' and 'datadict' formats
        will read them just fine and produce a wonderful boost in
        speed and efficiency.  There is no downside to using pytables()
        with small arrays.

        ---------------
        "format_" describes the format of the data given in the data
        parameter:

            'array'         =>  data is a numpy array.

            'textfile'      =>  data is stored in a text file.

            ['textfiles']   =>  data is stored in a set of text
                                files in a list.

            'datadict'      =>  data is a datadict -- a data dictionary
                                output by one of the Damon() methods.

            'hd5'           =>  data is in Hierarchical data format_, the
                                format in which pytables outputs are
                                stored.  Multiple arrays (in fact a
                                whole tree of outputs) can be stored
                                under the same hd5 file name.

                                If 'hd5' is specified, the Array2Read
                                parameter must be specified, but the
                                remaining parameters can be set to
                                None or ignored.

            'init_earray'    =>  It is anticipated that rows or columns
                                will be appended to an EArray (extendable)
                                array outside the pytables() function.
                                This option initializes the EArray and
                                returns it, as well as the file it belongs
                                to.  The append(), read(), and
                                close() pytables methods are applied
                                outside the function, if applicable.
                                'init_earray' requires that you
                                enter a parameter for the shape argument,
                                e.g., shape = (10,0), which means the
                                array will have 10 rows and be extendable
                                along the column dimension.

            'chunkfunc'     =>  data is created using a chunk-wise
                                generating function.  See syntax above.

        ---------------
        "putinfile" is the name of an hd5 file to be created that will store
        your groups and arrays.  It should have the suffix .hd5 .
        pytables is an interface to the HDF5 data format (Hierarchical
        data format_), a format designed to handle extremely large data
        files efficiently.  Example:  putinfile = 'MyFile.hd5'  .
        Multiple groups, arrays and other objects are stored under this
        .hd5 file name in a tree (directory) structure.

        putinfile can be a file name or an absolute file path.

        When format_ = 'hd5', putinfile is ignored and can be set to
        None.  The pytables file has already been created under the name
        given in data and you don't need to create a new one.

        ---------------
        "filemode" specifies the mode in which to open the file
        specified in 'putinfile'.  There are four modes:

            'w'     =>  write/create a new file
            'r'     =>  read an existing file
            'a'     =>  append data to an existing file, or create a new one
            'r+'    =>  like append, but file must already exist

        When creating a file for the first time, you will want to
        use 'w'.  When opening an existing file to add a group and
        new arrays, use 'r+'.  When reading an existing 'hd5' file,
        use 'r'.  It takes trial and error to figure out how these
        modes work.

        See the pytables docs for more information.

        ---------------
        "ingroup" specifies the name of a "group" that will be assigned
        to the file specified by putinfile; it is directly analogous to
        a directory, except that it isn't seen as such by your computer.
        pytables() currently requires that all groups be assigned
        to the same root directory; no groups within groups.

            ingroup = GroupName

            ingroup = 'Group1'
                            =>  Create a group (directory) called Group1
                                in the root directory (which corresponds
                                to the top level of the putinfile file.)

        If the group name already exists, it is accessed and read.
        If the group doesn't exist, it is created.

        ---------------
        "array_names" is a list of names under which one or more arrays
        are to be stored in the specified file and group.  Multiple arrays
        can be stored in one group only if format_ = 'datadict'.)  If
        format_ = 'hd5', array_names specifies the array(s) to open and read.
        Otherwise, it loads or creates a new array.

            array_names = ['MyArray']
                            =>  Create, load, or read the array called
                                'MyArray' in the specified file and group.

            array_names = ['Array1','Array2']
                            =>  Create, load, or read the 'Array1'
                                and 'Array2' arrays from the specified
                                datadict.

        ---------------
        "arraytype" is not yet supported.  It will allow
        pytables() to handle the "tables" and "VLArray" (variable
        length array -- each row with a different length) formats.
        The present version, supports only numpy arrays and extended
        arrays ("EArray"), which are arrays that can be extended by
        adding rows.

        That means you can only load rectangular arrays, where each
        row has the same number of columns.

        ---------------
        "atomtype" is the type of data to reside in each cell of the matrix
        (called an "atom" in pytables).  The three main AtomTypes are:
        'string', 'int', and 'float'.

        ---------------
        "atomsize" is how "large" each atom should be.  Each atomtype has its
        own size options, which are a little esoteric.  In a nutshell,
        atomsize for 'string' is how many characters should be allowed in
        a cell.  atomsize for 'int' is bit size -- 1, 2, 4, or 8.  atomsize
        for 'float' is also a bit size -- 4 or 8.  If you enter a wrong
        number, a pytables error message will report back the valid options.

        ---------------
        "shape" is only used when format_ = 'init_earray'.  (It might also
        support format_ = 'chunkfunc' in the future.)  It tells the shape
        of the array to initialize, where 0 indicates the extendable
        dimension.

            shape = None        =>  format_ is not 'init_earray'.

            shape = (10,0)      =>  Initialize an EArray that will have
                                    10 rows and be extendable along the
                                    column dimension, i.e., in anticipation
                                    that columns will be appended to the
                                    array.

            shape = (0,10)      =>  Initialize an EArray that will have
                                    10 columns and be extendable along the
                                    row dimension, i.e., in anticipation
                                    that rows will be appended to the
                                    array.

        ---------------
        "delimiter" only applies when data refers to a file.  It is the
        field delimiter, comma by default.

    Examples
    --------



    Paste function (plus template for chunkdict/ArgDict if format_ = 'chunkfunc')
    ----------------------------------------------------------------------------
        chunkdict = {'chunkfunc':_,
                     'nchunks':'Auto',
                     'chunksize':'Auto',
                     'nrows':_,
                     'ncols':_
                     }
        ArgDict = {'Arg1':_,
                   'Arg2':_,
                   'chunkstart':_
                   }
        data = {'chunkdict':chunkdict,'ArgDict':ArgDict}

        tools.pytables_(data,   # [<None, array, file,[files],datadict, hd5 file, or data generating function {'chunkdict':{...},'ArgDict':{...}}> ]
                     format_ = 'array',  # [<'array','textfile',['textfiles'],'datadict','hd5','init_earray','chunkfunc'>]
                     putinfile = 'data.hd5',    # [<None,'MyFileName.hd5',MyPyTable['fileh']>  => name of file in which to put groups and arrays> ]
                     filemode = 'w',    # [<None,'w','r','a','r+'>  => mode in which to open file (write, read, append, read+) ]
                     ingroup = 'Group1',   # [<None,GroupName>  => e.g., 'Group1' ]
                     array_names = ['array'],   # [ ['ArrayName0','ArrayName1']>  => list of one or more arrays to be created and/or read]
                     arraytype = None,   # [IGNORE, NOT YET SUPPORTED. <None,'Table','array','EArray','VLArray'> => type_ of array to create. ]
                     atomtype = None,   # [<None, <'string','int','float'> >  => type of data to create (see dtype arg in np.array)]
                     atomsize = None,   # [<None, int size of atom (generally 4)>.  atomtype='string'=> atomsize=nChars; 'int'=> [1,2,4,8]; 'float'=> [4,8]]
                     shape = None,  # [<None, (nrows,ncols)>  e.g. (10,0) => cols extendable.  Only supports format_ = 'init_earray'.]
                     delimiter = None,  # [<None, input file field delimiter>  => e.g. <',','\t'>]
                     )

    """

    # Import Pytables
    try:
        import tables as tab
    except ImportError:
        exc = 'Unable to find pytables.  If not installed, see www.pytables.org.  Consider the Enthought distribution.\n'
        raise pytables_Error(exc)


    ###################################
    # Initialize an EArray
    if format_ == 'init_earray':

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
            fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        # Atom specifies the size and type of each cell
        if atomtype is not None and atomsize is not None:
            Atom = tab.Atom.from_kind(atomtype,atomsize)
        else:
            exc = 'Need atomtype and atomsize parameters.\n'
            raise pytables_Error(exc)

        # Create an extendable array
        try:
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = shape,
                                          )
        except tab.NodeError:
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,array_names[0])
            GetArr.remove()
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = shape,
                                          )

        # Read array from disk
        GetGroup = getattr(fileh.root,ingroup)

        # Initialize new dictionary
        OutDict = {}
        for Name in array_names:
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        return {'arrays':OutDict,'fileh':fileh}



    ###################################
    # Read already existing 'hd5' file, copy to new file/group
    if format_ == 'hd5':

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
            fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        OutDict = {}

        for Name in array_names:
            # Handle dict containing hd5 arrays
            if isinstance(data,dict):
                data[Name].copy(newparent = Group,
                                newname = Name,
                                overwrite=True
                                )
            # No dictionary
            else:
                # Copy to specified group
                data.copy(newparent = Group,
                          newname = array_names[0],
                          overwrite=True
                          )

            # Read array from disk
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        # Load remaining keys into OutDict
        if isinstance(data,dict):
            for key in data.keys():
                if key not in array_names:
                    OutDict[key] = data[key]

        return {'arrays':OutDict,'fileh':fileh}


    ###################################
    # Convert array to PyTable array
    if format_ == 'array':

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
            fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        # Create an ordinary array
        try:
            hdfarray = fileh.createArray(where = Group,
                                          name = array_names[0],
                                          object = data,
                                          )
        except tab.NodeError:
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,array_names[0])
            GetArr.remove()
            hdfarray = fileh.createArray(where = Group,
                                         name = array_names[0],
                                         object = data,
                                         )

        # Read array from disk
        GetGroup = getattr(fileh.root,ingroup)

        # Initialize new dictionary
        OutDict = {}
        for Name in array_names:
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        return {'arrays':OutDict,'fileh':fileh}



    ###################################
    # Convert datadict to PyTable array
    if format_ == 'datadict':

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
            fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        # Initialize new dictionary
        OutDict = {}

        for Name in array_names:
            try:
                hdfarray = fileh.createArray(where = Group,
                                             name = Name,
                                             object = data[Name],
                                             )
            except tab.NodeError:
                GetGroup = getattr(fileh.root,ingroup)
                GetArr = getattr(GetGroup,Name)
                GetArr.remove()
                hdfarray = fileh.createArray(where = Group,
                                             name = Name,
                                             object = data[Name]
                                             )

            # Read array from disk
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        # Load remaining keys into OutDict
        for key in data.keys():
            if key not in array_names:
                OutDict[key] = data[key]

        return {'arrays':OutDict,'fileh':fileh}



    ####################################
    # Convert text file(s) to PyTable array
    if (format_ == ['textfiles']
        or format_ == 'textfile'
        ):

        # Force data to be a file list
        if format_ == 'textfile':
            Data1 = [data]
        elif format_ == ['textfiles']:
            Data1 = data

        # Pull out the first row of data from the data file
        try:
            inp = open(Data1[0])
        except IOError:
            exc = 'No such file or directory as '+Data1[0]+' .\n'
            raise pytables_Error(exc)

        # Get number of columns
        if delimiter is not None:
            FirstLine = inp.next()
            FirstLine = FirstLine.rstrip().split(delimiter)
            ncols = len(FirstLine)
            inp.close()
        else:
            exc = 'Need delimiter parameter.\n'
            raise pytables_Error(exc)

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
            fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        # Atom specifies the size and type of each cell
        if atomtype is not None and atomsize is not None:
            Atom = tab.Atom.from_kind(atomtype,atomsize)
        else:
            exc = 'Need atomtype and atomsize parameters.\n'
            raise pytables_Error(exc)

        # Create an extendable array
        try:
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = (0,ncols),
                                          )
        except tab.NodeError:
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,array_names[0])
            GetArr.remove()
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = (0,ncols),
                                          expectedrows = None   # Consider specifying these
                                          )

        # Consider:  file.readlines(10000) -- read in chunks

        # Read lines into extendable array from text file
        for textfile in Data1:
            inp = open(textfile)
            for line in inp:
                Line = [line.rstrip().split(delimiter)]
                try:
                    hdfarray.append(Line)
                except:
                    print 'Error in pytables():  Unable to append line to array.  Check consistency of number of columns.\n'
                    break

            inp.close()

        # Read array from disk
        GetGroup = getattr(fileh.root,ingroup)

        # Initialize new dictionary
        OutDict = {}
        for Name in array_names:
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        return {'arrays':OutDict,'fileh':fileh}


    ###################################
    # Generate new array from function
    elif format_ == 'chunkfunc':

        # Get generating function and array dimensions
        ncols = data['chunkdict']['ncols']

        # Create file to hold on-disk array
        if isinstance(putinfile,str):
##            try:
            fileh = tab.openFile(putinfile,filemode)
##            except:
##                os.remove(putinfile)
##                fileh = tab.openFile(putinfile,filemode)
        else:
            fileh = putinfile

        # Create group
        try:
            Group = fileh.createGroup('/',ingroup)
        except tab.NodeError:
            Group = getattr(fileh.root,ingroup)

        # Atom specifies the size and type of each cell
        if atomtype is not None and atomsize is not None:
            Atom = tab.Atom.from_kind(atomtype,atomsize)
        else:
            exc = 'Need atomtype and atomsize parameters.\n'
            raise pytables_Error(exc)

        # Create an extendable array
        try:
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = (0,ncols),
                                          )

        except tab.NodeError:
            GetGroup = getattr(fileh.root,ingroup)
            GetArr = getattr(GetGroup,array_names[0])
            GetArr.remove()
            hdfarray = fileh.createEArray(where = Group,
                                          name = array_names[0],
                                          atom = Atom,
                                          shape = (0,ncols),
                                          expectedrows = int(data['chunkdict']['nrows'])
                                          )

        # Get chunk variables
        f = data['chunkdict']['chunkfunc']
        nrows = long(data['chunkdict']['nrows'])
#        chunkstart = 0

        # Set chunk size
        if (data['chunkdict']['nchunks'] == 'Auto'
            or data['chunkdict']['chunksize'] == 'Auto'
            ):
            if nrows > 100000:
                chunksize = 25000.
            elif nrows > 10000:
                chunksize = 10000.
            elif nrows > 1000:
                chunksize = 1000.
            else:
                chunksize = 10.
            nchunks = np.ceil(nrows / chunksize)
        else:
            nchunks = int(data['chunkdict']['nchunks'])
            chunksize = int(data['chunkdict']['chunksize'])

        # Get ArgDict
        ArgDict = data['ArgDict']
#        VarChunkSize = True
        if 'chunksize' in ArgDict.keys():
            ArgDict['chunksize'] = int(chunksize)
#            VarChunkSize = False

        if 'nchunks' in ArgDict.keys():
            ArgDict['nchunks'] = int(nchunks)

        # Calc and append each chunk
        for i in xrange(int(nchunks)):
            Chunk = f(**ArgDict)
            LastChunkSize = np.size(Chunk,axis=0)
            hdfarray.append(Chunk)
            ArgDict['chunkstart'] += int(LastChunkSize)

        # Read array from disk
        GetGroup = getattr(fileh.root,ingroup)

        # Initialize new dictionary
        OutDict = {}
        for Name in array_names:
            GetArr = getattr(GetGroup,Name)
            OutDict[Name] = GetArr

        return {'arrays':OutDict,'fileh':fileh}




###########################################################################

def rand_chunk(nchunks, # [<'Auto',int>  => number of chunks across array]
               chunksize,   # [<'Auto',int>  => rows per chunk (except the last)]
               nrows,   # [int rows specified for whole core data array]
               ncols,   # [int columns specified for whole core data array]
               facmetric,   # [[m,b] => rand() * m + b, to set range of random numbers]
               seed = None,  # [<None => randomly pick starter coordinates; int => integer of "seed" random coordinates>]
               chunkstart = 0   # [number of first row of the first chunk, set at 0]
               ):
    """Create chunk of random numbers for pytables().

    Returns
    -------
        A chunk of rows consisting of random numbers
        to be accumulated into a pytables array of
        random numbers.

    Comments
    --------
        This function is defined specifically for use
        as a 'chunkfunc' option for pytables().  See
        pytables() docs.

        Chunks are defined in terms of a number of rows
        for a proposed array.  pytables() does not support
        building arrays out of column chunks.

        This function only creates core data, not labels.

        Random numbers are rounded to two decimal places
        to save space.

    Arguments
    ---------
        "nchunks" <'Auto',int> is the integer number
        of chunks needed to span the rows of the array.
        'Auto' allows pytables() to set nchunks internally.

        ------------
        "chunksize" <'Auto',int> is the number of rows per
        chunk.  It is expected that the last chunck may be
        smaller than the rest depending on the specified
        number of rows.

        ------------
        "nrows" is the integer number of rows specified for
        the whole core data array.

        ------------
        "ncols" is the integer number of columns specified for
        the whole core data array.

        ------------
        "facmetric" is used to set the range of ranom values.
        It multiplies a chunk of decimal random numbers by
        m and adds b:

            Output = rand() * m + b

        The syntax is [m,b].

            facmetric = [4,-2]  =>  multiply the chunk of random
                                    decimals by 4 and subtract 2.
        ------------
        "seed" <None,int> is an integer value used by Python to
        pick a specific set of random numbers.  If seed = None,
        the seed is chosen randomly and is different with every
        run.

        ------------
        "chunkstart" is the row number of the first row of the
        first chunk.  It should be set to zero.  The pytables()
        function automatically increments by chunksize after
        calculating each chunk.

    Examples
    --------

    Paste function Args as Dict
    ---------------------------
        ArgDict = {'nchunks':_,
                   'chunksize':_,
                   'nrows':_,
                   'ncols':_,
                   'facmetric':_,
                   'seed':_,
                   'chunkstart':0
                   }

    """
    # Variables
    Decimals = 2

    # Deal with seed
    if seed is not None:
        seed = seed + chunkstart

    # Get the first row of the top chunk
    TopChunk = (nchunks - 1) * chunksize

    # Get part of a chunk if last chunk
    if chunkstart >= TopChunk:
        ChunkPart0 = nrows % chunksize
        if ChunkPart0 == 0:
            ChunkPart = chunksize
        else:
            ChunkPart = ChunkPart0
        Chunk = (np.around(npr.RandomState(seed=seed).rand(ChunkPart,ncols),decimals=Decimals)
                 * facmetric[0] + facmetric[1])

    # Build rows for all complete chunks
    else:
        Chunk = (np.around(npr.RandomState(seed=seed).rand(chunksize,ncols),decimals=Decimals)
                 * facmetric[0] + facmetric[1])

    return Chunk


###########################################################################

def zeros_chunk(nchunks, # [<'Auto',int>  => number of chunks across array]
                chunksize,   # [<'Auto',int>  => rows per chunk (except the last)]
                nrows,   # [int rows specified for whole core data array]
                ncols,   # [int columns specified for whole core data array]
                Val,    # [<None,num> number to add to zeros]
                chunkstart = 0   # [number of first row of the first chunk, set at 0]
                ):
    """Create chunk of zeros for use by pytables().

    Returns
    -------
        A chunk of rows consisting of zeros
        to be accumulated into a pytables array of
        zeros.

    Comments
    --------
        This function is defined specifically for use
        as a 'chunkfunc' option for pytables().  See
        pytables() docs.

        Chunks are defined in terms of a number of rows
        for a proposed array.  pytables() does not support
        building arrays out of column chunks.

        This function only creates core data, not labels.

    Arguments
    ---------
        "nchunks" <'Auto',int> is the integer number
        of chunks needed to span the rows of the array.
        'Auto' allows pytables() to set nchunks internally.

        ------------
        "chunksize" <'Auto',int> is the number of rows per
        chunk.  It is expected that the last chunck may be
        smaller than the rest depending on the specified
        number of rows.

        ------------
        "nrows" is the integer number of rows specified for
        the whole core data array.

        ------------
        "ncols" is the integer number of columns specified for
        the whole core data array.

        ------------
        "Val" is a number to add to all the zeros.  If None,
        just the zeros are returned.

        ------------
        "chunkstart" is the row number of the first row of the
        first chunk.  It should be set to zero.  The pytables()
        function automatically increments by chunksize after
        calculating each chunk.

    Examples
    --------

    Paste function Args as Dict
    ---------------------------
        ArgDict = {'nchunks':_,
                   'chunksize':_,
                   'nrows':_,
                   'ncols':_,
                   'Val':_,
                   'chunkstart':0
                   }

    """
    # Get the first row of the top chunk
    TopChunk = (nchunks - 1) * chunksize

    # Get part of a chunk if last chunk
    if chunkstart >= TopChunk:
        ChunkPart0 = nrows % chunksize
        if ChunkPart0 == 0:
            ChunkPart = chunksize
        else:
            ChunkPart = ChunkPart0
        Chunk = np.zeros((ChunkPart,ncols))

    # Build rows for all complete chunks
    else:
        Chunk = np.zeros((chunksize,ncols))

    if Val is not None:
        Chunk += Val

    return Chunk


###########################################################################

def test(func, args, check='run', asserts=None, suffix=None, printout=True):
    return ut.test(func, args, check, asserts, suffix, printout)

# Assert functions
def obj_equal(obs, exp):
    return ut.obj_equal(obs, exp)

def allclose(obs, exp, tol=0.1):
    return ut.allclose(obs, exp, tol)

def damon_equal(obs, exp):
    return ut.damon_equal(obs, exp)

# Assign function docstring
test.__doc__ = ut.test.__doc__
allclose.__doc__ = ut.allclose.__doc__
damon_equal.__doc__ = ut.damon_equal.__doc__



###########################################################################

def test_damon(tests = ['All'], # [<['All', 'test_x', 'test_y'] => list of tests] 
               check = 'run', # [<'run', 'outputs', 'debug'> => test mode]
               printout = True # [<True, False> => console output]
               ):
    """Run specified damon unit tests.

    Return
    ------
        test_damon() returns a dict:

            {'test_func1':{'exceptions':{},
                           'n_tests':__,
                           'n_exceptions':__,
                           'time':__},
             'test_func2':__}

        As the function runs, it also prints out on the console a
        report of whether each covered Damon method or function passed
        each of a series of unit tests.

    Comments
    --------
        test_damon() is an automated way to check that Damon is working
        properly.  For each covered method or function, it sets up a
        scenario, makes sure it runs without exception, and checks its
        outputs against a file of outputs that have been visually
        inspected by the developer.

        test_damon() replaces test_damon_old(), which is retained only
        because it has a somewhat larger suite of tests.  test_damon()
        replaces the nose unit-testing framework with one customized
        for Damon called tester.py (see help(damon1.tester). damon1.tester
        is MUCH easier to write unit tests for, easier to debug, and
        it supports verification of outputs in a way that was impractical
        in the previous framework, and so rarely occurred.

        The Damon package comes with a directory called "test_asserts" that
        contains text files and pickle files intended to represent "verified"
        Damon ouptuts for each of a large number of data analysis scenarios.
        When these files are present, test_damon() checks that the current
        outputs (obtained by running test_damon()) match those on file,
        signified by the output phrase "Tests included assertions".  If they
        are not present for some reason (e.g., you deleted them), test_damon()
        only checks that the unit tests run without throwing exceptions.

        test_damon() gives you the power to overwrite these "test_asserts"
        files using the 'check' option (run with check='outputs' to overwrite,
        then check='run' to run the revised checks).  But there is really no
        reason you would want to do that.  If you want to, save off the
        developer files to another directory for safekeeping.

        If a unit test fails, you can run with check='debug' to get some
        handle on where the problem lies.

        As of this writing, test_damon() does a pretty good job of covering
        the essentials of Damon, but it is far from 100%. Feel free to
        write your own unit tests (see tools.test_func).

    Arguments
    ---------
        "tests" is the list of Damon functions or methods you want to check.
        The current unit test functions are in the damon1/tests.py module.

            tests = ['test_Damon', 'test_coord', 'test_create_data']
                        =>  run unit tests for Damon(), coord(), and
                            create_data().  Each unit test may run
                            hundreds of scenarios.

            tests = ['All'] (default)
                        =>  run all the existing unit tests, which cover
                            most of Damon's methods.  If these run without
                            exception, Damon's probably in pretty good
                            shape.
        --------
        "check" offers three unit testing modes:

            check = 'run' (default)
                        =>  Run the existing tests.  If the "test_asserts"
                            directory is populated, outputs for each scenario
                            will be checked and the console will say "Tests
                            included assertions."  Otherwise, the check is only
                            that each unit test ran without exception.

            check = 'outputs'
                        =>  Create a new set of outputs for each scenario,
                            overwriting any that may be in the "test_asserts"
                            directory.  You probably won't want this, but
                            if you do, save the existing .txt and .pkl files
                            to another directory first.

            check = 'debug'
                        =>  Run the unit test until an exception occurs.
                            Stop and print a traceback error report.

        --------
        "printout" controls how much information appears in the console
        report.

            printout = True (default)
                        =>  Gives a string of dots for each successful
                            run of a scenario, an 'E' if there is an
                            exception, plus some other printout info
                            depending on how check is set.

            printout = False
                        =>  Skips the dots.  Use this option for speed and
                            to produce more accurate time reports.

    Paste Function
    --------------
        test_damon(tests = ['All'], # [<['All', 'test_x', 'test_y'] => list of tests] 
                   check = 'run', # [<'run', 'outputs', 'debug'> => test mode]
                   printout = True # [<True, False> => console output]
                   )
    """
    import tests as ut
    
    dmn_tests = [f for f in dir(ut) if f.startswith('test_')]
    if tests == ['All']:
        tests = dmn_tests

    out = {}
    for test in dmn_tests:
        if test in tests:
            print '\nRunning ' + test + '():'
            out[test] = getattr(ut, test)(check, printout=printout)

    return out





###########################################################################

def test_damon_old(tests_ = ['All'],   # [<['All'],['test_bank.py','test_baseEAR.py',...]> => list of desired unit tests in 'damon/tests']
                   range_ = None,    # [<None, [3, 10]> => range of tests to run in file order]
                   ):
    """Run specified damon unit test using old (deprecated) unit testing tool.

    Returns
    -------
        test_damon_old() returns None but runs each of a specified
        set of unit test modules in the 'damon1/test' directory
        and prints out various reports for each.  Its main purpose
        is to run a fast check on all the primary damon methods
        before releasing a new version.

        If a given unit test does not return "OK", it is possible
        that the unit testing software is having trouble releasing
        one of the temporary files between runs.  Try running
        the unit test separately by opening the relevant unit
        test module and hitting F5 (run).  Sometimes it helps
        to reboot your machine.  If you still get an
        error, open the relevant test module, change the Mode
        to 'Inspect', run the module again, and try to reproduce
        the error.  You may see an easy work-around.  Debug if you
        feel courageous, or contact the Pythias-Damon forum.

        After test_damon() is run, all non-.py files are removed
        from the '/damon1/tests/' directory.

    Comments
    --------
        This function prints the results of running Damon unit
        tests.  These tests, so far, are not nearly as exhaustive
        as they could be.  They mainly check that a function or
        method runs properly across a variety of input scenarios
        and parameter configuration -- they don't compare against
        an ideal output to make sure the numbers are correct.  (With
        Damon, this type of test can be quite difficult to write.)
        Consider test_damon() as simply a heavy work-out of Damon.

        To make sure the numbers are right, the most convenient thing
        is run an individual unit test in "Inspect" mode and compare
        the inputs and outputs according to your understanding
        of the function.  In many cases the existing comparisons
        do not match simply because it's hard to independently replicate
        what the function should do, but if you read the documentation
        and play with your own examples you will get a good idea
        of what the results should look like and satisfy yourself
        that the function is giving you what you want.

        In time (perhaps with your help?), it is expected that
        the unit tests will become more rigorous and exhaustive.

        To run 'All' the tests takes around 10 minutes or so.

        Issues
        ------
        Sometimes test_damon() returns errors even though the
        individual unit tests run fine in isolation.  I'm not
        sure what causes this.  Make sure you don't have a
        directory named "damon" besides the one in
        site-packages/.  Sometimes it helps to run the
        function from a script rather than directly from the IDLE
        interpreter.  Sometimes reboot of IDLE helps, or a reboot
        of the system.  While this is inconvenient, you can
        always run the unit test individually by opening the
        relevant file and hitting F5.  That's the test that
        matters.

    Arguments
    ---------
        "tests_" is where you specify which unit tests to run.
        Specifying "All" will cause the function to run all
        of the unit test modules in 'damon/test' whose names
        lead with "test_".  Examples:

            tests_ = ['All']     =>  Run all available unit tests
                                    (with "test_" in the file name).

            tests_ = ['test_bank.py','test_base_resid.py']
                                =>  Run just the bank() and
                                    base_resid() unit tests.

            Notice that you have to type the full name of the
            unit test module as a string in quotes with a
            .py suffix.

        The order and selection of the tests does not matter.
        They are independent of each other.

        ---------
        "range_" is used when tests_ = ['All'] as a way to run a
        range of files within the total file list.

            range_ = None       =>  Don't select a range.

            range_ = [3, 5]     =>  Run unit tests number 3 (counting
                                    from zero) up to but not
                                    including 5, based on file
                                    order.

    Examples
    --------


    Paste function
    --------------
        test_damon(tests_ = ['All'],   # [<['All'],['test_bank.py','test_baseEAR.py',...]> => list of desired unit tests in 'damon/tests']
                   range_ = None,    # [<None, [3, 10]> => range of tests to run in file order]
                   )


    """

    # Import glob and test directory
    import damon1
    import glob
    import numpy.testing as npt

    print 'test_damon() is working...'

    if not isinstance(tests_,list):
        exc = "'tests_' parameter needs to be a list [...].\n"
        raise test_damon_Error(exc)

    # Set path and make it the current working directory
    path = damon1.__path__[0]+'/tests/'
    os.chdir(path)

    # Define list of tests
    if 'All' in tests_:
        run_tests = glob.glob(path+'test_*.py')

        if range_ is not None:
            run_tests = run_tests[range_[0]:range_[1]]
    else:
        run_tests = []
        for test in tests_:
            run_tests.append(path+test)

    for test in run_tests:
        print '\n\nRunning',os.path.basename(test),'...'
        npt.run_module_suite(test)

    # Clean up test outputs
    files = os.listdir(path)
    for file_ in files:
        if '.py' not in file_:
            try:
                os.remove(file_)
            except OSError:
                pass

    print '\ntest_damon() is done.\n'

    return None


###########################################################################

def get_damon_datadict(_damon_obj):
    '''Get the most recent datadict in the Damon workflow
    up to standardize_out.

    Returns
    -------
        A datadict containing keys like 'rowlabels', 'collabels',
        'coredata', etc.

    Comments
    --------

    Arguments
    ---------

    Paste Function
    --------------

    '''

    if not isinstance(_damon_obj,dmn.core.Damon):
        raise DamonException("_damon_obj argument of get_damon_datadict"+
                             " must be of type Damon")

    # Extract the correct data to analyze
    try:
        datadict = _damon_obj.standardize_out
    except AttributeError:
        try:
            datadict = _damon_obj.parse_out
        except AttributeError:
            try:
                datadict = _damon_obj.score_mc_out
            except AttributeError:
                try:
                    datadict = _damon_obj.extract_valid_out
                except AttributeError:
                    try:
                        datadict = _damon_obj.merge_info_out
                    except AttributeError:
                        try:
                            datadict = _damon_obj.data_out
                        except:
                            exc = 'Unable to find data to analyze.\n'
                            raise DamonException(exc)

    return datadict

###########################################################################

def get_datatype(damon_obj):
    '''
    this method returns the information about type of data of damon object.
    it returns a dictionary containing flags (isContinuous , isNumeric , isDichotomous)
    '''

    #initialise dictionary to be returned
    get_datatype_output={'isContinuous':False,
                         'isNumeric':False,
                         'isDichotomous':True}

    #get the correct data for damon object
    damon_datadict=get_damon_datadict(damon_obj)
    coredata=damon_datadict["coredata"]

    #get column keys of coredata
    col_keys= getkeys(damon_datadict,'Col','Core').tolist()

    #get valid chars for data.
    valid_chars=guess_validchars(coredata,
                                 col_keys,
                                 None,
                                 damon_obj.nanval)

    #set isnumeric flag
    if(len(valid_chars)==3 and valid_chars[2]=='Num'):
        get_datatype_output['isNumeric']=True
    else:
        get_datatype_output['isNumeric']=False

    #first take case , when valid chars are formated according to columns
    if(valid_chars[0]=='Cols'):
            #iterate over all columns in validchars
            for col_key in valid_chars[1]:

                #get column chars
                col_chars = valid_chars[1][col_key]

                #if any of cols has continous data, then flag whole data as continous
                if(len(col_chars)==1 and col_chars[0]=='All'):
                    get_datatype_output['isContinuous']=True
                    get_datatype_output['isDichotomous']=False
                else:
                    #if any of cols has non dichotomous data, then flag whole data as non dichotomous
                    if(len(col_chars)>2):
                        get_datatype_output['isDichotomous']=False
    else:
        if(valid_chars[0]=='All'):

            all_chars=valid_chars[1]

            if(len(all_chars)==1 and all_chars[0]=='All'):
                 get_datatype_output['isContinuous']=True
                 get_datatype_output['isDichotomous']=False
            else:
                #if any of cols has non dichotomous data, then flag whole data as non dichotomous
                if(len(all_chars)>2):
                    get_datatype_output['isDichotomous']=False

    return get_datatype_output


###########################################################################

def flag_items(col_ents_out,
               rel_lt = 0.10,
               ptbis_lt = 0.20,
               outfit_gt = 1.7,
               op = 'or',
               exclude = False
               ):
    """Flag items based on summstat statistics in col_ents_out.
    
    Returns
    -------
        flag_items() returns an array of item keys that meet
        specified statistical criteria.
    
    Comments
    --------
        An important part of psychometrics is editing a dataset
        until it conforms to the model's expectations.  flag_items()
        looks at three such criteria:
        
            reliability     =>  Is the item reliable enough, i.e., does
                                it differentiate persons in a way that
                                is robust to random error?
            
            point-biserial  =>  Are the items sufficiently correlated to
                                the construct?
                                
            outfit          =>  How well do the observations fit the
                                model estimates?
        
        Other criteria can also be used to evaluate items (e.g., item
        drift, differential item functioning, targeting, etc.), but these
        three cover most cases.  The point-biserial correlation actually
        plays a questionable role in evaluating multidimensional datasets
        and those where items may be negatively correlated with each
        other, but the great majority of assessments are constructed 
        to be more or less unidimensional with positively correlated
        items.
        
        The default statistical criteria will do a reasonably good job
        of flagging poorly performing items without flagging too many,
        but they aren't much better than rules of thumb and are somewhat
        dependent on the dataset.  Expect to tweak these parameters as
        you do item analysis.  Items that fail on multiple criteria at
        once tend to be the ones that are problematic.
        
        Rules of Thumb
        --------------
            1.  Bad items tend to have reliability = 0.0.  This means
                the average (RMS) error of the cell estimates for
                those items is larger than the spread of the estimates
                across persons.  In other words, the spead of the persons
                is indistinguishable from chance.  The item is telling
                us literally nothing.  However, reliability is
                sample-dependent.  It will fluctuate depending on
                whether the persons are widely spread out in ability
                or not.
            
            2.  Point-biserial < 0.20 (including negative) is widely
                used in the industry for flagging items that are
                insufficiently discriminating and don't adhere to the
                test construct.  The point-biserial is less meaningful
                for truly multidimensional datasets.
            
            3.  Outfit > 1.4 tends to signify that the estimates
                are "significantly" (p<0.05) different from the
                observations.  However, that also means you
                can expect about five such items by chance on a 100
                item test, which you probably wouldn't want to flag.
                So I would set the outfit higher and not rely on
                outfit in isolation to flag items.
            
            Of the three statistics, I find that item reliability
            is the most useful in flagging bad items.
    
        flag_items() requires as its input the col_ents_out Damon obj
        which is an output of Damon.summstat().  The function is
        intended to be passed to the Damon.flag() 'flag_rows' parameter.
        (In col_ents_out, items are rows rather than columns.)
    
    Arguments
    ---------
        "col_ents_out" is the Damon obj that is one of the outputs of
        Damon.summstat().  flag_items() won't work on any other input.
        
        ---------
        "rel_lt" stands for "reliability less than", where 0 <= rel <= 1.
            
            rel_lt = 0.50   =>  Flag items with reliability less than 0.50.
            
            rel_lt = 1.0    =>  All reliabilities are permitted.
        
        ---------
        "ptbis_lt" stands for "point-biserial correlation less than", where
        -1.0 < ptbis < 1.0.

            ptbis_lt = 0.20 =>  Flag items with point-biserials less than 0.20.
            
            ptbis_lt = -1.0 =>  All point-biserials are permitted.
        
        ---------
        "outfit_gt" stands for "outfit greater than", where 0.0 < outfit < inf.
        Ideally, outfit = 1.0, indicating that the amount of misfit is what
        you would expect given the noise of the dataset.
        
            outfit_gt = 1.5 =>  Flag items with misfit greater than 1.50.
            
        ---------
        "op" <and, or> stands for "logical operator".
        
            op = 'and'      =>  Flag items that meet criterion 1 AND criterion 2
                                AND criterion 3.
            
            op = 'or'       =>  Flag items that meet criterion 1 OR criterion 2
                                OR criterion 3.

        ---------
        "exclude" <bool> means return all items that are NOT flagged.
                            
            exclude = False =>  Return flagged items.
            
            exclude = True  =>  Return all items that are NOT flagged.
        
        Paste Function
        --------------
            flag_items(col_ents_out,
                       rel_lt = 0.50,
                       ptbis_lt = 0.20,
                       outfit_gt = 1.5,
                       op = 'or',
                       exclude = False
                       )
    """
    d = col_ents_out
    
    if op == 'or':
        ix = [(d.core_col['Rel'] < rel_lt) |
              (d.core_col['PtBis'] < ptbis_lt) |
              (d.core_col['Outfit'] > outfit_gt)]
            
    elif op == 'and':
        ix = [(d.core_col['Rel'] < rel_lt) &
              (d.core_col['PtBis'] < ptbis_lt) &
              (d.core_col['Outfit'] > outfit_gt)]
    
    keys = getkeys(d, 'Row', 'Core')
    
    if exclude:
        rows = np.setdiff1d(keys, keys[ix])
    else:
        rows = keys[ix]
    
    return rows
    
        

###########################################################################

def flag_invalid(datadict,
                 axis = 'rows', # [<'rows', 'cols'> => axis to evaluate]
                 min_count = 10, # [<None, int, float> => minimum values required]
                 min_sd = 0.001, # [<None, float> => minimum standard deviation]
                 rem = None, # [<None, [ents...]> => list of ents to flag]
                 ):
    """Flag rows and columns for removal from analysis.
    
    Returns
    -------
        List of invalid row or column entities.
    
    Comments
    --------
        This function is used by the extract_valid() and flag()
        methods.
    
    Arguments
    ---------
        "datadict" is a Damon data dictionary.
    
        ----------
        "axis" <'rows', 'cols'> is the axis of the coredata array
        containing row or column entities to be flagged.
        
        ----------
        "min_count" <None, 10, 0.25> is the minimum number of data values 
        that a row or column entity must contain.  If a decimal is
        specified, the minimum number is calculated by multiplying
        the total size of the row or column by min_count.
        
            min_count = 10      =>  entity must contain at least 10
                                    values.
            
            min_count = 0.25    =>  at least 25% of the values must
                                    be valid.
        
        ----------
        "min_sd" is the minimum required standard deviation of the values
        in the row or column.  The entity is not valid unless there is
        at least this amount of variation.
        
        ----------
        "rem" is a list of entities to remove, to be added to those
        flagged by the other criteria.
    
    Paste Function
    --------------
        flag_invalid(datadict,
                     axis = 'rows', # [<'rows', 'cols'> => axis to evaluate]
                     min_count = 10, # [<None, int, float> => minimum values required]
                     min_sd = 0.001, # [<None, float> => minimum standard deviation]
                     rem = None, # [<None, [ents...]> => list of ents to flag]
                     )
    
        
    """
    if isinstance(datadict, dict):
        d = dmn.core.Damon(datadict, 'datadict', verbose=None)
    else:
        d = datadict
    
    # Get axis variables
    ax = {'rows':1, 'cols':0}
    fac = {'rows':'Row', 'cols':'Col'}
    keys = getkeys(d, fac[axis], 'Core')
    
    # Get entities with insufficient counts
    ents = []
    if min_count is not None:    
        if isinstance(min_count, float) and min_count < 1.0:
            total = np.size(d.coredata, ax[axis])
            min_count = round(min_count * total)
        counts = count(d.coredata, ax[axis], d.nanval)
        ents.extend(list(keys[counts < min_count]))
    
    # Get entities with insufficient variation
    if min_sd is not None:
        stds = std(d.coredata, ax[axis], d.nanval)
        ents.extend(list(keys[stds < min_sd]))
    
    # Get specified entities
    if rem is not None:
        ents.extend(list(rem))
    
    # Return array of uniques
    ents = np.unique(ents)

    return ents


    
    

###########################################################################

def flag_item_drift(form,
                    bankfile,
                    prefix,
                    rmse_gt = 3.0
                    ):
    """Check degree to which new item parameters deviate from bank.
    
    Returns
    -------
        flag_item_drift() return a dict giving the strength of the
        relationship between item (column) coordinates as calculated 
        from data for a particular form and the item coordinates as
        stored in a bank.
        
            {'corr':__,         =>  bank vs form Pearson correlation
            
             'rmse':__,         =>  bank vs form, root mean distance from
                                    the identity line.
                                    
             'contrast':__,     =>  mean(bank) - mean(form) / rmse
             
             'drift':__,        =>  datadict of drift stats per item per
                                    dimension.  In a 2-dimensional
                                    coordinate system, the item will
                                    appear twice.
                                    
             'items':__         =>  list of items more than rmse_gt se's
                                    from the identity line
             }
        
        In addition, flag_item_drift() displays and prints out an
        identity-line scatterplot.
    
    Comments
    --------
        Items are not supposed to change their behavior over time or 
        across forms, but they do.  Examinees interact with them in 
        new ways, more notoriously by getting a peak at the item, or
        ones similar to it, before the test.  flag_item_drift() makes
        it possible to compare an item as it performs on a particular
        form and how it is recorded in the bank.
        
        This is done using coord()'s anchor/Refresh_All option.  In the
        absence of drift the refreshed version of the item parameters
        should match their anchor values in the bank.  To the degree
        they don't, Damon defines that as "item drift".
        
        flag_item_drift() outputs some strength-of-relationship summary 
        statistics, plus a "drift" statistic for each item on each
        dimension with a list of flagged items.  But the most important
        output is a scatterplot of "Form" values versus "Bank" values.
        Visual inspection -- looking for items that pop off the identity line 
        in a way that seems nonrandom -- is the best way to identify items 
        that have changed behavior.
        
        This point is important.  The function's rmse_gt parameter allows
        the user to specify how many standard errors from the identity line
        an item needs to be to get flagged.  However, one would expect by
        chance that, if rmse_gt = 2.0, some 5% of items will be flagged -- not
        because of something odd about the items, but just by chance.
        If rmse_gt = 3.0, one would expect 1% or less.  What you want is
        to identify non-chance departures from the identity line.  This is
        partly determined by counting how many, how far off the line, and
        whether there is possible explanation for changed behavior.
        
        So don't use the list of flagged items blindly.  Flagging and
        removing iteratively will ultimately result in removing most
        items from the test.
    
    Arguments
    ---------
        "form" is a datadict or Damon object containing person x item
        data for a particular "form" of the test.
        
        ------
        "bankfile" is the str filename or path of the bank pickle file.
        
        ------
        "prefix" is how you would like to prefix the filename of the
        output scatterplot.
        
        -----
        "rmse_gt" stands for "root mean square error greater than" the
        value you specify.  Flag items that have (for any of their
        dimensions) a distance from the identity line that is greater
        than the value you specify.  If you specify rmse_gt = 2.0, expect
        to see roughly 5% of your items flagged by chance.  rmse_gt = 3.0 
        should flag roughly 1% of your items by chance.
    
    Paste Function
    --------------
        flag_item_drift(form,
                        bankfile,
                        prefix,
                        rmse_gt = 3.0
                        )
    """
    if not isinstance(form, dmn.core.Damon):
        f = dmn.core.Damon(form, 'datadict', 'RCD_dicts_whole', verbose=None)
    else:
        f = form
    
    # Calculate coords
    f.extract_valid(minsd=0.001)
    f.standardize(std_params=bankfile)    
    f.coord(anchors={'Bank':bankfile, 'Facet':1, 'Entities':['All'], 
                     'Coord':'ent_coord', 'Refresh_All':True})
    bank = np.load(bankfile)
    bank_coord = bank['facet1']['ent_coord']
    ent_coord_ = dmn.core.Damon(f.coord_out['fac1coord'], 'datadict', verbose=None)
    ent_coord = ent_coord_.core_row
    
    # Pull relevant coordinates from bank
    f_keys = getkeys(f.coord_out['fac1coord'], 'Row', 'Core')
    b_coords_ = bank['facet1']['ent_coord'] 
    b_keys = b_coords_.keys()
    keys = np.intersect1d(f_keys, b_keys)
    f_c = np.array([ent_coord[key] for key in keys])
    b_c = np.array([bank_coord[key] for key in keys])
    
    # Get stats and plot
    x = np.ravel(b_c)
    y = np.ravel(f_c)
    xy_labels = np.repeat(keys, 2)
    out = plot_identity(x, y, xy_labels, 'Drift', 'Bank', 'Form', 
                        prefix+'_drift', f.nanval)
    
    # Get drift
    sin_45 = 0.70710678118
    fb_diffs = (sin_45 * (y - x)) / out['rmse']
    
    # Build drift datadict
    rowlabels = np.append(np.array([['Item']]), xy_labels[:, np.newaxis], axis=0)
    collabels = np.array([['item_dim', 'drift']])
    coredata = fb_diffs[:, np.newaxis]
    
    drift = {'rowlabels':rowlabels, 'collabels':collabels, 
             'coredata':coredata, 'key4rows':0, 'key4cols':0,
             'rowkeytype':'S60', 'colkeytype':'S60', 
             'validchars':['All', ['All'], 'Num'], 'nanval':f.nanval}
    
    flags = abs(fb_diffs) > rmse_gt
    drift_items = np.unique(xy_labels[flags])
    
    return {'corr':out['corr'], 'rmse':out['rmse'], 'contrast':out['contrast'],
            'drift':drift, 'items':drift_items}






###########################################################################

def check_equating(form_a, # [<Damon obj> Form A student responses]
                   form_b, # [<Damon obj> Form B student responses]
                   bankfile, # [<str> => name/path of bank file]
                   construct, # [<str> => construct or subscale in equate()]
                   prefix='eq', # [<str> => output scatterplot files prefix]
                   overlap=True # [<bool> => include links in form data]
                   ):
    """Check the degree to which two test forms are equated.
    
    Returns
    -------
        check_equating() exports two identity line scatterplots and returns 
        a dictionary of useful stats:
        
            - 'nForm_A': number of form a items
            - 'nForm_B': number of form b items
            - 'nlinks': number of items linking forms a and b
            
            - 'Form_A':{'corr': correlation, form_a vs links,
                        'rmse': root mean square error (diff), form_a vs links}
                        'contrast': (mean(form_a) - mean(links) / rmse)
                        
            - 'Form_B':{'corr': correlation, form_b vs links,
                        'rmse': root mean square error (diff), form_b vs links}
                        'contrast': (mean(form_b) - mean(links) / rmse)
            
            - 'Form_A_err': median student measurement error, Form A
            
            - 'Form_B_err': median student measurement error, Form B

        Note: In addition to outputting scatterplots, the Form B scatterplot
        will probably appear in the console window.

        Workflow:
            1) Calibrate items in a separate analysis
                d = dmn.Damon()
                d.coord()
                d.base_est()
                d.base_resid()
                d.base_ear()
                d.base_se()
                d.equate()  
                d.bank('my_bank.pkl')
            
            2) Check equating
                form_a = dmn.Damon(form_a data, ...)
                form_b = dmn.Damon(form_b_data, ...)
                eq = tools.check_equating(form_a, form_b, 'my_bank.pkl', ...)

    Comments
    --------
        How do we know that the scores of examinees that take different
        test forms (with linking items) are comparable?  In principle, the
        coordinates of the linking items force all items into a common space
        so that student scores are automatically comparable.  In practice,
        the noisiness of the data and violations of the model will cause
        a degradation in comparability.  check_equating() is a way to
        quantify and visualize the degree of degradation.
        
        Here's how it works.  check_equating() assumes a scenario where
        two test forms are linked by common items and students have taken
        only one of the test forms.  (Ideally, students would take 
        both forms but that rarely happens; this scenario is not supported.)
        It also assumes that all items have been calibrated and their
        coordinates stored in a bank (see the Damon.bank() docs).
        
        check_equating() tries to calculate how all students would have scored
        on both forms.  Say the students took Form A.  check_equating() uses
        their responses combined with the coordinates of the Form A items
        to calculate "Form A" measures.  It then uses their responses
        to just the linking items (linked to Form B) to calcualate scores
        that are a proxy for how they WOULD have scored on Form B.  These
        are their "Form B" measures.  The Form A and Form B measures are
        graphed against each other.  The degree to which they depart from
        the identity line is an estimate of how well the forms are equated.
        The scatterplots are supplemented by the following statistics:
        
            'corr'      =>  the correlation between Form A and "Form B" scores.
                            corr = 1.0 signifies perfect equating.
                            
            'rmse'      =>  the root mean squared error (difference) between 
                            scores for Forms A and B, or more accurately 
                            the distance between the (A, B) coordinate and
                            the identity line, which is a bit different.
                            rmse = 0.0 signifies perfect equating.
            
            'contrast'  =>  the Form A mean minus the Form B mean, divided by
                            the rmse statistic. (The term 'contrast' is 
                            borrowed from DIF analysis.) It answers the 
                            question, how many standard errors apart are
                            the scores for the two forms in a positive or
                            negative direction? contrast = 0.0 signifies 
                            that the forms are not biased for or against
                            the students who take them.
                            
            'Form_A_Err'=> the median student measurement error for Form A.

            'Form_B_Err'=> the median student measurement error for Form B.
            
        The process is repeated for students who took Form B, yielding
        similar scatterplots and equating statistics.  Here the 
        linking items are a proxy for student Form A scores.
        
        When doing just the Form A analysis, a high correlation does not
        guarantee that the forms are perfectly equated inasmuch as the 
        students may perform very differently on the Form B non-linking
        items.  However, when the Form B analysis is taken into account,
        this possibility can be ruled out.  In other words, if the Form A
        scores are strongly correlated with those of the linking items and 
        the Form B scores are also strongly correlated with the linking
        items, then we can infer that were students to take both complete
        forms that their Form A and Form B scores would also be strongly
        correlated.  The forms are unambiguously equated.
        
        This type of equating evaluation is, so far as I know, unique to
        Damon.  There are no guidelines yet reqarding "typical" correlations
        or contrasts or acceptable bounds.
        
        Overlapping Items
        -----------------
        Note that check_equating() supports two ways of defining the two
        test forms.  Say the students took Form A.  We can calculate their
        scores either by including the linking items or omitting them, 
        controlled by the overlap parameter.  Including the linking items
        in the overall form score naturally inflates the correlation with 
        the linking items.  While such an inflation could be viewed as
        somehow artifial, I have made it the default approach because it more
        realistically represents the similarity between scores generated
        from two test forms.  Forms with ALL items in common SHOULD approach 
        an equating correlation of 1.0.  
        
        That said, omitting the linking items from the Form A measure
        is informative in checking the degree to which the linking and 
        non-linking items get at the same construct.
        
        Equating Parameters
        -------------------
        The details of how to conduct the equating, how to rescale, etc.,
        is stored in the "bankfile" parameter; you shouldn't have to mess
        with that.  If the Damon.bank() method was run after Damon.equate()
        during the item calibration phase, and equate() included for example
        a rescaling to logits and transformation to a 100 - 600 scale, the
        corresponding scores should appear on the check_equating()
        scatterplot axes.
        
    Arguments
    ---------
        "form_a" <Damon object> should hold student numerical response data
        for Form A.  If the raw data is not numerical, eg, some items 
        have multiple choice responses, run Damon.score_mc() to score 
        the Form A data. You may also want to make sure all the rows and
        columns are valid. For example:
        
            data = dmn.Damon(...)
            data.extract_valid(10, 10, 0.001)
            data.score_mc(...)
            form_a = dmn.Damon(data.score_mc_out, 'datadict', verbose=None)
        
        -------------
        "form_b" <Damon object> holds numerical response data for Form B.

        -------------
        "bankfile" <str> is the path or name of the bank file containing
        item parameters and other information, built during the item
        calibration and equating phase as shown in "workflow" above.

        -------------
        "construct" <str> is the name of the construct or subscale for
        which equating is to be assessed.  It will be one of the "construct"
        column headers in my_obj.equate_out['Construct'], where my_obj
        was built during the item calibration phase.
        
        -------------
        "prefix" <str> is the prefix to be added to the scatterplot .png
        files.  prefix = 'eq' will generate files called 'eq_Form_A.png'
        and 'eq_Form_B.png' and drop them in the current working directory.
        
        -------------
        "overlap" <bool> specifies whether to include the linking items
        with all the other items in a given form when generating scores
        for that form.  overlap = True (default) will yield a higher
        correlation depending on the number of linking items, indicative
        of greater similarity with the opposing form.  overlap = False
        will remove the increase in correlation caused by overlapping items
        (All Form A items with Form A linking items) and clarify whether
        the linking and non-linking items in the Form embody the same 
        construct.
        
    Paste Function
    --------------
        check_equating(form_a, # [<Damon obj> Form A student responses]
                       form_b, # [<Damon obj> Form B student responses]
                       bankfile, # [<str> => name/path of bank file]
                       construct, # [<str> => construct or subscale in equate()]
                       prefix='eq', # [<str> => output scatterplot files prefix]
                       overlap=True # [<bool> => include links in form data]
                       )    

    """
    # Get construct measures from form_a
    se = False # True allows comparison of student errors, but not meaningful.
    form_a.name = 'a'
    form_b.name = 'b'  
    fa_items = getkeys(form_a, 'Col', 'Core')
    fb_items = getkeys(form_b, 'Col', 'Core')
    links = np.intersect1d(fa_items, fb_items)
    
    if len(links) == 0:
        exc = 'There need to be items in common between form_a and form_b.'
        raise ValueError(exc)
    
    nanval = form_a.nanval
    names = set(['a', 'b'])
    meas = {}
    err = {}
    
    for f_ in [form_a, form_b]:
        
        # Use only valid rows and columns
        f_.extract_valid(5, 5, 0.001)
        f = dmn.core.Damon(f_.extract_valid_out, 'datadict', verbose=None)
        f.name = f_.name
        
        # Extract the links to represent the opposing form
        x = f.extract(f, 
                      getrows={'Get':'AllExcept', 'Labels':'key', 'Rows':[None]},
                      getcols={'Get':'NoneExcept', 'Labels':'key', 'Cols':links})
        e = dmn.core.Damon(x, 'datadict', verbose=None)
        e.name = list(names - set(f.name))[0] + '_links'
        meas[f.name] = {}

        # To have no items in common between the two comparison groups
        if not overlap:
            f_items = getkeys(f_, 'Col', 'Core')
            uniques = np.setdiff1d(f_items, links)
            x_f = f.extract(f,
                            getrows={'Get':'AllExcept', 'Labels':'key', 
                                     'Rows':[None]},
                            getcols={'Get':'NoneExcept', 'Labels':'key', 
                                     'Cols':uniques})
            f = dmn.core.Damon(x_f, 'datadict', verbose=None)
            f.name = f_.name

        # For both item groups, get person measures using bank item anchors
        for d in [f, e]:
            d.standardize(std_params=bankfile)
            d.coord(None, anchors={'Bank':bankfile,
                                   'Facet':1,
                                   'Entities':['All'],
                                   'Freshen':None})
            d.base_est()
            d.base_resid()
            d.base_ear()
            d.base_se()
            d.equate('Bank')
            construct_ = dmn.core.Damon(d.equate_out['Construct'], 'datadict', 
                                       verbose=None)
            se_ = dmn.core.Damon(d.equate_out['SE'], 'datadict', verbose=None)
            m = construct_ if se is False else se_
            meas[f.name][d.name] = m.core_col[construct]
            
            if d == f:
                s = se_.core_col[construct]
                err[f.name] = np.median(s[s != nanval])

    # Get measures for form_a and form_b examinees
    m_a = meas['a']['a']
    m_a_blinks = meas['a']['b_links']
    m_b = meas['b']['b']
    m_b_alinks = meas['b']['a_links']
    a_err = round(err['a'], 3)
    b_err = round(err['b'], 3)
    
    # Scores or errors
    mtype = 'Scores' if se is False else 'Errors'
    
    # form_a relationship information
    if not overlap:
        xlabel = mtype + ' Using Only Nonlinking Form A Items'
    else:
        xlabel = mtype + ' Using All Form A Items'

    rel_a = plot_identity(m_a, m_a_blinks, 
                          title = 'Form A Students', 
                          xy_labels = None,
                          xlabel = xlabel,
                          ylabel = mtype + ' Using Links to Form B',
                          out_as = prefix + '_Form_A',
                          nanval = nanval
                          )

    # form_a relationship information
    if not overlap:
        xlabel = 'Scores Using Only Nonlinking Form B Items'
    else:
        xlabel = 'Scores Using All Form B Items'
        
    rel_b = plot_identity(m_b, m_b_alinks, 
                          title = 'Form B Students', 
                          xlabel = xlabel,
                          ylabel = mtype + ' Using Links to Form A',
                          out_as = prefix + '_Form_B',
                          nanval = nanval)
   
    return {'Form_A':rel_a, 'Form_B':rel_b, 
            'Form_A_err':a_err, 'Form_B_err':b_err,
            'nForm_A':len(fa_items), 'nForm_B':len(fb_items), 
            'nlinks':len(links)}





###########################################################################
def get_cuts(measures, ratings, nanval=-999):
    """Get scale cut-points given expert ratings.
    
    Returns
    -------
        get_cuts() returns a list of measure cut-points for 
        classifying measures into performance levels.
    
    Comments
    --------
        get_cuts() is intended to remove some of the pain
        associated with "standard-setting", the process of
        setting cut-scores for assigning each person a performance
        levels based on his or her measure.
        
        In conventional standard-setting (the "bookmark method"),
        items are arranged in difficulty order and a panel of
        experts decides at which scale score a student transitions
        from one performance level the next.  A student at that
        point is called the "minimally proficient student."
        
        The "Angoff method" has experts decide, for each item,
        whether the minimally proficient student would get it
        right.
        
        Both methods run into problems.  The bookmark method only
        works for unidimensional tests, ruling out most Damon
        datasets.  The Angoff method requires experts to make
        predictions that are, cognitively, very hard to get right.
        
        Damon adopts the "expert recommender" approach associated
        with machine learning.  Here, experts review each person's
        responses (or a representative sample thereof) and assign
        a rating to each student.  The algorithm then predicts the
        expert rating from the student responses.  Damon can do
        this internally by modeling the expert responses included
        as a column in the data file, or it can use get_cuts() to
        assign cut-points to a a continuous scale.  The function
        can be used stand-alone or via Damon's equate() method.
        
        The method is as follows:
            
            1.  Given an array of continuous measures and a 
                corresponding array of performance level ratings,
                for each rating calculate the median measure
                associated with it.
            
            2.  Calculate the cut-points by finding the midpoint
                between each successive pair of medians:
                    
                    cut[i] = (median[i] - median[i-1]) / 2
    
        The cuts are returned and can then be applied to the
        measures to assign person performance levels.
    
    Arguments
    ---------
        "measures" is an array of continuous measures.
        
        ---------
        "ratings" is an array of expert categorical ratings that 
        correspond to the measures.
        
        ---------
        "nanval" is the not-a-number value.

    """
    
    # Get rid of nanvals
    valix = (measures != nanval) & (ratings != nanval)
    measures = measures[valix]
    ratings = ratings[valix]

    # Calculate median measure for each rating
    uniques = np.unique(ratings)
    meds = []

    for rating in uniques:
        meds.append(np.median(measures[ratings == rating]))
    
    # Cut is defined as the measure equidistant between two medians
    cuts = []
    for i, med in enumerate(meds):
        if i > 0:
            cut = round((med + meds[i - 1]) / 2.0, 2)
            cuts.append(cut)
            
    return cuts
        

###########################################################################
# Next function









