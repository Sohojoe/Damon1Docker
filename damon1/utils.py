# -*- coding: utf-8 -*-
"""
The utils.py module contains functions which support
Damon() methods in the core.py module.

It is a module in the damon1 package.

"""
# Import system modules
import os
import sys

# Import numpy and other python modules
import cPickle
import csv
import ast

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.ma as npma

import damon1 as dmn
import damon1.tools as tools

try:
    import tables as tab
except:
    pass


# Define exceptions
class Damon_Error(Exception): pass
class create_data_Error(Exception): pass
class TopDamon_Error(Exception): pass
class merge_info_Error(Exception): pass
class extract_valid_Error(Exception): pass
class pseudomiss_Error(Exception): pass
class score_mc_Error(Exception): pass
class subscale_Error(Exception): pass
class parse_Error(Exception): pass
class standardize_Error(Exception): pass
class rasch_Error(Exception): pass
class best_dim_in_coord_Error(Exception): pass
class seed_in_coord_Error(Exception): pass
class coord_Error(Exception): pass
class sub_coord_Error(Exception): pass
class objectify_Error(Exception): pass
class base_est_Error(Exception): pass
class base_resid_Error(Exception): pass
class fin_resid_Error(Exception): pass
class base_ear_Error(Exception): pass
class est2logit_Error(Exception): pass
class base_se_Error(Exception): pass
class base_fit_Error(Exception): pass
class fin_fit_Error(Exception): pass
class fin_est_Error(Exception): pass
class fillmiss_Error(Exception): pass
class item_diff_Error(Exception): pass
class summstat_Error(Exception): pass
class merge_summstat_Error(Exception): pass
class plot_two_vars_Error(Exception): pass
class wright_map_Error(Exception): pass
class equate_Error(Exception): pass
class bank_Error(Exception): pass
class restore_invalid_Error(Exception): pass
class export_Error(Exception): pass
class extract_Error(Exception): pass



#######################################################################

def _data(_locals):
    "Basis of the Damon __init__ method."

    # Get self
    self = _locals['self']

    # Retrieve variables from _locals
    data = _locals['data']
    format_ = _locals['format_']
    workformat = _locals['workformat']
    validchars = _locals['validchars']
    nheaders4rows = _locals['nheaders4rows']
    key4rows = _locals['key4rows']
    rowkeytype = _locals['rowkeytype']
    nheaders4cols = _locals['nheaders4cols']
    colkeytype = _locals['colkeytype']
    key4cols = _locals['key4cols']
    check_dups = _locals['check_dups']
    dtype = _locals['dtype']
    nanval = _locals['nanval']
    missingchars = _locals['missingchars']
    miss4headers = _locals['miss4headers']
    recode = _locals['recode']
    cols2left = _locals['cols2left']
    selectrange = _locals['selectrange']
    delimiter = _locals['delimiter']
    pytables = _locals['pytables']
    verbose = _locals['verbose']
    OrigInputs = None
    fileh = None

    # Interpret variables
    if pytables is False:
        pytables = None

    if pytables is not None and verbose is True:
        print ('\nWarning in Damon(): Due to issues between PyTables and '
               'Windows the pytables argument is no longer being tested '
               'and will be phased out.  Out of memory dataset will be '
               'handled differently in damon2\n')

    # Overwrite label types if necessary:
    if (nheaders4rows > 0
        and (key4rows is None
             or rowkeytype is None)
        ):
        key4rows = 0 if key4rows is None else key4rows
        rowkeytype = 'S60' if rowkeytype in [None,object,str] else rowkeytype

    if (nheaders4cols > 0
        and (key4cols is None
             or colkeytype is None)
        ):
        key4cols = 0 if key4cols is None else key4cols
        colkeytype = 'S60' if colkeytype in [None,object,str] else colkeytype

    # Get and assign path name
    if os.getcwd() == '/':
        import damon1.tests.play
        Path = damon1.tests.play.__path__[0]
    else:
        Path = os.getcwd()
    self.path = Path+'/'

    if pytables is not None and isinstance(pytables,str):
        pytables = self.path+'temp_'+pytables


    # Redefine rowkeytype and colkeytype to prevent accidental truncation of strings
    if rowkeytype == str:
        rowkeytype = dtype[0]
    if colkeytype == str:
        colkeytype = dtype[0]


    #################
    ##  'Damon'    ##
    ##   format    ##
    #################

    if format_ == 'Damon':
        datadict = data.__dict__
        datadict['fileh'] = None

        if pytables is not None:
            TargArrays = ['rowlabels','collabels','coredata']

            if isinstance(pytables,str):
                Mode = 'w'
            else:
                Mode = None

            # Convert to pytables
            if isinstance(datadict['coredata'],np.ndarray):
                TabOut = tools.pytables_(datadict,'datadict',pytables,Mode,'data_out',
                                            TargArrays,None,None,None,None,None)
            # data is already a dictionary of pytables arrays
            else:
                TabOut = tools.pytables_(datadict,'hd5',pytables,Mode,'data_out',
                                            TargArrays,None,None,None,None,None)
            # Complete new datadict
            Result = {}
            for key in datadict.keys():
                if key in TargArrays:
                    Result[key] = TabOut['arrays'][key]
                else:
                    Result[key] = datadict[key]
            Result['pytables'] = pytables
            Result['fileh'] = TabOut['fileh']
        else:
            Result = {}
            datadict = data.__dict__
            for key in datadict.keys():
                Result[key] = datadict[key]

        return Result


    ################
    ##  'pickle'  ##
    ##   format   ##
    ################

    if format_ == 'pickle':
        ResultFile = open(data,'rb')
        Result = np.load(ResultFile)
        ResultFile.close()

        if verbose is True:
            print ("Note: format_ = 'pickle' just opens a pickle file and "
                   "loads data.  It doesn't format it.\n")

        return Result


    ######################
    ##  'datadict_link'  ##
    ##      format      ##
    ######################

    # Link to existing datadict
    if format_ == 'datadict_link':

        try:
            # Add missing fields
            data['verbose'] = None
            data['rl_row'] = None
            data['rl_col'] = None
            data['cl_row'] = None
            data['cl_col'] = None
            data['core_row'] = None
            data['core_col'] = None
            data['whole_row'] = None
            data['whole_col'] = None
            data['whole'] = None
            data['fileh'] = None
            data['dtype'] = [object, 3, None]

        except TypeError:
            exc = "'data' needs to be a datadict.\n"
            raise Damon_Error(exc)
            
        # Make sure rowlabels, collabels, and nHeaders parameters are present
        try:
            rowlabels = data['rowlabels']
        except KeyError:
            exc = 'datadict is missing rowlabels.\n'
            raise Damon_Error(exc)

        try:
            collabels = data['collabels']
        except KeyError:
            exc = 'datadict is missing collabels.\n'
            raise Damon_Error(exc)

        try:
            data['nheaders4rows']
        except KeyError:
            data['nheaders4rows'] = np.size(rowlabels,axis=1)

        try:
            data['nheaders4cols']
        except KeyError:
            data['nheaders4cols'] = np.size(collabels,axis=0)

        # Convert arrays to pytables / Read existing pytables
        if pytables is not None:
            TargArrays = ['rowlabels','collabels','coredata']

            if isinstance(pytables,str):
                Mode = 'w'
            else:
                Mode = None

            # Convert to pytables
            if isinstance(data['coredata'],np.ndarray):

                TabOut = tools.pytables_(data,'datadict',pytables,Mode,'data_out',
                                            TargArrays,None,None,None,None,None)

            # data is already a dictionary of pytables arrays
            else:
                TabOut = tools.pytables_(data,'hd5',pytables,Mode,'data_out',
                                            TargArrays,None,None,None,None,None)

            # Complete new datadict
            Result = {}
            for key in data.keys():
                if key in TargArrays:
                    Result[key] = TabOut['arrays'][key]
                else:
                    Result[key] = data[key]
            Result['pytables'] = pytables
            Result['fileh'] = TabOut['fileh']
            data = Result

        return data


    ##################
    ##  'dataframe' ##
    ##    format    ##
    ##################

    if format_ == 'dataframe':
        import pandas as pd
                
        # Get corner
        if data.index.name is None:
            corner = 'id'
        else:
            if isinstance(data.index.values[0], (tuple)):
                corner = str(tuple(data.index.name))
            else:
                corner = data.index.name
        
        # Get rowlabels
        rowlabels_ = data.index.values
        if isinstance(rowlabels_[0], tuple):
            rl = [str(t) for t in rowlabels_]
            rl.insert(0, corner)
        else:
            rl = rowlabels_.tolist()
            rl.insert(0, corner)
        rowlabels = np.array(rl, dtype='S60')[:, np.newaxis]
        
        # Get collabels
        collabels_ = data.columns.values
        if isinstance(collabels_[0], tuple):
            cl = [str(t) for t in collabels_]
            cl.insert(0, corner)
        else:
            cl = collabels_.tolist()
            cl.insert(0, corner)
        collabels = np.array(cl, dtype='S60')[np.newaxis, :]
        
        # Get coredata and other variables
        coredata = data.values
        nheaders4rows = 1
        key4rows = 0
        rowkeytype = 'S60'
        nheaders4cols = 1
        key4cols = 0
        colkeytype = 'S60'


    ##################
    ##  'datadict'  ##
    ##    format    ##
    ##################

    # Convert datadict to single array
    if (format_ == 'datadict'
        or format_ == 'datadict_whole'
        or (format_ == 'hd5'
            and isinstance(data,dict))
        ):
        try:
            data.keys()
        except AttributeError:
            exc = 'data arg does not match format_ arg.\n'
            raise Damon_Error(exc)

        # Store inputs in dictionary for later
        OrigInputs = {}
        for key in data.keys():
            OrigInputs[key] = data[key]

        # Check for presence of required keys
        if format_ == 'datadict':
            ReqKeys =  ['rowlabels',
                        'collabels',
                        'coredata',
                        'key4rows',
                        'rowkeytype',
                        'key4cols',
                        'colkeytype',
                        'nanval',
                        'validchars',
                        ]

            for key in ['rowlabels', 'collabels', 'coredata',
                        'key4rows', 'rowkeytype', 'key4cols',
                        'colkeytype', 'nanval', 'validchars']:
                try:
                    data[key]
                except KeyError:
                    exc = "datadict missing a required array or variable.  Must include: {'rowlabels'_,'collabels:_','coredata':_,'key4rows':_,'rowkeytype':_,'key4cols':_,'colkeytype':_,'nanval':_,'validchars':_}\n"
                    raise Damon_Error(exc)

        elif format_ == 'datadict_whole':
            for key in ['rowlabels', 'collabels', 'coredata']:
                try:
                    data[key]
                except KeyError:
                    exc = "datadict missing a required array.  Must include: {'rowlabels':_,'collabels:_','coredata':_}.  All other parameters must be specified using Damon(...) args.\n"
                    raise Damon_Error(exc)

        # Force coredata, rowlabels, collabels to have compatible dimensions
        coredata = OrigInputs['coredata']
        if coredata is None or np.size(coredata) == 0:
            exc = 'coredata is None or empty. Unable to create Damon object.'
            raise Damon_Error(exc)
        nCoreRows = np.size(coredata,axis=0)
        nCoreCols = np.size(coredata,axis=1)

        rowlabels = OrigInputs['rowlabels']
        nRLRows = np.size(rowlabels,axis=0)
        nRLCols = np.size(rowlabels,axis=1)

        collabels = OrigInputs['collabels']
        nCLRows = np.size(collabels,axis=0)
        nCLCols = np.size(collabels,axis=1)

        if nCLCols != nCoreCols + nRLCols:

            CLGap = np.zeros((nCLRows,nCoreCols + nRLCols - nCLCols)) + nanval
            collabels = np.append(CLGap,collabels,axis=1)
            if verbose is True:
                print 'Warning in Damon.__init__(): rowlabels, collabels, coredata have incompatible dimensions.  collabels resized.\n'

        if nRLRows != nCoreRows + nCLRows:
            RLGap = collabels[:nCLRows,:nRLCols]
            rowlabels = np.append(RLGap,rowlabels,axis=0)
            if verbose is True:
                print 'Warning in Damon.__init__(): rowlabels, collabels, coredata have incompatible dimensions.  rowlabels resized.\n'

        if pytables is not None:
            TempDict = {'coredata':coredata[:,:],'rowlabels':rowlabels[:,:],'collabels':collabels[:,:]}
            X_ = tools.pytables_(TempDict,'datadict',pytables,'w','data_out',['coredata','rowlabels','collabels'],
                                None,None,None,None,None)
            coredata = X_['arrays']['coredata']
            rowlabels = X_['arrays']['rowlabels']
            collabels = X_['arrays']['collabels']
            fileh = X_['fileh']
            format_ = 'datadict'

        # Note:  The parameters listed below are drawn from
        # the user's data = MyDataDict (here called OrigInputs),
        # which are required parameters UNLESS 'datadict_whole'
        # is specified, in which case they are drawn from the
        # Damon parameters.

        # Otherwise, it is assumed the user does not want
        # to change these.  All other parameters will be as
        # specified for this Damon.

        if format_ == 'datadict':
            nheaders4rows = np.size(rowlabels[:,:],axis=1)
            key4rows = OrigInputs['key4rows']
            rowkeytype = OrigInputs['rowkeytype']
            nheaders4cols = np.size(collabels[:,:],axis=0)
            colkeytype = OrigInputs['colkeytype']
            key4cols = OrigInputs['key4cols']
            nanval = OrigInputs['nanval']
            validchars = OrigInputs['validchars']

            try:
                dtype = OrigInputs['dtype']
            except KeyError:
                pass

            if verbose is True:
                print "Note: Pulling 'required' parameters from datadict specified in 'data =', not from your Damon() specs.  If you want to change them, edit your input datadict or specify format_ = 'datadict_whole' and specify the required args in Damon(...).\n"

        # Build "whole" array
        elif format_ == 'datadict_whole':

            # Build whole array
            if pytables is None:
                PyTab = None
            else:
                PyTab = True

            #if verbose is True:
                #print "tools.addlabels() is working.  To skip this step, try format_ = 'datadict' or 'datadict_link'.\n"
            NewData = tools.addlabels(coredata = coredata[:,:], # [2-d array of coredata]
                                      rowlabels = rowlabels[:,:],    # [2-d array of rowlabels]
                                      collabels = collabels[:,:],    # [2-d array of collabels]
                                      fill_top = 0,   # [<0 => No need to fill in top row labels, 1 => Fill in row labels to top>]
                                      fill_left = 0, # [0 => No need to fill in left col labels, 1 => Fill in col labels all the way to left]
                                      filler = None,     # [<None, value with which to fill empty corner cells if two preceding args = 1>]
                                      dtype = dtype,    # [int number of desired decimal places]
                                      filename = pytables,   # [<None, name of output text file or .hd5 file>]
                                      delimiter = None,  # [<None, field delimiter character if filename refers to a text file>]
                                      nanval = nanval,
                                      pytables = PyTab,  # [<None,True> ]
                                      )

            # Redefine required input variables, as appropriate
            data = NewData['whole']

            # Define format
            if pytables is None:
                format_ = 'array'
            else:
                format_ = 'hd5'

        # Redefine rowkeytype and colkeytype to prevent accidental truncation of strings
        if rowkeytype == str:
            rowkeytype = dtype[0]
        if colkeytype == str:
            colkeytype = dtype[0]


    ###################
    ##  Load array,  ##
    ##  textfile(s)  ##
    ##     'hd5'     ##
    ###################

    if (pytables is not None
        and (format_ == 'array'
             or format_ == 'textfile'
             or format_ == ['textfiles']
             or format_ == 'hd5')
        ):
        if dtype[0] == object:
            str_size = 60
            if format_ == 'array':
                data = data.astype('S60')
        else:
            str_size = int(dtype[0][1:])

        All_DatTab_ = tools.pytables_(data,format_,pytables,'w','data_out',
                                     ['All_Dat'],None,'string',str_size,None,delimiter)

        # Define fileh and whole string array.  Add fileh to object.
        All_Dat = All_DatTab_['arrays']['All_Dat']
        fileh = All_DatTab_['fileh']
        data = None

        # Close NewData (from datadict above) if possible
        try:
            NewData['fileh'].close()
        except:
            pass

    # Load array into data()
    elif format_ == 'array':

        if isinstance(data,np.ndarray):
            All_Dat = data
        else:
            exc = 'Unable to read array.  Check data and format_ parameters of Damon.\n'
            raise Damon_Error(exc)

    # Load file into data()
    elif (format_ == 'textfile'
          or format_ == ['textfiles']
          ):
        # Force data to be a file list
        if format_ == 'textfile':
            Data1 = [data]
        elif format_ == ['textfiles']:
            Data1 = data

        # Read each file, line by line
        All_DatRaw = []
        for textfile in Data1:

            # Use np.genfromtxt()
            try:
                All_Dat = np.genfromtxt(textfile,dtype=dtype[0],delimiter=delimiter)
            except:

                try:
                    Lines = open(textfile,'rb')
                except ValueError:
                    exc = 'Unable to open file.  Check data and format_ parameters of Damon.\n'
                    raise Damon_Error(exc)

                try:
                    All_DatRaw.extend(list(csv.reader(Lines, delimiter=delimiter)))                    
                except:
                    try:
                        [All_DatRaw.append(line.rstrip().split(delimiter)) for line in Lines]                        
                    except:
                        exc = 'Unable to read lines from file.\n'
                        raise Damon_Error(exc)

                Lines.close()

                # Convert data to an array
                try:
                    All_Dat = np.array(All_DatRaw,dtype=dtype[0])
                    np.shape(All_Dat[:, :])
                except (ValueError, IndexError):
                    print 'Warning: Possible non-rectangular data. data() will fill array aligning top/left.\n'

                    # Create rectangular array
                    nrows = len(All_DatRaw)
                    nRowElem = []
                    for i in xrange(nrows):
                        nRowElem.append(len(All_DatRaw[i]))
                    ncols = max(nRowElem)
                    All_Dat = np.zeros((nrows,ncols),dtype=dtype[0])
                    All_Dat[:,:] = nanval

                    # Load list data into array, automatically aligned to left and top margins.
                    for i in xrange(nrows):
                        All_Dat[i,0:nRowElem[i]] = All_DatRaw[i]

    elif format_ == 'hd5':
        All_Dat = data

    elif format_ == 'datadict':
        All_Dat = None
    
    elif format_ == 'dataframe':
        All_Dat = None

    else:
        exc = 'Unable to figure out format_ parameter.\n'
        raise Damon_Error(exc)


    ###################
    ##  Recode data  ##
    ###################

    if All_Dat is not None:

        # Recode data in specified ranges of All_Dat
        if recode is not None:
            alldat = np.copy(All_Dat).astype(dtype[0])
            ranges = recode.keys()
            for r in ranges:
                reco_range = recode[r][0]
                reco_dict = recode[r][1]

                for key in reco_dict.keys():
                    alldat[reco_range][alldat[reco_range] == key] = reco_dict[key]

            All_Dat = alldat


        ###########################
        ##  Shift Potential Row  ##
        ##    Labels to Left     ##
        ##   and Select range_    ##
        ###########################

        # Shift columns of rowlabels to left of array, if necessary
        if cols2left is not None:
            nRows0 = np.size(All_Dat[:,:],axis=0)

            # Create rowlabels array
            RowLabelsTemp = np.zeros((nRows0,len(cols2left))).astype(dtype[0])
            nheaders4rows = len(cols2left)
            ColLoc = []

            for i in range(nheaders4rows):
                try:
                    Index = np.where(All_Dat[key4cols,:].astype('S60') == str(cols2left[i]))[0][0]
                except IndexError:
                    try:
                        Index = np.where(All_Dat[key4cols,:] == cols2left[i])[0][0]
                    except IndexError:
                        exc = 'Unable to find cols2left elements in collabels.'
                        print 'Error in Damon.__init__(): ',exc
                        print 'cols2left: \n',cols2left
                        print 'collabels: \n',All_Dat[key4cols,:].tolist()
                        print '\n'
                        raise Damon_Error(exc)
                
                ColLoc.append(Index)
                RowLabelsTemp[:,i] = np.squeeze(All_Dat[:,Index])

            # Delete duplicate columns from main array
            ColLoc = np.array(ColLoc, dtype=int)
            All_Dat_del = np.delete(All_Dat[:,:],ColLoc,axis=1)
            All_Dat = np.append(RowLabelsTemp,All_Dat_del,axis=1)

        # Select range of incoming data that shall be official data array
        if selectrange is not None:
            All_Dat = All_Dat[selectrange]
            SelRange = eval(selectrange,{'__builtins__':None},{'slice':slice})
            All_Dat = All_Dat[SelRange]

        # Rebuild pytables if necessary
        if (pytables is not None
            and (cols2left is not None
                 or selectrange is not None
                 )
            ):
            #fileh.data_out.All_Dat.remove()
            All_DatTab_ = tools.pytables_(All_Dat.astype('S60'),'array',fileh,'r+','data_out',
                                         ['All_Dat1'],None,'string',str_size,None,delimiter)

            # Define fileh and whole string array.  Add fileh to object.
            All_Dat = All_DatTab_['arrays']['All_Dat1']


        ###########################
        # Useful variables

        try:
            nrows = np.size(All_Dat[:,:],axis=0)
            ncols = np.size(All_Dat[:,:],axis=1)
        except IndexError:
            exc = 'Unable to figure out data indices.  Check that data is convertible to a 2-D rectangular array.  Also, check delimiter.'
            print 'Error in Damon.__init__(): ',exc
            print 'Shape: ',np.shape(All_Dat),'\n'
            raise Damon_Error(exc)

        # Cast nanval
        try:
            float(nanval)
        except ValueError:
            print 'Warning in Damon: nanval not convertible to float.  Re-setting to -999.\n'
            nanval = -999.

        if nanval == np.nan:
            print 'Warning in Damon: "nanval = np.nan" being reset to -999.\n'
            nanval = -999.


        ##################
        ##  Append Row, ##
        ##  Col Labels  ##
        ##  if missing  ##
        ##################
        
        rowkeytype_flag = False
        colkeytype_flag = False
        
        if (nheaders4rows == 0
            or nheaders4cols == 0
            ):
            if nheaders4rows == 0:
                rowlabels = np.arange(1,nrows + 1)[:,np.newaxis]
                nheaders4rows = 1
                key4rows = 0
                #if rowkeytype is None:
                rowkeytype = int
                rowkeytype_flag = True

                # Avoid dups
                if nheaders4cols > 0:
                    colkeys = All_Dat[key4cols,:].astype('S60') #.astype(colkeytype)
                    X = rowlabels[key4cols,0].astype('S60') #.astype(colkeytype)
                    if X in colkeys:
                        s = set(colkeys)
                        t = set(np.array(range(-1000,1),dtype=colkeytype))
                        rowlabels[key4cols,0] = np.amax(np.array(list(t - s),dtype=long))
                All_Dat = np.append(rowlabels,All_Dat,axis=1)
                ncols = np.size(All_Dat,axis=1)

            if nheaders4cols == 0:
                collabels = np.arange(0,ncols)[np.newaxis,:]
                nheaders4cols = 1
                key4cols = 0
                #if colkeytype is None:
                colkeytype = int
                colkeytype_flag = True

                # Avoid dups
                if nheaders4rows > 0:
                    RowKeys = All_Dat[:,key4rows].astype('S60') #.astype(rowkeytype)
                    X = collabels[0,key4rows].astype('S60') #.astype(rowkeytype)
                    if X in RowKeys:
                        s = set(RowKeys)
                        t = set(np.array(range(-1000,1),dtype=rowkeytype))
                        collabels[0,key4rows] = np.amax(np.array(list(t - s),dtype=long))
                All_Dat = np.append(collabels,All_Dat,axis=0)

            nrows = np.size(All_Dat,axis=0)
            ncols = np.size(All_Dat,axis=1)

            # Convert back to PyTable
            if pytables is not None:
                All_Dat = tools.pytables_(All_Dat,'array',fileh,None,'data_out',
                                            ['All_Dat2'],None,None,None,None,None)['arrays']['All_Dat2']

        ################
        ## Row Labels ##
        ################

        # Build an array of row labels (pytables is handled)
        if rowkeytype_flag:
            rowkeytype = 'S60'
            rowlabels = All_Dat[:,:nheaders4rows].astype(int).astype(rowkeytype)
        else:
            rowlabels = All_Dat[:, :nheaders4rows]
        
        
        # Deal with missing labels
        if miss4headers is not None:
            for i in xrange(len(miss4headers)):
                rowlabels = np.where(rowlabels.astype(str) == str(miss4headers[i]),nanval,rowlabels)

        rowlabels = np.array(rowlabels,ndmin=2)
        nRLRows = np.size(rowlabels,axis=0)

        RCDict = {'rowlabels':rowlabels,'key4rows':key4rows,'rowkeytype':rowkeytype,
                  'key4cols':key4cols,'colkeytype':colkeytype,
                  'nheaders4rows':nheaders4rows,'nheaders4cols':nheaders4cols
                  }

        # Check for duplicate column keys
        if check_dups is not None:
            RCDict['rowlabels'] = rowlabels
            RowKeys = tools.getkeys(RCDict,'Row','All','S60',None)
            RowDups = tools.dups(RowKeys)

            if len(RowDups) != 0:
                if check_dups == 'stop':
                    exc = "Found duplicate row keys: "+str(RowDups)+"\n"
                    raise Damon_Error(exc)

                elif check_dups == 'warn':
                    print "Warning in Damon.__init__():  Found duplicate row keys.\n",
                    RowDups,"\nAssigning new IDs formatted '-'+ID+'00N' to differentiate them.\n"

                    for Dup in RowDups.keys():
                        nDups = RowDups[Dup]
                        NewKeys = ['-'+str(Dup)+'00'+str(i) for i in range(nDups)]
                        RowKeys[RowKeys == Dup] = NewKeys
                        rowlabels[:,key4rows] = RowKeys
                else:
                    exc = 'Unable to figure out check_dups parameter.\n'
                    raise Damon_Error(exc)

        # Convert to PyTable
        if pytables is not None:
            rowlabels = tools.pytables_(rowlabels,'array',fileh,None,'data_out',
                                      ['rowlabels'],None,None,None,None,None)['arrays']['rowlabels']

        ################
        ## Col Labels ##
        ################

        # Build an array of column labels (pytables is handled)
        if colkeytype_flag:
            colkeytype = 'S60'
            collabels = All_Dat[:nheaders4cols,:].astype(int).astype(colkeytype)
        else:
            collabels = All_Dat[:nheaders4cols,:]
            
        if miss4headers is not None:
            for i in xrange(len(miss4headers)):
                collabels = np.where(collabels.astype(str) == str(miss4headers[i]),nanval,collabels)
        collabels = np.array(collabels,ndmin=2)
        nCLCols = np.size(collabels,axis=1)

        # Check for duplicate column keys
        if check_dups is not None:
            RCDict['collabels'] = collabels
            colkeys = tools.getkeys(RCDict,'Col','All','S60',None)
            ColDups = tools.dups(colkeys)

            if len(ColDups) != 0:
                if check_dups == 'stop':
                    exc = "Found duplicate column keys: {0}\n".format(ColDups)
                    raise Damon_Error(exc)

                elif check_dups == 'warn':
                    print "Warning in Damon.__init__():  Found duplicate column keys:",ColDups,"\nAssigning new IDs formatted '-'+ID+'00N' to differentiate them.\n"

                    for Dup in ColDups.keys():
                        nDups = ColDups[Dup]
                        NewKeys = ['-'+str(Dup)+'00'+str(i) for i in range(nDups)]
                        colkeys[colkeys == Dup] = NewKeys
                        collabels[key4cols,:] = colkeys
                else:
                    exc = 'Unable to figure out check_dups parameter.\n'
                    raise Damon_Error(exc)

        # Convert to PyTable
        if pytables is not None:
            collabels = tools.pytables_(collabels,'array',fileh,None,'data_out',
                                      ['collabels'],None,None,None,None,None)['arrays']['collabels']


    ################
    ## Core data  ##
    ################

    # Build an array for the core data values, appropriately filtered (pytables is handled)

    ##########################################################################
    def clean_core(coredata, nanval, validchars, missingchars):
        "Clean and format coredata, whole array or row by row"

        # Convert object array to strings
        if coredata.dtype == object:
            coredata = coredata.astype('S60')

        # Check validchars
        if validchars is not None:
            if validchars[0] not in ['All','Cols']:
                exc = 'Error in Damon.__init__():  Could not figure out validchars.  Check syntax.\n'
                raise Damon_Error(exc)

        # Filter out missing characters -- TODO: verify fix
        if missingchars is not None:
            for char in missingchars:
                ix = dmn.tools.index_val(coredata, char)
                coredata = dmn.tools.apply_val(coredata, ix, nanval)
                
                
#                try:
#                    coredata = np.where((coredata == char) |
#                                        (coredata == str(char)) |
#                                        (coredata == str(float(char))),
#                                        nanval,coredata)
#                except ValueError:
#                    coredata = np.where((coredata == char) |
#                                        (coredata == str(char)),
#                                        nanval,coredata)
                            
        # Define float casting function
        def tryfloat(coredata,i,j):
            try:
                float(coredata[i,j])
            except:
                coredata[i,j] = nanval

            return coredata[i,j]

        # Replace np.nan with nanval
        try:
            coredata[np.isnan(coredata)] = nanval
        except TypeError:
            coredata[coredata == 'nan'] = nanval

        # Force coredata to be numeric
        if (validchars is not None
            and 'Num' in validchars
            and 'SkipCheck' not in validchars
            ):
            try:
                IsAlpha = np.core.defchararray.isalpha(coredata)
                IsPeriod = (coredata == '.')
                IsBlank = (coredata == '')
                IsSpace = (coredata == ' ')
                coredata = np.where(IsAlpha | IsPeriod | IsBlank | IsSpace,
                                    nanval, coredata)
                #coredata = np.where(np.logical_or(IsAlpha,IsPeriod,IsBlank),nanval,coredata)
            except TypeError:
                pass

            # Try to convert to float
            try:
                coredata = coredata[:,:].astype(np.float)
            except ValueError:
                if verbose is not None:
                    print 'Damon.__init__() found characters besides numbers, letters, ".", and " " -- cleaning using a slower method.\n'
                temp = [tryfloat(coredata[:,:],i,j) for i in xrange(nCoreRows) for j in xrange(nCoreCols)]
                temp = None
                try:
                    coredata = coredata.astype(np.float)
                except ValueError:
                    print 'Warning in Damon.__init__():  Unable to cast coredata to float.\n'
                    pass

        # Recast nanval type
        try:
            if isinstance(coredata[0,0],str):
                nanval = str(nanval)
        except IndexError:
            exc = 'coredata array ended up empty for some reason.\n'
            raise Damon_Error(exc)

        # Select only valid characters, forcing all chars to be string
        # NOTE: This will work on both string and numeric data.  If it's returning
        # all nanvals, check that you don't have a type mismatch (e.g., '3' vs. '3.0').
        if validchars is not None:
            dash = ' -- '
            LenDash = len(dash)
            Decimals = 2

            # Create ValidChar-based dictionaries
#            VCOut = tools.valchars(validchars,    # ['validchars' output of data() function]
#                                 dash = dash, # [Expression used to denote a range]
#                                 defnone = 'interval',   # [How to interpret metric when validchars = None]
#                                 retcols = None,    # [<None, [list of core col keys]>]
#                                 )

            if validchars[0] == 'All':

                # Not a range 'm -- n'
                if (dash not in str(validchars[1])):
                    if validchars[1][0] == 'All':
                        pass
                    else:
                        try:
                            ValidChars1 = np.around(np.array(validchars[1]).astype(float),
                                                    decimals = Decimals)
                        except:
                            ValidChars1 = np.array(validchars[1])

                        if 'Num' in validchars:
                            try:
                                RndCore = np.around(coredata.astype(float),decimals=Decimals)
                            except:
                                RndCore = coredata
                        else:
                            RndCore = coredata

                        AllChars = list(np.unique(RndCore))
                        NonValidChars = np.array(list(set(AllChars) - set(ValidChars1)))
                        for i in xrange(np.size(NonValidChars)):
                            coredata = np.where(RndCore == NonValidChars[i],nanval,coredata)

                # Is a range 'm -- n'
                elif (dash in str(validchars[1])):
                    Ran = validchars[1][0]

                    if Ran == '.'+dash+'.':
                        pass
                    else:
                        if Ran == dash:
                            coredata = np.around(coredata)      # No decimal, floats get rounded

                        elif Ran[-LenDash:] == dash and '.' in Ran:
                            MinChar = float(Ran[0:Ran.find(dash)])
                            coredata = np.where(coredata < MinChar,nanval,coredata)

                        elif Ran[-LenDash:] == dash and '.' not in Ran:
                            MinChar = int(Ran[0:Ran.find(dash)])
                            coredata = np.where(coredata < MinChar,nanval,np.around(coredata))    # In this case , floats get rounded

                        elif dash in Ran and '.' in Ran:
                            MinChar = float(Ran[0:Ran.find(dash)])
                            MaxChar = float(Ran[Ran.find(dash) + LenDash:])
                            coredata = np.where(np.logical_or(coredata < MinChar,coredata > MaxChar),nanval,coredata)

                        elif dash in Ran and '.' not in Ran:
                            MinChar = float(Ran[0:Ran.find(dash)])
                            MaxChar = float(Ran[Ran.find(dash) + LenDash:])
                            coredata = np.where(np.logical_or(coredata < MinChar,coredata > MaxChar),nanval,np.around(coredata)) # Floats get rounded

            # Check for valid characters for each column individually
            # NOTE: 'Num' is ignored.  It tries to cast to float if it can.
            elif validchars[0] == 'Cols':
                CharDict = validchars[1]
                #ColKeys0 = tools.getkeys(RCDict,'Col','Core','Auto',None)
                try:
                    ColKeys0 = collabels[key4cols, nheaders4rows:].astype(colkeytype)
                except ValueError:
                    exc = 'Unable to cast keys to the desired type.'
                    print 'Error in __init__():',exc
                    print 'keys = \n',collabels[key4cols,nheaders4rows:],type(collabels[key4cols,nheaders4rows:][0])
                    print 'colkeytype = ',colkeytype
                    raise Damon_Error(exc)

                for i in xrange(len(ColKeys0)):
                    try:
                        CharDict[ColKeys0[i]]
                    except KeyError:
                        exc = ('KeyError: Maybe the validchars parameter is '
                               'missing a coredata column.\n')
                        print 'validchars =\n',validchars
                        print 'validchars key type=', type(CharDict.keys()[0])
                        print 'column keys=\n',ColKeys0
                        print 'column key=', ColKeys0[i], type(ColKeys0[i])
                        raise Damon_Error(exc)

                    if CharDict[ColKeys0[i]] == ['All']:
                        pass

                    # Not a range 'm -- n'
                    elif (dash not in str(CharDict[ColKeys0[i]])):
                        try:
                            ValidChars1 = np.around(np.array(CharDict[ColKeys0[i]]).astype(float), 
                                                    decimals=Decimals)
                        except:
                            ValidChars1 = np.array(CharDict[ColKeys0[i]])
                        try:
                            RndCore = np.around(coredata[:,i].astype(float),
                                                decimals=Decimals)
                        except:
                            RndCore = coredata[:,i]

                        AllChars = list(np.unique(RndCore))
                        NonValidChars = np.array(list(set(AllChars) - set(ValidChars1)))                        
                        
                        for j in xrange(np.size(NonValidChars)):
                            ix = dmn.tools.index_val(RndCore, NonValidChars[j])
                            w_nan = dmn.tools.apply_val(RndCore, ix, nanval)
                            coredata[:, i] = w_nan

                    # Is a range 'm -- n'
                    elif (dash in str(CharDict[ColKeys0[i]])):
                        NaNValF = float(nanval)

                        try:
                            CoreDataF = coredata[:,i].astype(float)
                        except ValueError:
                            CoreDataF = coredata[:,i]

                        Ran = CharDict[ColKeys0[i]][0]

                        if Ran == '.'+dash+'.':
                            pass
                        else:
                            if Ran == dash:
                                coredata = np.around(coredata)      # No decimal, floats get rounded

                            elif Ran[-LenDash:] == dash and '.' in Ran:
                                MinChar = float(Ran[0:Ran.find(dash)])
                                coredata[:,i] = np.where(CoreDataF < MinChar,NaNValF,CoreDataF)

                            elif Ran[-LenDash:] == dash and '.' not in Ran:
                                MinChar = int(Ran[0:Ran.find(dash)])
                                coredata[:,i] = np.where(CoreDataF < MinChar,NaNValF,np.around(CoreDataF))    # In this case , floats get rounded

                            elif dash in Ran and '.' in Ran:
                                MinChar = float(Ran[0:Ran.find(dash)])
                                MaxChar = float(Ran[Ran.find(dash) + LenDash:])
                                coredata[:,i] = np.where(np.logical_or(CoreDataF < MinChar,CoreDataF > MaxChar),nanval,CoreDataF)

                            elif dash in Ran and '.' not in Ran:
                                MinChar = float(Ran[0:Ran.find(dash)])
                                MaxChar = float(Ran[Ran.find(dash) + LenDash:])
                                coredata[:,i] = np.where(np.logical_or(CoreDataF < MinChar,CoreDataF > MaxChar),NaNValF,np.around(CoreDataF)) # Floats get rounded

        return coredata
        ##########################################################################

    nCoreRows = np.size(rowlabels[nheaders4cols:,:],axis=0)
    nCoreCols = np.size(collabels[:,nheaders4rows:],axis=1)

    if All_Dat is not None:
        coredata = All_Dat[nheaders4cols:,nheaders4rows:]

    # Clean coredata as non-PyTable whole array
    coredata = clean_core(coredata[:,:], nanval, validchars, missingchars)

    # Guess validchars spec if necessary
    if isinstance(validchars,list) and 'Guess' in validchars:
        RCDict = {'rowlabels':rowlabels,'key4rows':key4rows,'rowkeytype':rowkeytype,
                  'collabels':collabels,'key4cols':key4cols,'colkeytype':colkeytype,
                  'nheaders4rows':nheaders4rows,'nheaders4cols':nheaders4cols
                  }
        colkeys = tools.getkeys(RCDict,'Col','Core','Auto',None)
        validchars = tools.guess_validchars(coredata,colkeys,500,nanval)

    # Convert to pytables
    if pytables is not None:
        coredata = tools.pytables_(coredata[:,:],'array',fileh,None,'data_out',
                                  ['coredata'],None,None,None,None,None)['arrays']['coredata']
    # Set type of nanval
    if isinstance(coredata[0,0],str):
        nanval = str(int(nanval))
    
    # Exit if array has insufficient valid data values
    if pytables is None:
        ix = tools.index_val(coredata, nanval)
        NaNCore = np.sum(ix)
##        NaNCore = np.sum((coredata[:,:] == nanval) |
##                         (coredata[:,:] == str(nanval)) |
##                         (coredata[:,:] == str(float(nanval)))
##                         )
#        if isinstance(coredata[0,0], (int, float)):
#            NaNCore = np.sum(coredata[:,:] == nanval)
#        elif isinstance(coredata[0,0], str):
#            NaNCore = np.sum((coredata[:, :] == str(nanval)) |
#                             (coredata[:, :] == str(float(nanval))))

        nDat = nCoreRows * nCoreCols
        if NaNCore >= 0.98 * nDat:
#            print ('Warning in Damon.__init__:  Small number (< 2%) of valid '
#                   'data values. Check validchars arg in Damon.')
#            print 'coredata=\n',coredata[:,:]

            exc = 'Small number (< 2%) of valid data values. Check validchars arg in Damon.'
            print 'Error in Damon: ',exc
            np.set_printoptions(precision=2, suppress=True)
            print 'coredata =\n',coredata[:,:]
            np.savetxt('check_eq_err.csv', coredata, '%s', ',')
            raise Damon_Error(exc)
    else:
        if verbose is True:
            print 'Warning: Skipping check for sufficient valid data due to use of pytables.\n'
        else:
            pass


    #################
    ## Whole array ##
    #################

    # return_ data array with row and col labels attached
    if (workformat == 'whole'
        or workformat == 'RCD_whole'
        or workformat == 'RCD_dicts_whole'
        ):
        if pytables is not None:
            PyTab = True
        else:
            PyTab = None

        WholeArray_ = tools.addlabels(coredata = coredata, # [2-d array of coredata]
                                      rowlabels = rowlabels,    # [2-d array of rowlabels]
                                      collabels = collabels,    # [2-d array of collabels]
                                      fill_top = 0,   # [<0 => No need to fill in top row labels, 1 => Fill in row labels to top>]
                                      fill_left = 0, # [0 => No need to fill in left col labels, 1 => Fill in col labels all the way to left]
                                      filler = None,     # [<None, value with which to fill empty corner cells if two preceding args = 1>]
                                      dtype = dtype,    # [int number of desired decimal places]
                                      filename = None,   # [<None, name of output text file or .hd5 file>]
                                      delimiter = None,  # [<None, field delimiter character if filename refers to a text file>]
                                      nanval = nanval,
                                      pytables = PyTab,  # [<None,True> ]
                                      )
        whole = WholeArray_['whole'][:,:]
    else:
        whole = None


    ###############################
    ## Label Lookup Dictionaries ##
    ###############################

    if workformat == 'RCD_dicts' or workformat == 'RCD_dicts_whole':
        if workformat == 'RCD_dicts_whole':
            whole_arg = whole
        else:
            whole_arg = None

        # Build all Damon dictionaries
        d_dicts = tools.damon_dicts(coredata,
                                    rowlabels,nheaders4rows,key4rows,rowkeytype,
                                    collabels,nheaders4cols,key4cols,colkeytype,
                                    range4labels = 'Core',
                                    strip_labkeys = None,
                                    whole = whole_arg
                                    )
    else:
        d_dicts = {'rl_row':None,
                   'rl_col':None,
                   'cl_row':None,
                   'cl_col':None,
                   'core_row':None,
                   'core_col':None,
                   'whole_row':None,
                   'whole_col':None
                   }

    # Close whole PyTable if necessary
    try:
        WholeArray_['fileh'].close()
    except:
        pass


    ################
    ##   Reports  ##
    ################

    if verbose is True:
        print 'Rows in coredata:',nCoreRows
        print 'Columns in coredata:',nCoreCols,'\n'

    Result = {'rowlabels':rowlabels,'collabels':collabels,'coredata':coredata,'whole':whole,
              'nheaders4rows':nheaders4rows,'key4rows':key4rows,'rowkeytype':rowkeytype,
              'nheaders4cols':nheaders4cols,'key4cols':key4cols,'colkeytype':colkeytype,
              'nanval':nanval,'validchars':validchars,'fileh':fileh
              }

    # Add dictionaries
    Result.update(d_dicts)

    # Add in extra bits from datadict
    if OrigInputs is not None:
        Results = Result.keys()
        for key in OrigInputs.keys():
            if key not in Results:
                Result[key] = OrigInputs[key]

    # Add in input variables
    Results = Result.keys()
    for key in _locals.keys():
        if (key not in Results
            and key != 'data'
            and key != 'self'
            ):
            Result[key] = _locals[key]

    return Result




######################################################################

def _create_data(_locals):
    "Basis of the core.create_data() function."

    # Retrieve variables from _locals
    nfac0 = _locals['nfac0']
    nfac1 = _locals['nfac1']
    ndim = _locals['ndim']
    seed = _locals['seed']
    facmetric = _locals['facmetric']
    noise = _locals['noise']
    validchars = _locals['validchars']
    mean_sd = _locals['mean_sd']
    p_nan = _locals['p_nan']
    nanval = _locals['nanval']
    condcoord_ = _locals['condcoord']
    nheaders4rows = _locals['nheaders4rows']
    nheaders4cols = _locals['nheaders4cols']
    extra_headers = _locals['extra_headers']
    input_array = _locals['input_array']
    apply_zeros = _locals['apply_zeros']
    outfile = _locals['outfile']
    delimiter = _locals['delimiter']
    output_as = _locals['output_as']
    bankf0 = _locals['bankf0']
    bankf1 = _locals['bankf1']
    verbose = _locals['verbose']
    Fileh0 = None    # Initialize
    Fileh1 = None    # Initialize

    # Interpret seed
    if not isinstance(seed, dict):
        SeedDict = {}
        SeedDict['Fac0'] = seed
        Seed1 = seed
        if isinstance(seed,int):
            SeedDict['Fac1'] = seed + 1
        else:
            SeedDict['Fac1'] = None
    else:
        SeedDict = seed
        if isinstance(SeedDict['Fac0'], int):
            Seed1 = SeedDict['Fac0']
        elif isinstance(SeedDict['Fac1'], int):
            Seed1 = SeedDict['Fac1']
        else:
            Seed1 = None

    def string_keys(d):
        "Force row and col keys to be string"
        e = {}
        for k in d:
            e[str(k)] = d[k]
        return e

    # Force noise keys to be string
    if noise is not None and isinstance(noise, dict):
        if isinstance(noise['Rows'], dict):
            noise['Rows'] = string_keys(noise['Rows'])
        if isinstance(noise['Cols'], dict):
            noise['Cols'] = string_keys(noise['Cols'])

    # Force validchars keys to be string
    if validchars is not None and isinstance(validchars[1], dict):
        validchars[1] = string_keys(validchars[1])

    # Force mean_sd keys to be string
    if mean_sd is not None and isinstance(mean_sd[1], dict):
        mean_sd[1] = string_keys(mean_sd[1])

    # Function to build subspaces
    def zero_C(C, collabels, apply_zeros):
        "Apply zeros to coordinates for specified subspaces"
        if apply_zeros is None:
            return C
        else:
            row = collabels[apply_zeros[0], nheaders4rows:]
            sub_dict = apply_zeros[1]
            new_C = np.ones((np.shape(C)))

            for sub in sub_dict.keys():
                z_arr = np.array(sub_dict[sub])[:, np.newaxis]
                try:
                    new_C[:, row == sub] = C[:, row == sub] * z_arr
                except ValueError:
                    exc = 'Mismatch between apply_zeros and ndim parameters.  Make sure they have the same number of dimensions.\n'
                    raise create_data_Error(exc)

            return new_C


    #################
    ##   Create    ##
    ##   Labels    ##
    #################

    KeyType = 'S60'

    # Create row labels
    rowlabels_ = np.array(xrange(0, nheaders4cols + nfac0), 
                          dtype=KeyType)[:,np.newaxis]
    if nheaders4rows > 1:

        if isinstance(extra_headers, int):
            XtraHeadRows = (npr.RandomState(seed=seed).rand(nheaders4cols + nfac0,
                                                            nheaders4rows - 1) * extra_headers).astype(int)

        elif isinstance(extra_headers, dict):
            xh_dict = extra_headers
            subs = sorted(xh_dict)
            arr = np.zeros((nheaders4cols + nfac0, nheaders4rows - 1), dtype=KeyType)
            start = nheaders4cols
            arr[:start, :] = '0'

            for sub in subs:
                n = nfac0 * xh_dict[sub]
                end = start + n
                arr[int(start):int(end), :] = sub
                start += n

            XtraHeadRows = arr
        rowlabels = np.append(rowlabels_,XtraHeadRows,axis=1)
    else:
        rowlabels = rowlabels_

    # Create col labels
    collabels_ = np.array(xrange(0,nheaders4rows + nfac1),dtype=KeyType)[np.newaxis,:]
    if nheaders4cols > 1:
        if isinstance(extra_headers, int):
            XtraHeadCols = (npr.RandomState(seed=seed).rand(nheaders4cols - 1,
                                                            nheaders4rows + nfac1) * extra_headers).astype(int)

        elif isinstance(extra_headers, dict):
            xh_dict = extra_headers
            subs = sorted(xh_dict)
            arr = np.zeros((nheaders4cols -1, nheaders4rows + nfac1), dtype=KeyType)
            start = nheaders4rows
            arr[:, :start] = '0'

            for sub in subs:
                n = int(nfac1 * xh_dict[sub])
                end = start + n
                arr[:, start:end] = sub
                start += n

            XtraHeadCols = arr
        collabels = np.append(collabels_,XtraHeadCols,axis=0)
    else:
        collabels = collabels_

    # Restore overwritten row and col keys
    rowlabels[:,0] = np.squeeze(rowlabels_)
    collabels[0,:] = np.squeeze(collabels_)
    rowlabels[0,:] = collabels[0,:nheaders4rows]
    collabels[:,0] = rowlabels[:nheaders4cols,0]

    # Add non-numerical corner value
    rowlabels[0,0] = 'id'
    collabels[0,0] = 'id'

    # Convert to PyTable (make separate copies for data and model)
    if output_as == 'hd5':

        RowLabels0 = tools.pytables_(rowlabels,'array',Fileh0,None,'create_data_out',
                                      ['rowlabels'],None,None,None,None,None)['arrays']['rowlabels']
        ColLabels0 = tools.pytables_(collabels,'array',Fileh0,None,'create_data_out',
                                      ['collabels'],None,None,None,None,None)['arrays']['collabels']

        RowLabels1 = tools.pytables_(rowlabels,'array',Fileh1,None,'create_data_out',
                                      ['rowlabels'],None,None,None,None,None)['arrays']['rowlabels']
        ColLabels1 = tools.pytables_(collabels,'array',Fileh1,None,'create_data_out',
                                      ['collabels'],None,None,None,None,None)['arrays']['collabels']


    #################
    ##   Create    ##
    ##  estimates  ##
    #################

    # Input array,if desired
    if input_array is not None:
        if isinstance(input_array,np.ndarray):
            Fac0 = None
            Fac1 = None
            Data0 = input_array
            facmetric = None
        elif isinstance(input_array,dict):
            Fac0 = input_array['fac0coord']
            Fac1 = np.transpose(input_array['fac1coord'])
            Fac1 = zero_C(Fac1, collabels, apply_zeros)
            Data0 = np.dot(Fac0,Fac1)
            facmetric = None
        else:
            exc = 'Unable to figure out input_array parameter.\n'
            raise create_data_Error(exc)

        nfac0 = np.size(Data0,axis=0)
        nfac1 = np.size(Data0,axis=1)
        if output_as == 'hd5':
            output_as == 'Damon'

    # Create coordinates and model array
    else:
        if output_as != 'hd5':

            # Create coordinates, then model estimates
            if isinstance(SeedDict['Fac0'],np.ndarray):
                Fac0 = SeedDict['Fac0']
            else:
                Fac0 = np.around(npr.RandomState(seed=SeedDict['Fac0']).rand(nfac0,ndim)
                                 * facmetric[0] + facmetric[1], decimals=8)

                if condcoord_ == 'Orthonormal':
                    CondCoord_ = tools.condcoord(fac0coord = Fac0,  # [None, nRowEnts x nDims facet1 coordinates array]
                                               fac1coord = None,   # [None, nColEnts x nDims Facet2 coordinates array]
                                               cond_facet = 'Fac0',    # [<'Fac0','Fac1'>]
                                               function = condcoord_,    # [<'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...} >]
                                               nanval = nanval   # [Not-a-number value]
                                               )
                    Fac0 = CondCoord_['F0Std']

            # Ensure different seed
            if isinstance(SeedDict['Fac1'],np.ndarray):
                Fac1 = np.transpose(SeedDict['Fac1'])
            else:
                Fac1 = np.transpose(np.around(npr.RandomState(seed=SeedDict['Fac1']).rand(nfac1,ndim),
                                              decimals=8)) * facmetric[0] + facmetric[1]

            # Apply zeros
            Fac1 = zero_C(Fac1, collabels, apply_zeros)

            # Calc model data
            Data0 = np.dot(Fac0,Fac1)


        ################
        ##   Create   ##
        ##  pytables  ##
        ################

        # IMPORTANT:  The pytables .hd5 file for "data" is initialized here
        else:
            # Get and assign path name
            if __name__ == "__main__":
                import damon1.tests.play
                Path = damon1.tests.play.__path__[0]
            else:
                Path = os.getcwd()+'/'

            if apply_zeros is not None:
                print 'Warning:  The apply_zeros arg does not work when output_as is "hd5".  Ignoring it.\n'

            if outfile is None:
                OutFile0 = Path+'model_create_data.hd5'
                OutFile1 = Path+'data_create_data.hd5'
            else:
                OutFile0 = Path+'model_'+outfile
                OutFile1 = Path+'data_'+outfile

            if condcoord_ == 'Orthonormal':
                print "Warning: output_as = hd5 does not support the condcoord_ = 'Orthonormal' option.  Changing condcoord_ to None.\n"
                condcoord_ = None

            # Calc Fac0
            if isinstance(SeedDict['Fac0'],np.ndarray):
                Fac0Tab_ = tools.pytables_(SeedDict['Fac0'],'array',OutFile0,'w','create_data_out',
                                      ['Fac0'],None,None,None,None,None)
                Fileh0 = Fac0Tab_['fileh']
                Fac0 = Fac0Tab_['arrays']['Fac0']
            else:
                F0_ChunkDict = {'chunkfunc':tools.rand_chunk,
                                 'nchunks':'Auto',
                                 'chunksize':'Auto',
                                 'nrows':nfac0,
                                 'ncols':ndim,
                                 }

                F0_ArgDict = {'nchunks':'Auto',
                               'chunksize':'Auto',
                               'nrows':nfac0,
                               'ncols':ndim,
                               'facmetric':facmetric,
                               'seed':SeedDict['Fac0'],
                               'chunkstart':0
                               }

                F0_Dicts = {'chunkdict':F0_ChunkDict,'ArgDict':F0_ArgDict}

                # Create Fac0 PyTable
                Fac0Tab_ = tools.pytables_(F0_Dicts,'chunkfunc',OutFile0,'w','create_data_out',
                                          ['Fac0'],None,'float',4,None,None)
                Fileh0 = Fac0Tab_['fileh']
                Fac0 = Fac0Tab_['arrays']['Fac0']

            # Calc Fac1
            if isinstance(SeedDict['Fac1'],np.ndarray):
                Fac1 = tools.pytables_(np.transpose(SeedDict['Fac1']),'array',Fileh0,'w','create_data_out',
                                      ['Fac1'],None,None,None,None,None)['arrays']['Fac1']
            else:
                F1_ChunkDict = {'chunkfunc':tools.rand_chunk,
                                 'nchunks':'Auto',
                                 'chunksize':'Auto',
                                 'nrows':nfac1,
                                 'ncols':ndim,
                                 }

                F1_ArgDict = {'nchunks':'Auto',
                               'chunksize':'Auto',
                               'nrows':nfac1,
                               'ncols':ndim,
                               'facmetric':facmetric,
                               'seed':SeedDict['Fac1'],
                               'chunkstart':0
                               }

                F1_Dicts = {'chunkdict':F1_ChunkDict,'ArgDict':F1_ArgDict}

                # Create Fac1 PyTable
                Fac1 = tools.pytables_(F1_Dicts,'chunkfunc',Fileh0,None,'create_data_out',
                                          ['Fac1'],None,'float',4,None,None)['arrays']['Fac1']

                Fac1 = np.transpose(Fac1)


            #############
            ##  model  ##
            #############

            # IMPORTANT:  The pytables .hd5 file for "model" is initialized here

            def dot_chunk(Fac0,Fac1,chunksize,chunkstart=0):
                Chunk = np.dot(Fac0[chunkstart:(chunkstart + chunksize),:],Fac1)
                return Chunk

            dot_ChunkDict = {'chunkfunc':dot_chunk,
                             'nchunks':'Auto',
                             'chunksize':'Auto',
                             'nrows':nfac0,
                             'ncols':nfac1,
                             }
            dot_ArgDict = {'Fac0':Fac0,
                           'Fac1':Fac1,
                           'chunksize':'Auto',
                           'chunkstart':0
                           }

            # Compute estimates
            dot_Dicts = {'chunkdict':dot_ChunkDict,'ArgDict':dot_ArgDict}
            Data0 = tools.pytables_(dot_Dicts,'chunkfunc',Fileh0,None,'create_data_out_preModel',
                                      ['coredata'],None,'float',4,None,None)['arrays']['coredata']


    #############
    ##   Add   ##
    ##  noise  ##
    #############


    #####################
    # data_chunk() is suspended until the new noise syntax is programmed

    # Define chunk-wise noise function for BOTH pytables and non-pytables
    def data_chunk(nchunks,chunksize,nrows,ncols,seed,model,noise,chunkstart=0):
        "Create chunk of data = mod + noise for specified rows/cols."

        facmetric = [1,0]

        # Create noise for whole array
        if (isinstance(noise,float)
            or isinstance(noise,int)
            ):
            NoiseArray = noise * (tools.rand_chunk(nchunks,chunksize,nrows,
                                                   ncols,facmetric,seed,
                                                   chunkstart) - 0.50)

        # Create noise for rows and cols individually
        elif isinstance(noise,dict):
            RowNoise = np.zeros((chunksize,1))
            if (isinstance(noise['Rows'],float)
                or isinstance(noise['Rows'],int)
                ):
                RowNoise += noise['Rows']

            elif isinstance(noise['Rows'],dict):
                for key in noise['Rows'].keys():
                    RowNoise[int(key) - nheaders4cols, :] = noise['Rows'][key]
                
#                # Ignored. Not chunking for now.
#                ChunkKeys = range(nheaders4cols + (chunksize * (nchunks - 1)),
#                                  nheaders4cols + (chunksize * nchunks))
#                                      
#                rowkeys = noise['Rows'].keys()
#                for key in rowkeys:
#                    if str(key) in noise['Rows'].keys():
#                        RowNoise[int(key) - nheaders4cols,:] = noise['Rows'][str(key)]
            else:
                exc = 'Unable to interpret noise parameter.\n'
                raise create_data_Error(exc)

            ColNoise = np.zeros((1,nfac1))
            if (isinstance(noise['Cols'],float)
                or isinstance(noise['Cols'],int)
                ):
                ColNoise += noise['Cols']

            elif isinstance(noise['Cols'],dict):
                for key in noise['Cols'].keys():
                    ColNoise[:,int(key) - nheaders4rows] = noise['Cols'][key]
            else:
                exc = 'Error in create_data(): Unable to interpret noise parameter.\n'
                raise create_data_Error(exc)

            RCNoise = RowNoise + ColNoise
            rand = npr.RandomState(seed=Seed1).rand(chunksize,nfac1) - 0.50
            NoiseArray = RCNoise * rand
        else:
            exc = 'Unable to interpret noise parameter.\n'
            raise create_data_Error(exc)

        DataChunk = model[chunkstart:chunkstart + chunksize,:] + NoiseArray

        return DataChunk
    #####################

    # Add no noise
    if noise is None:
        if output_as == 'hd5':
            Data1_ = tools.pytables_(Data0[:,:],'array',OutFile1,'w','create_data_out',
                                    ['coredata'],None,None,None,None,None)
            Data1 = Data1_['arrays']['coredata']
            Fileh1 = Data1_['fileh']
        else:
            Data1 = Data0

    # Add specified noise to array
    else:
        if output_as == 'hd5':

            # Define chunkdict and ArgDict
            data_ChunkDict = {'chunkfunc':data_chunk,
                               'nchunks':'Auto',
                               'chunksize':'Auto',
                               'nrows':nfac0,
                               'ncols':nfac1,
                               }
            data_ArgDict = {'nchunks':'Auto',
                             'chunksize':'Auto',
                             'nrows':nfac0,
                             'ncols':nfac1,
                             'seed':Seed1,
                             'model':Data0,
                             'noise':noise,
                             'chunkstart':0
                             }
            # Compute "data" array = model + noise
            data_Dicts = {'chunkdict':data_ChunkDict,'ArgDict':data_ArgDict}

            Data1_ = tools.pytables_(data_Dicts,'chunkfunc',OutFile1,'w','create_data_out',
                                   ['coredata'],None,'float',4,None,None)

            Data1 = Data1_['arrays']['coredata']
            Fileh1 = Data1_['fileh']

        else:
            Data1 = data_chunk(1,nfac0,nfac0,nfac1,Seed1,Data0,noise,0)


    #######################
    ##  Prep validchars  ##
    ##  for Targ Metrics ##
    #######################

    # Important variables
    if output_as != 'hd5':
        Keys = collabels[0, nheaders4rows:].astype('S60') 
    else:
        Keys = ColLabels1[0, nheaders4rows:].astype('S60')
    nKeys = np.size(Keys)
    dash = ' -- '

    try:
        if facmetric[1] == 0:
            Data1_ValRange = '0.0 -- '
        else:
            Data1_ValRange = 'All'
    except TypeError:
        Data1_ValRange = 'All'

    # Check validchars
    if validchars is not None:
        if validchars[0] not in ['All', 'Cols']:
            exc = 'Could not figure out validchars.  Check syntax.\n'
            raise create_data_Error(exc)

    # Convert validchars to dichotomous if alpha
    if (validchars is not None):

        # Values for data range
        if validchars[0] == 'All':
            Vals1 = validchars[1][:]
            if (isinstance(Vals1[0],str)
                and dash not in str(Vals1)
                and 'All' not in str(Vals1)
                ):
                Vals1_ = [0,1]
                ValidChars1 = ['All', Vals1_]
            else:
                ValidChars1 = validchars

        elif validchars[0] == 'Cols':
            ValDict1 = validchars[1].copy()
            for key in Keys:
                Vals1 = ValDict1[key]
                if (isinstance(Vals1[0],str)
                    and dash not in str(Vals1)
                    and 'All' not in str(Vals1)
                    ):
                    ValDict1[key] = [0, 1]
                else:
                    pass
            ValidChars1 = ['Cols',ValDict1]

        # Ensure continuous values for model range
        if ValidChars1[0] == 'All':
            Vals0 = ValidChars1[1][:]
            if (dash not in str(Vals0)
                and Vals0 != ['All']
                ):
                Min = str(float(min(Vals0)))
                Max = str(float(max(Vals0)))
                Vals0_ = [Min+dash+Max]
                ValidChars0 = ['All',Vals0_]
            else:
                ValidChars0 = ['All',Vals0]

        elif ValidChars1[0] == 'Cols':
            ValDict0 = ValidChars1[1].copy()
            for key in Keys:
                Vals0 = ValDict0[key]
                if (dash not in str(Vals0)
                    and Vals0 != ['All']
                    ):
                    Min = str(float(min(Vals0)))
                    Max = str(float(max(Vals0)))
                    Vals0_ = [Min+dash+Max]
                    ValDict0[key] = Vals0_
                else:
                    pass
            ValidChars0 = ['Cols',ValDict0]
    else:
        ValidChars1 = None
        ValidChars0 = None

    ###################
    ##  Data0, Data1 ##
    ##    Dicts      ##
    ###################

    # Put "model" in datadict format
    Data0RCD = {'rowlabels':RowLabels0 if output_as == 'hd5' else rowlabels,
                'collabels':ColLabels0 if output_as == 'hd5' else collabels,
                'coredata':Data0,
                'nheaders4rows':nheaders4rows,
                'key4rows':0,
                'rowkeytype':'S60',
                'nheaders4cols':nheaders4cols,
                'key4cols':0,
                'colkeytype':'S60',
                'nanval':nanval,
                'validchars':['All',[Data1_ValRange],'Num'],  # ValRange = 'All or '0 -- '
                }

    # Put "observed" in datadict format
    Data1RCD = {'rowlabels':RowLabels1 if output_as == 'hd5' else rowlabels,
                'collabels':ColLabels1 if output_as == 'hd5' else collabels,
                'coredata':Data1,
                'nheaders4rows':nheaders4rows,
                'key4rows':0,
                'rowkeytype':'S60',
                'nheaders4cols':nheaders4cols,
                'key4cols':0,
                'colkeytype':'S60',
                'nanval':nanval,
                'validchars':['All',[Data1_ValRange]],  # ValRange = 'All or '0 -- '
                }

    ####################
    ##    Scale to    ##
    ## target Metrics ##
    ####################

    if validchars is not None:

        ###########################
        # Define Functions

        # Define function to get means and standard deviations
        def meansd(data,key,referto):
            if referto == 'Cols':
                ColLoc = np.where(Keys == key)[0]
                ValDat = data[:,ColLoc]
                ColMean = np.mean(ValDat)
                ColSD = np.std(ValDat)
                return [ColMean,ColSD]

            elif referto == 'Whole':
                ValDat = data
                ArrMean = np.mean(ValDat)
                ArrSD = np.std(ValDat)
                return [ArrMean,ArrSD]

        # Define function to get 'Param' specifications
        def getparams(data,Keys,validchars,mean_sd):

            ParamVals = {}

            # TargMean / ValidChar scenarios
            if mean_sd is None:
                if validchars[0] == 'All':
                    referto = 'Whole'
                    if (validchars[1] == ['. -- .']
                        or validchars[1] == [' -- ']
                        or validchars[1] == ['All']
                        ):
                        ParamVals['All'] = meansd(data,None,'Whole')
                    else:
                        ParamVals['All'] = 'Refer2VC'

                elif validchars[0] == 'Cols':
                    referto = 'Cols'
                    for key in Keys:
                        if (validchars[1][key] == ['. -- .']
                            or validchars[1][key] == [' -- ']
                            or validchars[1][key] == ['All']
                            ):
                            ParamVals[key] = meansd(data,key,'Cols')
                        else:
                            ParamVals[key] = 'Refer2VC'

            elif mean_sd[0] == 'Cols':
                TargDict = {}
                if mean_sd[1] == 'Refer2VC':
                    for key in Keys:
                        TargDict[key] = mean_sd[1]
                else:
                    if isinstance(mean_sd[1], list):
                        for key in Keys:
                            TargDict[key] = mean_sd[1]
                    elif isinstance(mean_sd[1], dict):
                        for key in Keys:
                            TargDict[key] = mean_sd[1][key]

                if validchars[0] == 'All':
                    if (validchars[1] == ['. -- .']
                        or validchars[1] == [' -- ']
                        or validchars[1] == ['All']
                        ):
                        referto = 'Cols'
                        VCDict = {}
                        for key in Keys:
                            if TargDict[key] == 'Refer2VC':
                                ParamVals[key] = meansd(data,key,'Cols')
                            else:
                                ParamVals[key] = TargDict[key]
                            VCDict[key] = ['. -- .']
                        validchars = ['Cols',VCDict]
                    else:
                        referto = 'Whole'
                        ParamVals['All'] = 'Refer2VC'

                elif validchars[0] == 'Cols':
                    referto = 'Cols'
                    for key in Keys:
                        if (validchars[1][key] == ['. -- .']
                            or validchars[1][key] == [' -- ']
                            or validchars[1][key] == ['All']
                            ):
                            if TargDict[key] == 'Refer2VC':
                                ParamVals[key] = meansd(data,key,'Cols')
                            else:
                                ParamVals[key] = TargDict[key]
                        else:
                            ParamVals[key] = 'Refer2VC'

            elif mean_sd[0] == 'All':
                if validchars[0] == 'All':
                    referto = 'Whole'
                    if (validchars[1] == ['. -- .']
                        or validchars[1] == [' -- ']
                        or validchars[1] == ['All']
                        ):
                        if mean_sd[1] == 'Refer2VC':
                            ParamVals['All'] = meansd(data,None,'Whole')
                        else:
                            ParamVals['All'] = mean_sd[1]
                    else:
                        ParamVals['All'] = 'Refer2VC'

                elif validchars[0] == 'Cols':
                    referto = 'Cols'  # Considered 'Whole' but ran into trouble with unit tests
                    for key in Keys:
                        if (validchars[1][key] == ['. -- .']
                            or validchars[1][key] == [' -- ']
                            or validchars[1][key] == ['All']
                            ):
                            if mean_sd[1] == 'Refer2VC':
                                ParamVals[key] = meansd(data,None,'Whole')
                            else:
                                ParamVals[key] = mean_sd[1]
                        else:
                            ParamVals[key] = 'Refer2VC'

            else:
                exc = 'Unable to figure out the TargMean parameter.\n'
                raise create_data_Error(exc)

            return {'ParamVals':ParamVals,'validchars':validchars,'referto':referto}


        ####################
        # "observed data":  Get 'Param' specifications
        ParOut1 = getparams(Data1[:,:], Keys, ValidChars1, mean_sd)
        ParamVals1 = ParOut1['ParamVals']
        ValidChars1 = ParOut1['validchars']
        ReferTo1 = ParOut1['referto']

        ####################
        # "model data":  Get 'Param' specifications
        ParOut0 = getparams(Data0[:,:], Keys, ValidChars0, mean_sd)
        ParamVals0 = ParOut0['ParamVals']
        ValidChars0 = ParOut0['validchars']
        ReferTo0 = ParOut0['referto']

        ###################################
        # rescale data = model + noise

        # Convert "observed data" to '0-1' standardized metric
        if output_as == 'hd5':
            PyTables1 = 'data_std_create_data.hd5'
        else:
            PyTables1 = None

        preData1Obj = dmn.core.Damon(Data1RCD,
                                     'datadict_link',
                                     pytables=PyTables1,
                                     verbose=None)

        preData1Obj.standardize(metric = '0-1',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                                referto = ReferTo1,   # [<None,'Whole','Cols'>]
                                rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                                std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                                add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                                )

        StdParams1 = {'stdmetric':'0-1',
                      'validchars':ValidChars1,
                      'referto':ReferTo1,
                      'params':ParamVals1,
                      'rescale':None,
                      'orig_data':None,
                      }

        # Scale standardized data according to std_params
        preData1Obj.fin_est(orig_data = 'std_params', # [<'data','pseudomiss','parse','std_params' => fill out std_params arg>]
                            stdmetric = '0-1',  # [<'LogDat','SD','0-1','PMinMax','Logit','Percentile','PLogit','Orig'>]
                            ents2restore = 'All',   # [<'All',['AllExcept',[list of column entities to exclude from orig_data]]
                            referto = ReferTo1,    # [<'Whole','Cols'>]
                            std_params = StdParams1,     # [<None, standardization parameters from original data>]
                            )

        Data1 = preData1Obj.fin_est_out['coredata']
        

        ###################################
        # rescale data = model

        # Convert "model data" to '0-1' standardized metric
        if output_as == 'hd5':
            PyTables2 = 'model_std_create_data.hd5'
        else:
            PyTables2 = None

        preData0Obj = dmn.core.Damon(Data0RCD, 'datadict_link', pytables=PyTables2,
                                     verbose=None)

        preData0Obj.standardize(metric = '0-1',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                                referto = ReferTo0,   # [<None,'Whole','Cols'>]
                                rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                                std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                                add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                                )

        StdParams0 = {'stdmetric':'0-1',
                      'validchars':ValidChars0,
                      'referto':ReferTo0,
                      'params':ParamVals0,
                      'rescale':None,
                      'orig_data':None,
                      }

        preData0Obj.fin_est(orig_data = 'std_params', # [<'data','pseudomiss','parse','std_params' => fill out std_params arg>]
                              stdmetric = '0-1',  # [<'LogDat','SD','0-1','PMinMax','Logit','Percentile','PLogit','Orig'>]
                              ents2restore = 'All',   # [<'All',['AllExcept',[list of column entities to exclude from orig_data]]
                              referto = 'Cols',    # [<'Whole','Cols'>]
                              std_params = StdParams0,     # [<None, standardization parameters from original data>]
                              )

        Data0 = preData0Obj.fin_est_out['coredata']

    # report min and max data
    if verbose is True:
        print 'Number of Rows=',nfac0
        print 'Number of Columns=',nfac1
        print 'Number of Dimensions=',ndim

        if output_as == 'hd5':
            print 'Data Min/Max are not reported for hd5 files.'
        else:
            print 'Data Min=',round(np.amin(Data1),3)
            print 'Data Max=',round(np.amax(Data1),3)


    ###################
    ##    Convert    ##
    ##   to alpha    ##
    ###################

    if validchars is not None:

        # Skip if all data is numeric or a range
        if (validchars[0] == 'All'
            and (not isinstance(validchars[1][0],str)
                 or dash in str(validchars[1])
                 or 'All' in str(validchars[1])
                 )
            ):
            anskey = None
            Data1_a = Data1

        # Data is alpha
        else:
            # Convert 'All' to columns
            if (validchars[0] == 'All'
                and isinstance(validchars[1][0],str)
                and dash not in str(validchars[1])
                and 'All' not in str(validchars[1])
                ):
                ValDict = {}
                for i in xrange(nKeys):
                    ValDict[Keys[i]] = validchars[1]
            else:
                ValDict = validchars[1]

            # Access valid alpha characters for each column
            anskey = np.zeros((1,nKeys)).astype('S5')
            KeyRand = npr.RandomState(seed=Seed1).rand(nKeys)
            RespRand = npr.RandomState(seed=Seed1).rand(nfac0,nKeys)
            Data1_a = np.copy(Data1[:,:]).astype('S5')

            for i in xrange(nKeys):
                Vals = ValDict[Keys[i]]

                # Test for alpha
                if (isinstance(Vals[0],str)
                    and dash not in str(Vals)
                    and 'All' not in str(Vals)
                    ):

                    # Load dichotomous
                    Data1_Dich = Data1[:,i]

                    # Create random alpha's to replace zeros
                    nChars = len(Vals)
                    Ans = np.round(KeyRand[i] * (nChars - 1))
                    RandResp = np.round(RespRand[:,i] * (nChars - 1))

                    # Convert 0,1 matrix to integer response values
                    S = set(np.unique(RandResp))
                    T = set([Ans])
                    NonAnsKey = list(S - T)
                    if len(NonAnsKey) == 0:
                        NonAnsKey = [nanval]
                    npr.RandomState(seed=Seed1).shuffle(NonAnsKey)

                    # Before converting 1's to anskey, need to get all random values not to equal anskey.
                    RandResp = np.where(RandResp == Ans,NonAnsKey[0],RandResp)
                    RandResp = np.where(Data1_Dich == 1,Ans,RandResp)
                    RandResp1 = np.copy(RandResp)

                    # Replace integers with alpha
                    for j in xrange(nChars):
                        RandResp1 = np.where(RandResp == j,Vals[j],RandResp1)
                        if Ans == j:
                            Ans1 = Vals[j]

                    anskey[0,i] = Ans1
                    Data1_a[:,i] = RandResp1

                # Non-alpha columns
                else:
                    anskey[0,i] = str(int(nanval))
                    Data1_a[:,i] = Data1[:,i]

            # Settle answer key
            if all(np.squeeze(anskey) == str(int(nanval))):
                anskey = None
                Data1_a = Data1_a.astype(np.float)

    else:
        anskey = None
        Data1_a = Data1

    ###############
    ##    Add    ##
    ##  Missing  ##
    ###############

    # Make a percentage of core data values missing
    if p_nan == 0.0:
        Data2 = Data1_a[:, :]
    else:
        if anskey is not None:
            TempNaNVal = str(int(nanval))
        else:
            TempNaNVal = nanval

        CoreRand = npr.RandomState(seed=Seed1).rand(nfac0,nfac1)
        Data2 = np.where(CoreRand <= p_nan, TempNaNVal, Data1_a[:,:])
        ObsPercentNaN = np.round(np.sum(Data2 == TempNaNVal) / float(nfac0 * nfac1),
                                 decimals=3)

        # report missing stats
        if verbose is True:
            print 'Proportion made missing=',ObsPercentNaN
            print 'Not-a-number value (nanval)=',TempNaNVal, type(TempNaNVal)

    ################
    ##  Create    ##
    ##  Objects   ##
    ################

    ################################
    # Create 'data' dicts/obj/arrays

    # Convert Data2 ("observed data") to pytables
    if output_as == 'hd5':
        Data2_ = tools.pytables_(Data2,'array',Fileh1,None,'create_data_out_Data',
                               ['coredata'],None,None,None,None,None)
        Data2 = Data2_['arrays']['coredata']

    # Final data array in data() format
    Data1RCD['coredata'] = Data2
    Data1RCD['verbose'] = None
    Data1RCD['fileh'] = Fileh1
    Data1RCD['validchars'] = validchars

    # Assign 'Num' info to validchars
    if not isinstance(Data2[0, 0], str):
        try:
            if 'Num' not in Data1RCD['validchars']:
                Data1RCD['validchars'].append('Num')
        except TypeError:
            Data1RCD['validchars'] = ['All', ['All'], 'Num']

    # Convert nanval to string if necessary
    if 'Num' not in Data1RCD['validchars']:
        Data1RCD['nanval'] = str(int(nanval))

    # Create DamonObjects
    if (output_as == 'Damon'
        or output_as == 'Damon_textfile'
        or output_as == 'dataframe'
        ):
        Data1Obj = dmn.core.Damon(data = Data1RCD,
                                  format_ = 'datadict',
                                  workformat = 'RCD_dicts_whole',
                                  verbose = None,
                                  pytables = None
                                  )
        Data1Obj.fileh = Data1Obj.pytables = Fileh1
        Data1Obj.verbose = verbose

    # Assemble whole array
    Outs = ['array','textfile','array_textfile','Damon_textfile','datadict_textfile']
    if (output_as in Outs):
        Data1Whole = tools.addlabels(Data1RCD['coredata'],
                                     Data1RCD['rowlabels'],
                                     Data1RCD['collabels'],
                                     0, 0, None,
                                     ['S20',None,None],
                                     None, None,
                                     Data1RCD['nanval'],
                                     None)['whole']
    try:
        try:
            preData1Obj.fileh.close()
        except AttributeError:
            pass
    except UnboundLocalError:
        pass


    ################################
    # Create 'model' dicts/obj/arrays

    # Convert model to PyTable
    if output_as == 'hd5':
        Data0_ = tools.pytables_(Data0,'hd5',Fileh0,None,'create_data_out_Model',
                               ['coredata'],None,None,None,None,None)
        Data0 = Data0_['arrays']['coredata']

    Data0RCD['coredata'] = Data0
    Data0RCD['validchars'] = ['All',['All']]
    Data0RCD['verbose'] = None
    Data0RCD['fileh'] = Fileh0

    # Create Damon's
    if (output_as == 'Damon'
        or output_as == 'Damon_textfile'
        or output_as == 'dataframe'
        ):
        Data0Obj = dmn.core.Damon(data = Data0RCD,
                                  format_ = 'datadict',
                                  workformat = 'RCD_dicts_whole',
                                  verbose = None,
                                  pytables = None
                                  )
        Data0Obj.fileh = Data0Obj.pytables = Fileh0
        Data0Obj.verbose = verbose

    # Assemble whole array
    Outs = ['array','textfile','array_textfile','Damon_textfile','datadict_textfile']
    if (output_as in Outs):
        Data0Whole = tools.addlabels(Data0RCD['coredata'],
                                     Data0RCD['rowlabels'],
                                     Data0RCD['collabels'],
                                     0, 0, None,
                                     ['S20',None,None], None, None,
                                     Data0RCD['nanval'],
                                     None)['whole']

    try:
        try:
            preData0Obj.fileh.close()
        except AttributeError:
            pass
    except UnboundLocalError:
        pass


    ##################
    ## Create Banks ##
    ##################

    # No banks if input_array is used
    if input_array is None:
        if bankf0 is not None or bankf1 is not None:

            # Prepare fac0coord data object
            F0RowLabels = np.append(rowlabels[0,:][np.newaxis,:],rowlabels[nheaders4cols:,:],axis=0)
            F0ColLabels = np.append(rowlabels[0,:][np.newaxis,:],np.array(range(1,ndim + 1),ndmin=2),axis=1)
            fac0coord = Fac0[:,:]

            Fac0CoordRCD = {'rowlabels':F0RowLabels[:,:],
                            'collabels':F0ColLabels,
                            'coredata':fac0coord[:,:],
                            'nheaders4rows':nheaders4rows,
                            'key4rows':0,
                            'rowkeytype':'S60',
                            'nheaders4cols':1,
                            'key4cols':0,
                            'colkeytype':int,
                            'nanval':nanval,
                            'validchars':['All',['All'],'Num'],
                            'opp_count':nfac1
                            }

            # Prepare fac1coord data object
            F1RowLabels = np.transpose(np.append(collabels[:,0][:,np.newaxis],
                                                 collabels[:,nheaders4rows:],
                                                 axis=1))
            F1ColLabels = np.append(collabels[:,0][np.newaxis,:],
                                    np.array(range(1,ndim + 1), ndmin=2),
                                    axis=1)
            fac1coord = np.transpose(Fac1[:,:])

            Fac1CoordRCD = {'rowlabels':F1RowLabels[:,:],
                            'collabels':F1ColLabels,
                            'coredata':fac1coord[:,:],
                            'nheaders4rows':nheaders4cols,
                            'key4rows':0,
                            'rowkeytype':'S60',
                            'nheaders4cols':1,
                            'key4cols':0,
                            'colkeytype':int,
                            'nanval':nanval,
                            'validchars':['All',['All'],'Num'],
                            'opp_count':nfac0
                            }

            # Add to dummy data object
            BankObj = dmn.core.Damon(npr.rand(5,5),'array',verbose=None)
            BankObj.coord_out = {'fac0coord':Fac0CoordRCD,
                                  'fac1coord':Fac1CoordRCD,
                                  'ndim':ndim
                                  }

            # Output bank pickle file
            BankObj.bank('Bank.pkl',{'Remove':['All'], 'Add':bankf0},
                         {'Remove':['All'], 'Add':bankf1})


    #############
    ##  Answer ##
    ##   key   ##
    #############

    # Build answer key array
    if anskey is None:
        AnsKeyOut = None
        AnsKeyObj = None
        AnsKeyDict = None
    else:
        Keys = np.array(collabels)[0,nheaders4rows:].astype('S60') #astype(int)
        ValResp = np.zeros(np.shape(Keys),dtype='S60') # changed from object dtype
        for i in range(np.size(Keys)):
            ValResp[i] = str(ValDict[Keys[i]])

        # Assign cluster labels
        if isinstance(extra_headers, dict):
            Cluster = XtraHeadCols[0][nheaders4rows:]
        else:
            Cluster = npr.RandomState(seed=seed).randint(0,4,(np.shape(Keys))).astype('S60')
            ClustDict = {'0':'Clust0', '1':'Clust1', '2':'Clust2', '3':'Clust3'}
            for k in ClustDict.keys():
                Cluster[Cluster == k] = ClustDict[k]

        KeyRowLabels = np.array([['ItemID'] + range(nheaders4cols - 1) + ['Correct', 'Cluster', 'ValidResp']])
        AnsKeyOut0 = np.concatenate((np.array(collabels)[:,nheaders4rows:],
                                     anskey,
                                     Cluster[np.newaxis,:],
                                     ValResp[np.newaxis,:]),axis=0
                                    )
        AnsKeyOut0 = np.transpose(AnsKeyOut0)
        AnsKeyOut = np.append(KeyRowLabels,AnsKeyOut0,axis=0)

        # Remember answer key is transposed -- Cols = Rows
        AnsKeyObj = dmn.core.Damon(data = AnsKeyOut,    # [<file path name, array name, datadict name, Damon name>]
                                 format_ = 'array',    # [<'textfile','array','datadict','Damon'>]
                                 workformat = 'RCD_dicts_whole',   # ['RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole']
                                 validchars = None,   # [None; ['All',[valid chars]]; or ['Cols', {'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'], 'ID4':['0 -- '],...}] ]
                                 nheaders4rows = nheaders4cols,  # [N columns to hold row labels]
                                 key4rows = 0,   # [Nth column from left which holds row keys]
                                 rowkeytype = 'S60',     # [data type of row keys]
                                 nheaders4cols = 1,  # [N rows to hold column labels]
                                 key4cols = 0, # [Nth row from top which holds column keys]
                                 colkeytype = 'S60',     # [data type of column keys]
                                 dtype = [object,None,None], # [String data type of whole matrix when and core data when cast as string]
                                 nanval = nanval,    # [Value to which non-numeric/invalid characters should be converted.]
                                 verbose = None,
                                 )
        AnsKeyDict = AnsKeyObj.data_out


    #############
    ## outputs ##
    #############

    # Output files
    outputs = ['Damon', 'datadict', 'array', 'dataframe', 'textfile',
               'Damon_textfile', 'datadict_textfile', 'array_textfile', 'hd5']
    if output_as not in outputs:
        exc = 'Unable to figure out output_as arg.\n'
        raise create_data_Error(exc)

    if (output_as == 'Damon_textfile'
        or output_as == 'textfile'
        or output_as == 'array_textfile'
        or output_as == 'datadict_textfile'
        ):
        format_ = '%s'
        outprefix = 'a_'

        # Deal with inappriate parameters
        if outfile is None:
            outfile = 'created.csv'
        if delimiter is None:
            delimiter = ','

        # Add prefixes (generalized to handle paths)
        PathElem = outfile.split('/')
        ObsPathElem = PathElem[:]
        ObsPathElem[-1] = outprefix+'data_'+PathElem[-1]
        ObsPathName = '/'.join(ObsPathElem)

        ModPathElem = PathElem[:]
        ModPathElem[-1] = outprefix+'model_'+PathElem[-1]
        ModPathName = '/'.join(ModPathElem)

        AnsPathElem = PathElem[:]
        AnsPathElem[-1] = outprefix+'anskey_'+PathElem[-1]
        AnsPathName = '/'.join(AnsPathElem)

        # Save files
        np.savetxt(ObsPathName,Data1Whole,fmt = format_,delimiter=delimiter)
        if verbose:
            print ObsPathName,'has been saved.'

        np.savetxt(ModPathName,Data0Whole,fmt = format_,delimiter=delimiter)
        if verbose:
            print ModPathName,'has been saved.'

        # Save answer key file
        if AnsKeyOut is not None:
            np.savetxt(AnsPathName,AnsKeyOut,fmt = format_,delimiter=delimiter)
            if verbose:
                print AnsPathName,'has been saved.'

        if output_as == 'textfile':
            if verbose:
                print 'Returning only specified files, no arrays.\n'

            return {'data':None,'anskey':None,'model':None,
                    'fac0coord':None,'fac1coord':None}

    # Output arrays
    if output_as == 'array' or output_as == 'array_textfile':
        return {'data':Data1Whole,'anskey':AnsKeyOut,'model':Data0Whole,
                    'fac0coord':Fac0,'fac1coord':np.transpose(Fac1)}

    # Output as Damon
    if output_as == 'Damon' or output_as == 'Damon_textfile':
        return {'data':Data1Obj,'anskey':AnsKeyObj,'model':Data0Obj,
                    'fac0coord':Fac0,'fac1coord':np.transpose(Fac1)}

    if output_as == 'datadict' or output_as == 'datadict_textfile':
        return {'data':Data1RCD,'anskey':AnsKeyDict,'model':Data0RCD,
                    'fac0coord':Fac0,'fac1coord':np.transpose(Fac1)}

    if output_as == 'hd5':
        return {'data':Data1RCD,'anskey':AnsKeyDict,'model':Data0RCD,
                    'fac0coord':Fac0,'fac1coord':np.transpose(Fac1)}

    if output_as == 'dataframe':
        ak = None if AnsKeyObj is None else AnsKeyObj.to_dataframe()
        return {'data':Data1Obj.to_dataframe(), 'anskey':ak,
                'model':Data0Obj.to_dataframe(), 'fac0coord':Fac0,
                'fac1coord':np.transpose(Fac1)}
        


######################################################################

def _TopDamon(_locals):
    "Basis of the core.TopDamon() function."

    # Get variables
    data = _locals['data']
    recode = _locals['recode']
    collabels_ = _locals['collabels']
    getcols = _locals['getcols']
    labelcols = _locals['labelcols']
    rename = _locals['rename']
    key4rows = _locals['key4rows']
    getrows = _locals['getrows']
    validchars = _locals['validchars']
    delimiter = _locals['delimiter']
    pytables = _locals['pytables']
    verbose = _locals['verbose']

    dtype = [object,None,None]
    stype = 'S60' if collabels_[2] is None else collabels_[2]

    # Define check_dups
    if key4rows is not None:
        if 'warn_dups' in key4rows:
            check_dups = 'warn'
        elif 'stop_dups' in key4rows:
            check_dups = 'stop'
        else:
            check_dups = None

    # Get nanval
    nanval = -999
    if isinstance(validchars,list):
        for i in validchars:
            if (isinstance(i,dict)
                and 'nanval' in i
                ):
                nanval = i['nanval']

    # Get data format
    if isinstance(data,np.ndarray):
        format_ = 'array'
    elif isinstance(data,str):
        format_ = 'textfile'
    else:
        exc = "Unable to figure out format of data.  TopDamon() accepts only 'array' and 'textfile'.\n"
        raise TopDamon_Error(exc)

    # Modify pytables name
    pytables_ = pytables+'_' if pytables is not None else pytables

    # Load file as Damon object
    Whole = dmn.core.Damon(data,
                           format_,
                           workformat = 'RCD',
                           validchars = None,
                           nheaders4rows = 0,
                           key4rows = None,
                           rowkeytype = None,
                           nheaders4cols = collabels_[0],
                           key4cols = collabels_[1],
                           colkeytype = collabels_[2],
                           check_dups = None,
                           dtype = dtype,
                           recode = recode,
                           nanval = nanval,
                           delimiter = delimiter,
                           pytables = pytables_,
                           verbose = None
                           )

    # Overwrite collabels_ parameter if no collabels exist. Cast as string.
    if collabels_[0] == 0:
        collabels_ = [1,0,stype]

    # Extract data
    if getrows is None:
        GetRows_ = {'Get':'AllExcept','Labels':'key','Rows':[None]}
    else:
        GetRows_ = getrows

    if getcols is None:
        GetCols_ = {'Get':'AllExcept','Labels':'key','Cols':[None]}
    else:
        GetCols_ = getcols

    Extract = Whole.extract(Whole.data_out,
                            getrows = GetRows_,
                            getcols = GetCols_,
                            labels_only = None
                            )

    if np.size(Extract['coredata'][:,:]) == 0:
        exc = 'The extract failed.  First, check that the getrows and getcols keys you specified can be found, with no extraneous characters, in the data source.  Also check labelcols.\n'
        raise TopDamon_Error(exc)

    # rename labels
    collabels = Extract['collabels'].astype(stype)
    
    # Set up columns to move to left
    if (labelcols is None
        and 'Auto' not in key4rows
        ):
        labelcols = np.array([key4rows[0]],dtype=stype)
    elif labelcols is None:
        labelcols = np.array([],dtype=stype)
        if 'Auto' not in key4rows:
            labelcols = np.insert(labelcols,0,str(key4rows[0]))
    elif isinstance(labelcols,int):
        labelcols = collabels[Extract['key4cols'],1:labelcols + 1].astype(stype)

##  Subtle issues.  This block creates problems, doesn't SEEM to be necessary.
##        if 'Auto' not in key4rows:
##            labelcols = np.insert(labelcols,0,0) #str(key4rows[0]))

    else:
        labelcols = np.array(labelcols,dtype=stype)

    # Include the leading integers
    if 'Auto' in key4rows:
        InsertKey = Extract['rowlabels'][Extract['key4cols'],0]
        labelcols = np.insert(labelcols,0,InsertKey)

    # Rename keys
    if rename is not None:
        RFKeys = rename.keys()
        for key in RFKeys:
            collabels = np.where(collabels == str(key),rename[key],collabels)
            labelcols = np.where(labelcols == str(key),rename[key],labelcols)
        Extract['collabels'] = collabels

    # Row keys
    if 'Auto' in key4rows:
        RKeyCol = 0
        RKeyType = 'S20'
    else:
        try:
            RKeyCol = np.where(labelcols == str(key4rows[0]))[0][0] # Location after labelcols shifted left
            RKeyType = 'S20' if key4rows[1] is int else key4rows[1]
        except IndexError:
            print 'Warning in TopDamon(): Could not find specified key4rows in labelcols.  Using the leading column as row keys.\n'
            RKeyCol = 0
            RKeyType = 'S20' #int

    # Close first pytables file
    try:
        Whole.fileh.close()
    except:
        pass
    
    # Remove leading integers
    if 'Auto' not in key4rows and len(list(labelcols)) > 0:
        array = tools.addlabels(Extract['coredata'],
                                Extract['rowlabels'],
                                Extract['collabels'],
                                0,0,None,[object,None,None],None,None,Extract['nanval'],None
                                )['whole']
        Extract = array[:,1:]
        format_ = 'array'
        
        # Rename rowlabels
        if rename is not None:
            for key in rename.keys():
                rlabs = Extract[:, :len(list(labelcols))]
                rlabs[rlabs == str(key)] = rename[key]
    else:
        format_ = 'datadict_whole'
        
        # Rename rowlabels
        if rename is not None:
            for key in rename.keys():
                rlabs = Extract['rowlabels']
                rlabs[rlabs == str(key)] = rename[key]
    
    # Final Damon object
    data = dmn.core.Damon(data = Extract,
                      format_ = format_,
                      workformat = 'RCD_dicts_whole',
                      validchars = validchars,
                      nheaders4rows = len(list(labelcols)),
                      key4rows = RKeyCol,
                      rowkeytype = RKeyType,
                      nheaders4cols = collabels_[0],
                      key4cols = collabels_[1],
                      colkeytype = collabels_[2],
                      check_dups = check_dups,
                      dtype = [dtype[0], dtype[1], ''],
                      nanval = nanval,
                      cols2left = list(labelcols),
                      pytables = pytables,
                      verbose = None
                      )
    data.verbose = verbose  # duplicate verbiage when inside Damon() ???

    return data




######################################################################

def _merge_info(_locals):
    "Basis of the merge_info() method."

    # Get _locals
    self = _locals['self']
    target_axis = _locals['target_axis']
    get_validchars = _locals['get_validchars']

    data = self.data_out
    nanval = float(data['nanval'])

    # Convert info to datadict
    if isinstance(_locals['info'],dmn.core.Damon):
        info = _locals['info'].data_out
    else:
        info = _locals['info']

    if (info['whole'] is None
        or info['rl_row'] is None
        ):
        exc = "Need to specify 'workformat' = 'RCD_dicts_whole' when building the 'info' Damon object.\n"
        raise merge_info_Error(exc)

    # Get mergetool parameters.  Force column label type merge by tranposing if necessary
    if target_axis == 'Row':
        d = self.transpose(data)
    else:
        d = data

    source = np.transpose(info['whole'])
    target = d['collabels'][:,d['nheaders4rows']:]
    axis = 1

    # Get key indices
    source_ind = info['key4rows']
    targ_ind = d['key4cols']

    ####################
    ##  Do the merge  ##
    ####################

    new_collabels = tools.mergetool(source,target,axis,source_ind,targ_ind,object,nanval)
    new_nheaders4cols = np.size(new_collabels,axis=0)

    # Fill the corner with labels
    source_labels = info['collabels'][info['key4cols'],info['nheaders4rows']:]
    corner = np.zeros((new_nheaders4cols,d['nheaders4rows']),dtype=object) + nanval
    corner[-len(source_labels):,d['key4rows']] = source_labels
    corner[:d['nheaders4cols'],:] = d['rowlabels'][:d['nheaders4cols'],:]
    new_collabels = np.append(corner,new_collabels,axis=1)
    new_rowlabels = np.append(corner,d['rowlabels'][d['nheaders4cols']:,:],axis=0)

    # Build datadict
    datadict = {}
    keys = ['coredata','nheaders4rows','key4rows','rowkeytype',
            'key4cols','colkeytype','nanval','validchars'
            ]

    for key in d.keys():
        datadict[key] = d[key]

    datadict['rowlabels'] = new_rowlabels
    datadict['collabels'] = new_collabels
    datadict['nheaders4cols'] = np.size(new_collabels,axis=0)

    # Transpose back to original dimensions if necessary
    if target_axis == 'Row':
        merged = self.transpose(datadict)
    else:
        merged = datadict

    # Get validchars, apply to datadict
    if get_validchars is not None:

        # Get validchars
        keys = tools.getkeys(info,'Row','Core','Auto',None)
        vals = info['core_col'][get_validchars]
        vc_dict = dict(zip(keys,vals))

        # Convert them from string
        for key in vc_dict.keys():
            if isinstance(vc_dict[key],str):
                val = list(ast.literal_eval(vc_dict[key]))
                vc_dict[key] = val

        # Determine whether dataset can be cast to numeric
        numflag = True
        metric = tools.valchars(['Cols',vc_dict])['metric']

        if metric[0] == 'All':
            vc = vc_dict[vc_dict.keys()[0]]
            if metric[1] == 'nominal':
                validchars = ['All',vc]
            else:
                validchars = ['All',vc,'Num']
        elif metric[0] == 'Cols':
            met_dict = metric[1]
            for key in met_dict.keys():
                if met_dict[key] == 'nominal':
                    numflag = False
                    break

            if numflag is True:
                validchars = ['Cols',vc_dict,'Num']
            else:
                validchars = ['Cols',vc_dict]

        merged['validchars'] = validchars
        merged_obj = dmn.core.Damon(merged,'datadict',verbose=None)
        merged = merged_obj.data_out

    return merged





######################################################################

def _extract_valid(_locals):
    "Basis of the extract_valid() method."

    self = _locals['self']
    minperrow = _locals['minperrow']
    minpercol = _locals['minpercol']
    minsd = _locals['minsd']
    rem_rows = _locals['rem_rows']
    rem_cols = _locals['rem_cols']
    iterate = _locals['iterate']

    # Retrieve data
    try:
        data = self.score_mc_out
    except AttributeError:
        try:
            data = self.merge_info_out
        except AttributeError:
            try:
                data = self.data_out
            except AttributeError:
                exc = 'To run extract_valid() first run Damon().\n'
                raise extract_valid_Error(exc)

    d = dmn.core.Damon(data, 'datadict', verbose=None)
    
    def rem_ents(keys, bank, facet):
        "Build list of entities to remove if not in bank."
        
        if isinstance(bank, str):
            bank = np.load(bank)
        bkeys = bank[facet]['ent_coord'].keys()
        rem = []
        
        for key in keys:
            if key not in bkeys:
                rem.append(key)
        
        rem = None if len(rem) == 0 else rem
        return rem
    
    if isinstance(rem_rows, (str, dict)):
        keys = tools.getkeys(d, 'Row', 'Core')
        rem_rows = rem_ents(keys, rem_rows, 'facet0')
        
    if isinstance(rem_cols, (str, dict)):
        keys = tools.getkeys(d, 'Col', 'Core')
        rem_cols = rem_ents(keys, rem_cols, 'facet1')
    
    x = d.data_out
    nrows, ncols = 0, 0
    working = True
    it = 0
    
    # Clean array, recheck, clean again
    while working:
        nrows_, ncols_ = nrows, ncols
        d.flag(x,
               flag_rows = (dmn.tools.flag_invalid, {'axis':'rows',
                                                     'min_count':minperrow,
                                                     'min_sd':minsd,
                                                     'rem':rem_rows}),
               flag_cols = (dmn.tools.flag_invalid, {'axis':'cols',
                                                     'min_count':minpercol,
                                                     'min_sd':minsd,
                                                     'rem':rem_cols}),
               extract = {'rows':'AllExcept', 'cols':'AllExcept'}
               )
        nrows = len(d.flag_out['rows'])
        ncols = len(d.flag_out['cols'])

        if d.flag_out['extract'] is None:
#            print 'np.unique(my_data)=\n', np.unique(x['coredata'])
            exc = 'Could not find any valid rows or columns. Check args.'
            raise extract_valid_Error(exc)
        else:
            x = d.flag_out['extract'].data_out

        # Check if stabilized
        if iterate is False:
            working = False
        elif (nrows == nrows_) & (ncols == ncols_):
            working = False
        
        it += 1
        
    x['iterations'] = it
    
    return x





######################################################################

def _pseudomiss(_locals):
    "Basis of the pseudomiss() method."

    # Retrieve data, row, column variables from self
    self = _locals['self']
    fileh = self.fileh
    pytables = self.pytables

    try:
        datadict = self.extract_valid_out
    except AttributeError:
        try:
            datadict = self.merge_info_out
        except AttributeError:
            try:
                datadict = self.data_out
            except AttributeError:
                exc = 'To run pseudomiss() first run Damon().\n'
                raise pseudomiss_Error(exc)

    # Retrieve variables from _locals
    rand_range = _locals['rand_range']
    rand_nan = _locals['rand_nan']
    ents2nan = _locals['ents2nan']
    range2nan = _locals['range2nan']
    seed = _locals['seed']

    # Use read() if pytables are used.
    data = datadict['coredata'][:,:]
    nanval = datadict['nanval']

    if isinstance(data[0,0],str):
        nanval = str(int(nanval))

    # key labels for row and column entities
    if rand_range != 'All' or ents2nan is not None:
        DatColLabels = np.array(tools.getkeys(datadict,'Col','Core','Auto',None))
        DatRowLabels = np.array(tools.getkeys(datadict,'Row','Core','Auto',None))

    # PseudoNaNVal variable
    if pytables is None:
        Data1 = np.copy(data)
    else:
        Data1 = tools.pytables_(data,'array',fileh,None,'pseudomiss_out',
                                  ['coredata'],None,None,None,None,None)['arrays']['coredata']

    try:
        RandNaNVal = nanval - 1
        FixedNaNVal = nanval - 2
    except TypeError:
        RandNaNVal = str(int(float(nanval)) - 1)
        FixedNaNVal = str(int(float(nanval)) - 2)

    OrigMsIndex = np.where(Data1[:,:] == nanval)
    ArrayShape = np.shape(data)
    nfac0 = ArrayShape[0]
    nfac1 = ArrayShape[1]

    # Apply random pseudo-random to entire matrix
    if rand_range is None or rand_nan == 0.0:
        psmsindex = ([],[])

    elif rand_range == 'All':
        nTot = nfac0 * nfac1
        Cells = np.arange(nTot)
        npr.RandomState(seed=seed).shuffle(Cells)
        Rand1 = np.reshape(Cells,(nfac0,nfac1))
        RandNaNThreshold = round(nTot * rand_nan,0) + 1     # Add 1 to prevent div/0
        Rand1[np.where(Data1 == nanval)] = RandNaNThreshold + 1      # Sets real NaNVals outside threshold
        RandMsIndex = np.where(Rand1 < RandNaNThreshold)
        Data1[RandMsIndex] = RandNaNVal

        psmsindex = RandMsIndex

    # Apply pseudo-random to specified entities
    else:

        # Locate the entities in either rowlabels or collabels and assign pseudo-missing
        for i in range(len(rand_range[1])):

            if rand_range[1][i] not in DatRowLabels and rand_range[1][i] not in DatColLabels:
                exc = "Unable to find specified entity label.\n"
                raise pseudomiss_Error(exc)

            # Match to row labels
            elif rand_range[0] == 'Rows':
                RowEnt = np.where(DatRowLabels == rand_range[1][i])[0][0]
                RowEntDat = Data1[RowEnt,:]

                # Select cells at random from specified entities
                RowTot = len(RowEntDat)
                RowCells = np.arange(RowTot)
                npr.RandomState(seed=seed).shuffle(RowCells)
                RowRand = np.reshape(RowCells,(1,RowTot))
                RandNaNThreshold = round(RowTot * rand_nan,0) + 1    # Add 1 to prevent div/0
                RowRand[np.where(RowEntDat[np.newaxis,:] == nanval)] = RandNaNThreshold + 1   # Fac0 Sets real NaNVals outside threshold
                RowEntRandMsIndex = np.where(RowRand < RandNaNThreshold)
                RowEntDat[np.newaxis,:][RowEntRandMsIndex] = RandNaNVal                     # Fac0 Assigns RandNaNVals to cells in entity

            # Match to column labels
            elif rand_range[0] == 'Cols':
                ColEnt = np.where(DatColLabels == rand_range[1][i])[0][0]
                ColEntDat = Data1[:,ColEnt][:,np.newaxis]

                # Select cells at random from specified entities
                ColTot = len(ColEntDat)
                ColCells = np.arange(ColTot)
                npr.RandomState(seed=seed).shuffle(ColCells)
                ColRand = np.reshape(ColCells,(ColTot,1))
                RandNaNThreshold = round(ColTot * rand_nan,0) + 1    # Add 1 to prevent div/0
                ColRand[np.where(ColEntDat == nanval)] = RandNaNThreshold + 1   # Fac1 Sets real NaNVals outside threshold
                ColEntRandMsIndex = np.where(ColRand < RandNaNThreshold)
                ColEntDat[ColEntRandMsIndex] = RandNaNVal                     # Fac1 Assigns RandNaNVals to cells in entity

        # Map all RandNaNVals across Data1
        psmsindex = np.where(Data1[:,:] == RandNaNVal)

    # Make specified entity pairs (e.g., person/item) pseudo-missing
    if ents2nan is not None:
        for h in range(len(ents2nan)):

            # Make cell missing
            if (ents2nan[h][0] != 'All'
                and ents2nan[h][1] != 'All'
                ):
                Row = np.where(DatRowLabels == ents2nan[h][0])[0][0]
                Col = np.where(DatColLabels == ents2nan[h][1])[0][0]
                if Data1[Row,Col] != nanval:
                    Data1[Row,Col] = FixedNaNVal

            # Make whole row missing
            elif (ents2nan[h][0] == 'All'
                and ents2nan[h][1] != 'All'
                ):
                Col = np.where(DatColLabels == ents2nan[h][1])[0][0]
                Data1[:,Col] = np.where(Data1[:,Col] == nanval,nanval,FixedNaNVal)

             # Make whole col missing
            elif (ents2nan[h][0] != 'All'
                and ents2nan[h][1] == 'All'
                ):
                Row = np.where(DatRowLabels == ents2nan[h][0])[0][0]
                Data1[Row,:] = np.where(Data1[Row,:] == nanval,nanval,FixedNaNVal)

        psmsindex = np.where(Data1 == FixedNaNVal)

    # Make specified row/column pairs pseudo-missing
    elif range2nan is not None:
        Data1[range2nan] = FixedNaNVal      # Warning:  Allows fixed missing to be real missing.
        psmsindex = np.where(Data1[:,:] == FixedNaNVal)

    # If no cells are to be made pseudo-missing
    if rand_range == None and ents2nan == None and range2nan == None:
        psmsindex = None

    # Convert all pseudo-missing data to nanval
    NaNArray = np.select([Data1[:,:] == nanval,Data1[:,:] == RandNaNVal,Data1[:,:] == FixedNaNVal],[nanval,nanval,nanval])
    msindex = np.where(NaNArray == nanval)

    # Remove pytables from memory
    if pytables is not None:
        Data0 = Data1 = None

    return {'msindex':msindex,
            'true_msindex':OrigMsIndex,
            'psmsindex':psmsindex,
            'parsed_msindex':None,
            'parsed_true_msindex':None,
            'parsed_psmsindex':None,
            'seed':seed
            }



######################################################################

def _score_mc(_locals):
    "Basic of the score_mc method."

    # Retrieve parameters
    self = _locals['self']
    anskey = _locals['anskey']
    report = _locals['report']
    getrows = _locals['getrows']
    score_denom = _locals['score_denom']
    usecols = _locals['usecols']

    try:
        nanval = float(_locals['nanval'])
    except ValueError:
        exc = 'nanval needs to be a float or integer.\n'
        raise score_mc_Error(exc)

    # Get answer key from bank
    if isinstance(anskey, basestring):
        try:
            bankfile = open(anskey, 'rb')
            anskey = np.load(bankfile)['anskey_param']
            bankfile.close()
        except:
            exc = 'Unable to figure out anskey parameter.\n'
            raise score_mc_Error(exc)

        if anskey is None:
            exc = 'Unable to figure out anskey parameter.\n'
            raise score_mc_Error(exc)            

    if report is None:
        report = []

    if 'All' in report:
        report = ['RowFreq',
                  'ColFreq',
                  'MostFreq',
                  'AnsKey',
                  'MatchKey',
                  'RowScore',
                  'ColScore',
                  'PtBis'
                  ]
    # Get data
    try:
        data = self.extract_valid_out
    except AttributeError:
        try:
            data = self.merge_info_out
        except AttributeError:
            try:
                data = self.data_out
            except AttributeError:
                exc = 'Unable to find input data.\n'
                raise score_mc_Error(exc)

    data_obj = dmn.core.Damon(data,'datadict_link',verbose=None)
    ents = tools.getkeys(data,'Col','Core','Auto',None)

    # Dictionary to access columns by keys
    if data['core_col'] is None:
        vals = np.transpose(data['coredata'][:,:])
        col_dict = dict(zip(ents,vals))
    else:
        col_dict = data['core_col']

    # Get answer key dictionary
    if anskey[0] == 'All':
        ansdict = {}
        for ent in ents:
            ansdict[ent] = anskey[1]
    else:
        ansdict = anskey[1]

    # Flag scoring items
    score_flag = np.zeros(np.size(ents))
    nan_key = [None,nanval,data['nanval'],
               str(int(nanval)),str(int(data['nanval'])),
               str(nanval),str(data['nanval'])
               ]

    for i,ent in enumerate(ents):
        if not isinstance(ansdict[ent],(np.ndarray,list)):
            ansdict[ent] = [ansdict[ent]]
        correct = ansdict[ent]

        if correct[0] in nan_key:
            score_flag[i] = 0
        else:
            score_flag[i] = 1

    # Define subarray for data subjected to an answer key
    score_loc = np.where(score_flag == 1)[0]
    score_ents = ents[score_loc]

    # Column labels
    corner_id = np.array([[data['rowlabels'][data['key4rows'],data['key4cols']]]])
    leader = data['collabels'][:,:data['nheaders4rows']]
    scored_collabels = data['collabels'][:,data['nheaders4rows']:][:,score_loc]
    scored_collabels = np.append(leader,scored_collabels,axis=1)


    #####################
    ##  Create Scored  ##
    ##      Array      ##
    #####################

    shape = np.shape(data['coredata'])
    nrows = shape[0]
#    ncols = shape[1]
    scored = np.zeros(shape)
    valchar_dict = {}
#    meth_dict = {}

    for i,ent in enumerate(ents):
        correct = ansdict[ent]

        if score_flag[i] == 0:
            scored[:,i] = data['coredata'][:,i]
            valchar_dict[ent] = ['All']
        else:
            response = col_dict[ent]
            scored[:,i] = np.array([1 if response[j] in correct else 0 for j in xrange(nrows)])
            valchar_dict[ent] = [0,1]

    scored[data['coredata'] == data['nanval']] = nanval

    try:
        scored = scored.astype(float)
        coretype = 'Num'
    except ValueError:
        coretype = None
        pass

    # Created scored datadict
    results = {}
    outs = ['rowlabels','collabels','coredata',
            'nheaders4rows','key4rows','rowkeytype',
            'nheaders4cols','key4cols','colkeytype',
            'nanval','validchars'
            ]

    for key in data.keys():
        if key in outs:
            results[key] = data[key]

    results['coredata'] = scored
    results['nanval'] = nanval
    results['validchars'] = ['Cols',valchar_dict]

    if coretype == 'Num':
        results['validchars'].append('Num')


    ##################
    ##   Response   ##
    ##  Statistics  ##
    ##################

    # Define the data and column entities to use
    if ('RowFreq' in report
        or 'ColFreq' in report
        or 'MaxFreq' in report
        or 'AnsKey' in report
        or 'MatchKey' in report
        ):

        # Extract appropriate data
        try:
            usecols['Freqs']
        except:
            exc = 'Could not figure out usecols parameter.\n'
            raise score_mc_Error(exc)

        if usecols['Freqs'] == 'All':
            freq_data_ = data_obj.extract(data,
                                         getrows = getrows,
                                         getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}
                                         )
            freq_data = freq_data_['coredata']
            resp = np.unique(freq_data)
            freq_ents = ents
        elif usecols['Freqs'] == 'Scored':
            freq_data_ = data_obj.extract(data,
                                         getrows = getrows,
                                         getcols = {'Get':'NoneExcept','Labels':'key','Cols':score_ents}
                                         )
            freq_data = freq_data_['coredata']
            freq_data[freq_data == str(float(nanval))] = str(int(float(nanval)))
            resp = np.unique(freq_data)
            freq_ents = tools.getkeys(freq_data_,'Col','Core','Auto',None)
        else:
            exc = 'Could not figure out usecols parameter.\n'
            raise score_mc_Error(exc)

        n_resp = np.size(resp)
        n_userows = np.size(freq_data,axis=0)
        n_usecols = np.size(freq_data,axis=1)

    # Row response frequencies
    if 'RowFreq' in report:
        row_resp_freq = np.zeros((n_userows,n_resp))
        for i in xrange(n_userows):
            for j in range(n_resp):
                row_resp_freq[i,j] = np.sum(freq_data[i,:] == resp[j]) / float(n_usecols)
    else:
        row_resp_freq = None

    # Column response frequencies
    if ('ColFreq' in report
        or 'MaxFreq' in report
        or 'MatchKey' in report
        ):
        col_resp_freq = np.zeros((n_usecols,n_resp))
        for i in xrange(n_usecols):
            for j in range(n_resp):
                col_resp_freq[i,j] = np.sum(freq_data[:,i] == resp[j]) / float(n_userows)
    else:
        col_resp_freq = None

    # Most frequent response per column
    if ('MostFreq' in report
        or 'MatchKey' in report
        ):
        most_freq = np.zeros((n_usecols,1),dtype=object)
        for i in xrange(n_usecols):
            most_freq[i,0] = resp[col_resp_freq[i,:] == np.amax(col_resp_freq[i,:])][0]
    else:
        most_freq = None

    # Get answer key
    if 'AnsKey' in report:
        ans_key = np.array([ansdict[ent] for ent in freq_ents])
    else:
        ans_key = None

    # Does key match most frequent response
    if 'MatchKey' in report:
        match_key = np.zeros((n_usecols,1))
        match_key[:,0] = [1 if most_freq[i,0] in ansdict[ent] else 0 for i,ent in enumerate(freq_ents)]
    else:
        match_key = None


    #################
    ##    Score    ##
    ##  Statistics ##
    #################

    if ('RowScore' in report
        or 'ColScore' in report
        or 'PtBis' in report
        ):

        # Extract appropriate data
        try:
            usecols['Freqs']
        except:
            exc = 'Could not figure out usecols parameter.\n'
            raise score_mc_Error(exc)

        cols = ['All'] if usecols['Scores'] == 'All' else score_ents
        scored_data_ = data_obj.extract(results,
                                        getrows = getrows,
                                        getcols = {'Get':'NoneExcept','Labels':'key','Cols':cols}
                                        )

        scored_data = scored_data_['coredata']
        scored_collabels = scored_data_['collabels']
        scored_ents = tools.getkeys(scored_data_,'Col','Core','Auto',None)

        # Create masked version
        scored_data_ma = npma.masked_values(scored_data,nanval)

    # Calc row scores
    if 'RowScore' in report:
        if score_denom == 'All':
            rs_denom = np.size(scored_data,axis=1)
            rs_sum = npma.sum(scored_data_ma,axis=1).filled(nanval)
            row_score = rs_sum / float(rs_denom)
            row_score[rs_sum == nanval] = nanval
            row_score = row_score[:,np.newaxis]
        else:
            row_score = npma.mean(scored_data_ma,axis=1).filled(nanval)[:,np.newaxis]
    else:
        row_score = None

    # Calc column scores
    if 'ColScore' in report:
        if score_denom == 'All':
            cs_denom = np.size(scored_data,axis=0)
            cs_sum = npma.sum(scored_data_ma,axis=0).filled(nanval)
            col_score = cs_sum / float(cs_denom)
            col_score[cs_sum == nanval] = nanval
            col_score = col_score[:,np.newaxis]
        else:
            col_score = npma.mean(scored_data_ma,axis=0).filled(nanval)[:,np.newaxis]
    else:
        col_score = None

    # Calculate point biserials
    if 'PtBis' in report:
        pb_data = {}
        for key in data.keys():
            pb_data[key] = data[key]
        pb_data['coredata'] = scored_data
        pb_data['collabels'] = scored_collabels[scored_data_['key4cols'],:][np.newaxis,:]
        pb_data['validchars'] = ['All',['All'],'Num']
        pb_data['nanval'] = nanval
        ptbis = tools.ptbis(pb_data,'All',None,None)
    else:
        ptbis = None


    #################
    ##  Statistics ##
    ##  datadicts  ##
    #################

    #######################
    if 'RowFreq' in report:
        rkeys = tools.getkeys(freq_data_,'Row','Core','Auto',None)
        rl_rf = np.append(corner_id,rkeys[:,np.newaxis],axis=0)
        cl_rf = np.append(corner_id,np.array(resp)[np.newaxis,:],axis=1)

        row_freq_dict = {'rowlabels':rl_rf,'collabels':cl_rf,'coredata':row_resp_freq,
                         'key4rows':0,'rowkeytype':data['rowkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        row_freq_dict = None

    #######################
    if 'ColFreq' in report:
        rl_cf = np.append(corner_id,np.array(freq_ents)[:,np.newaxis],axis=0)
        cl_cf = np.append(corner_id,np.array(resp)[np.newaxis,:],axis=1)

        col_freq_dict = {'rowlabels':rl_cf,'collabels':cl_cf,'coredata':col_resp_freq,
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        col_freq_dict = None

    #######################
    if 'MostFreq' in report:
        rl_mf = np.append(corner_id,np.array(freq_ents)[:,np.newaxis],axis=0)
        cl_mf = np.append(corner_id,np.array(['MostFreq'])[np.newaxis,:],axis=1)

        most_freq_dict = {'rowlabels':rl_mf,'collabels':cl_mf,'coredata':most_freq,
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All']]
                         }
    else:
        most_freq_dict = None

    #######################
    if 'AnsKey' in report:
        rl_ak = np.append(corner_id,np.array(freq_ents)[:,np.newaxis],axis=0)
        cl_ak = np.append(corner_id,np.array(['AnsKey'])[np.newaxis,:],axis=1)

        ans_key_dict =  {'rowlabels':rl_ak,'collabels':cl_ak,'coredata':ans_key,
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All']]
                         }
    else:
        ans_key_dict = None

    #######################
    if 'MatchKey' in report:
        rl_mk = np.append(corner_id,np.array(freq_ents)[:,np.newaxis],axis=0)
        cl_mk = np.append(corner_id,np.array(['MatchKey'])[np.newaxis,:],axis=1)

        match_key_dict = {'rowlabels':rl_mk,'collabels':cl_mk,'coredata':match_key,
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        match_key_dict = None

    #######################
    if 'RowScore' in report:
        rkeys = tools.getkeys(scored_data_,'Row','Core','Auto',None)
        rl_rs = np.append(corner_id,rkeys[:,np.newaxis],axis=0)
        cl_rs = np.append(corner_id,np.array(['RowScore'])[np.newaxis,:],axis=1)

        row_score_dict = {'rowlabels':rl_rs,'collabels':cl_rs,'coredata':row_score,
                         'key4rows':0,'rowkeytype':data['rowkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        row_score_dict = None

    #######################
    if 'ColScore' in report:
        rl_cs = np.append(corner_id,np.array(scored_ents)[:,np.newaxis],axis=0)
        cl_cs = np.append(corner_id,np.array(['ColScore'])[np.newaxis,:],axis=1)

        col_score_dict = {'rowlabels':rl_cs,'collabels':cl_cs,'coredata':col_score,
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        col_score_dict = None


    #######################
    if 'PtBis' in report:
        rl_pb = np.append(corner_id,np.array(scored_ents)[:,np.newaxis],axis=0)
        cl_pb = np.append(corner_id,np.array(['PtBis'])[np.newaxis,:],axis=1)

        ptbis_dict =    {'rowlabels':rl_pb,'collabels':cl_pb,
                         'coredata':np.transpose(ptbis['coredata']),
                         'key4rows':0,'rowkeytype':data['colkeytype'],
                         'key4cols':0,'colkeytype':'S60',
                         'nanval':nanval,'validchars':['All',['All'],'Num']
                         }
    else:
        ptbis_dict = None

    #######################
    out_keys = ['RowFreq',
                'ColFreq',
                'MostFreq',
                'AnsKey',
                'MatchKey',
                'RowScore',
                'ColScore',
                'PtBis'
                ]

    out_vals = [row_freq_dict,
                col_freq_dict,
                most_freq_dict,
                ans_key_dict,
                match_key_dict,
                row_score_dict,
                col_score_dict,
                ptbis_dict
                ]

    for i,key in enumerate(out_keys):
        results[key] = out_vals[i]

    # Add anskey parameter
    results['anskey'] = anskey
    
    return results





######################################################################

def _subscale(_locals):
    "Basis if the subscale() method."

    self = _locals['self']
    data_ = _locals['data']
    subscales_ = _locals['subscales']
    method = _locals['method']
    rescale = _locals['rescale']
    meth_keys = method.keys()

    try:
        missing = method['mean']['missing']
    except KeyError:
        missing = 'row2nan'

    # Get data
    if 'filter' in meth_keys:
        if data_ is not 'base_est_out':
            print "Warning in subscale(): When method specifies 'filter', specify data = 'base_est_out'.  Changing accordingly.\n"
            data_ = 'base_est_out'
        try:
            coords = self.coord_out
        except AttributeError:
            exc = "Unable to find coordinates.  When method specifies 'filter', run coord() and base_est() first.\n"
            raise subscale_Error(exc)

    data = self.__dict__[data_]
    data = dmn.core.Damon(data,'datadict_link',verbose=None)
    data_x = data.extract(data,
                          getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                          getcols = subscales_,
                          labels_only = None
                          )
    nanval = data['nanval']

    # List subscales
    if subscales_['Labels'] == 'key' or subscales_['Labels'] == 'index':
        subs = np.array(['scale'])
        ind = None
    else:
        if isinstance(subscales_['Labels'],str):
            keys = tools.getkeys(data_x,'Row','All','Auto',None)
            try:
                ind = np.where(keys == subscales_['Labels'])[0][0]
            except IndexError:
                exc = "Unable to find the heading given in the 'Labels' parameter.\n"
                raise subscale_Error(exc)

        elif isinstance(subscales_['Labels'],int):
            ind = subscales_['Labels']
        else:
            exc = 'Unable to figure out subscales argument.\n'
            raise subscale_Error(exc)

        subs = np.unique(data_x['collabels'][ind,data_x['nheaders4rows']:])

    if isinstance(subs[0],str):
        subs = subs[subs != str(nanval)]
    else:
        subs = subs[subs != nanval]

    # Interpret rescale
    if rescale is not None:
        rkeys = rescale.keys()
        if ('All' in rkeys
            and len(rkeys) == 1
            ):
            rdict = rescale['All']
            r_subdict = {}
            for sub in subs:
                r_subdict[sub] = rdict
        else:
            r_subdict = rescale

    # Define a function for subscale raw scores
    def raw_mean(subdata, missing, nanval):
        "Compute raw score mean"

        subdata_ma = npma.masked_values(subdata['coredata'], nanval)
        scores = npma.mean(subdata_ma, axis=1)
        scores = scores.filled(nanval)

        if missing == 'row2nan':
            nancounts = np.sum(subdata['coredata'] == nanval, axis=1)
            scores[np.where(nancounts != 0)] = nanval

        return scores

    # Compute scores for each subscale
    nrows = np.size(data_x['coredata'],axis=0)
    ncols = np.size(subs)
    subscales = np.zeros((nrows,ncols),dtype=np.float64)
    rel_dict = {}
    dim_dict = {}

    for c,sub in enumerate(subs):
        if ('mean' in meth_keys
            or 'rasch' in meth_keys
            or 'coord' in meth_keys
            ):
            if subscales_['Labels'] == 'key' or subscales_['Labels'] == 'index':
                subdata = data_x
            else:
                subdata = data.extract(data_x,
                                       getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                       getcols = {'Get':'NoneExcept','Labels':ind,'Cols':[sub]},
                                       labels_only = None
                                       )

            if 'mean' in meth_keys:
                scores = raw_mean(subdata, missing, nanval)
                rel_dict['sub_'+sub] = None
                dim_dict['sub_'+sub] = None

            if 'rasch' in meth_keys:
                sub_obj = dmn.core.Damon(subdata,'datadict',verbose=None)
                sub_obj.rasch(**method['rasch'])
                scores = np.squeeze(sub_obj.rasch_out['fac0coord']['coredata'])
                rel = sub_obj.rasch_out['reliability']['coredata'][0][1]
                rel_dict['sub_'+sub] = rel
                dim_dict['sub_'+sub] = 1

            if 'coord' in meth_keys:
                sub_obj = dmn.core.Damon(subdata, 'datadict', verbose=None)
                sub_obj.standardize()
                sub_obj.coord(**method['coord'])
                sub_obj.base_est()
                sub_obj.base_resid()
                sub_obj.base_ear()
                sub_obj.base_se()

                # Get scores
                sub_est_ma = npma.masked_values(sub_obj.base_est_out['coredata'],nanval)
                scores = npma.mean(sub_est_ma,axis=1)
                scores = scores.filled(nanval)
                nancounts = np.sum(sub_obj.base_est_out['coredata'] == nanval,axis=1)
                scores[np.where(nancounts != 0)] = nanval

                # Get reliability
                sub_se = tools.rmsr(None, None, sub_obj.base_se_out['coredata'], nanval)
                rel = tools.reliability(None, scores, None, sub_se, nanval)
                rel_dict['sub_'+sub] = rel
                dim_dict['sub_'+sub] = sub_obj.coord_out['ndim']

        # Use the subscale_filter() tool
        elif 'filter' in meth_keys:
            if ind is None:
                corekeys = tools.getkeys(data,'Col','Core','Auto',None)
                subkeys = tools.getkeys(data_x,'Col','Core','Auto',None)
                cols = [np.where(corekeys == subkeys[key])[0][0] for key in range(np.size(subkeys))]
            else:
                cols = np.where(data.collabels[ind,data.nheaders4rows:] == sub)[0]
            scores = tools.subscale_filter(data,cols,coords,'UseCoord',method['filter']['lo_hi'])
            rel_dict['sub_'+sub] = None


        # TODO:  Rewrite the subscale section

        # Rescale if desired
        if rescale is not None:
            subkeys = r_subdict.keys()
            scores = tools.rescale(scores,
                                   straighten = r_subdict['straighten'] if 'straighten' in subkeys else None,
                                   logits = r_subdict['logits'] if 'logits' in subkeys else None,
                                   reverse = r_subdict['reverse'] if 'reverse' in subkeys else False,
                                   mean_sd = r_subdict['mean_sd'] if 'mean_sd' in subkeys else None,
                                   m_b = r_subdict['m_b'] if 'm_b' in subkeys else [1,0],
                                   clip = r_subdict['clip'] if 'clip' in subkeys else None,
                                   round_ = r_subdict['round_'] if 'round_' in subkeys else None,
                                   nanval = nanval
                                   )

        subscales[:,c] = scores


    ###############
    ##  Merge to ##
    ##    Data   ##
    ###############

    # Build subscale collabels
    subnames = []
    for sub in subs:
        subname = 'sub_' + sub
        subnames.append(subname)

    ncols = data.nheaders4rows + np.size(subs)
    sub_collabels = np.copy(data.collabels)[:,:ncols]
    sub_collabels[:, data.nheaders4rows:] = nanval
    sub_collabels[data.key4cols, data.nheaders4rows:] = subnames

    if ind is not None:
        sub_collabels[ind, data.nheaders4rows:] = subs

    # Add elements to validchars
    validchars = data.validchars
    if validchars is not None:
        if isinstance(validchars[1],dict):
            vc_dict = validchars[1]
            for subname in subnames:
                vc_dict[subname] = ['All']
        else:
            vc_dict = {}
            ikeys = tools.getkeys(data,'Col','Core','Auto',None)
            for key in ikeys:
                vc_dict[key] = validchars[1]
            for subname in subnames:
                vc_dict[subname] = ['All']

        # Convert keys to string
        vc_dict1 = {}
        for key in vc_dict.keys():
            vc_dict1[str(key)] = vc_dict[key]

        validchars = ['Cols',vc_dict1,'Num']


    # Build subdict
    subdict = {}
    for key in data.data_out.keys():
        subdict[key] = data.data_out[key]

    subdict['collabels'] = sub_collabels
    subdict['coredata'] = subscales[:,:]
    subdict['colkeytype'] = 'S60'
    subdict['validchars'] = validchars

    # Do the merge
    data.merge(source = subdict,
              axis = {'target':0,'source':0},
              targ_data = True,
              targ_labels = None,
              source_ids = None,
              nanval = nanval
              )

    data.merge_out['validchars'] = validchars
    data.merge_out['colkeytype'] = 'S60'

    # Include separate subdict
    data.merge_out['subscales'] = subdict
    data.merge_out['reliabilities'] = rel_dict
    data.merge_out['ndims'] = dim_dict

    return data.merge_out





######################################################################

def _parse(_locals):
    "Basis of the parse() method."

    # Retrieve self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Retrieve _locals variables
    ParseParams_ = _locals['parse_params']

    if ParseParams_ is None:
        items2parse = _locals['items2parse']
        resp_cat = _locals['resp_cat']
        extractkey = _locals['extractkey']
        ordinal = _locals['ordinal']
        nanval = _locals['nanval']
        parse_params = {'items2parse':items2parse,
                       'resp_cat':resp_cat,
                       'extractkey':extractkey,
                       'ordinal':ordinal,
                       'nanval':nanval,
                       }

    # Get from bank
    else:
        BankFile = open(ParseParams_,'rb')
        Bank = np.load(BankFile)
        BankFile.close()
        parse_params = Bank['parse_params']

        # Retrieve variables from bank
        items2parse = parse_params['items2parse']
        resp_cat = parse_params['resp_cat']
        extractkey = parse_params['extractkey']
        ordinal = parse_params['ordinal']
        nanval = parse_params['nanval']

    # Make sure nanval is numerical
    try:
        nanval = float(_locals['nanval'])
    except ValueError:
        print 'Error in parse(): nanval in needs to be a float or integer.\n'

    # Get datadict
    try:
        datadict = self.subscale_out
    except AttributeError:
        try:
            datadict = self.score_mc_out
        except AttributeError:
            try:
                datadict = self.extract_valid_out
            except AttributeError:
                try:
                    datadict = self.merge_info_out
                except AttributeError:
                    try:
                        datadict = self.data_out
                    except AttributeError:
                        exc = 'Unable to find data to parse.\n'
                        raise parse_Error(exc)

    # data variables, pytables is handled
    PreCoreData = datadict['coredata']
    rowlabels = datadict['rowlabels']
    ColLabels0 = datadict['collabels']

    nheaders4rows = datadict['nheaders4rows']
    PrenHeaders4Cols = datadict['nheaders4cols']
#    key4cols = datadict['key4cols']
    key4rows = datadict['key4rows']
    colkeytype = datadict['colkeytype']
    prenanval = datadict['nanval']
    ValChars = datadict['validchars']
    nDatRows = np.size(PreCoreData,axis=0)

    # ColLabel variables
    PreColLabels = ColLabels0[:,nheaders4rows:]
    PreColKeys = tools.getkeys(datadict,'Col','Core','Auto',None)

    #PreColKeys = ColLabels0[key4cols,nheaders4rows:].astype(colkeytype)
    nPreColRows = np.size(ColLabels0,axis=0)
    dash = ' -- '

    # Use Damon.validchars to define resp_cat
    if resp_cat is 'Auto':
        if ValChars is None:
            resp_cat = 'Find'
        else:
            resp_cat = ValChars[:]

    # Force items to be string
    if (isinstance(items2parse[1], (list, np.ndarray))
        and items2parse[1][0] not in [None, 'All']
        ):
        items2parse[1] = [str(k) for k in items2parse[1]]

    # Identify items to parse
    if items2parse[0] == 'AllExcept':
        if items2parse[1] == 'continuous':

            if resp_cat == 'Find':
                I2Parse = list(PreColKeys)
                nSamp = min(nDatRows,20)
                for i in range(np.size(PreColKeys)):
                    key = PreColKeys[i]
                    nUnique = np.size(np.unique(PreCoreData[:nSamp,i]))
                    if nUnique / float(nSamp) > 0.65:
                        I2Parse.remove(key)

            elif (resp_cat[0] == 'All'
                and (type(resp_cat[1][0]) is type(dash)
                     and (dash not in resp_cat[1][0]
                          and 'All' not in resp_cat[1]
                          )
                     or type(resp_cat[1][0]) is not type(dash)
                     )
                ):
                I2Parse = set(PreColKeys)

            elif resp_cat[0] == 'Cols':
                I2Parse = list(PreColKeys)
                RespCatDict = resp_cat[1]
                for key in PreColKeys:
                    RCats = RespCatDict[key]
                    if (type(RCats[0] is type(dash)
                             and (dash in RCats[0]
                                  or 'All' in RCats[0]
                                  )
                             )
                        ):
                        I2Parse.remove(key)

            I2Parse = set(I2Parse)

        elif items2parse[1][0] is None:
            I2Parse = set(PreColKeys)

        else:
            I2Parse = set(PreColKeys) - set(items2parse[1])

    elif items2parse[0] == 'NoneExcept':
        I2Parse = set(PreColKeys) & set(items2parse[1])

    else:
        exc = 'Unable to figure out items2parse.\n'
        raise parse_Error(exc)

    # Create response category columns (RCats) for each item
    if pytables is None:
        coredata = np.zeros((nDatRows,0))

    # Build PyTable
    else:
            CoreDataTab = tools.pytables_(None,'init_earray',fileh,None,'parse_out',
                                         ['coredata'],None,'float',4,(nDatRows,0),None)
            coredata = CoreDataTab['arrays']['coredata']

    collabels = np.zeros((nPreColRows+3,0))   # initializing col
    dash = ' -- '

    # Prep ExtractKeyDict
    if extractkey is not None:
        if extractkey[0] == 'All':
            ExtractKeyDict = {}
        elif extractkey[0] == 'Cols':
            ExtractKeyDict = extractkey[1]

    # Initialize method and ordinal dictionary
    MethDict = {}
    OrdDict = {}
    for key in PreColKeys:
        OrdDict[key] = False

    # Handle each original column entity
    nPreColKeys = len(PreColKeys)
    for i in xrange(nPreColKeys):

        # Create parsed data
        if PreColKeys[i] in I2Parse:

            # List response categories per item.
            if resp_cat != 'Find':

                if resp_cat[0] == 'All':
                    RCats = resp_cat[1]
                    if ((type(RCats[0]) is type(dash)
                        and dash in RCats[0])
                        or RCats == 'All'
                        ):
                        exc = 'Requires non-continuous response data.  Check items2parse and resp_cat args.\n'
                        raise parse_Error(exc)

                elif resp_cat[0] == 'Cols':
                    RespCatDict = resp_cat[1]
                    RCats = RespCatDict[PreColKeys[i]]

                    if ((type(RCats[0]) is type(dash)
                        and dash in RCats[0])
                        or RCats == 'All'
                        ):
                        exc = 'Requires non-continuous response data.  Check items2parse and resp_cat args.\n'
                        raise parse_Error(exc)

            # Include all answer key options in RCats.  Force int type if possible.
            elif resp_cat == 'Find':
                try:
                    RCats = list(np.unique(PreCoreData[:,i]).astype(float).astype(int))
                except ValueError:
                    RCats = list(np.unique(PreCoreData[:,i]))

                if extractkey is not None:
                    if ExtractKeyDict[PreColKeys[i]][0] not in RCats:
                        RCats.append(ExtractKeyDict[PreColKeys[i]][0])
                        RCats = list(np.unique(RCats))

            else:
                exc = 'Unable to figure out resp_cat parameter.\n'
                raise parse_Error(exc)

            # Remove nanval as a response
            try:
                RCats.remove(prenanval)
            except ValueError:
                try:
                    RCats.remove(str(nanval))
                except ValueError:
                    try:
                        RCats.remove(str(int(nanval)))
                    except ValueError:
                        pass

            # count response categories (temporary)
            nRCats = len(RCats)

            # Add responses to ExtractKeyDict, if necessary
            if (extractkey is not None
                and extractkey[0] == 'All'
                ):
                Resps = extractkey[1]
                Resps4Key = []
                for Resp in Resps:
                    if Resp in RCats:
                        Resps4Key.append(Resp)

                if len(Resps4Key) > 0:
                    ExtractKeyDict[PreColKeys[i]] = Resps4Key
                else:
                    ExtractKeyDict[PreColKeys[i]] = None

            # Add entity method to MethDict
            if ordinal is True:
                if (extractkey is not None
                    and ExtractKeyDict[PreColKeys[i]] is not None
                    ):
                    MethDict[PreColKeys[i]] = ['Extr',ExtractKeyDict[PreColKeys[i]]]
                    OrdDict[PreColKeys[i]] = True

                else:
                    # Apply 'Exp' or 'Pred' methods
                    try:
#                        Test = np.array(RCats).astype(int)
                        MethDict[PreColKeys[i]] = ['Exp']
                    except ValueError:
                        MethDict[PreColKeys[i]] = ['Pred']

            elif ordinal is not True:
                if (extractkey is not None
                    and ExtractKeyDict[PreColKeys[i]] is not None
                    ):
                    MethDict[PreColKeys[i]] = ['Extr',ExtractKeyDict[PreColKeys[i]]]
                else:
                    MethDict[PreColKeys[i]] = ['Pred']

            # If ordinal, remove smallest response category
            if (ordinal is True
                and not isinstance(RCats[0],str)
                and nRCats > 1
                ):
                MinCat = min(RCats)
                RCats.remove(MinCat)
                nRCats -= 1
                OrdDict[PreColKeys[i]] = True

            # Prep to code category colums as 0 or 1 or nanval
            ParsedItem = np.zeros((nDatRows,nRCats),dtype=float)
            ParsedColKey = np.repeat(np.array(PreColKeys[i]),nRCats)
            ParsedColLabel = np.repeat(PreColLabels[:,i][:,np.newaxis],nRCats,axis=1)
            ItemColKeys_ = list(np.zeros((nRCats)).astype(str))

            # Iterate through each response category
            for j in range(nRCats):
                if ordinal is True:

                    # ordinal requires a nanval that can be converted to integer
                    PreCoreData1 = np.where(PreCoreData[:,i] == prenanval,str(nanval),PreCoreData[:,i])

                    # Treat responses as ordinal, backfill
                    try:
                        ParsedItem[:,j] = np.where(PreCoreData1.astype(float).astype(int) < int(float(RCats[j])),0,1)
                        ParsedItem[:,j] = np.where(PreCoreData1 == str(nanval),
                                                   nanval,ParsedItem[:,j])

                    # Treat responses as nominal, no backfill
                    except ValueError:
                        ParsedItem[:,j] = np.where(PreCoreData[:,i] == RCats[j],1,0)
                        ParsedItem[:,j] = np.where(PreCoreData[:,i] == prenanval,
                                                   nanval,ParsedItem[:,j])

                # Treat responses as nominal, no backfill
                else:
                    ParsedItem[:,j] = np.where(PreCoreData[:,i] == RCats[j],1,0)
                    ParsedItem[:,j] = np.where(PreCoreData[:,i] == prenanval,
                                               nanval,ParsedItem[:,j])

                # Build new column keys, with response label
                ItemColKeys_[j] = str(ParsedColKey[j])+'_'+str(RCats[j])

        # if column is not supposed to be parsed
        else:
            RCats = [int(nanval)]
            ParsedItem = np.where(PreCoreData[:,i] == prenanval, nanval, 
                                  PreCoreData[:,i].astype(float))[:,np.newaxis]
            ParsedColKey = np.array(PreColKeys[i])
            ParsedColLabel = PreColLabels[:,i][:,np.newaxis]
            ItemColKeys_ = str(PreColKeys[i])  #+'_'  Was this serving a purpose?
            MethDict[PreColKeys[i]] = None

        # Add rows to collabels
        ItemColKeys = np.array(ItemColKeys_,ndmin=2)
        RCats = np.array(RCats,ndmin=2)
        ParsedColKey = np.array(ParsedColKey,ndmin=2).astype('S60')
        ParsedColLabel = np.concatenate((ItemColKeys, ParsedColKey, RCats, 
                                         ParsedColLabel),axis=0)

        # Append new column labels (per item) to existing collabels and accumulate
        collabels = np.append(collabels,ParsedColLabel,axis=1)

        if pytables is None:
            coredata = np.append(coredata,ParsedItem,axis=1)

        # Append to PyTable
        else:
            coredata.append(ParsedItem)

    # Delete initializing column
    #collabels = np.delete(collabels,np.s_[0],axis=1)
    nheaders4cols = np.size(collabels,axis=0)

    # Delete leading col
##    if pytables is None:
##        coredata = np.delete(coredata,np.s_[0],axis=1)
##
##    # From PyTable (no leading col)
##    else:
##        pass

    # Prepare row labels
    DatRowLabels = rowlabels[PrenHeaders4Cols:,:]

    # Complete rowlabels and collabels by adding header
    filler = np.zeros((nheaders4cols,nheaders4rows),dtype=int)
    ColLabels1 = np.append(filler,collabels,axis=1)
    ColLabels1[1:nheaders4cols,key4rows] =  -1 * np.array(range(1,nheaders4cols ))
    RowLabels1 = np.append(filler,DatRowLabels,axis=0)
    RowLabels1[1:nheaders4cols,key4rows] = ColLabels1[1:nheaders4cols,key4rows]

    # Get revised ValChars
    ParValDict = {}
    IColKeys = ColLabels1[0,nheaders4rows:]
    IColVals = ColLabels1[1,nheaders4rows:].astype(colkeytype)
    IColDict = dict(zip(IColKeys,IColVals))

    for i in xrange(np.size(IColKeys)):
        if IColDict[IColKeys[i]] in I2Parse:
            ParValDict[IColKeys[i]] = [0,1]
        else:
            # ValChars parsed ents may be missing the lowest category
            if ValChars is None:
                ParValDict[IColKeys[i]] = [None]
            elif ValChars[0] == 'All':
                ParValDict[IColKeys[i]] = ValChars[1]
            elif ValChars[0] == 'Cols':
                ParValDict[IColKeys[i]] = ValChars[1][IColDict[IColKeys[i]]]

    ParValChars = ['Cols',ParValDict,'Num']   # 'Cols' because column specific dict

    # Build new pseudo-missing indices for parsed data sets
    try:
        PsMsOrig = self.pseudomiss_out['psmsindex']

        def parsems(MsOrig):

            MsRows = MsOrig[0]
            MsCols = MsOrig[1]
            nMsCols = np.size(MsCols)
            if nMsCols > 1000:
                print 'Warning in parse():  Number of pseudo-missing cells',nMsCols,' > 1000.  If this number gets large it causes a speed bottleneck.  Consider resetting pseudomiss() to make fewer cells pseudomissing.\n'

            Col2Key = {}
            Key2ParCol = {}

            # Parsed col index = address of key obtained from psmsindex
            for Col in xrange(nPreColKeys):
                Col2Key[Col] = PreColKeys[Col]
                Key2ParCol[PreColKeys[Col]] = np.where(IColVals == PreColKeys[Col])

            # Translate original psmsindex to parsed psmsindex
            ParMsCols = np.array([],dtype=int)
            ParMsRows = np.array([],dtype=int)

            for i in xrange(nMsCols):

                # Augment PsMs col index
                ParMsCol = Key2ParCol[Col2Key[MsCols[i]]]
                nResp = np.size(ParMsCol)
                ParMsCols = np.append(ParMsCols,ParMsCol,axis=None)

                # Augment PsMs row index
                ParMsRow = np.repeat(MsRows[i],nResp)
                ParMsRows = np.append(ParMsRows,ParMsRow)

            parsed_msindex = (ParMsRows,ParMsCols)

            return parsed_msindex

        # Build pseudo index and stick in pseudomiss_out
        self.pseudomiss_out['parsed_psmsindex'] = parsems(PsMsOrig)

    # pseudomiss() not run
    except AttributeError:
        pass

    # Labels to pytables
    if pytables is not None:
        RowLabels1 = tools.pytables_(RowLabels1,'array',fileh,None,'parse_out',
                                    ['rowlabels'],None,'float',4,None,None)['arrays']['rowlabels']
        ColLabels1 = tools.pytables_(ColLabels1,'array',fileh,None,'parse_out',
                                    ['collabels'],None,'string',60,None,None)['arrays']['collabels']

    # Create {ParseKey:EntKey} lookup dict
    ParseKeys = ColLabels1[0,:]
    EntKeys = ColLabels1[1,:].astype('S60')
    KeyDict = dict(zip(ParseKeys,EntKeys))

    # Convert to data() format
    ParseDatRCD = {'rowlabels':RowLabels1,
                   'collabels':ColLabels1,
                   'coredata':coredata,
                   'nheaders4rows':nheaders4rows,
                   'key4rows':key4rows,
                   'rowkeytype':datadict['rowkeytype'],
                   'nheaders4cols':nheaders4cols,
                   'key4cols':0,
                   'colkeytype':'S60',
                   'nanval':nanval,
                   'validchars':ParValChars
                   }

    # Add MethDict to parse_out
    ParseDatRCD['MethDict'] = MethDict
    ParseDatRCD['OrdDict'] = OrdDict
    ParseDatRCD['KeyDict'] = KeyDict
    ParseDatRCD['EntRow'] = 1
    ParseDatRCD['RespRow'] = 2
    ParseDatRCD['parse_params'] = parse_params

    return ParseDatRCD



######################################################################

def _standardize(_locals):
    "Basis of the standardize() method."

    ################
    ##   Define   ##
    ##  Variables ##
    ################

    # Retrieve self and datadict
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Get data to standardize
    try:
        datadict = self.parse_out
    except AttributeError:
        try:
            datadict = self.subscale_out
        except AttributeError:
            try:
                datadict = self.score_mc_out
            except AttributeError:
                try:
                    datadict = self.extract_valid_out
                except AttributeError:
                    try:
                        datadict = self.merge_info_out
                    except AttributeError:
                        try:
                            datadict = self.data_out
                        except AttributeError:
                            exc = 'Unable to find data to standardize.\n'
                            raise standardize_Error(exc)

    # Retrieve Local variables
    metric = _locals['metric']
    rescale = _locals['rescale']
    referto = _locals['referto']
    std_params = _locals['std_params']
    RetStdParams = True
    add_datadict = _locals['add_datadict']

    # Define variables (pytables is handled)
    CoreData2 = datadict['coredata']    
#    rowlabels = datadict['rowlabels']
    collabels = datadict['collabels']

    nheaders4rows = datadict['nheaders4rows']
#    key4rows = datadict['key4rows']
#    rowkeytype = datadict['rowkeytype']
#    nheaders4cols = datadict['nheaders4cols']
#    key4cols = datadict['key4cols']
#    colkeytype = datadict['colkeytype']

    validchars = datadict['validchars']
    nanval = float(datadict['nanval'])
#    nRows2 = np.size(CoreData2,axis=0)
#    nCols2 = np.size(CoreData2,axis=1)

    LogDatMin = 0.0001

    # Redefine variables if std_params is used
    if std_params is not None:
        if isinstance(std_params,dict):

            # Overwrite existing parameters
            metric = std_params['stdmetric']
            validchars = std_params['validchars']
            referto = std_params['referto']
            rescale = std_params['rescale']
        else:
            try:
                std_params = np.load(std_params)['std_params']

                # Overwrite existing parameters
                metric = std_params['stdmetric']
                validchars = std_params['validchars']
                referto = std_params['referto']
                rescale = std_params['rescale']
            except:
                exc = 'Unable to figure out std_params parameter.\n'
                raise standardize_Error(exc)
                
    # type_ check coredata
    try:
        CoreData2[0,0] + 1
    except TypeError:
        exc = 'Requires numerical core data.  This is string.  Check type, set validchars switch to Num, consider using score_mc() or parse() if there are alpha chars.\n'
        raise standardize_Error(exc)


    ################
    ##  Original  ##
    ##  Metrics   ##
    ################

    # Get original data metrics for each column
    if referto == 'Cols':
        retcols = tools.getkeys(datadict,'Col','Core','Auto',None)
        ValidChars1 = validchars

    # Under 'Whole', validchars has to be rebuilt for one long column labeled 'All'.
    elif referto == 'Whole':
        if validchars is not None:
            if validchars[0] == 'All':
                ValCh = validchars[1]
            elif validchars[0] == 'Cols':
                RefKey = tuple(validchars[1])[0]
                ReferToWarning = False
                for key in validchars[1].keys():
                    if validchars[1][key] != validchars[1][RefKey]:
                        ReferToWarning = True
                        ValidChars1 = ['Cols',validchars[1]]
                        break
                ValCh = list(validchars[1][RefKey])
                if ReferToWarning is True:
                    print "Warning in standardize(): referto is 'Whole', but the validchars attribute has multiple metrics.  Resetting to 'Cols'.\n"
            else:
                exc = "'validchars' specification must be None, or include 'All' or 'Cols'.\n"
                raise standardize_Error(exc)

            ValidChars1 = ['Cols',{'All':ValCh}]
        else:
            ValidChars1 = validchars

        retcols = ['All']

    else:
        exc = 'Unable to figure out referto parameter.\n'
        raise standardize_Error(exc)

    # Extract metric, Round, and minmax dictionaries
    OrigMetrics = tools.valchars(ValidChars1,    # ['validchars' output of data() function]
                             dash = ' -- ', # [Expression used to denote a range]
                             defnone = 'interval',   # [How to interpret metric when validchars = None]
                             retcols = retcols,    # [<None, [list of core col keys]>]
                             )

    OrigMetric = OrigMetrics['metric'][1]
    minmax = OrigMetrics['minmax'][1]


    ##############
    ##  Resize  ##
    ##   data   ##
    ##############

    # Standardize by columns
    if referto == 'Cols':

        # shape according to referto arg
        coredata = CoreData2[:,:]
        collabels = collabels[:,nheaders4rows:]
        nrows = np.size(coredata[:,:],axis=0)
        ncols = np.size(coredata[:,:],axis=1)
        nDat = np.size(coredata[:,:],axis=None)

    # Standardize by whole array
    elif referto == 'Whole':

        # To resize at end
        nRows_Orig = np.size(CoreData2[:,:],axis=0)
        nCols_Orig = np.size(CoreData2[:,:],axis=1)

        # Convert array to a single column
        coredata = np.ravel(CoreData2[:,:])[:,np.newaxis]
        nrows = np.size(coredata[:,:],axis=0)
        ncols = np.size(coredata[:,:],axis=1)
#        nDat = np.size(coredata[:,:],axis=None)


    ##################
    ##    Prep for  ##
    ##  Percentiles ##
    ##################

    # Create ColDict to access columns from original core data, for Percentile and PLogit only.
    # Percentile and PLogit require the whole data set to be stored under 'params'

    if (std_params is not None and std_params['referto'] is 'Cols'
        and (std_params['stdmetric'] == 'Percentile'
             or std_params['stdmetric'] == 'PLogit')
        ):
        CoreData0 = std_params['params']['coredata']
        ColLabels0 = std_params['params']['collabels']
        Key4Cols0 = std_params['params']['key4cols']
        ColKeyType0 = std_params['params']['colkeytype']
        nHeaders4Rows0 = std_params['params']['nheaders4rows']
        NaNVal0 = std_params['params']['nanval']

        # Build dictionary
        ColKeys0 = ColLabels0[Key4Cols0,nHeaders4Rows0:].astype(ColKeyType0)
        ColVals0 = np.transpose(CoreData0)       # transpose because array is stratified by rows
        ColDict0 = dict(zip(ColKeys0,ColVals0))


    ##################
    ##  Initialize  ##
    ##   StdUnit    ##
    ##################

    if pytables is None:
        StdUnit = np.zeros((nrows,ncols)) + nanval

    # Initialize StdUnit as pytables array
    else:
        StdUnit = tools.pytables_(np.zeros((nrows,ncols)) + nanval,'array',fileh,None,'standardize_out',
                                 ['coredata'],None,None,None,None,None)['arrays']['coredata']

    # Initialize parameter dictionary
    ParamDict = {}

    # Initialize rescale
    if rescale is not None:
        RescaleDict = {}
    else:
        RescaleDict = None

    # collabels has invalid cols removed, including row headers
    if referto == 'Cols':
        colkeys = tools.getkeys(datadict,'Col','Core','Auto',None)
    elif referto == 'Whole':
        colkeys = np.array(['All'])


    ################
    ##  Define    ##
    ##  rescale   ##
    ################

    if rescale is not None:
        if referto == 'Cols':
            if 'All' in rescale.keys():
                mb = rescale['All']
                for key in colkeys:
                    rescale[key] = mb
                del rescale['All']
        elif referto == 'Whole':
            if 'All' not in rescale.keys():
                RefKey = tuple(rescale)[0]
                ReferToWarning = False
                for key in rescale.keys():
                    if rescale[key] != rescale[RefKey]:
                        exc = "'rescale' is by column and 'referto' is 'Whole'.  They need to be made consistent.\n"
                        raise standardize_Error(exc)

                rescale['All'] = rescale[RefKey]
                print "Warning in standardize(): 'referto' is 'Whole' and 'rescale' is by column.  Resetting rescale to include 'All'.\n"


    ################
    ##   Std by   ##
    ##   Columns  ##
    ################

    # Standardization performed column by column
    for i in xrange(ncols):

        # Issue warnings
        ValMetrics = ['ordinal','interval','sigmoid','ratio']
        
        try:
            OrigMetric[colkeys[i]]
        except KeyError:
            exc = 'Could not find {0} in validchars.'.format(colkeys[i])
            raise standardize_Error(exc)

        if OrigMetric[colkeys[i]] not in ValMetrics:
            exc = 'Column '+i+' cannot be standardized as it does not contain numerical data.\n'
            raise standardize_Error(exc)

        if ((OrigMetric[colkeys[i]] == 'ordinal'
             or OrigMetric[colkeys[i]] == 'sigmoid')
                and (metric == 'LogDat'
                     or metric == 'SD')
            ):
            if validchars is None:
                print "Warning in standardize(): You specified Damon.validchars = None.  standardize() is assuming an interval metric.\n"
            else:
                print "Warning in standardize(): For coredata Column",i,"your 'metric' spec is a poor fit with the original data metric.\n"

        # Index valid rows
        ValLoc = np.where(coredata[:,i] != nanval)[0]
        if np.size(ValLoc) == 0:
            exc = 'Column ',i,' has 0 valid observations.  Use extract_valid() to filter out sparse columns/rows.\n'
            raise standardize_Error(exc)


        ########################
        # Take the log if data are ratios or counts
        if metric == 'LogDat':
            CoreData_Ratio = np.clip(coredata[:,i][ValLoc],LogDatMin,np.inf)
            StdUnit[ValLoc,i] = np.log(CoreData_Ratio)

            # rescale
            if rescale is not None:
                m = rescale[colkeys[i]][0]
                b = rescale[colkeys[i]][1]
                StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

            # Save standardization parameters
            if RetStdParams is True:
                ParamDict[colkeys[i]] = None

                if rescale is not None:
                    RescaleDict[colkeys[i]] = [m,b]


        ########################
        # Standard deviation metric
        elif metric == 'SD':

            # Prep core data
            if OrigMetric[colkeys[i]] == 'ratio':
                CoreData_Ratio = np.clip(coredata[:,i][ValLoc],LogDatMin,np.inf)
                ValCoreData1 = np.log(CoreData_Ratio)
            else:
                ValCoreData1 = coredata[:,i][ValLoc]

            # Get std_params from previous dataset
            if (std_params is not None
                and std_params['stdmetric'] == 'SD'
                ):
                Mean = std_params['params'][colkeys[i]][0]
                SD = std_params['params'][colkeys[i]][1]
            else:
                Mean = np.mean(ValCoreData1,axis=None)
                SD = np.std(ValCoreData1,axis=None)

            # Handle SD = 0
            if SD == 0.0:
                SD = 1.0
                print 'Warning in standardize(): Column',i,'has standard deviation = 0.0.  Setting to 1.0 to avoid div/0 error.\n'

            # Standardize
            StdUnit[ValLoc,i] = (ValCoreData1 - Mean) / SD

            # rescale
            if rescale is not None:
                m = rescale[colkeys[i]][0]
                b = rescale[colkeys[i]][1]
                StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

            # Save standardization parameters
            if RetStdParams is True:
                ParamDict[colkeys[i]] = [Mean,SD]

                # Save rescale params
                if rescale is not None:
                    RescaleDict[colkeys[i]] = [m,b]


        ########################
        # PMinMax metric
        elif metric == 'PMinMax':
            ValCoreData = coredata[:,i][ValLoc]

            if (std_params is not None
                and std_params['stdmetric'] == 'PMinMax'
                ):
                Min = std_params['params'][colkeys[i]][0]
                Max = std_params['params'][colkeys[i]][1]
            else:
                try:                                # Possible: {'All':None} => TypeError
                    Min = minmax[colkeys[i]][0]
                    Max = minmax[colkeys[i]][1]
                except TypeError:
                    Min = np.amin(ValCoreData,axis=None)
                    Max = np.amax(ValCoreData,axis=None)

            # Handle Min = Max
            if Max == Min:
                Max = Min + 1.0
                print 'Warning in standardize(): The column',i,'minimum and maximum are the same.  Adding 1.0 to maximum to avoid div/0 error.\n'

            # Standardize cells
            StdUnit[ValLoc,i] = (coredata[:,i][ValLoc] - Min) / float(Max - Min)

            # Save standardization parameters
            if RetStdParams is True:
                ParamDict[colkeys[i]] = [Min,Max]


        ########################
        # '0-1' metric
        elif metric == '0-1':

            # Get std_params from previous dataset
            if (std_params is not None
                and std_params['stdmetric'] == '0-1'
                ):
                Mean = std_params['params'][colkeys[i]][0]
                SD = std_params['params'][colkeys[i]][1]

            # Get Min and Max, in case
            try:
                if validchars is not None:
                    Min = minmax[colkeys[i]][0]
                    Max = minmax[colkeys[i]][1]
                else:
                    ValCoreData = coredata[:,i][ValLoc]
                    Min = np.amin(ValCoreData,axis=None)
                    Max = np.amax(ValCoreData,axis=None)
            except TypeError:
                pass

            ###########
            # Convert ratio/interval data to PreLogits, then to probabilities ('0-1')
            if (OrigMetric[colkeys[i]] == 'ratio'
                or OrigMetric[colkeys[i]] == 'interval'
                ):

                # Linearize ratio data
                if OrigMetric[colkeys[i]] == 'ratio':
                    CoreData_Ratio = np.clip(coredata[:,i][ValLoc],LogDatMin,np.inf)
                    ValCoreData1 = np.log(CoreData_Ratio)

                # interval data
                elif OrigMetric[colkeys[i]] == 'interval':
                    ValCoreData1 = coredata[:,i][ValLoc]

                # Get Mean, SD
                if std_params is None:
                    Mean = np.mean(ValCoreData1,axis=None)
                    SD = np.std(ValCoreData1,axis=None)

                # Handle SD = 0
                if SD == 0:
                    SD = 1.0
                    print 'Warning in standardize(): Column',i,'has standard deviation = 0.0.  Setting to 1.0 to avoid div/0 error.\n'

                # Convert standard deviations to logits, then probabilities
                # Conversion factor = pi/sqrt(3) = 1.81379936423422
                PiSqrt3 = 1.81379936423422
                Log = PiSqrt3 * (ValCoreData1 - Mean) / SD
                StdUnit[ValLoc,i] = np.exp(Log) / (1.0 + np.exp(Log))

                # Save standardization parameters
                if RetStdParams is True:
                    ParamDict[colkeys[i]] = [Mean,SD]

            ###########
            # Convert ordinal or sigmoid data to sigmoid 0 - 1 using PMinMax formula
            else:

                # Handle Min = Max
                if Max == Min:
                    Max = Min + 1.0
                    print 'Warning in standardize(): The column',i,'minimum and maximum are the same.  Adding 1.0 to maximum to avoid div/0 error.\n'

                # Standardize cells
                StdUnit[ValLoc,i] = (coredata[:,i][ValLoc] - Min) / float(Max - Min)

                # Save standardization parameters
                if RetStdParams is True:
                    ParamDict[colkeys[i]] = 'VCMinMax'


        ########################
        # 'PreLogit' metric
        elif metric == 'PreLogit':

            # Get std_params from previous dataset
            if (std_params is not None
                and std_params['stdmetric'] == 'PreLogit'
                ):
                
                try:
                    Mean = std_params['params'][colkeys[i]][0]                
                    SD = std_params['params'][colkeys[i]][1]
                    
                    # Mean, SD are not actually called in this case
                    if isinstance(Mean, str):
                        Mean, SD = None, None
                        
                except KeyError:
                    Mean, SD = None, None
                    
            # Get Min and Max, in case
            try:
                if validchars is not None:
                    Min = minmax[colkeys[i]][0]
                    Max = minmax[colkeys[i]][1]
                else:
                    ValCoreData = coredata[:,i][ValLoc]
                    Min = np.amin(ValCoreData,axis=None)
                    Max = np.amax(ValCoreData,axis=None)
            except TypeError:
                pass

            ###########
            # Convert ratio/interval data to PreLogits
            if (OrigMetric[colkeys[i]] == 'ratio'
                or OrigMetric[colkeys[i]] == 'interval'
                ):

                # Linearize ratio data
                if OrigMetric[colkeys[i]] == 'ratio':
                    CoreData_Ratio = np.clip(coredata[:,i][ValLoc],LogDatMin,np.inf)
                    ValCoreData1 = np.log(CoreData_Ratio)

                # interval data
                elif OrigMetric[colkeys[i]] == 'interval':
                    ValCoreData1 = coredata[:,i][ValLoc]

                # Get Mean, SD (or obtained at top of block from std_params)
                if (std_params is None
                    or Mean is None):
                    Mean = np.mean(ValCoreData1,axis=None)
                    SD = np.std(ValCoreData1,axis=None)

                # Handle SD = 0
                if SD == 0:
                    SD = 1.0
                    print 'Warning in standardize(): Column',i,'has standard deviation = 0.0.  Setting to 1.0 to avoid div/0 error.\n'

                # Convert standard deviations to PreLogits
                # Conversion factor = pi/sqrt(3) = 1.81379936423422
                PiSqrt3 = 1.81379936423422
                StdUnit[ValLoc,i] = PiSqrt3 * (ValCoreData1 - Mean) / SD

                # rescale
                if rescale is not None:
                    m = rescale[colkeys[i]][0]
                    b = rescale[colkeys[i]][1]
                    StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

                # Save standardization parameters
                if RetStdParams is True:
                    ParamDict[colkeys[i]] = [Mean,SD]

                    # Save rescale params
                    if rescale is not None:
                        RescaleDict[colkeys[i]] = [m,b]

            ###########
            # Convert continuous sigmoid data to PreLogits
            elif OrigMetric[colkeys[i]] == 'sigmoid':

                # Handle Min = Max
                if Max == Min:
                    Max = Min + 1.0
                    print 'Warning in standardize(): The column',i,'minimum and maximum are the same.  Adding 1.0 to maximum to avoid div/0 error.\n'

                Normed = (coredata[:,i][ValLoc] - Min) / (Max - Min)
                ClipCore = np.clip(Normed,0.000001,0.999999)
                StdUnit[ValLoc,i] = np.log(ClipCore / (1.0 - ClipCore))

                # rescale
                if rescale is not None:
                    m = rescale[colkeys[i]][0]
                    b = rescale[colkeys[i]][1]
                    StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

                # Save standardization parameters
                if RetStdParams is True:
                    ParamDict[colkeys[i]] = 'VCMinMax'

                    # Save rescale params
                    if rescale is not None:
                        RescaleDict[colkeys[i]] = [m,b]

            ###########
            # Convert ordinal data to PreLogits (assumes categories are integers)
            elif OrigMetric[colkeys[i]] == 'ordinal':

                # Handle Min = Max
                if Max == Min:
                    Max = Min + 1.0
                    print 'Warning in standardize(): The column',i,'minimum and maximum are the same.  Adding 1.0 to maximum to avoid div/0 error.\n'

                # Set range of logits/probs
                MagicSqueeze = 0.10
                m = (Max + Min) / 2.0
                data = coredata[:,i][ValLoc]
                Normed = 0.5 + (data - m) / (Max - Min + MagicSqueeze)
                StdUnit[ValLoc,i] = np.log(Normed / (1.0 - Normed))

                # rescale
                if rescale is not None:
                    m = rescale[colkeys[i]][0]
                    b = rescale[colkeys[i]][1]
                    StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

                # Save standardization parameters
                if RetStdParams is True:
                    ParamDict[colkeys[i]] = 'VCMinMax'

                    # Save rescale params
                    if rescale is not None:
                        RescaleDict[colkeys[i]] = [m,b]


        ########################
        # PLogit or percentile metric
        elif metric == 'Percentile' or metric == 'PLogit':
            ValCoreData = coredata[:,i][ValLoc]

            if (std_params is not None
                and (std_params['stdmetric'] == 'Percentile'
                     or std_params['stdmetric'] == 'PLogit')
                ):

                # Use coredata from another data set stored in std_params
                ColData0 = ColDict0[colkeys[i]]
                ValCoreData = ColData0[np.where(ColData0 != NaNVal0)]

            # Sort column
            SortedCol = np.sort(ValCoreData,axis=None)
            nSortedCol = np.size(SortedCol)

            # Standardize each cell in this coredata column, down the rows
            for r in xrange(nrows):
                if coredata[r,i] == nanval:
                    StdUnit[r,i] = nanval
                else:
                    MinLoc = np.searchsorted(SortedCol,coredata[r,i],'Left')
                    MaxLoc = np.searchsorted(SortedCol,coredata[r,i],'Right')
                    nTies = MaxLoc - MinLoc     # number of ties
                    Below = MinLoc + nTies/2.
                    #Above = (nSortedCol - MaxLoc) + nTies/2.

                    # Get percentile
                    StdUnit[r,i] = Below / float(nSortedCol)

            #k = 1
            #h = 0
            Normed = np.clip(StdUnit[ValLoc,i],0.000001,0.999999)

            if metric == 'Percentile':
                StdUnit[ValLoc,i] = Normed

            elif metric == 'PLogit':
                StdUnit[ValLoc,i] = np.log( Normed / (1.0 - Normed))

            # rescale
            if rescale is not None:
                m = rescale[colkeys[i]][0]
                b = rescale[colkeys[i]][1]
                StdUnit[ValLoc,i] = m * StdUnit[ValLoc,i] + b

            # Save standardization parameters
            if RetStdParams is True:

                # Save rescale params
                if rescale is not None:
                    RescaleDict[colkeys[i]] = [m,b]

        # Unknown metric
        else:
            exc = ('Error in standardize():  Unable to figure out metric '
                   'parameter.\n')
            raise standardize_Error(exc)


    ########################
    # Save standardization parameters for whole data array
    if RetStdParams is True:
        if metric == 'Percentiles' or metric == 'PLogits':

            if metric == 'Percentile':
               StdParams1 = {'stdmetric':'Percentile',
                             'validchars':validchars,
                             'referto':referto,
                             'params':datadict,
                             'rescale':RescaleDict,
                             'orig_data':None
                             }

            elif metric == 'PLogit':
                StdParams1 = {'stdmetric':'PLogit',
                              'validchars':validchars,
                              'referto':referto,
                              'params':datadict,
                              'rescale':RescaleDict,
                              'orig_data':None
                              }

        # std_params for other metrics
        else:
            StdParams1 = {'stdmetric':metric,
                         'validchars':validchars,
                         'referto':referto,
                         'params':ParamDict,
                         'rescale':RescaleDict,
                         'orig_data':None
                         }
    else:
        StdParams1 = None


    #############
    ##  Valid  ##
    ##  Chars  ##
    #############

    # Update validchars specification (standardized ValChars applies to 'All' columns, by definition)
    if metric == 'LogDat':
        StdValChars = ['All',['0.0 -- ']]

    elif (metric == 'PreLogit'
          or metric == 'SD'
          or metric == 'PLogit'
          ):
        StdValChars = ['All',['All']]

    elif (metric == 'PMinMax'
          or metric == '0-1'
          or metric == 'Percentile'
          ):
        StdValChars = ['All',['0.0 -- 1.0']]


    ###############
    ##  Reports  ##
    ###############

    # Save original dataset in std_params
    if add_datadict == 1:
        std_params['orig_data'] = datadict

    # Reconstitute std matrix to match original coredata
    if referto == 'Cols':
        StdUnit2 = StdUnit[:,:]
##        StdUnit2 = np.copy(CoreData2[:,:])
##        StdUnit2[:,:] = nanval
##        np.transpose(StdUnit2)[ValColLoc] = np.transpose(StdUnit)

    elif referto == 'Whole':
        StdUnit2 = np.resize(StdUnit[:,:],(nRows_Orig,nCols_Orig))

    # Add to pytables
    if pytables is not None:
        StdUnit2 = tools.pytables_(StdUnit2,'array',fileh,None,'standardize_out',
                                    ['coredata'],None,None,None,None,None)['arrays']['coredata']

    # Convert to datadict
    StdUnitRCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars'
               ]

    for key in datadict.keys():
        if key in ValList:
            StdUnitRCD[key] = datadict[key]

        StdUnitRCD['coredata'] = StdUnit2
        StdUnitRCD['validchars'] = StdValChars
        StdUnitRCD['stdmetric'] = metric
        StdUnitRCD['std_params'] = StdParams1

    return StdUnitRCD





######################################################################

def _rasch(_locals):
    "Basis of the rasch() method."
    
    # TODO: Flag and handle extreme values better. Filter out when
    # calculating derivative stats.

    # Get variables
    self = _locals['self']
    groups = _locals['groups']
    anchors = _locals['anchors']
    runspecs = _locals['runspecs']
    minvar = _locals['minvar']
    maxchange = _locals['maxchange']
    labels = _locals['labels']
    extreme = _locals['extreme']

    # Get data
    try:
        data = self.score_mc_out
    except AttributeError:
        try:
            data = self.extract_valid_out
        except AttributeError:
            try:
                data = self.merge_info_out
            except AttributeError:
                try:
                    data = self.data_out
                except AttributeError:
                    exc = 'Unable to find data to analyze.\n'
                    raise rasch_Error(exc)

    nanval = data['nanval']

    # Build groups dictionary with indices and item keys
    all_items = tools.getkeys(data, 'Col', 'Core', 'Auto', None)
    groups_arg = groups

    if groups_arg is None:
        groups_arg = ['key', {'All':all_items}]

    if isinstance(groups_arg, dict):
        try:
            groups_row = data['collabels'][groups_arg['row'], data['nheaders4rows']:]
            groups_list = np.unique(groups_row).tolist()
            groups = {}

            for group in groups_list:
                index = np.where(groups_row == group)[0]
                g_items = all_items[index]
                groups[group] = {'index':index, 'items':g_items}

        except KeyError:
            exc = 'Unable to find "row" in groups parameter.\n'
            raise rasch_Error(exc)

    elif isinstance(groups_arg, list) and 'key' in groups_arg:
        groups_list = groups_arg[1].keys()
        groups = {}

        for group in groups_list:
            g_items = groups_arg[1][group]
            try:
                index = np.array([np.where(all_items == g_items[i])[0][0]
                                  for i in range(len(g_items))])
            except IndexError:
                print 'all_items=\n', list(all_items)
                print 'g_items=\n', g_items
                exc = ('Unable to index items in groups parameter for '
                       'some reason -- probably one of the items in the '
                       'group not appearing in the list of all items. '
                       'See printed info above.')
                raise rasch_Error(exc)

            groups[group] = {'index':index, 'items':g_items}

    elif isinstance(groups_arg, list) and 'index' in groups_arg:
        groups_list = groups_arg[1].keys()
        groups = {}

        for group in groups_list:
            index = groups_arg[1][group]
            try:
                g_items = all_items[index]
            except IndexError:
                exc = ('Unable to figure out index in groups parameter. Check that '
                       'indices are integers.\n')
                raise rasch_Error(exc)

            groups[group] = {'index':index, 'items':g_items}

    else:
        exc = 'Unable to figure out groups parameter.\n'
        raise rasch_Error(exc)


    #################
    ##  Initialize ##
    ##  Variables  ##
    #################

    # Get categories
    cats = {}
    for group in groups_list:
        if isinstance(data['validchars'][1], dict):
            g_items = groups[group]['items']
            cats_list = data['validchars'][1][g_items[0]]

            # Check that all items have same validchars
            for item in g_items:
                if data['validchars'][1][item] != cats_list:
                    exc = 'Not all items in group' + group + 'have the same range of responses. Check validchars.\n'
                    raise rasch_Error(exc)
        else:
            cats_list = data['validchars'][1]

        # Check that cats are integers
        for cat in cats_list:
            if cat % 1 != 0:
                exc = 'Scoring category is not integer-like. Check my_obj.validchars.'
                raise rasch_Error(exc)

        cats[group] = cats_list

    # Number of categories for each item
    cats_per_item = {}
    for group in groups_list:
        g_items = groups[group]['items']
        for item in g_items:
            cats_per_item[item] = cats[group]

    # Stopping
    stop_when_change = runspecs[0]
    max_iteration = runspecs[1]

    # List of entities
    R_ents = tools.getkeys(data, 'Row', 'Core', 'Auto', None)
    C_ents = all_items

    # Shape of observed data
    obs = data['coredata']
    nrows, ncols = np.shape(obs)

    # Estimates, variance, fit
    est = np.zeros((nrows, ncols))
    est_fin = np.zeros((nrows, ncols))
    var = np.zeros((nrows, ncols))
    var_fin = np.zeros((nrows, ncols))
    R_var = np.zeros((nrows, 1))
    R_res = np.zeros((nrows, 1))
    R_infit = np.zeros((nrows, 1))
    R_outfit = np.zeros((nrows, 1))
    C_var = np.zeros((1, ncols))
    C_res = np.zeros((1, ncols))
    C_infit = np.zeros((1, ncols))
    C_outfit = np.zeros((1, ncols))

    # Frequencies and steps
    def init_cat_stats(cats, groups_list):
        cat_init = {}
        for group in groups_list:
            ncats = len(cats[group])
            cat_init[group] = np.zeros((ncats))

        return cat_init

    exp_cat_freq = init_cat_stats(cats, groups_list)
    obs_step_rat = init_cat_stats(cats, groups_list)
    exp_step_rat = init_cat_stats(cats, groups_list)

    # Raw category freqs
    obs_cat_freq = {}
    for group in groups_list:
        g_cats = cats[group]
        obs_cat_freq[group] = np.zeros((len(g_cats)))

        # Count cats in observations
        for cat in g_cats:
            try:
                obs_cat_freq[group][cat] = (np.sum(obs[:, groups[group]['index']] 
                                            == cat))
            except IndexError:
                exc = ('Category {0} in group {1} turned up empty. Use '
                       'extract_valid() to remove items with '
                       'no variation or collapse categories.').format(cat, group)
                raise IndexError(exc)
                           
    # Category probability matrices
    cat_probs = {}
    for group in groups_list:
        cat_probs_ = {}

        for cat in cats[group]:
            cat_probs_[cat] = np.zeros((nrows, len(groups[group]['index'])))

        cat_probs[group] = cat_probs_

    # Create indices for valid data in each row/column
    R_locval = {}
    C_locval = {}
    for i in xrange(nrows):
        R_locval[i] = np.where(obs[i, :] != nanval)

    for i in xrange(ncols):
        C_locval[i] = np.where(obs[:, i] != nanval)

    # Get maximum raw score per row
    g_cols = []
    for group in groups_list:
        g_cols.append(len(groups[group]['items']) * len(cats[group]))

    # Set limits on extreme values for R and C
    j, k = float(extreme[0]), float(extreme[1])

    max_row_logit = np.log((sum(g_cols) - j) / j)   # Same for each row
    min_row_logit = -1 * max_row_logit

    max_col_logit = {}  # Differs across columns
    min_col_logit = {}
    for item in all_items:
        max_col_logit[item] = np.log((nrows * len(cats_per_item[item]) - k) / k)
        min_col_logit[item] = -1 * max_col_logit[item]


    #####################
    ##  Apply R, C, T  ##
    ##     Anchors     ##
    #####################

    # Initialize: R (row measures), C (col measures) , T (step measures)
    R = np.zeros((nrows, 1))
    C = np.zeros((1, ncols))

    T = {}
    for group in groups_list:
        T[group] = np.zeros((len(cats[group])))

    # Where R, C are not anchored
    R_nonanc_loc = np.where(R == 0)
    C_nonanc_loc = np.where(C == 0)
    calc_T = True

    # Impose entity anchors
    if anchors is not None:

        # Load bank
        bank = np.load(anchors['Bank'])

        # Initialize R, C anchors
        R_anc = np.copy(R)
        R_anc += nanval

        C_anc = np.copy(C)
        C_anc += nanval

        anc_fac = None

        # Row entity anchors
        if anchors['row_ents'] not in [None, [None]]:
            anc_fac = 0
            R_file = bank['facet0']['ent_coord']

            # Count matches with bank
            R_matched = set(R_ents) & set(R_file.keys())
            n_in_common = len(R_matched)

            if self.verbose is True:
                print ('N matches between row entities and bank =', n_in_common,
                       '. If N is low, look for key format discrepancies.\n')

            # Merge bank coordinates with anchored entities
            if anchors['row_ents'] in ['All', ['All']]:
                R_anc_ents = R_matched
            else:
                R_anc_ents = np.array(anchors['row_ents'],
                                      dtype=data['rowkeytype'])[:, np.newaxis]

            # Fill R from bank
            for i, R_ent in enumerate(R_ents):
                if R_ent in R_anc_ents:
                    try:
                        R_anc[i, :] = R_file[R_ent]
                    except KeyError:
                        pass

            # Index non-anchor and anchor locations
            R_nonanc_loc = np.where(R_anc == nanval)
            R_anc_loc = np.where(R_anc != nanval)

        # Column entity anchors
        if anchors['col_ents'] not in [None, [None]]:
            anc_fac = 1
            C_file = bank['facet1']['ent_coord']

            # Count matches with bank
            C_matched = set(C_ents) & set(C_file.keys())
            n_in_common = len(C_matched)

            if self.verbose is True:
                print ('N matches between column entities and bank =', n_in_common,
                       '. If N is low, look for key format discrepancies.\n')

            # Merge bank coordinates with anchored entities
            if anchors['col_ents'] in ['All', ['All']]:
                C_anc_ents = C_matched
            else:
                C_anc_ents = np.array(anchors['col_ents'],
                                      dtype=data['colkeytype'])[np.newaxis, :]

            # Fill C from bank
            for i, C_ent in enumerate(C_ents):
                if C_ent in C_anc_ents:
                    try:
                        C_anc[:, i] = C_file[C_ent]
                    except KeyError:
                        pass

            # Index non-anchor locations
            C_nonanc_loc = np.where(C_anc == nanval)
            C_anc_loc = np.where(C_anc != nanval)

        # Get step anchors T
        try:
            T_ = bank['step_coord']
            T_anc = {}
            for group in groups_list:
                T_anc[group] = np.squeeze(T_[group]['coredata'])
            calc_T = None

        except KeyError:
            exc = ('Warning in rasch(): Could not find step_coord or group in '
                   'anchor file. You may need to delete your bank file and '
                   'rebuild it with a dataset that is like the current dataset.\n')
            raise rasch_Error(exc)


    #################
    ##  Calculate  ##
    ##  R, C, T    ##
    #################

    # Iterate to calculate row, column, step measures
    it = 0
    stop = 0 if anchors is None else 1
    max_res = 1

    if self.verbose is True:
        print 'It\tChange'

    while stop < 2:

        est[:, :] = 0
        var[:, :] = 0

        # Calculate category probability numerators.  Accumulate for denominators
        for group in groups_list:

            # Impose anchors
            if anchors is not None:
                T = T_anc

                if anc_fac == 0:
                    R[R_anc_loc] = R_anc[R_anc_loc]
                elif anc_fac == 1:
                    C[C_anc_loc] = C_anc[C_anc_loc]

            # Pull group section of main arrays
            ind = groups[group]['index']
            g_C = C[:, ind]
            g_est = np.zeros((nrows, len(ind)))
            g_obs = obs[:, ind]
            g_est_fin = np.zeros((nrows, len(ind)))
            g_var = np.zeros((nrows, len(ind)))
            g_var_fin = np.zeros((nrows, len(ind)))

            # Initialize denominator = sum(all numerators)
            cat_prob_denom = np.zeros((nrows, len(groups[group]['items'])))



            # TODO:  Check the formula -- top and (top-1) categories have same sum
            # See MMEdits_Poly_Rasch_Demo_v3.xlsx

            for i, cat in enumerate(cats[group]):
                try:
                    cat_probs[group][cat] = np.exp(cat * (R - g_C) 
                                                   - np.sum(T[group][:i + 1]))
                except TypeError:
                    exc = ('Found non-integer values.  Make sure inputs are '
                           'integers.\n')
                    raise rasch_Error(exc)

                cat_prob_denom += cat_probs[group][cat]

            # For each category:  Numerator / Denominator
            for cat in cats[group]:
                cat_probs[group][cat] = cat_probs[group][cat] / cat_prob_denom

            # Calculate expected values (estimates)
            for cat in cats[group]:
                g_est = np.where(g_obs == nanval, nanval,
                                 g_est + (cat * cat_probs[group][cat]))

                # Estimates for final iteration, no nanvals
                if stop == 1:
                    g_est_fin += cat * cat_probs[group][cat]

            # Cell variances: sum(cat^2 * p[cat])[cats] - est^2)
            for cat in cats[group]:
                g_var = np.where(g_obs == nanval, nanval,
                                 g_var + (cat**2 * cat_probs[group][cat]))

                # Estimates for final iteration, no nanvals
                if stop == 1:
                    g_var_fin += (cat**2 * cat_probs[group][cat])

            # Add estimates term of variance formula above
            g_var = np.where(g_obs == nanval, nanval,
                             g_var - g_est**2)

            if stop == 1:
                g_var_fin += g_var_fin - g_est**2

            # Populate estimates array
            est[:, ind] = g_est
            var[:, ind] = g_var

            if stop == 1:
                est_fin[:, ind] = g_est_fin
                var_fin[:, ind] = g_var_fin

        # Get row/col sums of variances
        for i in xrange(nrows):
            R_var[i, :] = np.sum(var[i, :][R_locval[i]])

        for i in xrange(ncols):
            C_var[:, i] = np.sum(var[:, i][C_locval[i]])

        R_var = np.clip(R_var, minvar, np.inf)
        C_var = np.clip(C_var, minvar, np.inf)

        # Get residuals
        res = np.where(obs == nanval, nanval, obs - est)

        # Get row/col sums of residuals
        for i in xrange(nrows):
            R_res[i, :] = np.sum(res[i, :][R_locval[i]])

        for i in xrange(ncols):
            C_res[:, i] = np.sum(res[:, i][C_locval[i]])

        # Calculate new R, C. Constrain change, constrain R and C.
        R[R_nonanc_loc] = np.clip(R[R_nonanc_loc] +
                                  np.clip(R_res[R_nonanc_loc] / R_var[R_nonanc_loc],
                                          -1 * maxchange, maxchange),
                                  min_row_logit,
                                  max_row_logit)

        # C handled differently because it has a group component
        C[C_nonanc_loc] -= np.clip(C_res[C_nonanc_loc] / C_var[C_nonanc_loc],
                                   -1 * maxchange, maxchange)

        # Impose limits on C
        for i, item in enumerate(all_items):
            if C[0][i] != nanval:
                C[0][i] = np.clip(C[0][i], min_col_logit[item], max_col_logit[item])

        # Adjust C to have mean of zero
        if (anchors is None
            or anchors['col_ents'] is [None]):
            C -= np.mean(C)

        # Calculate new T
        if calc_T is True:

            for group in groups_list:
                if 0 in obs_cat_freq[group]:
                    exc = 'One of your rating categories is not represented in the data.  Adjust validchars attribute.\n'
                    raise rasch_Error(exc)

                for cat in cats[group]:
                    valloc = np.where(cat_probs[group][cat] != nanval)
                    exp_cat_freq[group][cat] = np.sum(cat_probs[group][cat][valloc])

                    # Bottom step category always set at zero
                    if cat == 0:
                        obs_step_rat[group][cat] = 0
                        exp_step_rat[group][cat] = 0
                        T[group][cat] = 0

                    # Get ratios of adjacent observed cats and expected cats to get steps
                    else:
                        obs_step_rat[group][cat] = (obs_cat_freq[group][cat] /
                                                    float(obs_cat_freq[group][cat - 1]))

                        exp_step_rat[group][cat] = (exp_cat_freq[group][cat] /
                                                    float(exp_cat_freq[group][cat - 1]))

                        T[group][cat] += np.log(exp_step_rat[group][cat] / obs_step_rat[group][cat])

                # Adjust T to have mean of zero
                T[group][1:] -= np.mean(T[group][1:])

        # Evaluate sums of residuals
        maxR_res = np.max(np.abs(R_res))
        maxC_res = np.max(np.abs(C_res))
        prev = max_res
        max_res = max(maxR_res, maxC_res)
        change = abs((max_res - prev))

        # Report
        if self.verbose is True:
            print it, '\t', round(change, 4)

        # Evaluate stopping conditions.  For extra iteration for final estimates.
        if (max_res < stop_when_change
            or it >= max_iteration - 1
            ):
            stop += 1

        # Increment iteration
        it += 1


    ################
    ##  Calculate ##
    ##   SE, Fit  ##
    ################

    # Get cell fit -- standardized residuals
    fit = np.copy(obs)
    valloc = np.where((res != nanval) & (var != nanval))  # was "or |" ??
    fit[valloc] = res[valloc] / np.sqrt(var[valloc])

    # Get standard errors
    R_se = np.sqrt(1 / R_var)
    C_se = np.sqrt(1 / C_var)

    # Get row infit
    for i in xrange(nrows):
        R_infit[i, :] = np.sum(res[i, :][R_locval[i]]**2) / R_var[i, :]

    # Get col infit
    for i in xrange(ncols):
        C_infit[:, i] = np.sum(res[:, i][C_locval[i]]**2) / C_var[:, i]

    # Get row outfit
    for i in xrange(nrows):
        R_outfit[i, :] = np.average(fit[i, :][R_locval[i]]**2)

    # Get col outfit
    for i in xrange(ncols):
        C_outfit[:, i] = np.average(fit[:, i][C_locval[i]]**2)

    # Get row separation
    R_rmsr = tools.rmsr(None, None, R_se, nanval)
    R_sep = tools.separation(None, R_rmsr, R, nanval)

    # Get col separation
    C_rmsr = tools.rmsr(None, None, C_se, nanval)
    C_sep = tools.separation(None, C_rmsr, C, nanval)

    # Get row reliability
    R_rel = tools.reliability(R_sep, None, None, None, nanval)

    # Get col reliability
    C_rel = tools.reliability(C_sep, None, None, None, nanval)

    # Assemble reliability table
    rel_tab = np.array([[R_sep, R_rel], [C_sep, C_rel]])


    #################
    ##   Build     ##
    ##  Datadicts  ##
    #################

    # Observations
    observed = data.copy()

    # Prepare rowlabels
    d = data
    d['dtype'] = [object, 3, '']
    R_rowlabels = np.append(d['rowlabels'][d['key4cols'], :][np.newaxis, :],
                            d['rowlabels'][d['nheaders4cols']:, :],
                            axis=0)

    C_rowlabels = np.transpose(np.append(d['collabels'][:, d['key4rows']][:, np.newaxis],
                                         d['collabels'][:, d['nheaders4rows']:],
                                         axis=1)
                               )

    # Add person/item header label
    if labels is not None:
        R_rowlabels[0, d['key4rows']] = labels['row_ents']
        C_rowlabels[0, d['key4cols']] = labels['col_ents']

        #for full arrays? d['rowlabels'][d['key4cols'], d['key4rows']]

    R_corner = R_rowlabels[0, :]
    C_corner = C_rowlabels[0, :]

    # Step coordinates (T), folded into fac0coord and fac1coord, not reported separately
    step_coord = {}

    for group in groups_list:
        step_coord[group] = {'rowlabels':np.array(range(len(cats[group])))[:, np.newaxis],
                             'collabels':np.array([['Step', 'Difficulty']]),
                             'coredata':T[group][:, np.newaxis],
                             'nheaders4rows':1,
                             'key4rows':0,
                             'rowkeytype':'S60',
                             'nheaders4cols':1,
                             'key4cols':0,
                             'colkeytype':'S60',
                             'nanval':nanval,
                             'validchars':['All', ['All'], 'Num']
                             }

    # Row coordinates R
    fac0coord = {'rowlabels':R_rowlabels,
                 'collabels':np.append(R_corner[np.newaxis, :], np.array([['Measure']]), axis=1),         
                 'coredata':R,
                 'nheaders4rows':d['nheaders4rows'],
                 'key4rows':d['key4rows'],
                 'rowkeytype':d['rowkeytype'],
                 'nheaders4cols':1,
                 'key4cols':0,
                 'colkeytype':'S60',
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num'],
                 'step_coord':step_coord,
                 'opp_count':len(C)
                 }

    # Col coordinates C
    fac1coord = {'rowlabels':C_rowlabels,
                 'collabels':np.append(C_corner[np.newaxis, :], np.array([['Measure']]), axis=1),       
                 'coredata':np.transpose(C),
                 'nheaders4rows':d['nheaders4cols'],
                 'key4rows':d['key4cols'],
                 'rowkeytype':d['colkeytype'],
                 'nheaders4cols':1,
                 'key4cols':0,
                 'colkeytype':'S60',
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num'],
                 'step_coord':step_coord,
                 'opp_count':len(R)
                 }

    # Estimates
    estimates = {'rowlabels':d['rowlabels'],
                 'collabels':d['collabels'],
                 'coredata':est_fin,
                 'nheaders4rows':d['nheaders4rows'],
                 'key4rows':d['key4rows'],
                 'rowkeytype':d['rowkeytype'],
                 'nheaders4cols':d['nheaders4cols'],
                 'key4cols':d['key4cols'],
                 'colkeytype':d['colkeytype'],
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num']
                 }

    # Residuals
    residuals = {'rowlabels':d['rowlabels'],
                 'collabels':d['collabels'],
                 'coredata':res,
                 'nheaders4rows':d['nheaders4rows'],
                 'key4rows':d['key4rows'],
                 'rowkeytype':d['rowkeytype'],
                 'nheaders4cols':d['nheaders4cols'],
                 'key4cols':d['key4cols'],
                 'colkeytype':d['colkeytype'],
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num']
                 }

    # Cell variance
    cell_var =  {'rowlabels':d['rowlabels'],
                 'collabels':d['collabels'],
                 'coredata':var,
                 'nheaders4rows':d['nheaders4rows'],
                 'key4rows':d['key4rows'],
                 'rowkeytype':d['rowkeytype'],
                 'nheaders4cols':d['nheaders4cols'],
                 'key4cols':d['key4cols'],
                 'colkeytype':d['colkeytype'],
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num']
                 }

    # Cell fit
    cell_fit =  {'rowlabels':d['rowlabels'],
                 'collabels':d['collabels'],
                 'coredata':fit,
                 'nheaders4rows':d['nheaders4rows'],
                 'key4rows':d['key4rows'],
                 'rowkeytype':d['rowkeytype'],
                 'nheaders4cols':d['nheaders4cols'],
                 'key4cols':d['key4cols'],
                 'colkeytype':d['colkeytype'],
                 'nanval':nanval,
                 'validchars':['All', ['All'], 'Num']
                 }

    # Facet 0 standard error (R_se)
    fac0_se = {'rowlabels':R_rowlabels,
               'collabels':np.append(R_corner[np.newaxis, :], np.array([['SE']]), axis=1),
               'coredata':R_se,
               'nheaders4rows':d['nheaders4rows'],
               'key4rows':d['key4rows'],
               'rowkeytype':d['rowkeytype'],
               'nheaders4cols':1,
               'key4cols':0,
               'colkeytype':'S60',
               'nanval':nanval,
               'validchars':['All', ['All'], 'Num']
               }

    # Facet 1 standard error (C_se)
    fac1_se = {'rowlabels':C_rowlabels,
               'collabels':np.append(C_corner[np.newaxis, :], np.array([['SE']]), axis=1),
               'coredata':np.transpose(C_se),
               'nheaders4rows':d['nheaders4cols'],
               'key4rows':d['key4cols'],
               'rowkeytype':d['colkeytype'],
               'nheaders4cols':1,
               'key4cols':0,
               'colkeytype':'S60',
               'nanval':nanval,
               'validchars':['All', ['All'], 'Num']
               }

    # Facet 0 infit
    fac0_infit = {'rowlabels':R_rowlabels,
                  'collabels':np.append(R_corner[np.newaxis, :], np.array([['Infit']]), axis=1),
                  'coredata':R_infit,
                  'nheaders4rows':d['nheaders4rows'],
                  'key4rows':d['key4rows'],
                  'rowkeytype':d['rowkeytype'],
                  'nheaders4cols':1,
                  'key4cols':0,
                  'colkeytype':'S60',
                  'nanval':nanval,
                  'validchars':['All', ['All'], 'Num']
                  }

    # Facet 1 infit
    fac1_infit = {'rowlabels':C_rowlabels,
                  'collabels':np.append(C_corner[np.newaxis, :], np.array([['Infit']]), axis=1),
                  'coredata':np.transpose(C_infit),
                  'nheaders4rows':d['nheaders4cols'],
                  'key4rows':d['key4cols'],
                  'rowkeytype':d['colkeytype'],
                  'nheaders4cols':1,
                  'key4cols':0,
                  'colkeytype':'S60',
                  'nanval':nanval,
                  'validchars':['All', ['All'], 'Num']
                  }

    # Facet 0 outfit
    fac0_outfit = {'rowlabels':R_rowlabels,
                   'collabels':np.append(R_corner[np.newaxis, :], np.array([['Outfit']]), axis=1),
                   'coredata':R_outfit,
                   'nheaders4rows':d['nheaders4rows'],
                   'key4rows':d['key4rows'],
                   'rowkeytype':d['rowkeytype'],
                   'nheaders4cols':1,
                   'key4cols':0,
                   'colkeytype':'S60',
                   'nanval':nanval,
                   'validchars':['All', ['All'], 'Num']
                   }

    # Facet 1 infit
    fac1_outfit = {'rowlabels':C_rowlabels,
                   'collabels':np.append(C_corner[np.newaxis, :], np.array([['Outfit']]), axis=1),
                   'coredata':np.transpose(C_outfit),
                   'nheaders4rows':d['nheaders4cols'],
                   'key4rows':d['key4cols'],
                   'rowkeytype':d['colkeytype'],
                   'nheaders4cols':1,
                   'key4cols':0,
                   'colkeytype':'S60',
                   'nanval':nanval,
                   'validchars':['All', ['All'], 'Num']
                   }

    # Summary reliability table
    reliability = {'rowlabels':np.array([['Facet'], [labels['row_ents']], [labels['col_ents']]]),
                   'collabels':np.array([['Facet', 'Separation', 'Reliability']]),
                   'coredata':rel_tab,
                   'nheaders4rows':1,
                   'key4rows':0,
                   'rowkeytype':'S60',
                   'nheaders4cols':1,
                   'key4cols':0,
                   'colkeytype':'S60',
                   'nanval':nanval,
                   'validchars':['All', ['All'], 'Num'],
                   'dtype':d['dtype']
                   }

    reliability = dmn.core.Damon(reliability,
                                 'datadict',
                                 'RCD_dicts_whole',
                                 verbose = None)

    
    # Row entity summary statistics
    row_ents = dmn.core.Damon(fac0coord, 'datadict', verbose=None)
    
    for stat in [fac0_se, fac0_infit, fac0_outfit]:
        row_ents.merge(stat, targ_labels=None)
        row_ents = dmn.core.Damon(row_ents.merge_out, 'datadict', verbose=None)

    row_ents = dmn.core.Damon(row_ents.data_out,
                              format_ = 'datadict_whole',
                              workformat = 'RCD_dicts_whole',
                              nheaders4rows = d['nheaders4rows'],
                              key4rows = d['key4rows'],
                              rowkeytype = d['rowkeytype'],
                              nheaders4cols = 1,
                              key4cols = 0,
                              colkeytype = 'S60',
                              nanval = nanval,
                              validchars = ['All', ['All'], 'Num'],
                              dtype = d['dtype'],
                              verbose=None)

    # Col entity summary statistics
    col_ents = dmn.core.Damon(fac1coord, 'datadict', verbose=None)
    
    for stat in [fac1_se, fac1_infit, fac1_outfit]:
        col_ents.merge(stat, targ_labels=None)
        col_ents = dmn.core.Damon(col_ents.merge_out, 'datadict', verbose=None)

    col_ents = dmn.core.Damon(col_ents.data_out,
                              format_ = 'datadict_whole',
                              workformat = 'RCD_dicts_whole',
                              nheaders4rows = d['nheaders4cols'],
                              key4rows = d['key4cols'],
                              rowkeytype = d['colkeytype'],
                              nheaders4cols = 1,
                              key4cols = 0,
                              colkeytype = 'S60',
                              nanval = nanval,
                              validchars = ['All', ['All'], 'Num'],
                              dtype = d['dtype'],
                              verbose=None)

    # Summary statistic output
    summstat = {'row_ents':row_ents,
                'col_ents':col_ents,
                'reliability':reliability
                }

    # Output labels
    out = {'fac0coord':fac0coord,
           'fac1coord':fac1coord,
           'observed':observed,
           'estimates':estimates,
           'residuals':residuals,
           'cell_var':cell_var,
           'cell_fit':cell_fit,
           'fac0_se':fac0_se,
           'fac1_se':fac1_se,
           'fac0_infit':fac0_infit,
           'fac1_infit':fac1_infit,
           'fac0_outfit':fac0_outfit,
           'fac1_outfit':fac1_outfit,
           'reliability':reliability,
           'summstat':summstat
           }

    return out





######################################################################

def _bestdim(_locals):
    "Get best dimensionality for the coord() method."

    # Get self
    self = _locals['self']
#    pytables = self.pytables
#    fileh = self.fileh

    # Extract the correct data to analyze
    try:
        data = self.standardize_out['coredata']
        nanval = self.standardize_out['nanval']
    except AttributeError:
        try:
            data = self.parse_out['coredata']
            nanval = self.parse_out['nanval']
        except AttributeError:
            try:
                data = self.subscale_out['coredata']
                nanval = self.subscale_out['nanval']
            except AttributeError:
                try:
                    data = self.score_mc_out['coredata']
                    nanval = self.score_mc_out['nanval']
                except AttributeError:
                    try:
                        data = self.extract_valid_out['coredata']
                        nanval = self.extract_valid_out['nanval']
                    except AttributeError:
                        try:
                            data = self.merge_info_out['coredata']
                            data = self.merge_info_out['nanval']
                        except AttributeError:
                            try:
                                data = self.data_out['coredata']
                                nanval = self.data_out['nanval']
                            except:
                                exc = 'Unable to find data to analyze.\n'
                                raise best_dim_in_coord_Error(exc)

    # Used to be optional, now mandatory
    PsMsMeth = True
    nondegen = False

    # Some variables
    nrows = np.size(data,axis=0)
    ncols = np.size(data,axis=1)

    # Get dimensionality specifications
    ndim = _locals['ndim']

    # Set _bestseed() parameters for finding best dimensionality
    if (_locals['seed'] == 'Auto'
        or 'homogenize' in ndim
        ):
        DimSeed = 'Auto4BestDim'
        SeedRStat = 'Stab'

    elif (isinstance(_locals['seed'],dict)
          and isinstance(_locals['seed']['MaxIt'],list)
          ):
        DimSeed = _locals['seed']
        DimSeed['MaxIt'] = DimSeed['MaxIt'][0]
        SeedRStat = DimSeed['Stats']
        if len(SeedRStat) == 1:
            SeedRStat = SeedRStat[0]
        else:
            SeedRStat = 'Stab'

    elif isinstance(_locals['seed'],dict):
        DimSeed = _locals['seed']
        SeedRStat = DimSeed['Stats']
        if len(SeedRStat) == 1:
            SeedRStat = SeedRStat[0]
        else:
            SeedRStat = 'Stab'
        if 'NonDegen' in SeedRStat:
            nondegen = True

    elif isinstance(_locals['seed'],int):
        DimSeed = _locals['seed'] #'Auto4BestDim_Fast'
        SeedRStat = None #'Stab'

    else:
        exc = 'Unable to figure out seed arg.\n'
        raise best_dim_in_coord_Error(exc)

    # Calculate maximum possible number of dimensions
    nInRow = [ncols - np.sum(data[i,:] == nanval) for i in range(nrows)]
    nInCol = [nrows - np.sum(data[:,i] == nanval) for i in range(ncols)]

    # Handle nInRow = 0
    if 0 in nInRow:
        Z = [i for i,val in enumerate(nInRow) if val == 0]
        exc = ('Rows {} have no valid data. Use extract_valid() to '
               'filter out sparse rows.\n').format(Z)
        raise best_dim_in_coord_Error(exc)

    # Handle nInRow = 0
    if 0 in nInCol:
        Z = [i for i,val in enumerate(nInCol) if val == 0]
        exc = ('Columns {} have no valid data. Use extract_valid() to '
               'filter out sparse columns.\n').format(Z)
        raise best_dim_in_coord_Error(exc)

    maxposdim = max([min([np.min(nInRow),np.min(nInCol)]) - 1, 1])

    # Extract list of dimensionalities
    for i in range(len(ndim)):
        if isinstance(ndim[i],list):
            Dims = ndim[i]

    # Remove dims greater than max possible dim or less than 0
    dim_zero = False
    if 0 in Dims:
        dim_zero = True

    Dims0 = Dims[:]
    MaxDims0 = np.max(Dims0)
    for Dim in Dims0:
        if Dim > maxposdim:
            Dims.remove(Dim)
        if Dim < 0:
            Dims.remove(Dim)
        else:
            pass

    if MaxDims0 > maxposdim:
        print 'Warning in _bestdim():  Not enough data to report stats for all your specified dimensions.'
        print 'Highest specified dimensionality: ', MaxDims0
        print 'Max possible dimensionality given available data: ', maxposdim
        print 'min in row=', np.min(nInRow)
        print 'min in col=', np.min(nInCol)
        print '\n'
        

    ######################
    ##  Build data for  ##
    ##   Running Dims   ##
    ######################

    # data if homogenize is specified
    if 'homogenize' in ndim:
        if self.verbose is True:
            print 'Homogenizing data...'

        h = _locals['homogenize']
        h_args = {'arr':data,'facet':1,'form':'Cov','max_':500,'nanval':nanval}
        if h is not None:
            for key in h.keys():
                if key == 'Facet':
                    h_args['facet'] = h[key]
                if key == 'Form':
                    h_args['form'] = h[key]
                if key == 'Max':
                    h_args['max_'] = h[key]

        Data2 = tools.homogenize(**h_args)

        self.homogenized = Data2

        if self.verbose is True:
            print 'Data has been homogenized.\n'

    else:
        Data2 = data

    # Get rand_nan
    Size = np.size(Data2)
    if Size > 10000:
        rand_nan = 1000. / float(Size)
    else:
        rand_nan = 0.10

    # Handle hd5 format
    if isinstance(Data2,np.ndarray):
        format_ = 'array'
        PyTab = None
    else:
        format_ = 'hd5'
        PyTab = '_bestdim.hd5'

    # format_ data as new Damon object
    DimObj = dmn.core.Damon(data = Data2,    # [<array, file, [file list], datadict, Damon, hd5 file>  => data in format specified by format_=]
                         format_ = format_,    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','Damon','hd5'>]
                         workformat = 'RCD',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                         validchars = None,   # [<None, ['All',[valid chars]], or ['Cols', {'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'], 'ID4':['0 -- '],...}]>]
                         nheaders4rows = 0,  # [N columns to hold row labels]
                         nheaders4cols = 0,  # [N rows to hold column labels]
                         nanval = nanval,    # [Value to which non-numeric/invalid characters should be converted.]
                         pytables = PyTab,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                         verbose = None,    # [<None, True> => report method calls]
                         )

    # Add pseudo-missing indices
    if 'homogenize' in ndim:
        if PsMsMeth is True:
            try:
                PsSeed = self.pseudomiss_out['seed']
            except AttributeError:
                PsSeed = 1

            DimObj.pseudomiss(rand_range = 'All',     # [None; 'All'; [<'Rows','Cols'>,['ID1','ID2',...]] ]
                              rand_nan = rand_nan,     # [None, proportion to make missing]
                              seed = PsSeed,  # [<None => any random selection; int => integer of "seed" random coordinates>]
                              )
    else:
        try:
            DimObj.pseudomiss_out = self.pseudomiss_out
        except AttributeError:
            DimObj.pseudomiss(rand_range = 'All',
                              rand_nan = rand_nan,
                              seed = 1
                              )

    # Get stats
    stats_ = ['Acc','Stab','Obj','Speed','Err','NonDegen']
    stats = []
    for stat in stats_:
        if stat in ndim:
            stats.append(stat)

        # best dim criterion
        if 'Obj' in stats:
            best_crit = 'Obj'
        elif 'Acc' in stats:
            best_crit = 'Acc'
        elif 'Stab' in stats:
            best_crit = 'Stab'
        elif 'Err' in stats:
            best_crit = 'Err'
        elif 'Speed' in stats:
            best_crit = 'Speed'
        elif 'NonDegen' in stats:
            best_crit = 'NonDegen'

    if len(stats) == 0:
        stats = ['Acc','Stab','Obj','Speed','Err']
        best_crit = 'Obj'

    # Get objperdim collabels
    collabels = stats[:]
    collabels.insert(0,'Dim')
    collabels = np.array(collabels)[np.newaxis,:]

    #################
    ##  Functions  ##
    #################

    def format_objperdim(collabels, objperdim):
        "Format objperdim as Damon object"

        rowlabels = np.arange(np.size(objperdim, axis=0) + 1)[:, np.newaxis]
        #rowlabels[0, 0] = 'id'
        collabels = np.append(np.array([[0]]), collabels, axis=1)
        datadict = {'rowlabels':rowlabels, 
                    'collabels':collabels,
                    'coredata':objperdim
                    }

        opd_obj = dmn.core.Damon(datadict,
                                format_ = 'datadict_whole',
                                workformat = 'RCD_dicts_whole',
                                validchars = ['All', ['All'], 'Num'],
                                nheaders4rows = 1,
                                key4rows = 0,
                                rowkeytype = 'S3',
                                nheaders4cols = 1,
                                key4cols = 0,
                                colkeytype = 'S60',
                                verbose=None,
                                dtype = ['S60', 3, '']
                                )
        return opd_obj

    def get_bestdim(objperdim,best_crit):
        "Get best int dimensionality"

        dims = objperdim.core_col['Dim']
        if 0 not in dims:
            crit_col = objperdim.core_col[best_crit]
            loc = np.where(crit_col == np.amax(crit_col))[0][0]
            bestdim = int(objperdim.core_col['Dim'][loc])
            objectivity = objperdim.core_col[best_crit][loc]
        else:
            crit_col0 = objperdim.core_col['Err']
            crit_col1 = objperdim.core_col[best_crit]
            
            loc_err = np.where(crit_col0 == np.amin(crit_col0))[0][0]
            bdim_err = int(objperdim.core_col['Dim'][loc_err])
            
            loc = np.where(crit_col1 == np.amax(crit_col1))[0][0]
            bdim_crit = int(objperdim.core_col['Dim'][loc])
            
            if bdim_err == 0:
                bestdim = 0
                objectivity = crit_col0[loc_err]
            else:
                bestdim = bdim_crit
                objectivity = crit_col1[loc]


#        if 0 in dims:
#            best_crit = 'Err'
#
#        crit_col = objperdim.core_col[best_crit]
#        if best_crit == 'Err':
#            loc = np.where(crit_col == np.amin(crit_col))[0][0]
#        else:
#            loc = np.where(crit_col == np.amax(crit_col))[0][0]
#
#        bestdim = int(objperdim.core_col['Dim'][loc])
#        objectivity = objperdim.core_col[best_crit][loc]

        return {'bestdim':bestdim,'objectivity':objectivity}


    def print_array(objperdim):
        shape = np.shape(objperdim.coredata)
        nrows = shape[0]
        ncols = shape[1]

        cl = objperdim.collabels[0][1:]
        core = objperdim.coredata
        print   # Insert blank line

        for c in range(ncols):
            white = '\n' if c == ncols - 1 else '\t'
            print cl[c], white,

        for r in range(nrows):
            for c in range(ncols):
                white = '\n' if c == ncols - 1 else '\t'
                print int(core[r,c]) if c == 0 else round(core[r,c],3), white,

        return None


    ###################
    ##  Brute Force  ##
    ##    method     ##
    ###################

    # Get coord_args
    coord_args = {'ndim':[[1]],                     # Overwritten below
                  'runspecs':_locals['runspecs'],
                  'seed':DimSeed,
                  'homogenize':None,
                  'anchors':None,
                  'quickancs':None,
                  'startercoord':None,
                  'pseudomiss':True,
                  'miss_meth':_locals['miss_meth'],
                  'solve_meth':_locals['solve_meth'],
                  'solve_meth_specs':_locals['solve_meth_specs'],
                  'condcoord_':_locals['condcoord_'],
                  'weightcoord':_locals['weightcoord'],
                  'jolt_':_locals['jolt_'],
                  'feather':_locals['feather']
                  }

    if self.verbose is True:
        print 'Getting best dimensionality...'

    if 'search' not in ndim:

        # Run coord() for each dimension
        dim_stats = np.zeros((len(Dims),np.size(collabels,axis=1)))
        for i,Dim in enumerate(Dims):
            coord_args['ndim'] = [[Dim]]

            if not isinstance(DimSeed,int):

                # Clear DimObj.seed for this dimensionality
                try:
                    del DimObj.seed
                except AttributeError:
                    pass

                DimObj.coord(**coord_args)

            if self.verbose is True:
                sys.stdout.write(str(Dim)+'..')

            stats_out = tools.stats_per_dim(DimObj,stats,Dim,coord_args,SeedRStat,nanval)
            dim_stats[i,0] = Dim
            for j,stat in enumerate(stats):
                dim_stats[i,j+1] = stats_out[stat]

        objperdim = format_objperdim(collabels, dim_stats)
        out = get_bestdim(objperdim,best_crit)
        bestdim = out['bestdim']
        objectivity = out['objectivity']


    ######################
    ##  Binary 'search' ##
    ##     method       ##
    ######################

    elif 'search' in ndim:

        # Initialize variables
        #Stop = False
        it = 0
        MinDim = np.min(Dims)
        MaxDim = np.max(Dims)
        range_ = MaxDim - MinDim
        DimCutRange = [#int(np.ceil(0.25 * range_)),
                       int(np.ceil(0.50 * range_)),
                       #int(np.ceil(0.75 * range_))
                       ]
        DimCut = DimCutRange[0]
        DimStep = 1
        slope = 1
        SlopeCut = 0.0
        StopWhenChange = 1
        MaxIteration = 10
        crit_col = np.where(collabels == best_crit)[1][0]
        objperdim = np.zeros((0,np.size(collabels)))

        # Get best dimensionality with binary search using three different starting DimCuts
        for DimCut in DimCutRange:
            Stop = False
            LoCut = MinDim
            HiCut = MaxDim

            while Stop is False:
                PrevDimCut = np.copy(DimCut)

                # Calc Dim1 error on first iteration to set error range
                if it == 0:
                    DimPair = [1,DimCut,DimCut + DimStep]
                else:
                    DimPair = [DimCut,DimCut + DimStep]

                # Calc error for pair of near-adjacent dims
                dim_stats = np.zeros((len(DimPair),np.size(collabels,axis=1)))

                for i,Dim in enumerate(DimPair):
                    Dim = int(Dim)
                    coord_args['ndim'] = [[Dim]]

                    if self.verbose is True:
                        sys.stdout.write(str(Dim)+'..')

                    if not isinstance(DimSeed,int):

                        # Clear DimObj.seed for this dimensionality
                        try:
                            del DimObj.seed
                        except AttributeError:
                            pass

                        DimObj.coord(**coord_args)

                    stats_out = tools.stats_per_dim(DimObj, stats, Dim,
                                                    coord_args, SeedRStat,
                                                    nanval)
                    dim_stats[i,0] = Dim

                    for j,stat in enumerate(stats):
                        dim_stats[i,j+1] = stats_out[stat]

                objperdim = np.append(objperdim, dim_stats, axis=0)
                base_err = np.abs(float(objperdim[0,crit_col]))
                left_err = objperdim[-2,crit_col]
                right_err = objperdim[-1,crit_col]
                slope = (left_err - right_err) / base_err

                if best_crit == 'Err':
                    slope = -1 * slope

                if slope >= SlopeCut:
                    HiCut = DimCut
                    DimCut = DimCut - (DimCut - LoCut) / 2.
                else:
                    LoCut = DimCut
                    DimCut = DimCut + (HiCut - DimCut) / 2.

                DimCut = np.clip(int(np.ceil(DimCut)),1,np.inf)
                LoCut = np.clip(int(np.ceil(LoCut)),1,np.inf)
                HiCut = np.clip(int(np.ceil(HiCut)),1,np.inf)

                # Change in DimCuts
                Change = abs(DimCut - PrevDimCut)

                # Stopping condition
                if (Change <= StopWhenChange
                    or it >= MaxIteration
                    ):
                    Stop = True

                it += 1

        objperdim = format_objperdim(collabels, objperdim)
        out = get_bestdim(objperdim,best_crit)
        bestdim = out['bestdim']
        objectivity = out['objectivity']

    #################
    ##    Build    ##
    ##   Reports   ##
    #################

    # Close bestdim hd5 file
    try:
        DimObj.fileh.close()
    except AttributeError:
        pass

    # Print reports
    if self.verbose is True:
        print_array(objperdim)
        print 'Best Dimensionality = ',bestdim,'\n'

    # Add attributes to self, no need for function outputs
    self.objperdim = objperdim
    self.bestdim = bestdim
    self.maxposdim = maxposdim
    self.objectivity = objectivity

    return None




######################################################################

def _bestseed(_locals):
    "Supports the coord() method."

    # Get self
    self = _locals['self']
#    pytables = self.pytables
    seed = _locals['seed']

    # anchors don't need seeds
    if (_locals['anchors'] is not None
        or _locals['quickancs'] is not None
        or _locals['startercoord'] is not None
        ):
        return None

    # Extract the correct data to analyze
    try:
        data = self.standardize_out
        nanval = self.standardize_out['nanval']
    except AttributeError:
        try:
            data = self.parse_out
            nanval = self.parse_out['nanval']
        except AttributeError:
            try:
                data = self.score_mc_out
                nanval = self.score_mc_out['nanval']
            except AttributeError:
                try:
                    data = self.extract_valid_out
                    nanval = self.extract_valid_out['nanval']
                except AttributeError:
                    try:
                        data = self.merge_info_out
                        nanval = self.merge_info_out['nanval']
                    except AttributeError:
                        try:
                            data = self.data_out
                            nanval = self.data_out['nanval']
                        except:
                            exc = 'Error in coord()/seed(): Unable to find data to analyze.\n'
                            raise seed_in_coord_Error(exc)

    # Handle hd5 format
    if isinstance(data['coredata'],np.ndarray):
        format_ = 'datadict'
        PyTab = None
    else:
        format_ = 'hd5'
        PyTab = '_seed.hd5'

    # Create new Damon object
    D = dmn.core.Damon(data,format_,pytables=PyTab,verbose=None)

    # Remove any existing seed bank
    try:
        os.remove(self.path+'_seedBank.pkl')
    except OSError:
        pass

    # Get dimensionality
    try:
        ndim = int(self.bestdim)
    except AttributeError:
        ndim = int(_locals['ndim'][0][0])

    # Variables
    nrows = np.size(D.coredata,axis=0) + D.nheaders4cols
    ncols = np.size(D.coredata,axis=1) + D.nheaders4rows

    # Calculate maximum possible number of dimensions
    try:
        SelfMaxPosDim = self.maxposdim
        maxposdim = np.floor((SelfMaxPosDim + 1) / 2.) - 1
    except AttributeError:
        nRows1 = nrows - D.nheaders4cols
        nCols1 = ncols - D.nheaders4rows
        nInRow = [nCols1 - np.sum(D.coredata[i,:] == nanval) for i in range(nRows1)]
        nInCol = [nRows1 - np.sum(D.coredata[:,i] == nanval) for i in range(nCols1)]
        maxposdim = np.floor(min([np.min(nInRow),np.min(nInCol)]) / 2.) - 1

    Warn1 = ("Warning in _bestseed(): Dataset is too small relative to the "
             "number of dimensions to calculate a seed stability stat. "
             "Setting seed = 1.\n")

    if ndim > maxposdim:

        #print Warn1

        # Close hd5 files
        try:
            D.fileh.close()
        except AttributeError:
            pass

##        self.seed = {'BestSeed':1,
##                     'R':None,
##                     'MinR':None,
##                     'Attempts':None,
##                     'RPerSeed':None,
##                     'StabDict':None,
##                     'AccDict':None,
##                     'ObjDict':None,
##                     'ErrDict':None,
##                     'StatsPerSeed':None
##                     }
        return None


    ##########################################
    # Define function to format objperseed array
    def format_objperseed(objperseed):
        "Format objperseed as Damon object"

        return dmn.core.Damon(objperseed,'array','RCD_dicts_whole',
                          nheaders4rows=0,key4rows=None,rowkeytype=int,
                          nheaders4cols=1,key4cols=0,colkeytype='S60',
                          validchars=['All',['All'],'Num'],verbose=None
                          )

    # Define function to print arrays
    def print_array(objperseed):
        "Print objperseed array"

        shape = np.shape(objperseed.coredata)
        nrows = shape[0]
        ncols = shape[1]

        cl = objperseed.collabels[0][1:]
        core = objperseed.coredata

        for c in range(ncols):
            white = '\n' if c == ncols - 1 else '\t'
            print cl[c],white,

        for r in range(nrows):
            for c in range(ncols):
                white = '\n' if c == ncols - 1 else '\t'
                print int(core[r,c]) if c == 0 else round(core[r,c],3),white,

        return None


    ###########################################
    # Get seed variables, setting default first
    Cols = range(ncols)
    MinR = 0.80
    MaxIt = 3
    Facet = 1
    Stats = ['Acc','Stab','Obj','Speed','Err']
    G1 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows::2]}
    G2 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows+1::2]}

    if isinstance(seed,dict):
        s_keys = seed.keys()
        MinR = seed['MinR'] if 'MinR' in s_keys else MinR
        Facet = seed['Facet'] if 'Facet' in s_keys else Facet
        Stats = seed['Stats'] if 'Stats' in s_keys else Stats
        G1 = seed['Group1'] if 'Group1' in s_keys else G1
        G2 = seed['Group2'] if 'Group2' in s_keys else G2

        if 'MaxIt' in s_keys:
            if isinstance(seed['MaxIt'],list):
                MaxIt = seed['MaxIt'][1]    # MaxIt[1] because _bestdim() sends an integer here, not list
            elif isinstance(seed['MaxIt'],int):
                MaxIt = seed['MaxIt']

    elif seed == 'Auto':
        Cols = range(ncols)
        MinR = 0.80
        MaxIt = 3
        Facet = 1
        #Stats = ['Acc','Stab','Obj','Err','Speed']
        G1 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows::2]}
        G2 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows+1::2]}

    elif seed == 'Auto4BestDim':
        Cols = range(ncols)
        MinR = 0.80
        MaxIt = 3
        Facet = 1
        #Stats = ['Acc','Stab','Obj','Err','Speed']
        G1 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows::2]}
        G2 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows+1::2]}

    elif seed == 'Auto4BestDim_Fast':
        Cols = range(ncols)
        MinR = 0.80
        MaxIt = 1
        Facet = 1
        #Stats = ['Acc','Stab','Obj','Err','Speed']
        G1 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows::2]}
        G2 = {'Get':'NoneExcept','Labels':'index','Entities':Cols[D.nheaders4rows+1::2]}

    else:
        exc = 'Unable to figure out seed parameter.\n'
        raise seed_in_coord_Error(exc)


    ###############
    ##  Iterate  ##
    ###############

    # Initialize
    StabDict = {}
    AccDict = {}
    ObjDict = {}
    PsMsDict = {}
    Stop = False
    It = 0
    Floor = 0.000001
    Ceiling = 0.999999

    # Get objperseed collabels
    collabels = Stats[:]
    collabels.insert(0,'Seed')
    collabels = np.array(collabels)[np.newaxis,:]
    objperseed = np.zeros((0,np.size(collabels,axis=1)))

    if ('Obj' in Stats
        or ('Stab' in Stats
            and ('Acc' in Stats
                 or 'PsMsResid' in Stats
                 or 'NonDegen' in Stats
                 )
             )
        ):
        acc_in_stab = ['Accuracy','Err']

        if 'NonDegen' in Stats:
            acc_in_stab.append('NonDegen')
    else:
        acc_in_stab = None

    # Run different coordinate seeds until Corr > R
    while Stop is False:

        coord_args = {'ndim':[[ndim]],
                      'runspecs':_locals['runspecs'],
                      'seed':It + 1,
                      'miss_meth':_locals['miss_meth'],
                      'solve_meth':_locals['solve_meth'],
                      'solve_meth_specs':_locals['solve_meth_specs'],
                      'condcoord_':_locals['condcoord_'],
                      'weightcoord':_locals['weightcoord'],
                      'feather':_locals['feather'],
                      'pseudomiss':True if 'Acc' in Stats else None
                      }

        # Delegate to stability() function
        if ('Stab' in Stats
            or 'Obj' in Stats
            ):
            stab_out = tools.stability(D,
                                       coord_args = coord_args,
                                       stats = acc_in_stab,
                                       facet = Facet,
                                       group1 = G1,
                                       group2 = G2,
                                       nanval = nanval,
                                       verbose = self.verbose
                                       )

            Stab = np.clip(stab_out['Stability'],Floor,Ceiling) if stab_out is not None else np.nan
            StabDict[It + 1] = Stab
            StabDict[It + 1] = Stab

            if len(Stats) == 1:
                R = Stab
                RDict = StabDict

        else:
            Stab = np.nan
            StabDict = None

        # Extract accuracy stats from stability() outputs
        if (acc_in_stab is not None
            and stab_out is not None
            ):
            Acc = np.clip(stab_out['Accuracy'],Floor,Ceiling)
            AccDict[It + 1] = Acc

            psmsresid = stab_out['Err']
            PsMsDict[It + 1] = psmsresid

            NonDegen_ = stab_out['NonDegen']

            Speed = stab_out['Speed']

        # Delegate to _accuracy() function
        elif ('Acc' in Stats
              or 'PsMsResid' in Stats
              or 'NonDegen'
              or 'Obj' in Stats
              ):
            nondegen = True if 'NonDegen' in Stats else None
            acc_out = tools.accuracy(D,coord_args,nondegen,nanval)

            if ('Acc' in Stats
                or 'Obj' in Stats
                ):
                Acc = np.clip(acc_out['Accuracy'],Floor, Ceiling)
                AccDict[It + 1] = Acc
                if len(Stats) == 1:
                    R = Acc
                    RDict = AccDict
            else:
                Acc = np.nan
                AccDict = None

            if 'Err' in Stats:
                psmsresid = acc_out['Err']
                PsMsDict[It + 1] = psmsresid
            else:
                psmsresid = np.nan
                PsMsDict = None

            if 'NonDegen' in Stats:
                NonDegen_ = acc_out['NonDegen']
            else:
                NonDegen_ = np.nan

            if 'Speed' in Stats:
                Speed = acc_out['Speed']
            else:
                Speed = np.nan
        else:
            Acc = np.nan
            AccDict = None

        if (('Stab' in Stats
             and 'Acc' in Stats)
            or 'Obj' in Stats
            ):
            stab_acc = np.log(np.array([Stab,Acc])[~np.isnan(np.array([Stab,Acc]))])
            Obj = np.exp(np.mean(stab_acc))
            ObjDict[It + 1] = Obj
            R = Obj
            RDict = ObjDict
        else:
            Obj = np.nan
            ObjDict = None

        # Collect stats
        stat_dict = {'Stab':Stab,'Acc':Acc,'Obj':Obj,'Err':psmsresid,'NonDegen':NonDegen_,'Speed':Speed}
        seed_stats = np.zeros((1,np.size(collabels,axis=1)))
        seed_stats[0,0] = It + 1

        for i,stat in enumerate(Stats):
            seed_stats[0,i + 1] = stat_dict[stat]

        objperseed = np.append(objperseed,seed_stats,axis=0)

        # Evaluate stop
        It += 1

        if (R > MinR
            or It == MaxIt
            ):
            Stop = True

    # Print objperseed array
    objperseed = format_objperseed(np.append(collabels,objperseed,axis=0))

    if self.verbose is True:
        print_array(objperseed)

    # Close pytables file
    try:
        D.fileh.close()
    except AttributeError:
        pass


    ############
    ##  Best  ##
    ##  seed  ##
    ############

    SeedRs = np.array(RDict.items())
    SeedRs = np.where(np.logical_or(np.isnan(SeedRs),np.isinf(SeedRs)),nanval,SeedRs)
    ValLoc = np.where(SeedRs[:,1] != nanval)[0]

    try:
        BestR = np.amax(SeedRs[:,1][ValLoc])
    except ValueError:
        exc = 'Unable to calculate valid seed statistics for some reason.\n'
        raise seed_in_coord_Error(exc)

    BestSeed = int(SeedRs[:,0][np.where(SeedRs[:,1] == BestR)][0])
    Attempts = np.size(SeedRs,axis=0)

    if self.verbose is True:
        print 'Best coordinate seed is',BestSeed,', out of',Attempts,'attempts.\n'

        if BestR < MinR:
            print "Warning in coord()/seed(): Unable to find starting coordinates that meet your 'seed' requirements.  It is possible the dataset cannot yield the desired objectivity.\n"

    self.seed = {'BestSeed':BestSeed,
                 'R':round(BestR,3),
                 'MinR':MinR,
                 'Attempts':Attempts,
                 'RPerSeed':RDict,
                 'StabDict':StabDict,
                 'AccDict':AccDict,
                 'ObjDict':ObjDict,
                 'ErrDict':PsMsDict,
                 'StatsPerSeed':objperseed
                 }


    return None





######################################################################

def _coord(_locals):
    "Basis of the coord() method."

    ########################
    ##  Prepare Variables ##
    ########################

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Extract the correct data to analyze
    try:
        datadict = self.standardize_out
    except AttributeError:
        try:
            datadict = self.parse_out
        except AttributeError:
            try:
                datadict = self.subscale_out
            except AttributeError:
                try:
                    datadict = self.score_mc_out
                except AttributeError:
                    try:
                        datadict = self.extract_valid_out
                    except AttributeError:
                        try:
                            datadict = self.merge_info_out
                        except AttributeError:
                            try:
                                datadict = self.data_out
                            except:
                                exc = 'Error in coord(): Unable to find data to analyze.\n'
                                raise coord_Error(exc)

    # Get dimensionality
    if _locals['ndim'] is not None:
        try:
            ndim = int(self.bestdim)
        except AttributeError:
            ndim = int(_locals['ndim'][0][0])

    elif (_locals['ndim'] is None
          and (_locals['anchors'] is None
               and _locals['quickancs'] is None
               and _locals['startercoord'] is None
               )
          ):
        exc = 'If ndim is None, anchors or quickancs need to be filled out.\n'
        raise coord_Error(exc)
    else:
        ndim = [[1]]        # Just placeholder, not used

    # Get seed
    try:
        seed = self.seed['BestSeed']
        try:
            seed = int(seed)
        except TypeError:
            pass
    except AttributeError:
        if _locals['seed'] in ['Auto','Auto4BestDim','Auto4BestDim_Fast']:    # Only triggered if _bestseed() fails
            seed = 1
        elif isinstance(_locals['seed'],dict):
            seed = 1
        else:
            seed = _locals['seed']

    # Get _locals
    runspecs = _locals['runspecs']
    StopWhenChange = _locals['runspecs'][0]
    MaxIteration = _locals['runspecs'][1]
    anchors = _locals['anchors']
    quickancs = _locals['quickancs']
    startercoord = _locals['startercoord']
    pseudomiss = _locals['pseudomiss']
    miss_meth = _locals['miss_meth']
    solve_meth = _locals['solve_meth']
    solve_meth_specs = _locals['solve_meth_specs']
    condcoord_ = _locals['condcoord']
    weightcoord = _locals['weightcoord']
    jolt_ = _locals['jolt_']
    feather = _locals['feather']

    # Define label variables (pytables can also be read in this context)
    data = datadict['coredata']
    rowlabels = datadict['rowlabels']
    collabels = datadict['collabels']
    nanval = float(datadict['nanval'])

    try:
        nRowLabels = np.size(rowlabels,axis=0)
        nColLabels = np.size(collabels,axis=1)
    except IndexError:
        exc = 'Data is not an array or a PyTable.  Check that MyObj.fileh is not closed.\n'
        raise coord_Error(exc)

    nheaders4rows = datadict['nheaders4rows']
    nheaders4cols = datadict['nheaders4cols']
    key4rows = datadict['key4rows']
    key4cols = datadict['key4cols']
    rowkeytype = datadict['rowkeytype']
    colkeytype = datadict['colkeytype']
    facs_per_ent = [1,2]

    # data array shape
    ArrayShape = np.shape(data)
    nfac0 = ArrayShape[0]
    nfac1 = ArrayShape[1]

    # Extract keys
    KeyRowLabels = tools.getkeys(datadict,'Row','Core','Auto',None)
    KeyColLabels = tools.getkeys(datadict,'Col','Core','Auto',None)

    # Deal with ndim = 0
    all_same = False
    if ndim == 0:
        ndim = 1
        all_same = True
        condcoord_ = {'Fac0':'AllSame', 'Fac1':'AllSame'}


    ##################
    ##  Deal with   ##
    ## Anchor Files ##
    ##################

    # If using quickancs, use it to define dimensionality
    if quickancs is not None:
        ndim = np.size(quickancs[1],axis=1)

    if startercoord is not None:
        ndim = np.size(startercoord[1],axis=1)

#                anchors = {'Bank':______    # no default, must be specified
#                           'Facet':1        # assumes bank stores column entities
#                           'Coord':'ent_coord'  # type of anchor
#                           'Entities':['All']   # use all possible anchors
#                           'Refresh_All':False  # A total refresh is rare
#                           }

    # Extract and define anchor values
    AncFile = AncFac = AncDim = AncList = AncFresh = None
    if anchors is not None:
        
        # Apply defaults where necessary
        ancs = {'Facet':1, 'Coord':'ent_coord', 'Entities':['All'], 
                'Refresh_All':False}

        for key in ancs:
            try:
                anchors[key]
            except KeyError:
                anchors[key] = ancs[key]

        if isinstance(anchors['Bank'],dict):
            AncFile = anchors['Bank']
        else:
            AncFile = np.load(anchors['Bank'])
        AncFac = anchors['Facet']
        AncList = anchors['Entities']
        AncFresh = anchors['Refresh_All']
        AncFresh = False if AncFresh is None else AncFresh
        AncCoordType = anchors['Coord']

        # Get dimensionality
        if (AncCoordType == 'ear_coord'
            or AncCoordType == 'se_coord'
            ):
            AncDim = 2
        else:
            AncDim = AncFile['ndim']

        # Prep variables
        if AncFac == 0:
            KeyLabels = KeyRowLabels
            KeyLabels_set = set(KeyLabels)
            AncDict = AncFile['facet0'][AncCoordType]
            F_ = 'row'
            nRows_ = nfac0
            facs_per_ent = [1,1]
        elif AncFac == 1:
            KeyLabels = KeyColLabels
            KeyLabels_set = set(KeyLabels)
            AncDict = AncFile['facet1'][AncCoordType]
            F_ = 'column'
            nRows_ = nfac1
            facs_per_ent = [0,1]
        else:
            exc = 'Unable to figure out anchors facet spec.\n'
            raise coord_Error(exc)

        AncDict_set = set(AncDict.keys())

        # count matches:  how many KeyColLabels or KeyRowLabels are in 
        # the anchor dictionary
        MatchedLabels = KeyLabels_set & AncDict_set
        NInCommon = len(MatchedLabels)
        if self.verbose is True:
            print ('Number N of matches between',F_,'keys and '
                   'bank =',NInCommon,'. If N is near 0 look for key type '
                   'discrepancies.\n')
        if NInCommon == 0:
            exc = 'Zero matches between '+F_+' keys and bank.  Check type.\n'
            print 'Error in coord(): ',exc
            print 'KeyLabels_set=\n',KeyLabels,type(list(KeyLabels_set)[0])
            print 'AncDict_set=\n',AncDict_set,type(list(AncDict_set)[0])
            raise coord_Error(exc)

        # Clarify list of desired anchors
        if AncList == ['All']:
            AncList = list(MatchedLabels)
        elif isinstance(AncList, dict):
            if not isinstance(AncList['AllExcept'], (np.ndarray, list)):
                exc = "Unable to figure out anchors 'Entities' parameter."
                raise ValueError(exc)
                
            AncList = list(MatchedLabels - set(AncList['AllExcept']))
        AncEnts = np.array(AncList)[:,np.newaxis]

        # Force ndim = AncDim
        ndim = AncDim
        if self.verbose is True:
            print 'Anchors require locating entities in a space of',ndim,'dimensions.\n'

        # Get anchor coordinates
        CoordAnc = np.zeros((nRows_, AncDim)) + nanval
        for i in xrange(np.size(KeyLabels, axis=0)):
            if KeyLabels[i] in AncEnts:
                try:
                    CoordAnc[i, :] = AncDict[KeyLabels[i]]
                except KeyError:
                    pass
        AncLoc = np.where(CoordAnc[:, 0] != nanval)[0]

        # Record index of non-NaN anchor coordinates
        AncFacDict = {}
        AncValIndexDict = {}
        if AncFac == 0:
            AncFacDict[0] = CoordAnc
            AncFacDict[1] = None
            AncValIndexDict[0] = AncLoc
            AncValIndexDict[1] = None
        elif AncFac == 1:
            AncFacDict[0] = None
            AncFacDict[1] = CoordAnc
            AncValIndexDict[0] = None
            AncValIndexDict[1] = AncLoc

    # Check that array is large enough
    if (anchors is None
        and quickancs is None
        ):
        if (nfac1 < ndim
            or nfac0 < ndim
            ):
            exc = 'Insufficient data for number of dimensions.  Try smaller ndim.\n'
            print 'Error in coord(): ',exc
            print 'nfac1 =',nfac1
            print 'nfac0 =',nfac0
            print 'ndim =',ndim if all_same is False else 0
            raise coord_Error(exc)


    ########################
    ##  Prep for Missing  ##
    ##  Cell Values, etc. ##
    ########################

    # Make a copy and clean (working array cannot be hard-wired to input array)
    data = data[:,:]

    try:
        Data0 = np.where(np.isinf(data), nanval, 
                         np.where(np.isnan(data), nanval, data))    # Makes a copy
    except:
        exc = 'Unable to read data. Make sure it is numerical.\n'
        raise coord_Error(exc)

    data = None

    # Create or get missing index, impose on data
    try:
        if pseudomiss is True:
            if self.pseudomiss_out['parsed_psmsindex'] is not None:
                psmsindex = self.pseudomiss_out['parsed_psmsindex']
            else:
                psmsindex = self.pseudomiss_out['psmsindex']
            Data0[psmsindex] = nanval
            msindex = np.where(Data0 == nanval)
        else:
            msindex = np.where(Data0 == nanval)
    except AttributeError:
        msindex = np.where(Data0 == nanval)

    nMsIndex = len(msindex[0])
    ValIndex = np.where(Data0 != nanval)

    # Exit coord() if array has no valid values
    if (np.sum(ValIndex) == 0):
        exc = 'Found no valid data values.\n'
        raise coord_Error(exc)

    # feather (add random noise) to data
    if feather is not None:
        Data0 = np.where(Data0 == nanval, nanval, 
                         Data0 + (npr.rand(nfac0, nfac1) * feather - feather / 2.))

    # Exit coord() if insufficient variation
    DataSD = np.std(Data0[ValIndex])
    if (DataSD < 0.00000000001):
        exc = 'Insufficient variation in data array.'
        print 'Error in coord(): ',exc
        print 'data standard deviation =',round(DataSD,20)
        print 'feather =',feather
        raise coord_Error(exc)

    # Fill missing cells with array mean to implement 'ImputeCells' method
    if miss_meth == 'ImputeCells':
        ArrAvg = np.mean(Data0[ValIndex])
        Data1 = np.where(Data0 == nanval, ArrAvg, Data0)

    # Index missing cells for 'IgnoreCells' method
    elif miss_meth == 'IgnoreCells':
        Data1 = Data0


    #######################
    ##   Create Row/     ##
    ##  Col Coordinates  ##
    #######################

    # Create Fac0, Fac1 starter coordinates
    Data1SD = DataSD
    facmetric = Data1SD

    # Use startercoord (user-specified starter values)
    if startercoord is not None:
        if startercoord[0] == 0:
            fac0coord = np.copy(startercoord[1])
            fac1coord = np.zeros((nfac1,ndim))
        elif startercoord[0] == 1:
            fac0coord = np.zeros((nfac0,ndim))
            fac1coord = np.copy(startercoord[1])

    # Use quickancs (user-specified quick anchors -- already sized to the data array)
    elif quickancs is not None:
        if quickancs[0] == 0:
            fac0coord = np.copy(quickancs[1])
            fac1coord = np.zeros((nfac1,ndim))
            facs_per_ent = [1,1]
        elif quickancs[0] == 1:
            fac0coord = np.zeros((nfac0,ndim))
            fac1coord = np.copy(quickancs[1])
            facs_per_ent = [0,1]

    # anchors were pulled from banks above
    elif anchors is not None:
        if AncFac == 0:
            fac0coord = np.copy(AncFacDict[0])   # Copy to preserve original bank coordinates.
            fac1coord = np.zeros((nfac1,ndim))
        elif AncFac == 1:
            fac0coord = np.zeros((nfac0,ndim))
            fac1coord = np.copy(AncFacDict[1])

    # Create random starter values
    else:
        if all_same is False:
            if seed is None:
                fac0coord = npr.rand(nfac0,ndim) * facmetric
                fac1coord = npr.rand(nfac1,ndim) * facmetric

            # Seeds specify that a particular set of randoms is always used
            else:
                fac0coord = npr.RandomState(seed=seed).rand(nfac0,ndim) * facmetric
                fac1coord = npr.RandomState(seed=seed+1).rand(nfac1,ndim) * facmetric

            if condcoord_ is not None:
                fac0coord = tools.condcoord(fac0coord, None, 'Fac0', 
                                            condcoord_['Fac0'], nanval)['F0Std']
                fac1coord = tools.condcoord(None, fac1coord, 'Fac1',
                                            condcoord_['Fac1'], nanval)['F1Std']
        else:
            fac0coord = np.ones((nfac0, 1))
            fac1coord = np.ones((nfac1, 1))

    # Convert coords to NaNVals if data does not vary or has too many missing cells
    # CHECK: ARE MASKS WORKING RIGHT?
    if miss_meth == 'IgnoreCells':
        Data1_ma = npma.masked_values(Data1,nanval)
        if nfac1 == 1:
            Fac0SD = np.ones((nfac0))
        else:
            Fac0SD = npma.std(Data1_ma,axis=1)

        if nfac0 == 1:
            Fac1SD = np.ones((nfac1))
        else:
            Fac1SD = npma.std(Data1_ma,axis=0)

        # Counts
        Fac0Count = np.sum((Data1 != nanval),axis=1)
        Fac1Count = np.sum((Data1 != nanval),axis=0)

        # Set insufficient counts/variation to nanval (counts do not matter with anchors in some cases)
        if (anchors is not None or quickancs is not None):
            fac0coord[:,:][np.where(Fac0SD == 0)] = nanval
            fac1coord[:,:][np.where(Fac1SD == 0)] = nanval
        else:
            fac0coord[:,:][np.where(np.logical_or(Fac0SD == 0,
                                                  Fac0Count < ndim))] = nanval
            fac1coord[:,:][np.where(np.logical_or(Fac1SD == 0,
                                                  Fac1Count < ndim))] = nanval

    # Convert coordinates to pytables
    if pytables is not None:
        Fac0CoordTab = tools.pytables_(fac0coord[:,:],'array',fileh,None,'coord_out',
                                      ['Fac0Coord_temp'],None,None,None,None,None)['arrays']['Fac0Coord_temp']
        fac0coord = Fac0CoordTab[:,:]

        # Convert fac1coord
        Fac1CoordTab = tools.pytables_(fac1coord[:,:],'array',fileh,None,'coord_out',
                                      ['Fac1Coord_temp'],None,None,None,None,None)['arrays']['Fac1Coord_temp']
        fac1coord = Fac1CoordTab[:,:]

    # Add arrays to facet dictionary
    FacDict = {}
    FacDict[0] = fac0coord
    FacDict[1] = fac1coord


    ###################
    ##  Create data  ##
    ##    Indices    ##
    ###################

    # Index non-missing cells for each row
    RowDatIndex = []
    RowCountIndex = []

    RowCoord = FacDict[0]
    rownan = np.zeros(np.size(RowCoord,axis=0))
    rownan[np.where(RowCoord == nanval)[0]] = nanval

    ColCoord = FacDict[1]
    colnan = np.zeros(np.size(ColCoord,axis=0))
    colnan[np.where(ColCoord == nanval)[0]] = nanval

    for i in xrange(nfac0):
        RowArr = Data0[i,:]
        RowNonMissLoc = np.where(np.logical_and(RowArr != nanval,colnan != nanval))[0]
        nRowNonMiss = len(RowNonMissLoc)
        RowDatIndex.append(RowNonMissLoc)
        RowCountIndex.append(nRowNonMiss)

    # Index non-missing cells for each column
    ColDatIndex = []
    ColCountIndex = []

    for i in xrange(nfac1):
        ColArr = Data0[:,i]
        ColNonMissLoc = np.where(np.logical_and(ColArr != nanval,rownan != nanval))[0]
        nColNonMiss = len(ColNonMissLoc)
        ColDatIndex.append(ColNonMissLoc)
        ColCountIndex.append(nColNonMiss)

    # Put missing data and count indices in dictionary
    DataIndexDict = {}
    DataIndexDict[0] = RowDatIndex
    DataIndexDict[1] = ColDatIndex

    # Counts
    CountIndexDict = {}
    CountIndexDict[0] = RowCountIndex
    CountIndexDict[1] = ColCountIndex

    # Put data in dictionary row-wise and column-wise
    datadict = {}

    if pytables is not None:
        Data1_Tab = tools.pytables_(Data1,'array',fileh,None,'coord_out',
                                  ['coredata'],None,None,None,None,None)['arrays']['coredata']
        Data1T_Tab = tools.pytables_(np.transpose(Data1),'array',fileh,None,'coord_out',
                                  ['CoreData_T'],None,None,None,None,None)['arrays']['CoreData_T']

        datadict[0] = Data1_Tab
        datadict[1] = Data1T_Tab

    else:
        datadict[0] = Data1
        datadict[1] = np.transpose(Data1)

    Data0 = None


    #####################
    ##   Calculate     ##
    ##  Coordinates    ##
    #####################

    Stop = 0
    Change = StopWhenChange + 1.
    it = 0
    FacIt = 0
    MaxFacIt = 2
    Warn1 = Warn2 = Warn3 = Warn4 = None
    IRLSMode = 0
    it_dich = 0
    Fac0_done = 0
    Fac1_done = 0
    ChangeDict = {}
    WarnDict = {}
    joltflag = False
    changelog = np.zeros((0,4))

    if 'EstConverge' in runspecs:
        RMSR = 10.0
        RMSRChange = 10.0

    # Dichotomous/IRLS stopping conditions
    if solve_meth == 'IRLS':
        StopWhenChangeDich = solve_meth_specs['runspecs'][0]
        MaxIterationDich = solve_meth_specs['runspecs'][1]

        print 'Running Iteratively Reweighted Least Squares...\n'

    if self.verbose is True:
        if 'EstConverge' in runspecs:
            print 'Dim', '\t', 'Fac', '\t', 'Iter', '\t', 'Change', '\t', 'RMSR', '\t', 'RMSR Change'
        else:
            print 'Dim', '\t', 'Fac', '\t', 'Iter', '\t', 'Change', '\t', 'jolt_'

    # Iterate between row and col coordinates until stopping condition is met
    while Stop == 0:

        #####################
        # Set order for computing facet coordinates
        if (quickancs is not None and quickancs[0] == 0):
            Facs = [1]
            MaxFacIt = 1
            MaxIteration = 1
            W_All = None
            condcoord_ = None

        elif (quickancs is not None and quickancs[0] == 1):
            Facs = [0]
            MaxFacIt = 1
            MaxIteration = 1
            W_All = None
            condcoord_ = None

        elif (anchors is not None and AncFac == 0):
            Facs = [1, 0]
            MaxFacIt = 2
            MaxIteration = 1
            W_All = None
            condcoord_ = None

        elif (anchors is not None and AncFac == 1):
            Facs = [0, 1]
            MaxFacIt = 2
            MaxIteration = 1
            W_All = None
            condcoord_ = None

        elif (startercoord is not None and startercoord[0] == 0):
            Facs = [1, 0]
            MaxFacIt = 2
            MaxIteration = MaxIteration
            W_All = weightcoord

        elif (startercoord is not None and startercoord[0] == 1):
            Facs = [0, 1]
            MaxFacIt = 2
            MaxIteration = MaxIteration
            W_All = weightcoord

        # Compute facet coordinates without anchoring
        else:
            Facs = [0, 1]
            if (isinstance(condcoord_, dict)
                and 'first' in condcoord_
                and condcoord_['first'] == 0):
                Facs = [1, 0]
            MaxFacIt = 2
            MaxIteration = MaxIteration
            W_All = weightcoord


        ######################
        ##  Cycle through   ##
        ##     Facets       ##
        ######################

        for Fac in Facs:

            # Baseline to measure change
            if (quickancs is None and anchors is None):
                PrevFacCoord = np.copy(FacDict[Fac])
            else:
                PrevFacCoord = None

            # Define opposing facet
            oppfac = list(set([0,1]) - set([Fac]))[0]
            OppFacCoord = FacDict[oppfac]

             # IRLS mode
            if (solve_meth == 'IRLS'
                and IRLSMode == 0
                ):
                Method1 = 'LstSq'
            else:
                Method1 = solve_meth

            # Prepare weights for 'LstSq'
            if W_All is not None:
                if Method1 == 'LstSq':
                    OppCount = np.array(CountIndexDict[oppfac])[:,np.newaxis]
                    logn = np.where(OppCount < 1, 0, np.log(OppCount + 1))    # Prevent LogFacN = 0
                    W_All = tools.weight_coord(OppFacCoord, logn, 'R', nanval)

            # jolt_ coordinates of opposing (basis) facet if they look degenerate
            if jolt_ is not None:
                JoltOut = tools.jolt(U = OppFacCoord,     # [ent x dims coordinates array]
                                     sigma = jolt_[0], # [(Max - Mean)/SD above which jolting is indicated]
                                     jolt_ = jolt_[1],  # [jolt_*rand() amount of randomness to add to U]
                                     joltflag = joltflag,  # [<True,False> => True if previous facet called for a jolt]
                                     condcoord_ = condcoord_,  # [<None,condcoord() args> => condition jolt noise]
                                     facet = 'Fac0' if oppfac == 0 else 'Fac1',  # [<None,'Fac0','Fac1'> => facet of U (used only by condcoord_)]
                                     nanval = nanval # [Not-a-number value]
                                     )
                OppFacCoord = JoltOut[0]
                joltflag = JoltOut[1]

            # Tells faccoord not to skip anchored nanval coords
            anchored = True if Fac == AncFac else False

            # Calc coordinates for all entities in Fac facet
            Out = tools.faccoord(targfac = [Fac,FacDict[Fac], anchored], # [ [FacetNum,FacetArray], e.g., [0,FacetArray0] => existing facet array to recalculate] ]
                                 targdatindex = DataIndexDict[Fac], # [None,targdatindex, e.g., TargDatInd0 => index of valid data per targ entity]
                                 data = datadict[Fac],   # [2-D targfac x oppfac array of data (rotates so that targfac is always rows)]
                                 oppfac = OppFacCoord, # [Ents x dims array of coordinates of opposite facet(s), or their product]
                                 oppweights = W_All,  # [array of weights corresponding to opposite facet array]
                                 solve_meth = Method1, # [method for calculating coordinates <'LstSq','IRLS'>]
                                 solve_meth_specs = solve_meth_specs,  # [None, dictionary of specs specific to method, e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],...}]
                                 condcoord_ = condcoord_,  # [None,'Std','Orthonormal','Pos_1D_Dichot',funcstep dict => {0:'Fac = f0(Fac)',1:'Fac = f1(Fac)',...}>,} ]
                                 miss_meth = miss_meth, # ['ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells entirely (preferred)]
                                 nanval = nanval, # [Not-a-Number value, to label non-numerical outputs]
                                 )

            FacCoord = Out['FacCoord']

            # Update facet dictionary
            FacDict[Fac] = FacCoord

            # Update warning dictionary
            WarnDict[Fac] = [Out['Warn1'],Out['Warn2']]

            # report on iteration
            if self.verbose is True:
                try:
                    if JoltOut[1] is True:
                        JoltMsg = 'jolt_ >>>>'
                    else:
                        JoltMsg = ''
                except NameError:
                    JoltMsg = ''

                if 'EstConverge' in runspecs:
                    print ndim, '\t', Fac, '\t', it, '\t', round(Change, 5), '\t', RMSR, '\t', RMSRChange
                else:
                    print ndim, '\t', Fac, '\t', it, '\t', round(Change, 5), '\t', JoltMsg

            # Set facet stop status
            FacIt += 1
            if FacIt >= MaxFacIt:
                FacIt = 0
                break

            # Record change for this facet
            if (quickancs is None and anchors is None):
                ChangeDict[Fac] = (np.sqrt(np.mean((FacCoord - PrevFacCoord)**2)) 
                                   / float(facmetric))


        ########################
        ##  Evaluate Results  ##
        ##   of Iteration i   ##
        ########################

        # Used for checking estimate convergence only
        if 'EstConverge' in runspecs:
            ValRows = np.where(FacDict[0][:,0] != nanval)[0]
            ValCols = np.where(FacDict[1][:,0] != nanval)[0]

            R = FacDict[0][ValRows]
            C = FacDict[1][ValCols]
            Est = np.dot(R,np.transpose(C))
            Obs = Data1[ValRows]
            Obs = np.transpose(np.transpose(Obs)[ValCols])
            Val = np.where(Obs != nanval)
            RMSRTemp = np.sqrt(np.mean((Obs[Val] - Est[Val])**2))
            RMSRChange = RMSR - RMSRTemp
            RMSR = RMSRTemp
        else:
            RMSR = np.nan
            RMSRChange = np.nan

        # Update missing cells, if necessary
        if miss_meth == 'ImputeCells':

            # Update missing cells with expected values
            for h in xrange(nMsIndex):
                i = msindex[0][h]
                j = msindex[1][h]
                if np.logical_or(FacDict[0][i,0] == nanval,FacDict[1][j,0] == nanval):
                    Data1[i,j] = nanval
                else:
                    Data1[i,j] = np.dot(FacDict[0][i,:],np.transpose(FacDict[1][j,:]))

        # Reimpose anchors unless AncFresh is True, in which case all items
        # will be refreshed.  AncFresh is False means all items not specified
        # in anchors['Entities'] will keep their refreshed values.
        if (anchors is not None
            and AncFresh is False
            ):
            ix_ = AncValIndexDict[AncFac]
            FacDict[AncFac][ix_] = AncFacDict[AncFac][ix_, :]

        # Calculate change in coordinates
        if (quickancs is None
            and anchors is None
            ):
            if StopWhenChange == 0:
                Change = 1
            else:
                Change = max(ChangeDict.values())

        # Change table
        changelog = np.append(changelog, 
                              np.array([[it, Change, RMSR, RMSRChange]]),
                              axis=0)

        # Increment iteration
        it += 1

        # Set stop flag
        if solve_meth != 'IRLS':
            if (it < MaxIteration
                and Change > StopWhenChange
                ):
                Stop = 0
            else:
                Stop = 1

        # Reset stop parameters for IRLS mode
        elif solve_meth == 'IRLS':
            if (it < MaxIteration
                and Change > StopWhenChange
                and IRLSMode == 0   # To prevent sliding back from IRLSMode = 1 to IRLSMode = 0
                ):
                IRLSMode = 0
                Stop = 0
            else:
                IRLSMode = 1
                it_dich += 1

        # Set stop flag if in IRLS mode
        if IRLSMode == 1:
            if (it_dich < MaxIterationDich
                and Change > StopWhenChangeDich
                ):
                Stop = 0
            else:
                Stop = 1
        

    ###############
    ##  Prepare  ##
    ##  outputs  ##
    ###############

    if self.verbose is True:
        print '\n'

    # Fac 0 warnings
    try:
        if WarnDict[0][0] is True:
            print "Warning: Encountered a linear algebra error calculating a column entity's coordinates.  Converting coordinates to nanval."
##        if WarnDict[0][1] is True:
##            print "Warning: A column entity's coordinates contain NaNVals."
    except KeyError:
        pass

    # Fac 1 warnings
    try:
        if WarnDict[1][0] is True:
            print "Warning: Encountered a linear algebra error calculating a column entity's coordinates.  Converting coordinates to nanval."
##        if WarnDict[1][1] is True:
##            print "Warning: A column entity's coordinates contain NaNVals."
    except KeyError:
        pass

    # Remove NaN and Inf from arrays
    # fac0coord
    F0NaNLoc = np.where(np.isnan(FacDict[0]))
    F0InfLoc = np.where(np.isinf(FacDict[0]))
    FacDict[0][F0NaNLoc] = nanval
    FacDict[0][F0InfLoc] = nanval

    # fac1coord
    F1NaNLoc = np.where(np.isnan(FacDict[1]))
    F1InfLoc = np.where(np.isinf(FacDict[1]))
    FacDict[1][F1NaNLoc] = nanval
    FacDict[1][F1InfLoc] = nanval

    # Prepare fac0coord data object
    F0RowLabels = np.append(rowlabels[key4cols,:][np.newaxis,:],rowlabels[nheaders4cols:,:],axis=0)
    F0ColLabels = np.append(rowlabels[key4cols,:][np.newaxis,:],np.array(range(1,ndim + 1),ndmin=2),axis=1)
    fac0coord = FacDict[0]

    # pytables form
    if pytables is not None:
        fac0coord = tools.pytables_(fac0coord.astype(float),'array',fileh,None,'coord_out',
                                      ['fac0coord'],None,None,None,None,None)['arrays']['fac0coord']
        F0RowLabels = tools.pytables_(F0RowLabels.astype('S60'),'array',fileh,None,'coord_out',
                                      ['F0RowLabels'],None,None,None,None,None)['arrays']['F0RowLabels']
        F0ColLabels = tools.pytables_(F0ColLabels.astype('S60'),'array',fileh,None,'coord_out',
                                      ['F0ColLabels'],None,None,None,None,None)['arrays']['F0ColLabels']

    Fac0CoordRCD = {'rowlabels':F0RowLabels, 'collabels':F0ColLabels,
                    'coredata':fac0coord, 'nheaders4rows':nheaders4rows,
                    'key4rows':key4rows, 'rowkeytype':rowkeytype,
                    'nheaders4cols':1, 'key4cols':0, 'colkeytype':int,
                    'nanval':nanval, 'validchars':['All',['All'],'Num'],
                    'opp_count':len(FacDict[1])
                    }

    # Prepare fac1coord data object
    F1RowLabels = np.transpose(np.append(collabels[:,key4rows][:,np.newaxis],collabels[:,nheaders4rows:],axis=1))
    F1ColLabels = np.append(collabels[:,key4rows][np.newaxis,:],np.array(range(1,ndim + 1),ndmin=2),axis=1)
    fac1coord = FacDict[1]

    if pytables is not None:
        fac1coord = tools.pytables_(fac1coord.astype(float),'array',fileh,None,'coord_out',
                                      ['fac1coord'],None,None,None,None,None)['arrays']['fac1coord']
        F1RowLabels = tools.pytables_(F1RowLabels.astype('S60'),'array',fileh,None,'coord_out',
                                      ['F1RowLabels'],None,None,None,None,None)['arrays']['F1RowLabels']
        F1ColLabels = tools.pytables_(F1ColLabels.astype('S60'),'array',fileh,None,'coord_out',
                                      ['F1ColLabels'],None,None,None,None,None)['arrays']['F1ColLabels']

    Fac1CoordRCD = {'rowlabels':F1RowLabels, 'collabels':F1ColLabels,
                    'coredata':fac1coord, 'nheaders4rows':nheaders4cols,
                    'key4rows':key4cols, 'rowkeytype':colkeytype,
                    'nheaders4cols':1, 'key4cols':0, 'colkeytype':int,
                    'nanval':nanval, 'validchars':['All',['All'],'Num'],
                    'opp_count':len(FacDict[0])
                    }

    return {'fac0coord':Fac0CoordRCD,
            'fac1coord':Fac1CoordRCD,
            'ndim':ndim if all_same is False else 0,
            'changelog':changelog,
            'anchors':anchors,
            'facs_per_ent':facs_per_ent
            }




######################################################################

def _sub_coord(_locals):
    "Basis of the sub_coord() method"

    # Get args
    self = _locals['self']
    subspaces = _locals['subspaces']
    coord_subs = _locals['coord_subs']
    coord_resids = _locals['coord_resids']
    unique_weights_ = _locals['unique_weights']
    share_if = _locals['share_if']
    min_rel = _locals['min_rel']
    rpt_optimal = _locals['rpt_optimal']
    verbose = self.verbose

    # Get data
    try:
        data = self.standardize_out
    except AttributeError:
        try:
            data = self.parse_out
        except AttributeError:
            try:
                data = self.subscale_out
            except AttributeError:
                try:
                    data = self.score_mc_out
                except AttributeError:
                    try:
                        data = self.extract_valid_out
                    except AttributeError:
                        try:
                            data = self.merge_info_out
                        except AttributeError:
                            try:
                                data = self.data_out
                            except AttributeError:
                                exc = 'Unable to find data to analyze.\n'
                                raise sub_coord_Error(exc)

    try:
        d = dmn.core.Damon(data, 'datadict', verbose=None)
    except Damon_Error:
        d = data

    nanval = d.nanval
    nrows = np.size(d.coredata, axis=0)

    # Add subspace information to data if necessary, define subs
    if isinstance(subspaces, list):
        infodict = subspaces[1]
        items = list(tools.getkeys(d, 'Col', 'Core', 'Auto', None))
        info = np.zeros((len(items) + 1, 2), dtype='S60')
        info[0, :] = np.array([['ItemID', 'Cluster']])
        info[1:, 0] = items
        subs = infodict.keys()

        if 'key' in subspaces:
            for i, item in enumerate(items):
                for sub in subs:
                    if item in infodict[sub]:
                        info[i + 1, 1] = sub
                    else:
                        pass
        elif 'index' in subspaces:
            for sub in subs:
                info[1:, 1][infodict[sub]] = sub

        info_obj = dmn.core.Damon(info, 'array', 'RCD_dicts_whole', None,
                                  nheaders4rows=1, key4rows=0, rowkeytype=d.colkeytype,
                                  nheaders4cols=1, key4cols=0, colkeytype='S60',
                                  verbose=None)

        d.merge_info(info_obj, 'Col', None)
        d = dmn.core.Damon(d.merge_info_out, 'datadict', verbose=None)
        subrow = np.size(d.collabels, axis=0) - 1
    else:
        try:
            subrow = subspaces['row']
            subs = np.unique(d.collabels[subrow, d.nheaders4rows:]).tolist()
        except KeyError:
            exc = 'Unable to find "row" in subspaces.\n'
            raise sub_coord_Error(exc)

    # Assign coord() parameters to each subscale
    sub_params = {}
    if 'All' in coord_subs.keys():
        coord_subs['All']['feather'] = 0.0001
        for sub in subs:
            sub_params[sub] = coord_subs['All']
    else:
        for sub in subs:
            if coord_subs[sub] is not None:
                coord_subs[sub]['feather'] = 0.0001
        sub_params = coord_subs

    # Assign coord() parameters for each unique analysis, include feathering
    resid_params = {}
    if 'All' in coord_resids.keys():
        coord_resids['All']['feather'] = 0.0001
        for sub in subs:
            resid_params[sub] = coord_resids['All']
    else:
        for sub in subs:
            if coord_resids[sub] is not None:
                coord_resids[sub]['feather'] = 0.0001
        resid_params = coord_resids

    # Define unique_weights
    unique_weights = {}
    if 'All' in unique_weights_.keys():
        for sub in subs:
            unique_weights[sub] = unique_weights_['All']
    else:
        unique_weights = unique_weights_

    # List of optimal weights (used for comparing a range of weights with estimated optimum)
    optimal_weights = {}
    for sub in subs:
        optimal_weights[sub] = None

    # coord and estimates dicts for output
    sub_coord_out = {}
    est_dict = {}

    for key in d.data_out.keys():
        est_dict[key] = d.data_out[key]

    est_dict['coredata'] = np.zeros(np.shape(d.coredata))
    est_keys = tools.getkeys(est_dict, 'Col', 'Core', 'Auto', None)

    # Get R coordinates for each subspace (treated as "common")
    sub_dict = {}
    for sub in subs:
        sub_x = d.extract(d,
                          getrows = {'Get':'AllExcept', 'Labels':'key', 'Rows':[None]},
                          getcols = {'Get':'NoneExcept', 'Labels':subrow, 'Cols':[sub]}
                          )
        sub_obj = dmn.core.Damon(sub_x, 'datadict', verbose=verbose)           

        if verbose is True:
            print '\n\nRunning coord() on subspace', sub

        # Add subspace object to sub_dict
        if sub_params[sub] is not None:
            sub_obj.coord(**sub_params[sub])
            sub_obj.base_est()
            sub_dict[sub] = sub_obj

        # Handle case where no coord() parameters are given
        else:
            pseudo_coords = {}
            pseudo_coords['fac0coord'] = sub_x
            sub_obj.coord_out = pseudo_coords
            sub_obj.base_est_out = sub_x
            sub_dict[sub] = sub_obj

    # Compute coordinates and estimates for each subspace
    for targ in subs:
        
        preds = list(set(subs) - set([targ]))

        if len(preds) == len(subs):
            exc = ('Buggy behavior:  number of predictors should be one less '
                   'than number of subspaces.')
            raise sub_coord_Error(exc)
        
        ntarg =  np.size(sub_dict[targ].coredata, axis=1)
        targ_est = np.zeros((nrows, ntarg + ntarg * len(preds)))
        est = sub_dict[targ].base_est_out['coredata']
        end = np.size(est, axis=1)
        targ_est[:, 0:end] = est

        # Get estimates from each predictor subspace
        for i, pred in enumerate(preds):
            npred = np.size(sub_dict[pred].coredata, axis=1)

            if (ntarg < share_if['targ_<']
                and npred > share_if['pred_>']
                and sub_dict[pred] is not None
                ):
                R_comm = sub_dict[pred].coord_out['fac0coord']['coredata']

                # Get target residuals
                targ_obj = dmn.core.Damon(sub_dict[targ].data_out, 'datadict', verbose=None)
                targ_obj.coord(quickancs = [0, R_comm], feather=0.0001)
                targ_obj.base_est()
                targ_obj.base_resid()
                res = targ_obj.base_resid_out

                # Calculate unique R
                if verbose is True:
                    print '\n\nRunning coord() on residuals. Target:', targ, ', Predictor:', pred

                res_obj = dmn.core.Damon(res, 'datadict', verbose=verbose)
                
                try:
                    res_obj.coord(**resid_params[targ])

                    # If searching, does ndim = 0 give lowest error?  If so, there is no unique dimension.
                    try:
                        dim = res_obj.objperdim.core_col['Dim']
                        err = res_obj.objperdim.core_col['Err']
                        low_dim = dim[np.amin(err) == err[0]]

                        if low_dim == 0:
                            R_both = R_comm
                            dim_unique = 0
                            unique_weight_ = 0
                        else:
                            unique_weight_ = tools.get_unique_weight(targ, targ_obj, res_obj,
                                                                     unique_weights, min_rel,
                                                                     rpt_optimal)
                            R_unique = res_obj.coord_out['fac0coord']['coredata']
                            dim_unique = np.size(R_unique, axis=1)
                            R_both = np.append(R_comm, R_unique, axis=1)

                    except AttributeError:
                        unique_weight_ = tools.get_unique_weight(targ, targ_obj, res_obj,
                                                                 unique_weights, min_rel,
                                                                 rpt_optimal)
                        R_unique = res_obj.coord_out['fac0coord']['coredata']
                        dim_unique = np.size(R_unique, axis=1)
                        R_both = np.append(R_comm, R_unique, axis=1)

                except (coord_Error, TypeError):
                    R_unique = res_obj.coredata
                    dim_unique = np.size(R_unique, axis=1)
                    R_both = np.append(R_comm, R_unique, axis=1)
                    uw = unique_weights[targ] if unique_weights[targ] is not None else 0.0
                    unique_weight_ = {'unique_weight':uw, 'optimal_weight':None}

                dim_both = np.size(R_both, axis=1)

                # Apply weights to R
                weights = np.ones((dim_both))
                if dim_unique != 0:
                    weights[-dim_unique:] = unique_weight_['unique_weight']
                optimal_weights[targ] = unique_weight_['optimal_weight']

                # Get weighted estimates from combined R
                targ_obj_ = dmn.core.Damon(targ_obj, 'Damon', verbose=None)
                targ_obj_.coord(quickancs = [0, R_both], feather = 0.001)
                C = targ_obj_.coord_out['fac1coord']['coredata']
                est = tools.estimate(R_both * weights, C, nanval)

                # Add latest estimates to targ_est
                start = ntarg + i * ntarg
                end = start + ntarg
                targ_est[:, start:end] = est

        # Lop off extra zeros from targ_est (caused by unused predictors)
        zeros = np.all(targ_est == 0, axis=0)
        targ_est = targ_est[:, zeros == False]

        # Analyze targ_est
        if verbose is True:
            print '\n\nRunning coord() on composite estimates. Target:', targ

        targ_est_obj = dmn.core.Damon(targ_est, 'array', verbose=verbose)

        try:
            targ_ndim = sub_params[targ]['ndim']
        except TypeError:
            targ_ndim = [[1]]

        targ_est_obj.coord(ndim=targ_ndim)
        R = targ_est_obj.coord_out['fac0coord']['coredata']

        # Apply R back to original target data
        targ_obj_2 = dmn.core.Damon(sub_dict[targ].data_out, 'datadict', verbose=None)
        targ_obj_2.coord(quickancs = [0, R], feather = 0.001)
        targ_obj_2.base_est()
        targ_est_obj = dmn.core.Damon(targ_obj_2.base_est_out, 'datadict', verbose=None)

        # Output coordinates
        sub_coord_out[targ] = targ_obj_2.coord_out

        # Load estimates into est_obj array
        targ_keys = tools.getkeys(targ_obj_2, 'Col', 'Core', 'Auto', None)

        for key in targ_keys:
            est = est_dict['coredata']
            est[:, est_keys == key] = targ_est_obj.core_col[key][:, np.newaxis]

    sub_coord_out['estimates'] = est_dict
    sub_coord_out['optimal_weights'] = optimal_weights

    # Round optimal weights
    opt_wt = optimal_weights.copy()
    for key in optimal_weights.keys():
        if optimal_weights[key] is not None:
            opt_wt[key] = round(optimal_weights[key], 3)

    if verbose is True:
        print 'Optimal weights for each unique dimension:\n', opt_wt, '\n'

    # For calculating standard errors in base_se()
    sub_coord_out['facs_per_ent'] = [1, 2]  # Unsure about correct number of facets
    sub_coord_out['ndim'] = len(subs) + 1   # Contains a LOT of assumptions

    return sub_coord_out




######################################################################

def _objectify(_locals):
    "Basis of the objectify() method"


    # Get self
    self = _locals['self']

    # Get locals
    targ_ents_ = _locals['targ_ents']
    pred_ents_ = _locals['pred_ents']
    ndim = _locals['ndim']
    runspecs = _locals['runspecs']
    seed = _locals['seed']
    starters_ = _locals['starters']
    summdim = _locals['summdim']
    center = _locals['center']
    overwrite = _locals['overwrite']

    # Extract the correct data to analyze
    try:
        data = self.standardize_out
    except AttributeError:
        try:
            data = self.parse_out
        except AttributeError:
            try:
                data = self.subscale_out
            except AttributeError:
                try:
                    data = self.score_mc_out
                except AttributeError:
                    try:
                        data = self.extract_valid_out
                    except AttributeError:
                        try:
                            data = self.merge_info_out
                        except AttributeError:
                            try:
                                data = self.data_out
                            except AttributeError:
                                exc = 'Unable to find data to analyze.\n'
                                raise objectify_Error(exc)

    # Convert to main damon object
    d = dmn.core.Damon(data,'datadict_link',verbose=None)

    # Try to get coordinates and ndim
    try:
        fac0coord = self.coord_out['fac0coord']
        fac1coord = self.coord_out['fac1coord']

        if ndim != 'Refer2Coord':
            if self.verbose is True:
                print "Warning in objectify(): ndim is not 'Refer2Coord', yet you ran coord(). objectify() will use its own ndim parameter to get estimates.\n"
        else:
            ndim = [[self.coord_out['ndim']]]
    except AttributeError:
        fac0coord = fac1coord = None
        if ndim == 'Refer2Coord':
            exc = "ndim = 'Refer2Coord', but could not find coord() outputs.  Specify dimensionality or run coord().\n"
            raise objectify_Error(exc)

    # Get seed
    if seed == 'Refer2Coord':
        try:
            seed = self.seed['BestSeed']
        except AttributeError:
            if self.verbose is True:
                print 'Warning in objectify():  Could not find previously estimated "best seed".  Setting seed = "Auto".\n'
            seed = 'Auto'


    ##################
    ##  Get Targs,  ##
    ##    Preds     ##
    ##################

    # Get target entities
    targ_x = d.extract(data,
                       getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                       getcols = targ_ents_,
                       labels_only = True
                       )
    targ_ents = list(tools.getkeys(targ_x,'Col','Core','Auto',None))

    # Prepare to get different predictor entities for each target entity
    pred_keys = pred_ents_.keys()
    pred_vals = pred_ents_.values()
    pred_dict = {}

    # preds = all except targ
    if ('AllTargs' in pred_keys
        and 'AllExceptTarg' in pred_vals
        ):
        pred_x = d.extract(data,
                           getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                           getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]},
                           labels_only = True
                           )
        predx_keys = list(tools.getkeys(pred_x,'Col','Core','Auto',None))
        for targ_ent in targ_ents:
            pred_dict[targ_ent] = {}
            pred_dict[targ_ent]['Altogether'] = list(set(predx_keys) - set([targ_ent]))

    # preds = specified ents
    elif ('AllTargs' in pred_keys
          and isinstance(pred_vals[0],dict)
          ):
        pred_x = d.extract(data,
                           getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                           getcols = pred_ents_['AllTargs'],
                           labels_only = True
                           )
        predx_keys = list(tools.getkeys(pred_x,'Col','Core','Auto',None))
        for targ_ent in targ_ents:
            pred_dict[targ_ent] = {}
            pred_dict[targ_ent]['Altogether'] = predx_keys

    # preds specified for each targ individually
    elif len(pred_keys) > 1:
        for targ_ent in targ_ents:
            try:
                pred_x = d.extract(data,
                                   getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                   getcols = pred_ents_[targ_ent],
                                   labels_only = True
                                   )
            except KeyError:
                exc = 'Unable to figure out the predictor entities (pred_ents) you want for each targ_ent.\n'
                raise objectify_Error(exc)

            pred_dict[targ_ent] = {}
            pred_dict[targ_ent]['Altogether'] = list(tools.getkeys(pred_x,'Col','Core','Auto',None))

    # Group predictors by subscale
    elif 'Subscales' in pred_keys:
        try:
            ind = int(pred_ents_['Subscales']['Labels'])
        except ValueError:
            exc = "When 'pred_ents' specifies 'Subscales', 'Labels' must be an integer giving the row containing subscale labels.  'Cols' must be a list of predictor subscales or ['All'], for all.\n"
            raise objectify_Error(exc)

        # Get list of predictor subs
        all_subs = list(np.unique(d.collabels[ind,d.nheaders4rows:]).astype('S60'))
        if pred_ents_['Subscales']['Cols'] == ['All']:
            pred_subs_ = all_subs
        else:
            if pred_ents_['Subscales']['Get'] == 'NoneExcept':
                pred_subs_ = pred_ents_['Subscales']['Cols']
            else:
                pred_subs_ = list(set(all_subs) - set(pred_ents_['Subscales']['Cols']))

        # Get subscale-specific predictor entities for each target
        for targ_ent in targ_ents:

            # Clarify whether target is a subscale.  If so, ignore all its subscale items.
            if 'sub_' not in str(targ_ent):
                pred_subs_x = pred_subs_
            else:
                pred_subs_x = list(set(pred_subs_) - set([targ_ent[4:]]))

            pred_dict[targ_ent] = {}
            for pred_sub in pred_subs_x:
                pred_x = d.extract(data,
                                   getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                   getcols = {'Get':'NoneExcept','Labels':ind,'Cols':[pred_sub]},
                                   labels_only = True
                                   )
                preds = list(tools.getkeys(pred_x,'Col','Core','Auto',None))

                if 'sub_' not in str(targ_ent):
                    if targ_ent in preds:
                        preds.remove(targ_ent)

                for pred in preds:
                    if 'sub_' in pred:
                        preds.remove(pred)

                pred_dict[targ_ent][pred_sub] = preds

    else:
        exc = "Unable to figure out 'pred_ents' parameters.\n"
        print 'Error in objectify(): ',exc
        print 'pred_keys =\n',pred_keys
        print 'pred_vals =\n',pred_vals
        raise objectify_Error(exc)

    # Ensure consistency with ndim [not online yet]
    if isinstance(ndim,dict):
        if 'Subscales' in pred_keys:
            exc = "When 'ndim' specifies parameters for multiple predictor sets, 'pred_ents' must use the 'Subscales' option.  However, to handle subscales, ndim does not need multiple specifications.  See if one will work.\n"
            raise objectify_Error(exc)


    #################
    ##  Starters,  ##
    ##  Estimates, ##
    ##   unbias()  ##
    #################

    # Initialize
    facs_per_ent = [1,2]

    # Calculate estimates if necessary
    if fac1coord is None:
        d.coord(ndim,[0.00001,30],seed,feather=0.0001)
        d.base_est()
        fac1start = d.coord_out['fac1coord']
        dim = [[d.coord_out['ndim']]]
        est_dict = d.base_est_out
        facs_per_ent_ = d.facs_per_ent
    else:
        fac1start = fac1coord
        dim = ndim
        facs_per_ent_ = self.facs_per_ent
        try:
            est_dict = self.base_est_out
        except AttributeError:
            self.base_est()
            est_dict = self.base_est_out

    est_keys = tools.getkeys(est_dict,'Col','Core','Auto',None)
    obj_est = est_dict['coredata']

    nrows = np.size(obj_est,axis=0)
    ncols = np.size(obj_est,axis=1)

    # Define facs_per_ent (see tools.obspercell())
    if facs_per_ent_[0] == 0:
        if not isinstance(facs_per_ent_[1],np.ndarray):
            facs_per_ent = [facs_per_ent_[0],np.zeros((nrows),dtype=float) + facs_per_ent_[1]]
        else:
            facs_per_ent = facs_per_ent_
    elif facs_per_ent_[0] == 1:
        if not isinstance(facs_per_ent_[1],np.ndarray):
            facs_per_ent = [facs_per_ent_[0],np.zeros((ncols),dtype=float) + facs_per_ent_[1]]
        else:
            facs_per_ent = facs_per_ent_

    # Define function to unbias estimates for given entity
    def unbias(targ_ent,R,targ_data,targ_est,nanval):
        "Compute estimates as if corresponding observations were deleted first."

        # Prepare for unbiasing
        sub_est = np.zeros((np.size(R,axis=0))) + nanval
        valloc = np.where(R[:,0] != nanval)[0]
        invRTR = tools.invUTU(R[valloc],'R',None)
        h = tools.h_stat(R[valloc],'R',invRTR)
        resid = tools.residuals(targ_data['coredata'],targ_est['coredata'],None,None,None,nanval)

        # Get estimates as if corresponding observations were deleted
        unbiased_est = tools.unbiasest(targ_est['coredata'][valloc],resid[valloc],h,nanval)

        # Update subscale estimates
        sub_est[valloc] = np.squeeze(unbiased_est)

        return sub_est

    #################
    ##  Extract,   ##
    ##   Get R,    ##
    ##  Get Ests   ##
    #################

    # Calculate objectivized estimates for each target entity
    for targ_ent in targ_ents:

        # Get target entity data
        targ_data = d.extract(data,
                              getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                              getcols = {'Get':'NoneExcept','Labels':'key','Cols':[targ_ent]},
                              labels_only = None
                              )
        targ_obj = dmn.core.Damon(targ_data,'datadict_link',verbose=None)

        # Get estimates for each set of predictor entities
        pred_subs = pred_dict[targ_ent].keys()
        sub_est = np.zeros((nrows,len(pred_subs))) + d.nanval

        for i,pred_sub in enumerate(pred_subs):
            pred_data = d.extract(d,
                                  getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                  getcols = {'Get':'NoneExcept','Labels':'key','Cols':pred_dict[targ_ent][pred_sub]},
                                  labels_only = None,
                                  )

            if starters_ is True:
                pred_coords = d.extract(fac1start,
                                        getrows = {'Get':'NoneExcept','Labels':'key','Rows':pred_dict[targ_ent][pred_sub]},
                                        getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]},
                                        labels_only = None
                                        )
                starter = [1,pred_coords['coredata']]
                condcoord_ = None

            else:
                starter = None
                condcoord_ = {'Fac0':'Orthonormal','Fac1':None}

            # Calculate R
            pred_obj = dmn.core.Damon(pred_data,'datadict_link',verbose=None)
            pred_obj.coord(dim,runspecs,seed,startercoord=starter,condcoord_=condcoord_)
            R = np.copy(pred_obj.coord_out['fac0coord']['coredata'])

            # Center estimates on observations
            if center is True:
                R = np.append(np.ones((nrows,1)),R,axis=1)

            # Get unbiased estimates for this predictor subscale
            targ_obj.coord(quickancs=[0,R])
            targ_obj.base_est()
            sub_est[:,i] = unbias(targ_ent,R,targ_data,targ_obj.base_est_out,d.nanval)

        # Apply coord() to predictor subscale estimates array, anchor R, apply to targ_data to give objective estimates
        if len(pred_subs) > 1:

            # Use coord to combine predictor estimates for given target
            sub_est_obj = dmn.core.Damon(sub_est,'array',verbose=None)
            sub_est_obj.coord([[1]],runspecs,seed)
            R1 = np.copy(sub_est_obj.coord_out['fac0coord']['coredata'])
            targ_obj.coord(quickancs=[0,R1])
            targ_obj.base_est()

            # Average target estimates (multiple columns will be rare)
            targ_est_ma = npma.masked_values(targ_obj.base_est_out['coredata'],targ_obj.nanval)
            sub_est_0 = npma.mean(targ_est_ma,axis=1).filled(targ_obj.nanval)

            # As backup, use regular mean to combine predictor estimates for the target (to fill some nanvals)
            targ_est_ma_1 = npma.masked_values(sub_est,targ_obj.nanval)
            sub_est_1 = npma.mean(targ_est_ma_1,axis=1).filled(targ_obj.nanval)

            # Subscale estimates
            sub_est_ = np.where(sub_est_0 == targ_obj.nanval,sub_est_1,sub_est_0)
        else:
            sub_est_ = np.squeeze(sub_est)

        try:
            entloc = np.where(est_keys == targ_ent)[0][0]
            obj_est[:,entloc] = sub_est_
            facs_per_ent[1][entloc] = 1
        except IndexError:
            try:
                entloc = np.where(est_keys == str(targ_ent))[0][0]
                obj_est[:,entloc] = sub_est_
                facs_per_ent[1][entloc] = 1
            except IndexError:
                exc = 'Having trouble matching column entity types.\n'
                raise objectify_Error(exc)

    # Update objective estimates datadict
    est_dict['coredata'] = obj_est

    # Report dimensionality
    if len(dim[0]) == 1:
        est_dict['ndim'] = dim[0][0]
    else:
        est_dict['ndim'] = np.size(R,axis=1)    # WARNING:  Only gets rank of most recent R!


    ##################
    ##  Summary     ##
    ## Coordinates, ##
    ##   Reports    ##
    ##################

    # Compute coordinates that summarize objectivized estimates array
    if summdim is not None:
        if summdim == 'ndim' and ndim != 'Refer2Coord':
            summdim = ndim
        est_obj = dmn.core.Damon(est_dict,'datadict_link',verbose=None)
        est_obj.coord(summdim,[0.00001,30],seed)
        est_coords = est_obj.coord_out
        self.obj_dim = est_coords['ndim']

        if overwrite is True:
            try:
                self.objperdim = est_obj.objperdim
            except AttributeError:
                pass

            self.coord_out = est_coords
    else:
        est_coords = None
        try:
            self.coord_out
        except AttributeError:
            if self.verbose is True:
                print "Warning in objectify(): Because summdim was not used, no coord_out has been assigned to the Damon object.\n"

    # Update base_est_out
    if overwrite is True:
        self.base_est_out = est_dict
        self.objectify_overwrite = True
        self.facs_per_ent = facs_per_ent

    return {'obj_est':est_dict,
            'obj_coord':est_coords,
            }




######################################################################

def _base_est(_locals):
    "Basis of the base_est() method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Local variables
    fac_coords = _locals['fac_coords']
    ecut = _locals['ecutmaxpos']
    refit = _locals['refit']
    nondegen = _locals['nondegen']

    # Check if rasch() was run
    try:
        self.rasch_out
        if self.verbose is True:
            print 'Warning in base_est(): No need to run base_est() if rasch() has been run.  base_est_out already exists.\n'

        self.base_est_out['ecutmaxpos'] = ecut

        return self.base_est_out

    except AttributeError:
        pass

    # Pass through estimates if sub_coord() was run
    try:
        BaseEstRCD = self.sub_coord_out['estimates']
        BaseEstRCD['ecutmaxpos'] = ecut
        return BaseEstRCD
    except AttributeError:
        pass

    # Pass through estimates if objectify() was run and overwrite is True
    try:
        self.objectify_out
        try:
            if self.objectify_overwrite is True:
                BaseEstRCD = self.objectify_out['obj_est']
                BaseEstRCD['ecutmaxpos'] = ecut
                return BaseEstRCD
            else:
                pass
        except AttributeError:
            pass
    except AttributeError:
        pass

    # Extract data that went into coord()
    try:
        datadict = self.standardize_out
    except AttributeError:
        try:
            datadict = self.parse_out
        except AttributeError:
            try:
                datadict = self.subscale_out
            except AttributeError:
                try:
                    datadict = self.score_mc_out
                except AttributeError:
                    try:
                        datadict = self.extract_valid_out
                    except AttributeError:
                        try:
                            datadict = self.merge_info_out
                        except AttributeError:
                            try:
                                datadict = self.data_out
                            except AttributeError:
                                exc = 'Unable to find data.\n'
                                raise base_est_Error(exc)


    #################
    ##  Calc Base  ##
    ##  estimates  ##
    #################

    # Get coordinates (handles pytables)
    if fac_coords == 'Auto':
        try:
            fac0coord = self.coord_out['fac0coord']['coredata']
            fac1coord = self.coord_out['fac1coord']['coredata']
        except AttributeError:
            exc = 'Unable to find coordinates for calculating estimates.\n'
            raise base_est_Error(exc)

        nanval = float(self.nanval)

    # get from _locals
    else:
        try:
            fac0coord = fac_coords[0]
            fac1coord = fac_coords[1]
            nanval = fac_coords[2]
        except:
            exc = 'Unable to find coordinates for calculating estimates.\n'
            raise base_est_Error(exc)

    # Calculate dot product (includes NaNVals in calculation)
    if pytables is None:
        BaseEst = np.dot(fac0coord,np.transpose(fac1coord))

        # Get locations of missing coordinates
        F0NaN_Loc = np.where(fac0coord == nanval)[0]
        F1NaN_Loc = np.where(fac1coord == nanval)[0]

        # Apply NaNVals to rows with missing coordinates
        BaseEst[F0NaN_Loc] = nanval
        np.transpose(BaseEst)[F1NaN_Loc] = nanval

    # Use pytables chunkfunc to do dot product
    else:
        nfac0 = np.size(fac0coord[:,:],axis=0)
        nfac1 = np.size(fac1coord[:,:],axis=0)

        def dot_chunk(Fac0,Fac1T,chunksize,chunkstart=0):
            Fac0Chunk = Fac0[chunkstart:(chunkstart + chunksize),:]

            BaseEst = np.dot(Fac0Chunk,Fac1T)

            # Get locations of missing coordinates
            F0NaN_Loc = np.where(Fac0Chunk == nanval)[0]
            F1NaN_Loc = np.where(Fac1T == nanval)[0]

            # Apply NaNVals to rows with missing coordinates
            BaseEst[F0NaN_Loc] = nanval
            np.transpose(BaseEst)[F1NaN_Loc] = nanval

            return BaseEst

        dot_ChunkDict = {'chunkfunc':dot_chunk,
                         'nchunks':'Auto',
                         'chunksize':'Auto',
                         'nrows':nfac0,
                         'ncols':nfac1,
                         }
        dot_ArgDict = {'Fac0':fac0coord,
                       'Fac1T':np.transpose(fac1coord),
                       'chunksize':'Auto',
                       'chunkstart':0
                       }

        # Compute estimates
        dot_Dicts = {'chunkdict':dot_ChunkDict,'ArgDict':dot_ArgDict}
        BaseEst = tools.pytables_(dot_Dicts,'chunkfunc',fileh,None,'base_est_out',
                                  ['coredata'],None,'float',4,None,None)['arrays']['coredata']

    # refit estimates to data, if desired
    if refit is not None:
        if refit in ['Lstsq', 'lstsq']:
            deg = 1
        else:
            deg = refit
            
        obs = np.ravel(datadict['coredata'])
        est = np.ravel(BaseEst)
        ix = (obs != nanval) & (est != nanval)
        x = np.polyfit(est[ix], obs[ix], deg)
        p = np.poly1d(x)
        y = p(est)
        y[est == nanval] = nanval
        BaseEst = np.reshape(y, np.shape(BaseEst))
            
        # Build est object and rebuild coordinates with new estimates
        e = dmn.core.Damon(BaseEst, 'array', validchars=None, nanval=nanval, 
                           verbose=None)
        try:
            e.extract_valid(2, 2, 0.0000001)
        except:
            e.extract_valid(1, 1, 0.0)
        
        ndim = self.coord_out['ndim']
        e.coord([[ndim]], seed=1)
        e.restore_invalid(['coord_out'])

        # Assign attributes
        self.bestdim = ndim
        self.coord_out['fac0coord']['coredata'] = e.fac0coord['coredata']
        self.coord_out['fac1coord']['coredata'] = e.fac1coord['coredata']

    # Check for degenerate solutions
    if nondegen is True:
        msindex = np.where(datadict['coredata'][:,:] == nanval)
        NonMsIndex = np.where(datadict['coredata'][:,:] != nanval)

        if len(msindex[0]) > 0:

            # Missing estimates
            EstMiss = BaseEst[msindex]
            EstMiss = EstMiss[np.where(EstMiss != nanval)]
            EstMiss_MSq = np.mean(EstMiss**2)

            # NonMissing observations
            NonMiss = datadict['coredata'][NonMsIndex]
            NonMiss_MSq = np.mean(NonMiss**2)
            NonMiss_SDSq = np.std(NonMiss**2)

            # Are missing estimates significantly different from nonmissing?
            Contrast = abs(EstMiss_MSq - NonMiss_MSq) / NonMiss_SDSq
            NonDegeneracy = 1.0 - (Contrast / (1.0 + Contrast))
            self.nondegeneracy = NonDegeneracy

            if Contrast > 1.5:
                print "Warning in base_est(): NonDegeneracy",NonDegeneracy,"<= 0.40.  Missing cell estimates are in a different range than remaining observations -- a possible degenerate solution.  See my_DamonObj.NonDegeneracy for the statistic.\n"

        else:
            print 'Warning in base_est():  Unable to calculate NonDegeneracy. No missing cells.\n'

    # BaseEst datadict
    BaseEstRCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in datadict.keys():
        if key in ValList:
            BaseEstRCD[key] = datadict[key]
        BaseEstRCD['coredata'] = BaseEst
        BaseEstRCD['validchars'] = ['All',['All'],'Num']

    BaseEstRCD['ecutmaxpos'] = ecut

    return BaseEstRCD




######################################################################

def _base_resid(_locals):
    "Basis of the base_resid() method."

    # Get Local variables
    self = _locals['self']
    nearest_val = _locals['nearest_val']
    psmiss = _locals['psmiss']
    fileh = self.fileh
    pytables = self.pytables

    # Check if rasch() was run
    try:
        self.rasch_out
        if self.verbose is True:
            print 'Warning in base_resid(): Overwriting the base_resid_out already created by rasch().\n'
    except AttributeError:
        pass

    # Extract data that went into coord()
    try:
        ObsRCD = self.standardize_out
    except AttributeError:
        try:
            ObsRCD = self.parse_out
        except AttributeError:
            try:
                ObsRCD = self.subscale_out
            except AttributeError:
                try:
                    ObsRCD = self.score_mc_out
                except AttributeError:
                    try:
                        ObsRCD = self.extract_valid_out
                    except AttributeError:
                        try:
                            ObsRCD = self.merge_info_out
                        except AttributeError:
                            try:
                                ObsRCD = self.data_out
                            except AttributeError:
                                exc = 'Unable to find "observed" data.\n'
                                raise base_resid_Error(exc)

    # Get estimates
    try:
        EstRCD = self.base_est_out
    except AttributeError:
        exc = 'Unable to find "estimates" data.  Run base_est().\n'
        raise base_resid_Error(exc)

    # estimates variables (handles pytables)
    collabels = EstRCD['collabels']
    rowlabels = EstRCD['rowlabels']

    nheaders4rows = EstRCD['nheaders4rows']
    nheaders4cols = EstRCD['nheaders4cols']
    key4cols = EstRCD['key4cols']
    colkeytype = EstRCD['colkeytype']
    nanval = EstRCD['nanval']
    nrows = np.size(rowlabels[nheaders4cols:,0])
    ncols = np.size(collabels[0,nheaders4rows:])

    # Get psmsindex
    if psmiss is not None:
        try:
            if self.pseudomiss_out['parsed_psmsindex'] is None:
                psmsindex = self.pseudomiss_out['psmsindex']
            else:
                psmsindex = self.pseudomiss_out['parsed_psmsindex']
        except AttributeError:
            exc = 'Unable to find pseudo-missing cell index. First run pseudomiss().\n'
            raise base_resid_Error(exc)
    else:
        psmsindex = None

    # Get nearest_val parameter for use in tools.residuals() below
    if nearest_val == 'ECut':
        NearestVal_ = None
    else:
        NearestVal_ = nearest_val

    # Get ecut
    if nearest_val != 'ECut':
        ECut_ = None
    else:
        try:
            ECut1 = EstRCD['ecutmaxpos']
        except KeyError:
            exc = 'Unable to find "ecutmaxpos".  Enter as argument in base_est().\n'
            raise base_resid_Error(exc)

        # ecut is None
        if ECut1 is None:
            ECut_ = ['Cols','Med']

        # ecut is column dict
        elif isinstance(ECut1[1],dict):
            ECutDict = ECut1[1]

            try:
                EntRow = self.parse_out['EntRow']
                Ents = collabels[EntRow,nheaders4rows:]
            except AttributeError:
                Ents = collabels[key4cols,nheaders4rows:].astype(colkeytype)

            ECuts = []
            for Ent in Ents:
                ECuts.append(ECutDict[Ent][0])
            ECut_ = [ECut1[0],ECuts]

        # Use ecut as is
        else:
            ECut_ = [ECut1[0],ECut1[1][0]]

    # Calculate residuals
    if pytables is None:
        Resid = tools.residuals(observed = ObsRCD['coredata'], # [2D array of observed values]
                              estimates = EstRCD['coredata'],    # [2D array of cell estimates]
                              psmsindex = psmsindex, # [<None, where()-style index of cells made pseudo-missing>]
                              nearest_val = NearestVal_,     # [<None,'Nearest'> => first convert estimate to nearest valid observed value]
                              ecut = ECut_,  # [<None, [['All',ecut], ['Cols',[ECut1,ECut2,'Med',...]]> ]
                              nanval = nanval,   # [Not-a-Number value for cells missing in observed]
                              )

    # pytables chunkfunc
    else:
        def resid_chunk(Obs,Est,chunksize,nanval,chunkstart=0):
            "Calculate row-wise chunk of residuals."

            ResidChunk = tools.residuals(Obs[chunkstart:(chunkstart + chunksize),:],
                                        Est[chunkstart:(chunkstart + chunksize),:],
                                        psmsindex = None,   # Doesn't work with chunks.  Handle outside function.
                                        nearest_val = NearestVal_,
                                        ecut = ECut_,
                                        nanval = nanval
                                        )
            return ResidChunk

        Resid_ChunkDict = {'chunkfunc':resid_chunk,
                           'nchunks':'Auto',
                           'chunksize':'Auto',
                           'nrows':nrows,
                           'ncols':ncols
                           }
        Resid_ArgDict = {'Obs':ObsRCD['coredata'],
                         'Est':EstRCD['coredata'],
                         'chunksize':'Auto',
                         'nanval':nanval,
                         'chunkstart':0
                         }
        ResidTemp0_DataDicts = {'chunkdict':Resid_ChunkDict,'ArgDict':Resid_ArgDict}

        if psmsindex is None:
            Arr = 'coredata'
        else:
            Arr = 'Temp0'

        ResidTemp0Tab = tools.pytables_(ResidTemp0_DataDicts,'chunkfunc',fileh,None,'base_resid_out',
                                  [Arr],None,'float',4,None,None)['arrays'][Arr]

        # Handle pseudomissing option with another PyTable, if necessary
        if psmsindex is None:
            Resid = ResidTemp0Tab
        else:
            PsMs_ChunkDict = {'chunkfunc':tools.zeros_chunk,
                              'nchunks':'Auto',
                              'chunksize':'Auto',
                              'nrows':nrows,
                              'ncols':ncols
                              }
            PsMs_ArgDict = {'nchunks':'Auto',
                            'chunksize':'Auto',
                            'nrows':nrows,
                            'ncols':ncols,
                            'Val':nanval,
                            'chunkstart':0
                            }
            PsMs_DataDicts = {'chunkdict':PsMs_ChunkDict,'ArgDict':PsMs_ArgDict}
            Resid_ = tools.pytables_(PsMs_DataDicts,'chunkfunc',fileh,None,'fin_resid_out',
                                    ['coredata'],None,'float',4,None,None)['arrays']['coredata']
            Resid_[psmsindex] = ResidTemp0Tab[psmsindex]
            Resid = Resid_

    # Applies a binomial correction if the data are ordinal (see pq_resid)
    pq_resid = True
    if (pq_resid 
        and hasattr(self, 'standardize_out')
        and self.standardize_out['stdmetric'] == 'PreLogit'):
        
        # Separate ecuts per col not implemented
        ear = None
        Resid = tools.pq_resid(EstRCD['coredata'], 
                               Resid,
                               colkeys=tools.getkeys(EstRCD, 'Col', 'Core'),
                               ecut=['All', 0.0],
                               ear=ear,
                               new_logits=False,
                               validchars=self.validchars,
                               nanval=nanval)['new_resid']
                               
    # Build dict
    ResidRCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in EstRCD.keys():
        if key in ValList:
            ResidRCD[key] = EstRCD[key]
        ResidRCD['coredata'] = Resid
        ResidRCD['validchars'] = ['All',['All'],'Num']

    return ResidRCD




######################################################################

def _fin_resid(_locals):
    "Basis of the fin_resid() method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Get Local variables
    resid_type = _locals['resid_type']
    psmiss = _locals['psmiss']

    # String keys
    if isinstance(resid_type[1], dict):
        e = {}
        for key in resid_type[1]:
            e[str(key)] = resid_type[1][key]
        resid_type[1] = e

    # Extract original data
    try:
        ObsRCD = self.subscale_out
    except AttributeError:
        try:
            ObsRCD = self.extract_valid_out
        except AttributeError:
            try:
                ObsRCD = self.merge_info_out
            except AttributeError:
                try:
                    ObsRCD = self.data_out
                except AttributeError:
                    exc = 'Unable to find "observed" data (MyObj.data_out).\n'
                    raise fin_resid_Error(exc)

    # Get estimates
    try:
        EstRCD = self.fin_est_out
    except AttributeError:
        exc = 'Unable to find final estimates (MyObj.fin_est_out).\n'
        raise fin_resid_Error(exc)

    # Apply psmsindex to whole array
    if psmiss is True:
        try:
            if self.pseudomiss_out['parsed_psmsindex'] is None:
                psmsindex = self.pseudomiss_out['psmsindex']
            else:
                psmsindex = self.pseudomiss_out['parsed_psmsindex']
        except AttributeError:
            exc = 'Unable to find pseudo-missing cell index. First run pseudomiss().\n'
            raise fin_resid_Error(exc)

    # estimates variables
    rowlabels = EstRCD['rowlabels']
    collabels = EstRCD['collabels']
    nheaders4rows = EstRCD['nheaders4rows']
    nheaders4cols = EstRCD['nheaders4cols']
    nanval = EstRCD['nanval']

    nrows = np.size(rowlabels[nheaders4cols:,:],axis=0)
    ncols = np.size(collabels[nheaders4rows:,:],axis=1)

    # Get Ents
    ObsEnts = tools.getkeys(ObsRCD,'Col','Core','Auto',None)
    Ents = tools.getkeys(EstRCD,'Col','Core', type(ObsEnts[0]),None)
    Ents0 = tools.getkeys(EstRCD,'Col','Core','Auto',None)
    Edict = dict(zip(Ents,Ents0))

    # Determine if Ents = ObsEnts
    try:
        Eq0 = (Ents == ObsEnts).all()
    except:
        Eq0 = (Ents == ObsEnts)

    # Check viability of psmiss option
    if Eq0 is False:
        ObsEnts = ObsEnts.astype('S60')
        try:
            Eq1 = (Ents == ObsEnts).all()
        except:
            Eq1 = (Ents == ObsEnts)

        if Eq1 is False:
            print 'Warning in fin_resid(): observed keys and fin_est keys do not all match.  Forcing psmiss = False as a precaution.\n'
            psmiss = False

    # Convert 'All' resid_type to 'Cols'
    if resid_type[0] == 'All':
        ResDict = {}
        for Ent in Ents:
            ResDict[Ent] = resid_type[1]
        resid_type = ['Cols',ResDict]

    elif resid_type[0] == 'Cols':
        ResDict = {}
        ResDict0 = resid_type[1]
        for Ent in Ents:
            try:
                ResDict[Ent] = ResDict0[Edict[Ent]]
            except KeyError:
                break
        ResDict = ResDict0

    # Initialize array
    if pytables is None:
        Resid = np.zeros((nrows,1))
    else:
        if psmiss is True:
            filename = 'Temp'
        else:
            filename = 'coredata'

        Resid = tools.pytables_(None,'init_earray',fileh,None,'fin_resid_out',[filename],
                               None,'float',4,(nrows,0),None)['arrays'][filename]

    # Estimate residuals for each column entity
    for Ent in Ents:

        # Extract col of obs/est
        EntEst = EstRCD['coredata'][:,np.where(Ent == Ents)[0]]

        # Obs array may not have all same keys as Est array
        if Ent in ObsEnts:
            EntObs = ObsRCD['coredata'][:,np.where(Ent == ObsEnts)[0]]

        elif (str(int(float(Ent))) in ObsEnts
              and int(float(Ent))/float(Ent) == 1.0
              ):
            EntObs = ObsRCD['coredata'][:,np.where(str(int(float(Ent))) == ObsEnts)[0]]
        else:
            EntObs = nanval

        # Get just the difference
        if ResDict[Ent] == ['Diff']:
            nearest_val = None
            ecut = None

        # Get nearest valid value
        elif ResDict[Ent] == ['Nearest']:
            nearest_val = 'Nearest'
            ecut = None

        # Look for match
        elif ResDict[Ent] == ['Match']:
            nearest_val = 'Match'
            ecut = None

        # Are observation and estimate on same side as cut-point?
        elif ResDict[Ent][0] == 'ECut':
            nearest_val = None
            ecut = ['Cols',[ResDict[Ent][1]]]

        else:
            exc = "Unable to figure out resid_type parameter.\n"
            raise fin_resid_Error(exc)

        # Calculate residuals (PsMs handled outside loop)
        if isinstance(EntObs,np.ndarray):
            EntRes = tools.residuals(EntObs,EntEst,None,nearest_val,ecut,nanval)
        else:
            EntRes = np.zeros((nrows,1)) + nanval

        # Append entity residuals to initialize array
        if pytables is None:
            Resid = np.append(Resid,EntRes.astype(float),axis=1)
        else:
            Resid.append(EntRes.astype(float))

    # Delete leading zeros if needed
    if pytables is None:
        Resid = np.delete(Resid,0,axis=1)

    # Select only pseudo-missing options
    if pytables is None:
        if psmiss is True:
            Temp = np.zeros((nrows,ncols)) + nanval
            Temp[psmsindex] = Resid[psmsindex]
            Resid = Temp

    # pytables chunkfunc
    else:
        # Handle pseudomissing option with another PyTable, if necessary
        if psmiss is True:
            PsMs_ChunkDict = {'chunkfunc':tools.zeros_chunk,
                              'nchunks':'Auto',
                              'chunksize':'Auto',
                              'nrows':nrows,
                              'ncols':ncols
                              }
            PsMs_ArgDict = {'nchunks':'Auto',
                            'chunksize':'Auto',
                            'nrows':nrows,
                            'ncols':ncols,
                            'Val':nanval,
                            'chunkstart':0
                            }
            PsMs_DataDicts = {'chunkdict':PsMs_ChunkDict,'ArgDict':PsMs_ArgDict}
            Resid_ = tools.pytables_(PsMs_DataDicts,'chunkfunc',fileh,None,'fin_resid_out',
                                    ['coredata'],None,'float',4,None,None)['arrays']['coredata']
            Resid_[psmsindex] = Resid[psmsindex]
            Resid = Resid_

    # Build dict
    ResidRCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in EstRCD.keys():
        if key in ValList:
            ResidRCD[key] = EstRCD[key]
        ResidRCD['coredata'] = Resid
        ResidRCD['validchars'] = ['All',['All'],'Num']

    return ResidRCD




######################################################################

def _base_ear(_locals):
    "Basis of the base_ear() method."

    # Get Local variables
    self = _locals['self']
    ndim = _locals['ndim']
#    pytables = self.pytables
#    fileh = self.fileh

    if ndim != 2:
        print ('Warning in base_ear(): The number of dimensions (ndim) is '
               'not 2.  While this is permitted for experimentation, '
               'downstream methods such as equate() assume two dimensions.\n')

    # Check if rasch() was run
    try:
        self.rasch_out
        if self.verbose is True:
            print ('Warning in base_ear(): Overwriting the base_ear_out '
                   'already created by rasch().\n')
    except AttributeError:
        pass

    # Get residuals
    try:
        resid = self.base_resid_out
    except AttributeError:
        exc = 'Unable to find base_resid_out.  Run base_resid().\n'
        raise base_ear_Error(exc)

    # Get variables
    coredata = resid['coredata']
    nanval = resid['nanval']
    nrows = np.size(coredata,axis=0)
    ncols = np.size(coredata,axis=1)

    # Add a little noise to head off all zeros.  noise is > 0.
    noise = npr.RandomState(seed=999).rand(nrows,ncols) / 1000.0
    abs_resid_ = np.where(coredata == nanval, nanval,
                          np.clip(np.abs(coredata) + noise, 0.0, np.inf))  #**2

    # Convert to datadict and Damon
    abs_resid = {}
    for key in resid.keys():
        abs_resid[key] = resid[key]
        abs_resid['coredata'] = abs_resid_
        abs_resid['validchars'] = ['All', ['0.0 -- '], 'Num']
    
    # Get EAR anchor specs, if available.  Note that 'ear_coord' is specified
    try:
        coord_anc = self.coord_out['anchors']
    except AttributeError:
        coord_anc = None
    
    out = tools.estimate_error(abs_resid, 'ear', coord_anc)
        
    return out
      



######################################################################

def _est2logit(_locals):
    "Basis of the est2logit method."

    # Get _locals
    self = _locals['self']
    estimates = _locals['estimates']
    ECutMaxPos_ = _locals['ecutmaxpos']
    logitform = _locals['logitform']
    obspercellmeth = _locals['obspercellmeth']

    # Get valid original observations for determining counts
    try:
        orig_obs = self.extract_valid_out['coredata'][:,:]
    except AttributeError:
        try:
            orig_obs = self.merge_info_out['coredata'][:,:]
        except AttributeError:
            try:
                orig_obs = self.data_out['coredata'][:,:]
            except AttributeError:
                exc = 'Unable to find "observed" data.\n'
                raise est2logit_Error(exc)

    # Get estimates
    if estimates == 'base_est_out':
        try:
            EstimatesRCD = self.base_est_out
        except AttributeError:
            exc = 'Unable to find base_est() outputs.\n'
            raise est2logit_Error(exc)
    elif estimates == 'fin_est_out':
        try:
            EstimatesRCD = self.fin_est_out
        except AttributeError:
            exc = 'Unable to find fin_est() outputs.\n'
            raise est2logit_Error(exc)
    elif estimates == 'equate_out':
        try:
            EstimatesRCD = self.equate_out['Construct']
        except AttributeError:
            exc = 'Unable to find equate() outputs.\n'
            raise est2logit_Error(exc)
    else:
        try:
            EstimatesRCD = self.__dict__[estimates]
        except KeyError:
            exc = 'Unable to find specified estimates datadict.\n'
            raise est2logit_Error(exc)

    # Get EARs
    if estimates == 'base_est_out':
        try:
            EARDict = self.base_ear_out
        except AttributeError:
            EARDict = None
            if self.verbose is True:
                print 'Warning in est2logit(): Could not find base_ear() outputs.  Calculating metric logits without them.\n'
    elif estimates == 'fin_est_out':
        try:
            EARDict = self.fin_ear_out
        except AttributeError:
            EARDict = None
            if self.verbose is True:
                print 'Warning in est2logit(): Could not find fin_ear() outputs.  Calculating metric logits without them.\n'
    elif estimates == 'equate_out':
        try:
            EARDict = self.equate_out['EAR']
        except AttributeError:
            EARDict = None
            if self.verbose is True:
                print 'Warning in est2logit(): Could not find fin_ear() outputs.  Calculating metric logits without them.\n'
    else:
        EARDict = None
        if self.verbose is True:
            print 'Warning in est2logit(): No EAR array available.  Calculating metric logits without them.\n'

    if EARDict is not None:
        EAR = EARDict['coredata']

    # Get ecutmaxpos
    if ECutMaxPos_ == 'Auto':
        if estimates == 'base_est_out':
            EstParam = self.base_est_out['ecutmaxpos']
            if EstParam is None:
                ecutmaxpos = ['Cols',['Med','Max']]
            else:
                ecutmaxpos = EstParam
        else:
            ecutmaxpos = ['Cols',['Med','Max']]
    else:
        ecutmaxpos = ECutMaxPos_

    # Define variables
    Est = EstimatesRCD['coredata']
    nanval = EstimatesRCD['nanval']
#    ncols = np.size(Est,axis=1)

    # Get column keys
    colkeys = tools.getkeys(EstimatesRCD,'Col','Core','Auto',None)


    #####################
    ##   Calculate     ##
    ##  Probabilities  ##
    ##     EARs        ##
    #####################

    # ecut param for cumnormprob()
    if ecutmaxpos[0] == 'All':
        ecut_ = [ecutmaxpos[0], ecutmaxpos[1][0]]
    elif ecutmaxpos[0] == 'Cols' and isinstance(ecutmaxpos[1], dict):
        col_dict = {}
        for k in ecutmaxpos[1]:
            col_dict[k] = ecutmaxpos[1][k][0]
        ecut_ = [ecutmaxpos[0], col_dict]
    elif ecutmaxpos[0] == 'Cols' and not isinstance(ecutmaxpos[1], dict):
        ecut_ = [ecutmaxpos[0], ecutmaxpos[1][0]]
        
    # metric logit, no errors
    if ((logitform == 'Metric'
        or logitform == 'Statistical')
        and EARDict is None
        ):
        MProbOut = tools.metricprob(estimates = Est,  # [array of estimates for which we want a cumulative probability]
                                    colkeys = colkeys,    # [1-D array of column keys]
                                    ecutmaxpos = ecutmaxpos, # [<['All',[ecut,MaxPos]], ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                                    pcut = 0.50, # [Probability separating "success" from "failure"]
                                    logits = True,    # [<None, True> => return logits instead of probabilities]
                                    nanval = nanval, # [Not-a-Number Value]
                                    )

        LogOut = MProbOut['Logit']
        ProbOut = MProbOut['Prob']
        LogEAR = None
        LogSE = None

        if logitform == 'Statistical':
            print "Warning in est2logit(): To use logitform = 'Statistical', EARs are needed.  Since they are missing, computing 'Metric' logits instead.\n"

    # metric logit, with errors (use cumnormprob(), EARs fixed at col mean)
    elif logitform == 'Metric' and EARDict is not None:
        EAR_ma = npma.masked_values(EAR,nanval)

        # Get mean column EARs, apply to cells
        EARMeans = npma.mean(EAR_ma,axis=0)
        FixedEAR = np.zeros(np.shape(EAR))
        FixedEAR[:,:] = EARMeans

        # Get measures
        CumProbOut = tools.cumnormprob(estimates = Est,  # [array of estimates for which we want a cumulative probability]
                                       ear = FixedEAR,    # [array of Expected Absolute residuals]
                                       colkeys = colkeys,    # [<None, 1-D array of column keys>]
                                       ecut = ecut_, # [<['All', ecut], ['Cols',{'ID1':ecut, ...}]> ]
                                       logits = True,   # [<None, True> => return logits with probabilities]
                                       nanval = nanval, # [Not-a-Number Value]
                                       )

        LogOut = CumProbOut['Logit']
        ProbOut = CumProbOut['Prob']

        # Get logit EARs by brute force
        NaNLoc = np.where(LogOut == nanval)
        EstPlusEAR = np.where(LogOut >= 0.0,Est + EAR,Est - EAR)
        EstPlusEAR[NaNLoc] = nanval

        LogPlusEAR = tools.cumnormprob(estimates = EstPlusEAR,  # [array of estimates for which we want a cumulative probability]
                                       ear = FixedEAR,    # [array of Expected Absolute residuals]
                                       colkeys = colkeys,    # [1-D array of column keys]
                                       ecut = ecut_, # [<['All',[ecut,MaxPos]], ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                                       logits = True,   # [<None, True> => return logits with probabilities]
                                       nanval = nanval, # [Not-a-Number Value]
                                       )['Logit']


        ValLoc = np.where(np.logical_and(LogOut != nanval,LogPlusEAR != nanval))
        LogEAR = np.zeros(np.shape(EAR)) + nanval
        LogEAR[ValLoc] = np.abs(LogPlusEAR[ValLoc] - LogOut[ValLoc])

    # Statistical logit, with errors (use cumnormprob(), EARs different for each cell)
    elif logitform == 'Statistical' and EARDict is not None:

        # Get measures
        LogProbOut = tools.cumnormprob(estimates = Est,  # [array of estimates for which we want a cumulative probability]
                                       ear = EAR,    # [array of Expected Absolute residuals]
                                       colkeys = colkeys,    # [1-D array of column keys]
                                       ecut = ecut_, # [<['All',[ecut,MaxPos]], ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                                       logits = True,   # [<None, True> => return logits with probabilities]
                                       nanval = nanval, # [Not-a-Number Value]
                                       )

        LogOut = LogProbOut['Logit']
        ProbOut = LogProbOut['Prob']

        # Get EARs using 1 / sqrt(p * (1 - p))  Clip to mitigate extreme logits
        LogEAR = np.where(ProbOut == nanval, nanval,
                          1 / np.clip(np.sqrt(ProbOut * (1.0 - ProbOut)),
                                      0.005, 1.0))

    #####################
    ##   Calculate     ##
    ##      SEs        ##
    #####################

    if LogEAR is not None:

        try:
            ndim = self.coord_out['ndim']
        except AttributeError:
            try:
                ndim = self.sub_coord_out['ndim']
            except AttributeError:
                try:
                    ndim = self.objectify_out['obj_est']['ndim']
                except AttributeError:
                    exc = 'Could not find coord_out, sub_coord_out, or objectify_out in order to obtain dimensionality.\n'
                    raise est2logit_Error(exc)

        # Get count of observations per cell
        opc_fact = tools.obspercell(orig_obs, 'obs', 'obs', 'arr', ndim,
                                    self.facs_per_ent, count_chars=False,
                                    max_chars=5, p_items=None,
                                    meth='CombineFacs', nanval=nanval)
        obspercell_factor = opc_fact

        # Calculate standard errors
        LogSE = np.where(LogEAR == nanval, nanval,
                         obspercell_factor * LogEAR)

    else:
        LogSE = None


    ####################
    ##   Assemble     ##
    ##   DataDicts    ##
    ####################

    # Logit estimates
    LogitEstDict = {}
    for key in EstimatesRCD.keys():
        LogitEstDict[key] = EstimatesRCD[key]

    LogitEstDict['coredata'] = LogOut
    LogitEstDict['validchars'] = ['All',['All'],'Num']

    # Probabilities
    ProbDict = {}
    for key in EstimatesRCD.keys():
        ProbDict[key] = EstimatesRCD[key]

    ProbDict['coredata'] = ProbOut
    ProbDict['validchars'] = ['All',['0.0 -- 1.0'],'Num']

    # Store Prob datadict, source estimates name, in logit datadict
    LogitEstDict['Prob'] = ProbDict
    LogitEstDict['SourceEst'] = estimates

    # EARs
    LogitEARDict = {}
    for key in EstimatesRCD.keys():
        LogitEARDict[key] = EstimatesRCD[key]

    LogitEARDict['coredata'] = LogEAR
    LogitEARDict['validchars'] = ['All',['All'],'Num']

    # SEs
    LogitSEDict = {}
    for key in EstimatesRCD.keys():
        LogitSEDict[key] = EstimatesRCD[key]

    LogitSEDict['coredata'] = LogSE
    LogitSEDict['validchars'] = ['All',['All'],'Num']

    # Assign to object
    self.logit_ear_out = LogitEARDict
    self.logit_se_out = LogitSEDict

    return LogitEstDict



######################################################################

def _base_se(_locals):
    "Basis of the base_se() method."

    # Get Local variables
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh
    obspercellmeth = _locals['obspercellmeth']
    max_chars = 2
    p_items = None

    # Get valid original observations for determining counts
    try:
        orig_obs = self.extract_valid_out['coredata'][:,:]
    except AttributeError:
        try:
            orig_obs = self.merge_info_out['coredata'][:,:]
        except AttributeError:
            try:
                orig_obs = self.data_out['coredata'][:,:]
            except AttributeError:
                exc = 'Unable to find "observed" data.\n'
                raise base_se_Error(exc)

    # Deal with parsed items
    try:
        obs = self.parse_out['coredata'][:, :]
        p_items = np.size(orig_obs, axis=1) / float(np.size(obs, axis=1))
        obs_nanval = self.parse_out['nanval']
    except AttributeError:
        obs = orig_obs
        p_items = None
        obs_nanval = self.nanval
        
    # Get residuals
    try:
        data = self.base_resid_out
    except AttributeError:
        exc = 'Unable to find base_resid() outputs.\n'
        raise base_se_Error(exc)

#    coredata = data['coredata'][:,:]
    nanval = data['nanval']

    # Get EAR info
    try:
        EAROut = self.base_ear_out
    except AttributeError:
        exc = 'Unable to find base_ear() output.\n'
        raise base_se_Error(exc)

    EAR = EAROut['coredata']
    
    # Calculate obspercell_factor
    try:
        ndim = self.coord_out['ndim']
    except AttributeError:
        try:
            ndim = self.sub_coord_out['ndim']
        except AttributeError:
            try:
                ndim = self.objectify_out['obj_est']['ndim']
            except AttributeError:
                exc = 'Could not find coord_out, sub_coord_out, or objectify_out in order to obtain dimensionality.\n'
                raise base_se_Error(exc)
                
    # Note:  For anchored designs, obspercell is still calculated in order
    # to provide input to coord() below.  The resulting row coords should
    # be multiplied by the col params to get person se's.
    opc = tools.obspercell(obs = obs, # [<None, obs array> ]
                           by_rows = 'obs',   # [<'obs', ncols> => for obs counts by row]
                           by_cols = 'obs',  # [<'obs', nrows>] => for obs counts by col]
                           out_as = 'arr', # [<'arr', 'row', 'col', 'int'> => output shape]
                           ndim = ndim,    # [<int> dimensionality of estimates]
                           facs_per_ent = self.facs_per_ent,    # [<[<0,1>,<1,2,[1,1,2,...]]> => facet and number of unanchored facets per entity]
                           count_chars = True,   # [<bool, int, [ints]> => count n unique characters per cell]
                           max_chars = max_chars, # [<int>, maximum chars per cell]
                           p_items = p_items, # [<None, float] => percent items independent]
                           meth = obspercellmeth, # [<'PickMinFac','CombineFacs'>]
                           nanval = obs_nanval,   # [Not-a-Number value]
                           )
    obspercell_factor = opc

    # Calc SE:  SE = EAR / sqrt( sqrt(nRowDats)*sqrt(nColDats) )
    SE = np.zeros(np.shape(EAR)) + nanval
    ValLoc = np.where(np.logical_and(EAR[:,:] != nanval,
                                     obspercell_factor != nanval))

    # Applies a binomial correction if the data are ordinal (see tools.pq_resid)
    pq_resid = True
    if (pq_resid 
        and hasattr(self, 'standardize_out')
        and self.standardize_out['stdmetric'] == 'PreLogit'):
        est = self.base_est_out
        ear = 1  # Magic Number -- adjusts prelogit errors, not fully understood
    
        # Separate ecuts per col not implemented
        EAR = tools.pq_resid(est['coredata'], 
                             resid=EAR, 
                             colkeys=tools.getkeys(est, 'Col', 'Core'),
                             ecut=['All', 0.0],
                             ear=ear,
                             new_logits=False,
                             validchars=self.validchars,
                             nanval=nanval)['new_resid']

    SE[ValLoc] = EAR[ValLoc] * obspercell_factor[ValLoc]
    SE = np.where(np.logical_or(np.isnan(SE), np.isinf(SE)), nanval, SE)

    # Build datadict
    SE_RCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in EAROut.keys():
        if key in ValList:
            SE_RCD[key] = EAROut[key]
    SE_RCD['coredata'] = SE

    ###########################
    # Calculate SE coordinates and recalculate SE accordingly
    try:
        coord_anc = self.coord_out['anchors']
    except AttributeError:
        coord_anc = None
    
    se_dict = tools.estimate_error(SE_RCD, 'se', coord_anc)
    se_dict['obspercell_factor'] = obspercell_factor
          
    return se_dict
         




######################################################################

def _base_fit(_locals):
    "Basis of the base_fit() method."

    # Get self
    self = _locals['self']
    ear = _locals['ear']
    pytables = self.pytables
    fileh = self.fileh

    # Check if rasch() was run
    try:
        self.rasch_out
        if self.verbose is True:
            print ('Warning in base_fit(): Overwriting the base_fit_out '
                   'already created by rasch().\n')
    except AttributeError:
        pass

    # Get variables
    nanval = float(self.nanval)

    try:
        Resid = self.base_resid_out
    except AttributeError:
        exc = 'Unable to find base_resid() outputs.\n'
        raise base_fit_Error(exc)

    try:
        EAR = self.base_ear_out
    except AttributeError:
        exc = 'Unable to find base_ear() outputs.\n'
        raise base_fit_Error(exc)

    # Refine ear param
    if ear is None:
        ear = EAR['coredata'][:,:]
    elif ear == 'median':
        valix = EAR['coredata'] != nanval
        ear = np.median(EAR['coredata'][valix])
    elif ear == 'mean':
        valix = EAR['coredata'] != nanval
        ear = np.mean(EAR['coredata'][valix])
        
    # Calculate fit
    fit = tools.fit(None, None, ear, Resid['coredata'][:,:],
                    None, None, nanval)['cellfit']

    if pytables is not None:
        fit = tools.pytables_(fit,'array',fileh,None,'base_fit_out',
                                 ['coredata'],None,None,None,None,None)['arrays']['coredata']

    # Build datadict
    FitDict = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in Resid.keys():
        if key in ValList:
            FitDict[key] = Resid[key]
        FitDict['coredata'] = fit

    return FitDict



######################################################################

def _fin_fit(_locals):
    "Basis of the fin_fit() method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh
    nanval = float(self.nanval)

    try:
        Resid = self.fin_resid_out
    except AttributeError:
        exc = 'Unable to find fin_resid() outputs.\n'
        raise fin_fit_Error(exc)

    EAR = self.fin_ear_out
    if EAR is None:
        exc = ('Unable to find fin_ear_out. Remember to run '
               'base_se().  See fin_est() docs.\n')
        raise fin_fit_Error(exc)

    # Calculate fit
    fit = tools.fit(None, None, EAR['coredata'][:,:], Resid['coredata'][:,:],
                    None, None, nanval)['cellfit']

    if pytables is not None:
        fit = tools.pytables_(fit,'array', fileh, None, 'fin_fit_out',
                              ['coredata'], None, None, None, None,
                              None)['arrays']['coredata']

    # Build datadict
    FitDict = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in Resid.keys():
        if key in ValList:
            FitDict[key] = Resid[key]
        FitDict['coredata'] = fit

    return FitDict




######################################################################

def _fin_est(_locals):
    "Basis of the fin_est method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Retrieve _locals variables
    OrigData_ = _locals['orig_data']
    stdmetric = _locals['stdmetric']
    ents2restore = _locals['ents2restore']
    referto = _locals['referto']
    continuous = _locals['continuous']
    std_params = _locals['std_params']
    StdParsed = None #_locals['StdParsed']   # Deleted from main method docs on a trial basis.  Post-standardization can be done using est2logit().
    RespAlpha = _locals['alpha']

    ####################
    ##      Get       ##
    ##    stdmetric   ##
    ####################

    if stdmetric == 'Auto':
        try:
            self.est2logit_out
            stdmetric = 'Logit'
        except AttributeError:
            try:
                stdmetric = self.standardize_out['stdmetric']
            except AttributeError:
                exc = 'Unable to find the standardization metric. Run standardize() or specify a metric.\n'
                raise fin_est_Error(exc)


    ####################
    ##      Get       ##
    ##  Standardized  ##
    ##     data       ##
    ####################

    try:
        StdData = self.est2logit_out
        if (stdmetric == 'PMinMax'
            or stdmetric == '0-1'
            or stdmetric == 'Percentile'
            ):
            StdData = self.est2logit_out['Prob']
    except AttributeError:
        try:
            StdData = self.base_est_out
        except AttributeError:
            try:
                StdData = self.standardize_out
            except AttributeError:
                exc = ('Unable to find data for fin_est().  Run standardize() or '
                       'estimate() first.\n')
                raise fin_est_Error(exc)

    # Define StdData variables (handles pytables)
    coredata = StdData['coredata']
    rowlabels = StdData['rowlabels']
    collabels = StdData['collabels']
    nheaders4rows = StdData['nheaders4rows']
    key4rows = StdData['key4rows']
    rowkeytype = StdData['rowkeytype']
    nanval = StdData['nanval']
    nheaders4cols = StdData['nheaders4cols']
    key4cols = StdData['key4cols']
    colkeytype = StdData['colkeytype']
    LogDatMin = 0.0001

    # Get row containing column entity IDs, response options
    try:
        EntRow = self.parse_out['EntRow']
        RespRow = self.parse_out['RespRow']
    except AttributeError:
        EntRow = key4cols
        RespRow = None

    # Force referto = 'Cols' if data were parsed
    try:
        self.parse_out
        if referto == 'Whole':
            print ("Warning: fin_est() sees the data were parsed, so "
                   "referto = 'Whole' has been changed to 'Cols'. ('Whole' is not "
                   "supported in this context.)  Consider making referto = 'Cols' "
                   "in standardize().\n")
        referto = 'Cols'
    except AttributeError:
        pass


    ####################
    ##      Get       ##
    ##    Original    ##
    ##      data      ##
    ####################

    StdParamsFlag = False   # Until switched on

    if (OrigData_ != 'std_params'
        ):
        # Define original data
        if OrigData_ == 'parse_out':
            try:
                orig_data = self.parse_out
            except AttributeError:
                exc = 'Unable to find parse() outputs.\n'
                raise fin_est_Error(exc)

        elif OrigData_ == 'subscale_out':
            try:
                orig_data = self.subscale_out
            except AttributeError:
                exc = 'Unable to find subscale() outputs.\n'
                raise fin_est_Error(exc)

        elif OrigData_ == 'score_mc_out':
            try:
                orig_data = self.score_mc_out
            except AttributeError:
                exc = 'Unable to find score_mc() outputs.\n'
                raise fin_est_Error(exc)

        elif OrigData_ == 'data_out':
            try:
                orig_data = self.extract_valid_out
                try:
                    orig_data = self.score_mc_out
                    print "Warning in fin_est(): Your original data was nominal, scored using score_mc(). Changing orig_data to 'score_mc'.\n"
                    OrigData_ = 'score_mc'
                except AttributeError:
                    pass
            except AttributeError:
                try:
                    orig_data = self.merge_info_out
                    try:
                        orig_data = self.score_mc_out
                        print "Warning in fin_est(): Your original data was nominal, scored using score_mc(). Changing orig_data to 'score_mc'.\n"
                        OrigData_ = 'score_mc'
                    except AttributeError:
                        pass
                except AttributeError:
                    try:
                        orig_data = self.data_out
                        try:
                            orig_data = self.score_mc_out
                            print "Warning in fin_est(): Your original data was nominal, scored using score_mc(). Changing orig_data to 'score_mc'.\n"
                            OrigData_ = 'score_mc'
                        except AttributeError:
                            pass
                    except AttributeError:
                        exc = 'Unable to find original data.\n'
                        raise fin_est_Error(exc)

        else:
            exc = 'Unable to resolve orig_data parameter.\n'
            raise fin_est_Error(exc)

        # Extract orig_data variables (handles pytables)
        OrigCoreData = orig_data['coredata']
        OrigRowLabels = orig_data['rowlabels']
        OrigColLabels = orig_data['collabels']

        OrignHeaders4Rows = orig_data['nheaders4rows']
        OrignHeaders4Cols = orig_data['nheaders4cols']
        OrigKey4Rows = orig_data['key4rows']
        OrigKey4Cols = orig_data['key4cols']
        OrigRowKeyType = orig_data['rowkeytype']
        OrigColKeyType = orig_data['colkeytype']
        OrigValidChars = orig_data['validchars']
        OrigNaNVal = float(orig_data['nanval'])      # String NaNVals don't work

    # Get std_params dictionary
    if (OrigData_ == 'std_params'
        and std_params is None
        ):
        try:
            std_params = self.standardize_out['std_params']
        except AttributeError:
            exc = 'Unable to find the std_params dictionary.\n'
            raise fin_est_Error(exc)

    # Overwrite top-level variables with StdParam variables, if necessary
    if OrigData_ == 'std_params':
        stdmetric = std_params['stdmetric']
        OrigValidChars = std_params['validchars']
        referto = std_params['referto']

        # Get original data from std_params
        if std_params['orig_data'] is not None:
            orig_data = std_params['orig_data']

            # Original data variables
            OrigCoreData = orig_data['coredata']
            OrigColLabels = orig_data['collabels']
            OrigRowLabels = orig_data['rowlabels']
            OrignHeaders4Rows = orig_data['nheaders4rows']
            OrignHeaders4Cols = orig_data['nheaders4cols']
            OrigKey4Rows = orig_data['key4rows']
            OrigKey4Cols = orig_data['key4cols']
            OrigRowKeyType = orig_data['rowkeytype']
            OrigColKeyType = orig_data['colkeytype']
            OrigValidChars = orig_data['validchars']     # Overwrite the overwrite above
            OrigNaNVal = float(orig_data['nanval'])

        # All destandardization will occur without looking at orig_data
        elif OrigData_ == 'std_params':
            StdParamsFlag = True

            # No original data available, so some data variables defined in terms of standardized variables
            # Not used until the end for labels
            OrigCoreData = coredata
            OrigColLabels = collabels
            OrigRowLabels = rowlabels
            OrignHeaders4Rows = nheaders4rows
            OrignHeaders4Cols = nheaders4cols
            OrigKey4Rows = key4rows
            OrigKey4Cols = key4cols
            OrigRowKeyType = rowkeytype
            OrigColKeyType = colkeytype
            OrigNaNVal = nanval

    # Get list of entities
    if referto == 'Cols':
        if StdParamsFlag is False:
            Ents = tools.getkeys(orig_data,'Col','Core','Auto',None)
        else:
            Ents = OrigColLabels[OrigKey4Cols, OrignHeaders4Rows:].astype(OrigColKeyType)

#            This screwed up column assignment
#            Ents = np.array(OrigValidChars[1].keys()).astype(OrigColKeyType)

        nEnts = np.size(Ents,axis=0)

    # array is treated as a single entity called 'All'
    elif referto == 'Whole':
        Ents = np.array(['All'])
#        nEnts = 1

    # Get number of rows
    nrows = np.size(OrigCoreData,axis=0)
    ncols = np.size(OrigCoreData,axis=1)
    nColLabelRows = np.size(OrigColLabels,axis=0)


    ####################
    ##      Get       ##
    ##    Original    ##
    ##     metric     ##
    ####################

    if referto == 'Whole':

        VCOut = tools.valchars(validchars = OrigValidChars,    # ['validchars' output of data() function]
                             dash = ' -- ', # [Expression used to denote a range]
                             defnone = 'interval',   # [How to interpret metric when validchars = None]
                             retcols = None,    # [<None, [list of core col keys]>]
                             )

        if VCOut['metric'][0] == 'Cols':

            # OrigMetric
            OrigMetricDict = {}
            FirstKey0 = VCOut['metric'][1].keys()[0]
            OrigMetricDict['All'] = VCOut['metric'][1][FirstKey0]
            OrigMetric = ['Cols',OrigMetricDict]         # 'Cols' to allow key lookups below

            # Round
            RoundDict = {}
            FirstKey1 = VCOut['round_'][1].keys()[0]
            RoundDict['All'] = VCOut['round_'][1][FirstKey1]
            Round = ['Cols',RoundDict]

            # Get min and max
            MinMaxDict = {}
            mm_vals = VCOut['minmax'][1].values()
            min_all = np.amin(mm_vals)
            max_all = np.amax(mm_vals)
            FirstKey2 = VCOut['minmax'][1].keys()[0]
            MinMaxDict['All'] = [min_all, max_all]
            minmax = ['Cols',MinMaxDict]

        elif VCOut['metric'][0] == 'All':

            # OrigMetric
            OrigMetricDict = {}
            OrigMetricDict['All'] = VCOut['metric'][1]
            OrigMetric = ['Cols',OrigMetricDict]         # 'Cols' to allow key lookups below

            # Round
            RoundDict = {}
            RoundDict['All'] = VCOut['round_'][1]
            Round = ['Cols',RoundDict]

            # Get min and max
            MinMaxDict = {}
            MinMaxDict['All'] = VCOut['minmax'][1]
            minmax = ['Cols',MinMaxDict]


    elif referto == 'Cols':
        VCOut = tools.valchars(validchars = OrigValidChars,    # ['validchars' output of data() function]
                             dash = ' -- ', # [Expression used to denote a range]
                             defnone = 'interval',   # [How to interpret metric when validchars = None]
                             retcols = Ents,    # [<None, [list of core col keys]>]
                             )

        OrigMetric = VCOut['metric']
        Round = VCOut['round_']
        minmax = VCOut['minmax']


    #################
    ##    Build    ##
    ##   MethDict  ##
    #################

    if referto == 'Whole':
        MethDict = {'All':['DeStd']}
        if ents2restore != 'All':
            print "Warning in fin_est(): If referto = 'Whole', ents2restore must be 'All'. Making that change.\n"
            ents2restore = 'All'

    elif referto == 'Cols':

        # Get from parse() or build
        try:
            MethDict = self.parse_out['MethDict']
            for Ent in Ents:
                if MethDict[Ent] == None:
                    MethDict[Ent] = ['DeStd']

        except AttributeError:
            MethDict = {}
            for Ent in Ents:
                MethDict[Ent] = ['DeStd']

        # Exclude specified entities
        if (ents2restore != 'All'
            and ents2restore[0] == 'AllExcept'
            ):
            ExcludedEnts = ents2restore[1]
            for Ent in Ents:
                if (Ent in ExcludedEnts
                    and MethDict[Ent] == ['Exp']
                    ):
                    # For ordinal, return response probs, not cumulative probs
                    MethDict[Ent] = ['Probs']

                elif Ent in ExcludedEnts:
                    MethDict[Ent] = None

        # Exclude specified entities
        elif (ents2restore != 'All'
              and ents2restore[0] == 'NoneExcept'
              ):
            IncludedEnts = ents2restore[1]
            for Ent in Ents:
                if (Ent not in IncludedEnts
                    and MethDict[Ent] == ['Exp']
                    ):
                    # For ordinal, return response probs, not cumulative probs
                    MethDict[Ent] = ['Probs']

                elif Ent not in IncludedEnts:
                    MethDict[Ent] = None


    #################
    ##     Get     ##
    ##   Standard  ##
    ##    Errors   ##
    ##     EARs    ##
    #################

    # Get standard errors only if base_se() was run
    # 'ObsPerCell' is actually a more complicated obspercell_factor.  See base_se() docs.
    GetSE = False
    EntCoreSE = None
    EntCoreEAR = None

    try:
        CoreDataSE = self.base_se_out['coredata']
        ObsPerCell = self.base_se_out['obspercell_factor']
        GetSE = True
        CoreDataEAR = self.base_ear_out['coredata']
    except AttributeError:
        pass


    ######################
    ##   INITIALIZE     ##
    ##    by Column     ##
    ##     Entity       ##
    ######################

    # Deparse methods
    DeparseMeths = ['Extr','Exp','Pred','Probs']

    # Initialize prediction dictionary
    PredDict = {}
    Warning1 = False
#    Warning2 = False

    # Initialize first column, no pytables
    if pytables is None:

        if referto == 'Cols':
            # Initialize coredata and collabels (create dummy lead column)
            NewCoreData = np.zeros((nrows,1),dtype=int)
            NewColLabels = np.zeros((nColLabelRows,1),dtype=int)

            # Initialize SE
            if GetSE is True:
                NewCoreDataSE = np.zeros((nrows,1))
                NewCoreDataEAR = np.zeros((nrows,1))
            else:
                NewCoreDataSE = None
                NewCoreDataEAR = None

        elif referto == 'Whole':
            nCells = np.size(coredata)
            NewCoreData = np.zeros((nCells,1))

            if GetSE is True:
                NewCoreDataEAR = np.copy(NewCoreData)
                NewCoreDataSE = np.copy(NewCoreData)
            else:
                NewCoreDataEAR = None
                NewCoreDataSE = None

    # Initialize pytables array
    else:
        if referto == 'Cols':
            NewCoreDataTab = tools.pytables_(None,'init_earray',fileh,None,'fin_est_out',
                                            ['coredata'],None,'float',4,(nrows,0),None)
            NewCoreData = NewCoreDataTab['arrays']['coredata']
            NewColLabels = np.zeros((nColLabelRows,1))  # No PyTable for collabels here

            if GetSE is True:
                NewCoreDataSETab = tools.pytables_(None,'init_earray',fileh,None,'fin_se_out',
                                         ['coredata'],None,'float',4,(nrows,0),None)
                NewCoreDataSE = NewCoreDataSETab['arrays']['coredata']

                NewCoreDataEARTab = tools.pytables_(None,'init_earray',fileh,None,'fin_ear_out',
                                          ['coredata'],None,'float',4,(nrows,0),None)
                NewCoreDataEAR = NewCoreDataEARTab['arrays']['coredata']
            else:
                NewCoreDataSE = None
                NewCoreDataEAR = None

        elif referto == 'Whole':
            nCells = np.size(coredata[:,:])
            NewCoreDataTab = tools.pytables_(None,'init_earray',fileh,None,'fin_est_out',
                                            ['coredata'],None,'float',4,(nCells,0),None)
            NewCoreData = NewCoreDataTab['arrays']['coredata']
            NewColLabels = np.zeros((nColLabelRows,1))  # No PyTable for collabels here

            if GetSE is True:
                NewCoreDataSETab = tools.pytables_(None,'init_earray',fileh,None,'fin_est_out',
                                                  ['SE'],None,'float',4,(nCells,0),None)
                NewCoreDataSE = NewCoreDataSETab['arrays']['SE']

                NewCoreDataEARTab = tools.pytables_(None,'init_earray',fileh,None,'fin_est_out',
                                                   ['EAR'],None,'float',4,(nCells,0),None)
                NewCoreDataEAR = NewCoreDataEARTab['arrays']['EAR']
            else:
                NewCoreDataSE = None
                NewCoreDataEAR = None


    ######################################################################################
    # Define needed functions prior to iterating through entities

    #####################
    ##   function to   ##
    ##  Convert Linear ##
    ##   Measures to   ##
    ##      Probs      ##
    #####################

    def linear2prob(Ent,entcore,EntCoreSE,EntCoreEAR):
        "Convert a linear stdmetric to a probability."

        AdjEntCore = np.zeros(np.shape(entcore)) + nanval
        ValLoc = np.where(entcore != nanval)

        if GetSE is True:
            ValLocSE = np.where(EntCoreSE != nanval)
            ValLocEAR = np.where(EntCoreEAR != nanval)
            AdjEntCoreSE = np.zeros(np.shape(EntCoreSE)) + nanval
            AdjEntCoreEAR = np.zeros(np.shape(EntCoreEAR)) + nanval

        # Get rescaling parameters, necessary here but not above
        if (StdParamsFlag is True
            and std_params['rescale'] is not None
            ):
            [m,b] = std_params['rescale'][Ent]
        else:
            m,b = 1,0

        # Get metric into logits
        if (stdmetric == 'SD'
            or stdmetric == 'LogDat'
            ):

            # Convert from SD units to logits using Pi / sqrt(3)
            PiSqrt3 = 1.81379936423422

            if stdmetric == 'LogDat':
                ValEntCore_LogDat = entcore[ValLoc]
                Mean = np.mean(ValEntCore_LogDat)
                SD = np.std(ValEntCore_LogDat)
                AdjEntCore[ValLoc] = PiSqrt3 * (entcore[ValLoc] - Mean) / SD

                if GetSE is True:
                    AdjEntCoreSE[ValLocSE] = PiSqrt3 * (EntCoreSE[ValLocSE]) / SD
                    AdjEntCoreEAR[ValLocEAR] = PiSqrt3 * (EntCoreEAR[ValLocEAR]) / SD

            elif stdmetric == 'SD':
                AdjEntCore[ValLoc] = PiSqrt3 * (entcore[ValLoc] - b) / m

                if GetSE is True:
                    AdjEntCoreSE[ValLocSE] = PiSqrt3 * (EntCoreSE[ValLocSE]) / m
                    AdjEntCoreEAR[ValLocEAR] = PiSqrt3 * (EntCoreEAR[ValLocEAR]) / m

        elif (stdmetric == 'Logit'
              or stdmetric == 'PreLogit'
              or stdmetric == 'PLogit'
              ):
            AdjEntCore[ValLoc] = (entcore[ValLoc] - b) / m

            if GetSE is True:
                AdjEntCoreSE[ValLocSE] = (EntCoreSE[ValLocSE]) / m
                AdjEntCoreEAR[ValLocEAR] = (EntCoreEAR[ValLocEAR]) / m

        else:
            exc = 'Unable to figure out stdmetric.\n'
            raise fin_est_Error(exc)

        # Initialize 0-1 outputs
        p_EntCore = np.copy(AdjEntCore)

        if GetSE is True:
            p_EntCoreSE = np.copy(AdjEntCoreSE)
            p_EntCoreEAR = np.copy(AdjEntCoreEAR)
        else:
            p_EntCoreSE = None
            p_EntCoreEAR = None

        # Case1:  convert to 0 - 1 without using the probability formula (dichotomous data)
        if OrigMetric[1][Ent] == 'ordinal':
            Min = minmax[1][Ent][0]
            Max = minmax[1][Ent][1]
            nCats = Max - Min + 1
            SigThresh = 2   # Magic number of categories below which the ordinal scale is treated like an interval scale

            if nCats <= SigThresh:
                MinEst = np.amin(AdjEntCore[ValLoc])
                MaxEst = np.amax(AdjEntCore[ValLoc])

                # PLogits are forced to extend from 0.0 to 1.0 (otherwise, they tend to be too narrow)
                if stdmetric == 'PLogit':
                    Min_p = 0.0
                    Max_p = 1.0

                # For other metrics, get min and max p from the estimates
                else:
                    Min_p = np.exp(MinEst) / (1.0 + np.exp(MinEst))
                    Max_p = np.exp(MaxEst) / (1.0 + np.exp(MaxEst))

                p_ = (AdjEntCore[ValLoc] - MinEst) / (MaxEst - MinEst)
                p_EntCore[ValLoc] = p_ * (Max_p - Min_p) + Min_p

                if GetSE is True:
                    k = (Max_p - Min_p) / float(MaxEst - MinEst)
                    p_EntCoreSE[ValLocSE] = AdjEntCoreSE[ValLocSE] * k
                    p_EntCoreEAR[ValLocEAR] = AdjEntCoreEAR[ValLocEAR] * k

            else:
                # Same as Case2
                p_EntCore[ValLoc] = np.exp(AdjEntCore[ValLoc]) / (1.0 + np.exp(AdjEntCore[ValLoc]))

                if GetSE is True:
                    p_EntCoreSE[ValLocSE] = (np.exp(AdjEntCore[ValLocSE]) * AdjEntCoreSE[ValLocSE]) / ((1.0 + np.exp(AdjEntCore[ValLocSE]))**2)
                    p_EntCoreEAR[ValLocEAR] = (np.exp(AdjEntCore[ValLocEAR]) * AdjEntCoreEAR[ValLocEAR]) / ((1.0 + np.exp(AdjEntCore[ValLocEAR]))**2)

        # Case2: convert to 0 - 1 using the p = exp(x) / (1 + exp(x)) formula
        else:
            p_EntCore[ValLoc] = np.exp(AdjEntCore[ValLoc]) / (1.0 + np.exp(AdjEntCore[ValLoc]))

            if GetSE is True:
                ValLoc1 = np.where(np.logical_and(AdjEntCore != nanval,
                                                 AdjEntCoreSE != nanval,
                                                 AdjEntCoreEAR != nanval)
                                   )
                p_EntCoreSE[ValLoc1] = (np.exp(AdjEntCore[ValLoc1]) * AdjEntCoreSE[ValLoc1]) / ((1.0 + np.exp(AdjEntCore[ValLoc1]))**2)
                p_EntCoreEAR[ValLoc1] = (np.exp(AdjEntCore[ValLoc1]) * AdjEntCoreEAR[ValLoc1]) / ((1.0 + np.exp(AdjEntCore[ValLoc1]))**2)

        return {'p_EntCore':p_EntCore,'p_EntCoreSE':p_EntCoreSE,'p_EntCoreEAR':p_EntCoreEAR}


    #####################
    ##   function to   ##
    ##  Convert Probs  ##
    ##    to Linear    ##
    ##    Measures     ##
    #####################

    def prob2linear(ProbCore,stdmetric):
        "Convert deparse probabilities to stdmetric."

        LinOut = np.zeros(np.shape(ProbCore)) + nanval
        ValLoc = np.where(ProbCore != nanval)

        # Convert to logits
        Probs = np.clip(ProbCore[ValLoc],0.0001,0.9999)
        Log = np.log(Probs / (1.0 - Probs))
        PiSqrt3 = 1.81379936423422

        # Logit metric
        if (stdmetric == 'Logit'
            or stdmetric == 'PreLogit'
            or stdmetric == 'PLogit'
            ):
            LinOut[ValLoc] = Log

        elif stdmetric == 'SD':
            LinOut[ValLoc] = Log / PiSqrt3

        else:
            LinOut = ProbCore

        return LinOut
    ######################################################################################
    # End of defined functions


    # Destandardize each entity
    ######################
    for Ent in Ents:

        # Locate entity in standardized array
        EntRowElems = collabels[EntRow,nheaders4rows:].astype(OrigColKeyType)
        EntLoc = np.where(EntRowElems == Ent)

        # Locate entity in original array
        if StdParamsFlag is False:
            OrigEntRowElems = tools.getkeys(orig_data,'Col','Core','Auto',None)
            OrigEntLoc = np.where(OrigEntRowElems == Ent)

        # Define entity data for 'Whole' (not allowed when data contains parsing).
        if referto == 'Whole':
            entcore = np.ravel(coredata[:,:])[:,np.newaxis]

            if GetSE is True:
                EntCoreSE = np.ravel(CoreDataSE[:,:])[:,np.newaxis]
                EntObsPerCell = np.ravel(ObsPerCell[:,:])[:,np.newaxis]
                EntCoreEAR = np.ravel(CoreDataEAR[:,:])[:,np.newaxis]

            if StdParamsFlag is False:
                try:
                    OrigEntCore = np.ravel(OrigCoreData).astype(float)
                except ValueError:
                    OrigEntCore = np.ravel(OrigCoreData)

        # Define data for one entity
        elif referto == 'Cols':
            entcore = coredata[:,EntLoc[0]]

            if GetSE is True:
                EntCoreSE = CoreDataSE[:,EntLoc[0]]
                EntObsPerCell = ObsPerCell[:,EntLoc[0]]
                EntCoreEAR = CoreDataEAR[:,EntLoc[0]]

            if StdParamsFlag is False:
                try:
                    OrigEntCore = OrigCoreData[:,OrigEntLoc[0]].astype(float)
                except ValueError:
                    OrigEntCore = OrigCoreData[:,OrigEntLoc[0]]


        ######################
        # Keep entity as is:  no destandardizing or deparsing
        if (MethDict[Ent] is None
            or all(np.ravel(entcore[:,:]) == nanval)
            ):
            NewCore = entcore

            if GetSE is True:
                NewCoreSE = EntCoreSE
                NewCoreEAR = EntCoreEAR

            # Get column labels
            try:
                NewLabels = OrigColLabels[:,OrigEntLoc[0] + nheaders4rows]
            except NameError:
                NewLabels = collabels[:,EntLoc[0] + nheaders4rows]

            nRepeatCols = np.size(NewCore,axis=1)
            NewLabels = np.repeat(NewLabels,nRepeatCols,axis=1)


        ######################
        # Keep entity as is:  deparsing only.  'DeStd' just passes through the standardized data.
        elif (stdmetric is None
            and OrigData_ is not 'std_params'
            and MethDict[Ent] == ['DeStd']
            ):
            NewCore = entcore

            if GetSE is True:
                NewCoreSE = EntCoreSE
                NewCoreEAR = EntCoreEAR

            # Get column labels
            try:
                NewLabels = OrigColLabels[:,OrigEntLoc[0] + nheaders4rows]
            except NameError:
                NewLabels = collabels[:,EntLoc[0] + nheaders4rows]

            nRepeatCols = np.size(NewCore,axis=1)
            NewLabels = np.repeat(NewLabels,nRepeatCols,axis=1)


        ################
        ##  Deparse   ##
        ##  Approach  ##
        ################

        elif MethDict[Ent][0] in DeparseMeths:

            # Get parameters for resp_prob
            resp_cats = list(collabels[:,nheaders4rows:][RespRow,EntLoc[0]])
            return_ = MethDict[Ent][0]
            extr_est = None

            # If the standardized metric is unbounded
            if (stdmetric == 'SD'
                or stdmetric == 'LogDat'
                or stdmetric == 'Logit'
                or stdmetric == 'PreLogit'
                or stdmetric == 'PLogit'
                ):
                if (return_ == 'Extr'
                     and StdParsed is True
                    ):
                    p_EntCore = entcore
                    extr_est = True

                    if GetSE is True:
                        p_EntCoreSE = EntCoreSE
                        p_EntCoreEAR = EntCoreEAR
                    else:
                        p_EntCoreSE = None
                        p_EntCoreEAR = None
                else:
                    lin2_out = linear2prob(Ent,entcore,EntCoreSE,EntCoreEAR)
                    p_EntCore = lin2_out['p_EntCore']

                    if GetSE is True:
                        p_EntCoreSE = lin2_out['p_EntCoreSE']
                        p_EntCoreEAR = lin2_out['p_EntCoreEAR']
                    else:
                        p_EntCoreSE = None
                        p_EntCoreEAR = None

            else:
                p_EntCore = entcore

                if GetSE is True:
                    p_EntCoreSE = EntCoreSE
                    p_EntCoreEAR = EntCoreEAR
                else:
                    p_EntCoreSE = None
                    p_EntCoreEAR = None


            #####################
            ##  Apply deparse  ##
            ##    method       ##
            #####################

            # resp2extr
            if return_ == 'Extr':
                resp2extr = MethDict[Ent][1]
                GetSD = None
            else:
                resp2extr = None

            # pred_key
            if return_ == 'Pred':
                Ints = range(len(resp_cats))
                pred_key = dict(zip(resp_cats,Ints))
                GetSD = None
            else:
                pred_key = None

            # metric
            if return_ == 'Exp':
                RptMetric = 'ordinal'
                if GetSE is True:
                    GetSD = 'ExpSD'
                else:
                    GetSD = None
            else:
                RptMetric = 'nominal'

            # Calc destandardized data for entity
            RespProbOut = tools.resp_prob(entcore = p_EntCore,    # [nrows x nRespInts-1 2D array of probabilities corresponding to a given column entity]
                                          resp_cats = resp_cats,   # [list of valid response integers in increasing order less the minimum possible integer, or list of alpha responses]
                                          return_ = [return_,GetSD],     # [<['Extr','Exp','Pred','Probs','ExpSD']> => list desired]
                                          resp2extr = resp2extr,  # [<None, response whose probability to extract>]
                                          extr_est = extr_est,    # [<None,True> => extract from entcore instead of probs]
                                          pred_key = pred_key,   # [<None,{'a':0,'b':1,...}> => dict relating string responses to ints]
                                          metric = RptMetric,   # [<'ordinal','nominal'>]
                                          dropcol = None,  # [<None,True> => drop lead column of ordinal probabilities]
                                          nanval = nanval, # [float Not-a-Number value]
                                          )

            # Handle case where resp_prob is forced to return 'Pred' instead of 'Exp'
            if return_ == 'Exp' and RespProbOut['Exp'] is None:
                return_ = 'Pred'

            # Extract desired report
            if StdParsed is not True:
                NewCore = RespProbOut[return_]
            else:
                if return_ == 'Extr':
                    NewCore = RespProbOut[return_]

                elif return_ == 'Exp':
                    MinCat = float(min(resp_cats)) - 1.0
                    MaxCat = float(max(resp_cats))
                    Exp2Prob = (RespProbOut[return_] - MinCat) / float(MaxCat - MinCat)
                    NewCore = prob2linear(Exp2Prob,stdmetric)

                # return_ prob of most likely response in stdmetric
                else:
                    PredProbs = RespProbOut['PredProb']
                    Probs = np.array([PredProb[1] for PredProb in PredProbs])[:,np.newaxis]
                    NewCore = prob2linear(Probs,stdmetric)

            # Get standard errors:  if return_ == 'Exp' use resp_prob SD, else use base_se_out
            if GetSE is True:

                if return_ == 'Extr':
                    ExtrLoc = RespProbOut['ExtrIndex']
                    NewCoreSE = p_EntCoreSE[:,ExtrLoc]
                    NewCoreEAR = p_EntCoreEAR[:,ExtrLoc]

                elif return_ == 'Pred':
                    PredInd = RespProbOut['PredIndex']

                    def pullse(Row):
                        if PredInd[Row] == nanval:
                            return nanval
                        else:
                            return [p_EntCoreSE[Row][PredInd[Row]][0],p_EntCoreEAR[Row][PredInd[Row]][0]]

                    NewCoreSE = np.array([pullse(Row)[0] for Row in range(nrows)])[:,np.newaxis]
                    NewCoreEAR = np.array([pullse(Row)[1] for Row in range(nrows)])[:,np.newaxis]

                    if StdParsed is True:
                        print 'Warning in fin_est():  SE and EAR are not yet supported for the fin_est() StdParsed option.  Reporting the base_se() and base_ear() errors.\n'

                elif return_ == 'Exp':
                    ExpSD = RespProbOut['ExpSD']
                    NewCoreEAR = ExpSD
                    NewCoreSE = ExpSD * EntObsPerCell[:,0][:,np.newaxis]   # Just first column of EntObsPerCell, i.e. obspercell_factor K

                    if StdParsed is True:
                        print 'Warning in fin_est():  SE and EAR are not yet supported for the fin_est() StdParsed option.  Reporting the base_se() and base_ear() errors.\n'

                else:
                    exc = 'Having trouble with the MethDict from parse().\n'
                    raise fin_est_Error(exc)

            # Handle prediction info
            if return_ == 'Pred':
                PredDict[Ent] = {'pred_key':RespProbOut['pred_key'],
                                 'pred_prob':RespProbOut['PredProb']
                                 }

            # Get column labels
            try:
                NewLabels = OrigColLabels[:,OrigEntLoc[0] + nheaders4rows]
            except NameError:
                NewLabels = collabels[:,EntLoc[0] + nheaders4rows]

            nRepeatCols = np.size(NewCore,axis=1)
            NewLabels = np.repeat(NewLabels,nRepeatCols,axis=1)

            # Assign keys to repeater columns
            if nRepeatCols > 1:
                colkeys = collabels[key4cols,EntLoc[0] + nheaders4rows]
                NewLabels[OrigKey4Cols,:] = colkeys

            # NOTE:  "Append to existing" is handled at the end


        #####################
        ##    Regular      ##
        ##  Destandardize  ##
        ##    Approach     ##
        #####################

        elif MethDict[Ent][0] == 'DeStd':
            ValLoc = np.where(entcore != nanval)
            NewCore = np.zeros(np.shape(entcore)) + nanval
            NewCoreSE = np.zeros(np.shape(entcore)) + nanval
            NewCoreEAR = np.zeros(np.shape(entcore)) + nanval

            ###################################
            # Originally ratio or interval data
            if (OrigMetric[1][Ent] == 'interval'
                or OrigMetric[1][Ent] == 'ratio'
                ):

                # If the standardized metric is unbounded
                if (stdmetric == 'SD'
                    or stdmetric == 'LogDat'
                    or stdmetric == 'Logit'
                    or stdmetric == 'PreLogit'
                    or stdmetric == 'PLogit'
                    ):
                    ValEntCore = entcore[ValLoc]

                # If the standardized metric is bounded between 0 and 1
                elif (stdmetric == '0-1'
                      or stdmetric == 'PMinMax'
                      or stdmetric == 'Percentile'
                      ):

                    # Linearize probabilities (but check to make sure they are)
                    TryProbFlag = True
                    try:
                        self.coord_out
                    except AttributeError:
                        TryProbFlag = False
                        pass

                    if TryProbFlag is True:
                        try:
                            self.est2logit_out['Prob']
                        except AttributeError:
                            exc = 'Expects probabilities with this stdmetric. Run est2logit() first.\n'
                            raise fin_est_Error(exc)

                    ValEntCore = np.log((entcore[ValLoc]) / (1.0 - entcore[ValLoc]))

                # Pull Mean/SD parameters from std_params
                if (StdParamsFlag is True
                    and (std_params['stdmetric'] == 'SD'
                         or std_params['stdmetric'] == 'LogDat'
                         or std_params['stdmetric'] == 'PreLogit'
                         or std_params['stdmetric'] == '0-1'
                         )
                    ):
                    try:
                        if std_params['params'][Ent] != 'Refer2VC':
                            OrigMean = std_params['params'][Ent][0]
                            OrigSD = std_params['params'][Ent][1]
                        else:
                            OrigMean = 0.0
                            OrigSD = 1.0
                            #print 'Warning: std_params[params] did not provide a Mean/SD for Entity',Ent,'.  Assigning Mean = 0.0, SD = 1.0.\n'

                    except AttributeError:
                        print 'Error in fin_est(): Unable to get Mean/SD from std_params.\n'

                # Pull Mean/SD parameters from orig_data
                else:
                    OrigValLoc = np.where(OrigEntCore != OrigNaNVal)

                    if OrigMetric[1][Ent] == 'interval':
                        OrigValEntCore = OrigEntCore[OrigValLoc]

                    elif OrigMetric[1][Ent] == 'ratio':
                        OrigValEntCore = np.log(np.clip(OrigEntCore[OrigValLoc],
                                                        LogDatMin,np.inf))

                    # Get mean, sd
                    OrigMean = np.mean(OrigValEntCore,axis=None)
                    OrigSD = np.std(OrigValEntCore,axis=None)

                # Rescale back to original interval scale
                if OrigMetric[1][Ent] == 'interval':
                    StdSD = np.std(ValEntCore,axis=None)
                    ValNewCoreTemp = ValEntCore * (OrigSD / StdSD)
    
                    # Make rescaled Std data have same mean and SD as original data
                    NewCoreMean = np.mean(ValNewCoreTemp,axis=None)
                    NewCore[ValLoc] = ValNewCoreTemp + (OrigMean - NewCoreMean)
    
                    if GetSE is True:
                        NewCoreSE[ValLoc] = EntCoreSE[ValLoc] * (OrigSD / StdSD)
                        NewCoreEAR[ValLoc] = EntCoreEAR[ValLoc] * (OrigSD / StdSD)

                # Rescale back to original ratio scale.
                # Previous version adjusted to original mean, sd
                # Propogation of errors:  f = e^A, s[f]/f = s[A], s[f] = f * s[A]
                if OrigMetric[1][Ent] == 'ratio':
                    NewCore[ValLoc] = np.exp(ValEntCore)

                    if GetSE is True:
                        NewCoreSE[ValLoc] = NewCore[ValLoc] * EntCoreSE[ValLoc]
                        NewCoreEAR[ValLoc] = NewCore[ValLoc] * EntCoreEAR[ValLoc]
                        
#                        NewCoreSE[ValLoc] = NewCoreSE[ValLoc] * NewCore[ValLoc]
#                        NewCoreEAR[ValLoc] = NewCoreEAR[ValLoc] * NewCore[ValLoc]


            ####################################
            # Originally ordinal or sigmoid data

            elif (OrigMetric[1][Ent] == 'ordinal'
                  or OrigMetric[1][Ent] == 'sigmoid'
                  ):

                # If the standardized metric is unbounded, use function defined above
                if (stdmetric == 'SD'
                    or stdmetric == 'LogDat'
                    or stdmetric == 'Logit'
                    or stdmetric == 'PreLogit'
                    or stdmetric == 'PLogit'
                    ):
                    lin2_out1 = linear2prob(Ent,entcore,EntCoreSE,EntCoreEAR)
                    p_EntCore = lin2_out1['p_EntCore']

                    if GetSE is True:
                        p_EntCoreSE = lin2_out1['p_EntCoreSE']
                        p_EntCoreEAR = lin2_out1['p_EntCoreEAR']

                # If the standardized metric is already bounded from 0 to 1 (no spilling)
                elif (stdmetric == 'PMinMax'
                      or stdmetric == '0-1'
                      or stdmetric == 'Percentile'
                      ):

                    if (OrigMetric[1][Ent] == 'sigmoid'
                        or OrigMetric[1][Ent] == 'ordinal'
                        ):
                        Warning1 = True

                    # Make sure they are probabilities
                    TryProbFlag = True
                    try:
                        self.coord_out
                    except AttributeError:
                        TryProbFlag = False
                        pass

                    if TryProbFlag is True:
                        try:
                            self.est2logit_out['Prob']
                        except AttributeError:
                            exc = 'Expects probabilities with this stdmetric. Run est2logit() first.\n'
                            raise fin_est_Error(exc)
                    p_EntCore = entcore

                    if GetSE is True:
                        p_EntCoreSE = EntCoreSE
                        p_EntCoreEAR = EntCoreEAR

                # Convert to original metric using appropriate Min/Max
                if (StdParamsFlag is True
                    and (std_params['stdmetric'] == 'PMinMax'
                         or std_params['stdmetric'] == '0-1'
                         or std_params['stdmetric'] == 'Percentile'
                         )
                    ):
                    Min = minmax[1][Ent][0]
                    Max = minmax[1][Ent][1]

                # Extract Min/Max from original valid chars
                elif OrigValidChars is not None:
                    Min = minmax[1][Ent][0]
                    Max = minmax[1][Ent][1]

                # Get Min/Max from data
                else:
                    OrigValEntCore = OrigEntCore[np.where(OrigEntCore != OrigNaNVal)]
                    Min = np.amin(OrigValEntCore,axis=None)
                    Max = np.amax(OrigValEntCore,axis=None)

                # Build the destandardized column
                NewCore[ValLoc] = p_EntCore[ValLoc] * (Max - Min)

                if GetSE is True:
                    NewCoreSE[ValLoc] = p_EntCoreSE[ValLoc] * (Max - Min)
                    NewCoreEAR[ValLoc] = p_EntCoreEAR[ValLoc] * (Max - Min)

                # Round to nearest valid integer, if desired
                if continuous == 'Auto':
                    if Round[1][Ent] == 1:
                        NewCore = np.clip(np.around(NewCore),Min,Max)
            else:
                exc = 'OrigMetric[Ent] is "nominal" but treated as numerical. Check MethDict.\n'
                raise fin_est_Error(exc)

            # Get column labels
            if referto == 'Cols':
                try:
                    NewLabels = OrigColLabels[:,OrigEntLoc[0] + nheaders4rows]
                except NameError:
                    NewLabels = collabels[:,EntLoc[0] + nheaders4rows]

                nRepeatCols = np.size(NewCore,axis=1)
                NewLabels = np.repeat(NewLabels,nRepeatCols,axis=1)

        else:
            exc = 'Unable to figure out the MethDict output from parse().\n'
            raise fin_est_Error(exc)


        #################
        ##   Append    ##
        ##   NewCore   ##
        #################

        # WARNING:  Consider scenario where some columns are string and numerical cols contain scientific notation.

        # Append new data -- no pytables
        if pytables is None:
            NewCoreData = np.append(NewCoreData,NewCore,axis=1)

            if GetSE is True:
                NewCoreDataSE = np.append(NewCoreDataSE,NewCoreSE,axis=1)
                NewCoreDataEAR = np.append(NewCoreDataEAR,NewCoreEAR,axis=1)

            if referto == 'Cols':
                NewColLabels = np.append(NewColLabels,NewLabels,axis=1)

        # Append using pytables
        else:
            NewCoreDataTab['arrays']['coredata'].append(NewCore)

            if GetSE is True:
                NewCoreDataSETab['arrays']['coredata'].append(NewCoreSE)
                NewCoreDataEARTab['arrays']['coredata'].append(NewCoreEAR)

            if referto == 'Cols':
                NewColLabels = np.append(NewColLabels,NewLabels,axis=1)

    #################
    ##  Warnings   ##
    #################

    if Warning1 is True:
        try:
            self.base_est_out
            print "Warning in fin_est(): The",stdmetric,"standardized metric, when applied to sigmoid or ordinal data with more than two categories, may yield estimates that have an 'ogival' nonlinear relationship to the 'true' values.  'PreLogit' may work better.\n"
        except AttributeError:
            pass

    ###################
    ##  Clean data   ##
    ###################

    if pytables is None:

        # Remove leading dummy columns
        NewCoreData = np.delete(NewCoreData,0,axis=1)

        if GetSE is True:
            NewCoreDataSE = np.delete(NewCoreDataSE,0,axis=1)
            NewCoreDataEAR = np.delete(NewCoreDataEAR,0,axis=1)
        else:
            NewCoreDataSE = None
            NewCoreDataEAR = None

    # Read from PyTable
    else:
        NewCoreData = NewCoreDataTab['arrays']['coredata']

        if GetSE is True:
            NewCoreDataSE = NewCoreDataSETab['arrays']['coredata']
            NewCoreDataEAR = NewCoreDataEARTab['arrays']['coredata']
        else:
            NewCoreDataSE = None
            NewCoreDataEAR = None

    # Deal with collabels.  Not necessary if coredata was collapsed to one long column.
    if referto == 'Cols':
        NewColLabels = np.delete(NewColLabels,0,axis=1)

    ###################
    ##  Alpha Resp   ##
    ###################

    # Build new validchar parameter to go in outputs
    new_validchar = ['All',['All'],'Num']
    NaNValFin = nanval

    if RespAlpha is True:
        try:
            Preds = np.zeros((nrows,1),dtype='S20')
            Rows = range(nrows)
            i = 0
            for Ent in Ents:
                try:
                    PredCol = np.array([PredDict[Ent]['pred_prob'][Row][0] for Row in Rows])[:,np.newaxis]
                except KeyError:
                    PredCol = NewCoreData[:,i][:,np.newaxis]
                Preds = np.append(Preds,PredCol,axis=1)
                i += 1
            NewCoreData = np.delete(Preds,0,axis=1)

            new_validchar.remove('Num')
            NaNValFin = str(int(nanval))

        except TypeError:
            print 'Warning in fin_est():  Could not convert any estimates to alpha.\n'
            pass


    ###################
    ##  Build Dict   ##
    ###################

    # Resize NewCoreData to original dimensions, if necessary
    if referto == 'Whole':
        NewCoreData = np.resize(NewCoreData[:,:],(nrows,ncols))

        if pytables is not None:
            NewCoreData = tools.pytables_(NewCoreData,'array',fileh,None,'fin_est_out',
                                     ['coredata'],None,None,None,None,None)['arrays']['coredata']

        try:
            NewColLabels = OrigColLabels
        except:
            NewColLabels = collabels

    # Append lead columns
    elif referto == 'Cols':
        try:
            Leader = OrigRowLabels[:OrignHeaders4Cols,:].astype(int)
        except ValueError:
            Leader = OrigRowLabels[:OrignHeaders4Cols,:]
        NewColLabels = np.append(Leader,NewColLabels,axis=1)

    # Try to cast each ID in NewColLabels to int
    try:
        NewColLabels = NewColLabels[:,:].astype(float).astype(int)
    except ValueError:
        for i in range(np.size(NewColLabels[:,:],axis=1)):
            try:
                NewColLabels[0,i] = int(float(NewColLabels[0,i]))
            except:
                pass

    # Convert to pytables
    if pytables is not None:

        # Get collabels (whether array or PyTable) into fin_est_out pytable
        NewColLabels = tools.pytables_(NewColLabels[:,:],'array',fileh,None,'fin_est_out',
                                     ['collabels'],None,None,None,None,None)['arrays']['collabels']

        # Get OrigRowLabels (already PyTable) into fin_est_out pytable
        OrigRowLabels = tools.pytables_(OrigRowLabels[:,:],'array',fileh,None,'fin_est_out',
                                     ['rowlabels'],None,None,None,None,None)['arrays']['rowlabels']

    # Create output DataDicts
    DestdDict = {'rowlabels':OrigRowLabels,
                 'collabels':NewColLabels,
                 'coredata':NewCoreData,
                 'nheaders4rows':OrignHeaders4Rows,
                 'key4rows':OrigKey4Rows,
                 'rowkeytype':OrigRowKeyType,
                 'nheaders4cols':OrignHeaders4Cols,
                 'key4cols':OrigKey4Cols,
                 'colkeytype':colkeytype,     # if deparsed, colkeytype = 'S60'
                 'nanval':NaNValFin,
                 'validchars':new_validchar,  #OrigValidChars,
                 }

    # Add SE, EAR dicts to self
    if GetSE is True:
        SEDict = {}
        for key in DestdDict.keys():
            SEDict[key] = DestdDict[key]
            SEDict['coredata'] = NewCoreDataSE

        EARDict = {}
        for key in DestdDict.keys():
            EARDict[key] = DestdDict[key]
            EARDict['coredata'] = NewCoreDataEAR

        # Add SEDict, EARDict, PredDict to self
        self.fin_se_out = SEDict
        self.fin_ear_out = EARDict
    else:
        self.fin_se_out = None
        self.fin_ear_out = None

    # Add prediction dictionary to self
    self.pred_dict = PredDict

    return DestdDict



######################################################################

def _fillmiss(_locals):
    "Basis of the fillmiss() method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Extract original data
    try:
        ObsRCD = self.extract_valid_out
    except AttributeError:
        try:
            ObsRCD = self.merge_info_out
        except AttributeError:
            try:
                ObsRCD = self.data_out
            except AttributeError:
                exc = 'Unable to find a "observed" data (MyObj.data_out).\n'
                raise fillmiss_Error(exc)

    # Get estimates
    try:
        EstRCD = self.fin_est_out
    except AttributeError:
        try:
            EstRCD = self.base_est_out
        except AttributeError:
            exc = 'Unable to find estimates.  Run base_est() or fin_est().\n'
            raise fillmiss_Error(exc)

    # Observations variables
    ObsData = ObsRCD['coredata']
    ObsColLabels = ObsRCD['collabels']
    ObsnHeaders4Rows = ObsRCD['nheaders4rows']
    ObsKey4Cols = ObsRCD['key4cols']
    ObsColKeyType = ObsRCD['colkeytype']
    ObsNaNVal = ObsRCD['nanval']

    # estimates variables
    EstData = EstRCD['coredata']
    rowlabels = EstRCD['rowlabels']
    collabels = EstRCD['collabels']
    nheaders4rows = EstRCD['nheaders4rows']
    nheaders4cols = EstRCD['nheaders4cols']
    key4cols = EstRCD['key4cols']
    colkeytype = EstRCD['colkeytype']
#    nanval = EstRCD['nanval']

    nrows = np.size(rowlabels[nheaders4cols:,:],axis=0)

    # Get Ents
    EstEnts = collabels[key4cols,nheaders4rows:].astype(colkeytype)
    ObsEnts = ObsColLabels[ObsKey4Cols,ObsnHeaders4Rows:].astype(ObsColKeyType)

    # Initialize array
    if pytables is None:
        Fill = np.zeros((nrows,0))
    else:
        Fill = tools.pytables_(None,'init_earray',fileh,None,'fillmiss_out',['coredata'],
                               None,'float',4,(nrows,0),None)['arrays']['coredata']

    # Fill missing for each column entity
    for Ent in ObsEnts:

        # Extract observations
        EntObs = ObsData[:,np.where(Ent == ObsEnts)[0]]

        # Extract estimates
        if Ent in EstEnts:
            EntEst = EstData[:,np.where(Ent == EstEnts)[0]]
        elif Ent in EstEnts.astype(int):
            EntEst = EstData[:,np.where(Ent == EstEnts.astype(int))[0]]
        else:
            EntEst = None

        # Calculate EntFill
        if EntEst is None:
            EntFill = EntObs
        else:
            EntFill = np.where(EntObs == ObsNaNVal,EntEst,EntObs)

        # Append entity residuals to initialize array
        if pytables is None:
            Fill = np.append(Fill,EntFill,axis=1)
        else:
            Fill.append(EntFill)

    # Build dict
    FillMissRCD = {}
    ValList = ['rowlabels','collabels','coredata',
               'nheaders4rows','key4rows','rowkeytype',
               'nheaders4cols','key4cols','colkeytype',
               'nanval','validchars']
    for key in ObsRCD.keys():
        if key in ValList:
            FillMissRCD[key] = ObsRCD[key]
        FillMissRCD['coredata'] = Fill

    return FillMissRCD




######################################################################

def _item_diff(_locals):
    "Basis of the item_diff() method."

    self = _locals['self']
    scores = _locals['scores']
    curve = _locals['curve']
    pcut = _locals['pcut']
    rescale = _locals['rescale']
    minmax = _locals['minmax']

    # Get score keys, data
    s_keys = tools.getkeys(scores,'Col','Core','Auto',None)
    ScoreData = scores['coredata']

    # Get probabilities (actually logits)
    try:
        PCoreData_ = self.est2logit_out
        nanval = PCoreData_['nanval']
    except AttributeError:
        exc = 'Unable to find logit estimates.  Run est2logit() first.\n'
        raise item_diff_Error(exc)

    # Extract desired columns
    PCoreData = self.extract(PCoreData_,
                             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                             getcols = {'Get':'NoneExcept','Labels':'key','Cols':s_keys},
                             labels_only = None
                             )

    ProbData = self.extract(PCoreData_['Prob'],
                             getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                             getcols = {'Get':'NoneExcept','Labels':'key','Cols':s_keys},
                            labels_only = None
                             )

    # Rescale scores, if necessary
    if rescale is not None:
        rescaled = np.zeros(np.shape(ScoreData))
        for i in xrange(len(s_keys)):
            try:
                rescaled[:,i] = tools.rescale(ScoreData[:,i],
                                              straighten=None,
                                              logits=None, # TODO: Check this
                                              mean_sd = [rescale[0],rescale[1]],
                                              m_b = None,
                                              clip = None,
                                              nanval = nanval
                                              )
            except:
                rescaled[:,i] = nanval

        ScoreData = rescaled


    #####################
    ##  Linearization  ##
    #####################

    # Convert logit to probability
    if curve == 'Sigmoid':
        PVar = ProbData['coredata']
        PCut1 = pcut

    # Leave logit as is
    elif curve == 'Linear':
        PVar = PCoreData['coredata']
        PCut1 = np.log(pcut / (1 - pcut))

    else:
        exc = 'Unable to figure out curve parameter.\n'
        raise item_diff_Error(exc)


    #####################
    ##   Calculate     ##
    ##  Coefficients   ##
    #####################

    # Apply least squares to each column
    ncols = np.size(PVar,axis=1)
    ICoeff = np.zeros((2,ncols))
    for i in range(ncols):
        Loc = np.where(np.logical_and(PVar[:,i] != nanval,ScoreData[:,i] != nanval))
        Ones = np.ones((np.size(Loc[0]),1))
        P = np.append(PVar[:,i][Loc][:,np.newaxis],Ones,axis=1)
        S = ScoreData[:,i][Loc][:,np.newaxis]

        # Handle vectors with no valid values
        if list(P) == [] or list(S) == []:
            x = np.zeros((2,1)) + nanval
        else:
            # Get column coefficients
            x = npla.lstsq(P,S)[0]

        # Populate ICoeff array
        ICoeff[:,i] = x[:,0]

    # Calculate item difficulty:  [pcut,1] * ICoeff
    TargProb = np.array([[PCut1,1]])
    IDiff = np.transpose(tools.estimate(TargProb,np.transpose(ICoeff),nanval))

    # rescale and clip
    if minmax is None:
        Min = -np.inf
        Max = np.inf
    else:
        Min = minmax[0]
        Max = minmax[1]

    IDiff = np.where(IDiff == nanval,nanval,np.clip(IDiff,Min,Max))

    # Labels
    IKeys = np.array(tools.getkeys(scores,'Col','Core','Auto',None))[:,np.newaxis]
    PRowLabels1 = np.append(np.array([['ItemID']]),IKeys,axis=0)
    PColLabels1 = np.array([['ItemID','IDiff']])

    # Put in data() format
    IDiffDict = {'rowlabels':PRowLabels1,'collabels':PColLabels1,
                'coredata':IDiff,'nheaders4rows':1,'key4rows':0,
                'rowkeytype':scores['colkeytype'],'nheaders4cols':1,
                'key4cols':0,'colkeytype':'S10',
                'nanval':nanval,'validchars':['All',['All'],'Num']
                }

    return IDiffDict






######################################################################

def _summstat(_locals):
    "Basis of the summstat() method."

    # Get self
    self = _locals['self']

    # Get _locals
    data = _locals['data']
    getstats = _locals['getstats']
    getrows = _locals['getrows']
    getcols = _locals['getcols']
    itemdiff = _locals['itemdiff']
    outname = _locals['outname']
    labels = _locals['labels']
    group_se = _locals['group_se']
    correlated = _locals['correlated']
    
    if correlated is not None:
        exc = 'The correlated option has been deprecated. Set to None.'
        raise summstat_Error(exc)

    if labels is None:
        labels = {'row_ents':None, 'col_ents':None}

    # Check if rasch() was run
    try:
        self.rasch_out
        if self.verbose is True:
            print 'Warning in summstat(): Overwriting the summstat_out already created by rasch().\n'
    except AttributeError:
        pass

    # Get variables
    nanval = float(self.nanval)

    # List of possible Stats
    AllStats = ['Mean',
                'SE',
                'SD',
                'Corr',
                'PtBis',
                'Resid',
                'RMSEAR',
                'Sep',
                'Rel',
                'Outfit',
                'Fit_Perc>2',
                'Count',
                'Min',
                '25Perc',
                'Median',
                '75Perc',
                'Max',
                'Coord'
                ]

    if getstats == ['All'] or getstats == 'All':
        Stats = AllStats
    else:
        Stats = getstats

    # Head off missing coordinates exception
    if 'Coord' in Stats:
        try:
            self.coord_out
        except AttributeError:
            Stats.remove('Coord')

    # Get data that went into coord()
    try:
        StdObs = self.standardize_out
    except AttributeError:
        try:
            StdObs = self.parse_out
        except AttributeError:
            try:
                StdObs = self.subscale_out
            except AttributeError:
                try:
                    StdObs = self.score_mc_out
                except AttributeError:
                    try:
                        StdObs = self.extract_valid_out
                    except AttributeError:
                        try:
                            StdObs = self.merge_info_out
                        except AttributeError:
                            try:
                                StdObs = self.data_out
                            except AttributeError:
                                exc = 'Unable to find "observed" data.\n'
                                raise summstat_Error(exc)

    # Get reported observed/estimates, set type of outputs ('Base' or 'Final')
    if data == 'est2logit_out':
        try:
            UseEst = self.est2logit_out
            UseObs = None
        except AttributeError:
            exc = "Unable to find logit estimates.  Run est2logit().\n"
            raise summstat_Error(exc)

    elif data == 'fin_est_out':
        try:
            UseEst = self.fin_est_out
            UseObs = self.data_out
        except AttributeError:
            exc = "Unable to find final estimates.  Run fin_est().\n"
            raise summstat_Error(exc)

    elif data == 'base_est_out':
        try:
            UseEst = self.base_est_out
            UseObs = StdObs
        except AttributeError:
            exc = "Unable to find base estimates.  Run base_est().\n"
            raise summstat_Error(exc)

    else:
        try:
            UseEst = self.__dict__[data]
            UseObs = None
        except:
            exc = "Unable to find data.\n"
            raise summstat_Error(exc)

    # Calculate point biserial correlations
    if 'PtBis' in Stats:
        self.ptbis_out = tools.ptbis(UseObs, 'All')


    #####################################################################

    ###################
    ##    Define     ##
    ##  rangestat()  ##
    ##   functions   ##
    ###################

    def rangestat(GetStats_ = ['All'], # [Select from -> summstat list]
                  GetRows_ = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                  GetCols_ = {'Get':'AllExcept','Labels':'key','Cols':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                  ItemDiff_ = None, # [<None,True> => reverse sign for items to imply difficulty]
                  ):
        "Get summary row and column statistics for one specified range."

        # Initialize entity dictionary
        EntDict = {}

        # Re-used alot
        EstRange = self.extract(UseEst,GetRows_,GetCols_,None)['coredata']
        ValEstRange = EstRange[np.where(EstRange != nanval)]

        ############
        ##  Mean  ##
        ############

        if ('Mean' in GetStats_
            or 'All' in GetStats_
            ):
            if ItemDiff_ is True:
                IDiffWarning = "Warning in summstat(): itemdiff = True generally makes sense only when data = 'est2logit_out'.  You may also want to consider the item_diff() method.\n"
                if data != 'est2logit_out':
                    if self.verbose is True:
                        print IDiffWarning
                else:
                    try:
                        self.est2logit_out
                    except AttributeError:
                        if self.verbose is True:
                            print IDiffWarning
                Mean = -1.0 * np.mean(ValEstRange)

            else:
                Mean = np.mean(ValEstRange)

            EntDict['Mean'] = Mean


        ############
        ##   SD   ##
        ############

        if ('SD' in GetStats_
            or 'All' in GetStats_
            or 'Sep' in GetStats_
            or 'Rel' in GetStats_
            ):
            SD = np.std(ValEstRange)
            EntDict['SD'] = SD


        ############
        ##  Corr  ##
        ############

        if ('Corr' in GetStats_
            or 'All' in GetStats_
            ):
            if data not in ['base_est_out','fin_est_out']:
                EntDict['Corr'] = nanval
            else:
                Obs = self.extract(UseObs,GetRows_,GetCols_,None)['coredata']
                Est = self.extract(UseEst,GetRows_,GetCols_,None)['coredata']

                try:
                    Corr = tools.correl(Obs,Est,nanval,'Corr')
                    EntDict['Corr'] = Corr
                except:
                    EntDict['Corr'] = nanval


        ################
        ##  FinResid  ##
        ################

        if ('Resid' in GetStats_
            or 'All' in GetStats_
            ):
            if data not in ['base_est_out','fin_est_out']:
                EntDict['Resid'] = nanval

            else:
                if data == 'fin_est_out':
                    try:
                        Resid = self.fin_resid_out
                    except AttributeError:
                        exc = 'Unable to find fin_resid_out. Run fin_resid().\n'
                        raise summstat_Error(exc)

                elif data == 'base_est_out':
                    try:
                        Resid = self.base_resid_out
                    except AttributeError:
                        exc = 'Unable to find base_resid_out. Run base_resid().\n'
                        raise summstat_Error(exc)

                xResid = self.extract(Resid,GetRows_,GetCols_,None)['coredata']

                RMSR = tools.rmsr(None,None,xResid,nanval)
                EntDict['Resid'] = RMSR


        ##############
        ##  RMSEAR  ##
        ##############

        if ('RMSEAR' in GetStats_
            or 'All' in GetStats_
            or 'OutFit' in GetStats_
            ):
            if data not in ['base_est_out','fin_est_out','est2logit_out']:
                EntDict['RMSEAR'] = nanval

            elif data == 'est2logit_out':
                try:
                    EAR = self.logit_ear_out
                except AttributeError:
                    exc = 'Unable to find logit_ear_out.  Run est2logit() to build it.\n'
                    raise summstat_Error(exc)

            elif data == 'fin_est_out':
                try:
                    EAR = self.fin_ear_out
                except AttributeError:
                    exc = 'Unable to find fin_ear_out.  Run fin_est() to build it.\n'
                    raise summstat_Error(exc)

            elif data == 'base_est_out':
                try:
                    EAR = self.base_ear_out
                except AttributeError:
                    exc = 'Unable to find base_ear_out.  Run base_ear().\n'
                    raise summstat_Error(exc)

            xEAR = self.extract(EAR,GetRows_,GetCols_,None)['coredata']
            RMSEAR = tools.rmsr(None,None,xEAR,nanval)
            EntDict['RMSEAR'] = RMSEAR


        ############
        ##  RMSE  ##
        ############

        if ('SE' in GetStats_
            or 'All' in GetStats_
            or 'Sep' in GetStats_
            or 'Rel' in GetStats_
            ):
            if data not in ['base_est_out','fin_est_out','est2logit_out']:
                EntDict['SE'] = nanval

            elif data == 'est2logit_out':
                try:
                    CellSE = self.logit_se_out
                except AttributeError:
                    exc = 'Unable to find logit_se_out.  Use est2logit().\n'
                    raise summstat_Error(exc)

            elif data == 'fin_est_out':
                try:
                    CellSE = self.fin_se_out
                except AttributeError:
                    exc = 'Unable to find fin_se_out.  Use fin_est().\n'
                    raise summstat_Error(exc)

            elif data == 'base_est_out':
                try:
                    CellSE = self.base_se_out                    
                except AttributeError:
                    exc = 'Unable to find base_se_out.  Run base_se().\n'
                    raise summstat_Error(exc)
            
            xCellSE = self.extract(CellSE, GetRows_, GetCols_, 
                                   None)['coredata']

            ndim = self.coord_out['ndim']
            rmse_ = tools.rmsr(None, None, xCellSE, nanval)
            
            axis = 1
            if GetCols_['Get'] == 'NoneExcept' and len(GetCols_['Cols']) == 1:
                axis = 0
            nrows, ncols = np.shape(CellSE['coredata'])
            
            if group_se:
                SE = tools.group_se(rmse_, group_se, ndim, axis, nrows, ncols)
            else:
                SE = rmse_
            EntDict['SE'] = SE


        ##################
        ##  Separation  ##
        ##################

        if ('Sep' in GetStats_
            or 'All' in GetStats_
            or 'Rel' in GetStats_
            ):
            Sep = tools.separation(SD, SE)
            EntDict['Sep'] = Sep


        ##################
        ##  Reliability ##
        ##################

        if ('Rel' in GetStats_
            or 'All' in GetStats_
            ):
            Rel = tools.reliability(Sep)
            EntDict['Rel'] = Rel


        ##############
        ##    Fit   ##
        ##############

        if ('Outfit' in GetStats_
            or 'Fit_Perc>2' in GetStats_
            or 'All' in GetStats_
            ):
            if data not in ['base_est_out','fin_est_out']:
                EntDict['Outfit'] = nanval
                EntDict['Fit_Perc>2'] = nanval

            else:
                if data == 'fin_est_out':
                    try:
                        Fit_ = self.fin_fit_out
                    except AttributeError:
                        exc = 'Unable to find fin_fit_out.  Run fin_fit().\n'
                        raise summstat_Error(exc)

                elif data == 'base_est_out':
                    try:
                        Fit_ = self.base_fit_out
                    except AttributeError:
                        exc = 'Unable to find base_fit_out.  Run base_fit().\n'
                        raise summstat_Error(exc)

                # Get range
                range_ = self.extract(Fit_,GetRows_,GetCols_,None)['coredata']

                # Average absolute fit
                if 'Outfit' in GetStats_:
                    Fit1 = tools.fit(None,None,None,None,range_,'MeanSq',nanval)['summfit']
                    EntDict['Outfit'] = Fit1

                if 'Fit_Perc>2' in GetStats_:
                    Fit2 = tools.fit(None,None,None,None,range_,'Perc>2.0',nanval)['summfit']
                    EntDict['Fit_Perc>2'] = Fit2


        #############
        ##  count  ##
        #############

        if ('Count' in GetStats_
            or 'All' in GetStats_
            ):
            NaNVal0 = self.data_out['nanval']
            try:
                KeyDict = self.parse_out['KeyDict']
                GetRows1 = GetRows_
                GetCols1 = GetCols_.copy()
                ncols = len(GetCols_['Cols'])
                Cols1 = np.zeros(ncols)
                for i in range(ncols):
                    Cols1[i] = KeyDict[GetCols_['Cols'][i]]
                GetCols1['Cols'] = np.unique(Cols1)
            except:
                GetRows1 = GetRows_
                GetCols1 = GetCols_

            # extract range and count
            range_ = self.extract(self.data_out,GetRows1,GetCols1,None)['coredata']
            count = np.sum(range_ != NaNVal0)

            if count == 0:
                count = nanval

            EntDict['Count'] = count


        ############
        ##  Min   ##
        ############

        if ('Min' in GetStats_
            or 'All' in GetStats_
            ):
            try:
                Min = np.amin(ValEstRange)
            except ValueError:
                Min = nanval
            EntDict['Min'] = Min


        ##############
        ##  median_  ##
        ##############

        if ('Median' in GetStats_
            or 'All' in GetStats_
            or '25Perc' in GetStats_
            or '75Perc' in GetStats_
            ):
            median_ = np.median(ValEstRange)
            EntDict['Median'] = median_


        ###############
        ##   25Perc  ##
        ###############

        if ('25Perc' in GetStats_
            or 'All' in GetStats_
            ):
            Perc25 = tools.percent25(ValEstRange,median_,nanval)
            EntDict['25Perc'] = Perc25


        ###############
        ##   75Perc  ##
        ###############

        if ('75Perc' in GetStats_
            or 'All' in GetStats_
            ):
            Perc75 = tools.percent75(ValEstRange,median_,nanval)
            EntDict['75Perc'] = Perc75


        ############
        ##  Max   ##
        ############

        if ('Max' in GetStats_
            or 'All' in GetStats_
            ):
            try:
                Max = np.amax(ValEstRange)
            except ValueError:
                Max = nanval
            EntDict['Max'] = Max


        ###############
        ##   PtBis   ##
        ###############

        if ('PtBis' in GetStats_ or 'All' in GetStats_):
            proceed = False
            if len(GetRows_['Rows']) == 1:
                pass
            elif len(GetCols_['Cols']) == 1:
                if GetCols_['Labels'] == 'key':
                    proceed = True
                ptbis = self.ptbis_out
                getcols = {'Get':'NoneExcept',
                           'Labels':'key',
                           'Cols':GetCols_['Cols']}
            if proceed:
                ptbis = self.extract(ptbis, 
                                     getrows = {'Get':'AllExcept',
                                                      'Labels':'key',
                                                      'Rows':[None]},
                                     getcols = getcols)
                EntDict['PtBis'] = ptbis['coredata'][0][0]
            else:
                EntDict['PtBis'] = nanval


        ###############
        ##   Coord   ##
        ###############

        if ('Coord' in GetStats_
            or 'All' in GetStats_
            ):
            try:
                self.coord_out
                Proceed = True

                # Targ entity is a row entity
                if len(GetRows_['Rows']) == 1:
                    if GetRows_['Labels'] != 'key':
                        Proceed = False
                    elif GetRows_['Labels'] == 'key':
                        Coords = self.coord_out['fac0coord']
                        GetCoRows = GetRows_
                    else:
                        exc = "Unable to interpret 'Coord' extraction parameters.\n"
                        raise summstat_Error(exc)

                # Targ entity is a column entity
                elif len(GetCols_['Cols']) == 1:
                    if GetCols_['Labels'] != 'key':
                        Proceed = False
                    elif GetCols_['Labels'] == 'key':
                        Coords = self.coord_out['fac1coord']
                        GetCoRows = {'Get':'NoneExcept','Labels':'key','Rows':GetCols_['Cols']}     # Needed to change 'Rows' to 'Cols'

                        # Can't extract parsed col coords
                        if np.size(Coords['coredata'],axis=0) != np.size(UseEst['coredata'],axis=1):
                            Proceed = False
                    else:
                        exc = "Unable to interpret 'Coord' extraction parameters.\n"
                        raise summstat_Error(exc)
                else:
                    print 'GetCols_=\n', GetCols_
                    exc = "Unable to extract coordinates. Either run coord() or omit 'Coord' from getstats parameter.\n"
                    raise summstat_Error(exc)
                    Proceed = False

            except AttributeError:
                exc = "Unable to extract coordinates. Either run coord() or omit 'Coord' from getstats parameter.\n"
                raise summstat_Error(exc)
                Proceed = False

            if Proceed is True:
                Coord = self.extract(Coords, GetCoRows, 
                                     {'Get':'AllExcept','Labels':'key','Cols':[None]},
                                     None)
                EntDict['Coord'] = Coord['coredata']
            else:
                EntDict['Coord'] = np.array([nanval])

        return EntDict


        #####################################################################


    ################
    ##    Prep    ##
    ##  rangestat ##
    ################

    SummWholeRow = False
    SummWholeCol = False

    # Interpret 'SummWhole'
    if getrows == 'SummWhole' or getrows is None:
        SummWholeRow = True if getrows == 'SummWhole' else None
        getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}

    if getcols == 'SummWhole' or getcols is None:
        SummWholeCol = True if getcols == 'SummWhole' else None
        getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}

    # Extract relevant row and column labels
    X = self.extract(UseEst,getrows,getcols)

    # Check for indexing errors
    try:
        X['rowlabels'][X['nheaders4cols']:, X['key4rows']]
        X['rowlabels'][X['key4cols'],:]        
        X['collabels'][X['key4cols'], X['nheaders4rows']:]
        X['collabels'][:, X['key4rows']]
    except:
        exc = 'Unable to figure out getcols or getrows parameters.'
        raise summstat_Error(exc)

    # Extract array of row entities
    if (getrows['Labels'] == 'key'
        or getrows['Labels'] == 'index'
        ):
        #row_ents = tools.getkeys(X,'Row','Core','Auto',None)
        try:
            row_ents = X['rowlabels'][X['nheaders4cols']:,X['key4rows']].astype(X['rowkeytype'])  # Will have to redo
        except ValueError:
            row_ents = X['rowlabels'][X['nheaders4cols']:,X['key4rows']] #.astype(float).astype(X['rowkeytype'])
        RLabels = 'key'

    # array of unique attributes
    elif isinstance(getrows['Labels'],int):

        try:
            row_ents = np.unique(X['rowlabels'][X['nheaders4cols']:,getrows['Labels']])
        except IndexError:
            exc = 'Unable to figure out getrows or getcols parameters. Check that all specified rows or cols apply only to the coredata.'
            raise summstat_Error(exc)
        
        RLabels = getrows['Labels']

    elif isinstance(getrows['Labels'],str):
        ind = np.where(X['rowlabels'][X['key4cols'],:] == getrows['Labels'])[0][0]
        row_ents = np.unique(X['rowlabels'][X['nheaders4cols']:,ind])
        RLabels = getrows['Labels']

    else:
        exc = 'Unable to figure out getrows parameter.\n'
        raise summstat_Error(exc)

    # Extract array of col entities
    if (getcols['Labels'] == 'key'
        or getcols['Labels'] == 'index'
        ):
        #col_ents = tools.getkeys(X,'Col','Core','Auto',None)
        try:
            col_ents = X['collabels'][X['key4cols'],X['nheaders4rows']:].astype(X['colkeytype'])
        except ValueError:
            col_ents = X['collabels'][X['key4cols'],X['nheaders4rows']:] #.astype(float).astype(X['colkeytype'])
        CLabels = 'key'

    # array of unique attributes
    elif isinstance(getcols['Labels'],int):
        try:
            col_ents = np.unique(X['collabels'][getcols['Labels'],X['nheaders4rows']:])
        except IndexError:
            exc = 'Unable to figure out getrows or getcols parameters. Check that all specified rows or cols apply only to the coredata.'
            raise summstat_Error(exc)
        
        CLabels = getcols['Labels']

    elif isinstance(getcols['Labels'],str):
        ind = np.where(X['collabels'][:,X['key4rows']] == getcols['Labels'])[0][0]
        col_ents = np.unique(X['collabels'][ind,X['nheaders4rows']:])
        CLabels = getcols['Labels']

    else:
        exc = 'Unable to figure out getcols parameter.\n'
        raise summstat_Error(exc)


    ################
    ##    Feed    ##
    ##  rangestat ##
    ################


    # Get summary stats for row entities.  Note: dictionary comprehension wasn't faster.
    RowStats = {}

    if SummWholeRow is True:
        RowStats['AllRows'] = rangestat(GetStats_ = Stats, # [Select from -> summstat list]
                                  GetRows_ = {'Get':'NoneExcept','Labels':RLabels,'Rows':list(row_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                                  GetCols_ = {'Get':'NoneExcept','Labels':CLabels,'Cols':list(col_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                                  ItemDiff_ = None,
                                  )
    elif SummWholeRow is None:
        pass
    else:
        for r in row_ents:
            RowStats[r] = rangestat(GetStats_ = Stats, # [Select from -> summstat list]
                                  GetRows_ = {'Get':'NoneExcept','Labels':RLabels,'Rows':[r]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                                  GetCols_ = {'Get':'NoneExcept','Labels':CLabels,'Cols':list(col_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                                  ItemDiff_ = None,
                                  )

    # Get summary stats for col entities
    ColStats = {}
    if SummWholeCol is True:
        ColStats['AllCols'] = rangestat(GetStats_ = Stats, # [Select from -> summstat list]
                                  GetRows_ = {'Get':'NoneExcept','Labels':RLabels,'Rows':list(row_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                                  GetCols_ = {'Get':'NoneExcept','Labels':CLabels,'Cols':list(col_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                                  ItemDiff_ = itemdiff,
                                  )
    elif SummWholeCol is None:
        pass
    else:
        for c in col_ents:
            ColStats[c] = rangestat(GetStats_ = Stats, # [Select from -> summstat list]
                                  GetRows_ = {'Get':'NoneExcept','Labels':RLabels,'Rows':list(row_ents)}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                                  GetCols_ = {'Get':'NoneExcept','Labels':CLabels,'Cols':[c]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                                  ItemDiff_ = itemdiff,
                                  )


    ##############
    ##  Build   ##
    ##  Tables  ##
    ##############

    nRowEnts = np.size(row_ents)
    nColEnts = np.size(col_ents)
    nStats = np.size(Stats)
    RCoord = False
    CCoord = False
    CLCoord = False
    DType = 'S60'

    # Initialize table
    if 'Coord' in Stats:
        try:
            nDims = np.size(RowStats[row_ents[0]]['Coord'])
            RECore = np.zeros((nRowEnts,nStats + nDims - 1))
            CECore = np.zeros((nColEnts,nStats + nDims - 1))
            StatColLabels = np.zeros((nStats + nDims - 1),dtype=DType)
        except KeyError:
            StatsList = list(Stats)
            StatsList.remove('Coord')
            Stats = np.array(StatsList)
            nStats = np.size(Stats)

            RECore = np.zeros((nRowEnts,nStats))
            CECore = np.zeros((nColEnts,nStats))
            StatColLabels = np.zeros((nStats),dtype=DType)
    else:
        RECore = np.zeros((nRowEnts,nStats))
        CECore = np.zeros((nColEnts,nStats))
        StatColLabels = np.zeros((nStats),dtype=DType)

    # SummWholeRow
    if SummWholeRow is True:
        RECore = np.zeros((1,nStats))
        for s in range(nStats):
            RECore[0,s] = RowStats['AllRows'][Stats[s]]
    elif SummWholeRow is None:
        RECore = None
    else:
        # Get RowEnt stats
        for s in range(nStats):
            for r in range(nRowEnts):
                if Stats[s] == 'Coord':
                    RECore[r,s:s + nDims] = RowStats[row_ents[r]][Stats[s]]
                    RCoord = True
                else:
                    if RCoord is True:
                        RECore[r,s + nDims - 1] = RowStats[row_ents[r]][Stats[s]]
                    else:
                        RECore[r,s] = RowStats[row_ents[r]][Stats[s]]

    # SummWholeCol
    if SummWholeCol is True:
        CECore = np.zeros((1,nStats))
        for s in range(nStats):
            CECore[0,s] = ColStats['AllCols'][Stats[s]]
    elif SummWholeCol is None:
        CECore = None
    else:
        # Get ColEnt stats
        for s in range(nStats):
            for c in range(nColEnts):
                if Stats[s] == 'Coord':
                    CECore[c,s:s + nDims] = ColStats[col_ents[c]][Stats[s]]
                    CCoord = True
                else:
                    if CCoord is True:
                        CECore[c,s + nDims - 1] = ColStats[col_ents[c]][Stats[s]]
                    else:
                        CECore[c,s] = ColStats[col_ents[c]][Stats[s]]

    # Get StatColLabels
    for s in range(nStats):
        if Stats[s] == 'Coord':
            StatColLabels[s:s + nDims] = 'Coord'
            CLCoord = True
        else:
            if CLCoord is True:
                StatColLabels[s + nDims - 1] = Stats[s]
            else:
                StatColLabels[s] = Stats[s]

    # Deal with nan's in coredata
    if RECore is not None:
        RECore = np.where(np.isnan(RECore),nanval,RECore)
    if CECore is not None:
        CECore = np.where(np.isnan(CECore),nanval,CECore)

    # Abbrev for StdObs datadict
    st = StdObs
    
    # Corner label for row_ents
    R_descriptor = np.array([[data+' row ents where cols are: '+str(getcols['Get'])+' '+str(getcols['Cols'])]])

    if labels['row_ents'] is None:
        R_label = R_descriptor
    else:
        R_label = np.array([[labels['row_ents']]])

    # Get rowlabels for row_ents
    if SummWholeRow is True:
        R_RowLabels = np.array([['All Rows']])
        RCorner = R_label

    # Row group names as rowlabels
    elif getrows['Labels'] != 'key' and getrows['Labels'] != 'index':
        R_RowLabels = np.array(row_ents)[:, np.newaxis]
        RCorner = R_label

    # Subset of row keys as rowlabels
    elif np.any(tools.getkeys(st, 'Row', 'Core', 'Auto', None) != np.array(row_ents)):
        R_dict = tools.damon_dicts(st['coredata'], st['rowlabels'], st['nheaders4rows'],
                                   st['key4rows'], st['rowkeytype'], st['collabels'],
                                   st['nheaders4cols'], st['key4cols'], st['colkeytype'],
                                   range4labels = 'Core',
                                   strip_labkeys = None,
                                   whole = None)['rl_row']
        R_RowLabels = np.zeros((len(row_ents), st['nheaders4rows']), dtype='S60')
        
        for i, key in enumerate(row_ents):
            R_RowLabels[i, :] = R_dict[key]

        RCorner = st['rowlabels'][st['key4cols'], :][np.newaxis, :]
        RCorner[:, st['key4rows']] = R_label[0, 0]


    # All rowkeys as rowlabels
    else:
        R_RowLabels = st['rowlabels'][st['nheaders4cols']:, :]    #R_RowLabels = np.array(row_ents)[:,np.newaxis]
        RCorner = st['rowlabels'][st['key4cols'], :][np.newaxis, :]
        RCorner[:, st['key4rows']] = R_label[0, 0]

    R_RowLabels = np.append(RCorner,R_RowLabels,axis=0)
    R_ColLabels = np.append(RCorner,StatColLabels[np.newaxis,:],axis=1)


    #######
    # Corner label for col_ents
    C_descriptor = np.array([[data+' col ents where rows are: '+str(getrows['Get'])+' '+str(getrows['Rows'])]])

    if labels['col_ents'] is None:
        C_label = C_descriptor
    else:
        C_label = np.array([[labels['col_ents']]])

    # Get rowlabels for col_ents
    if SummWholeCol is True:
        C_RowLabels = np.array([['All Cols']])
        CCorner = C_label

    # Col group names as rowlabels
    elif getcols['Labels'] != 'key' and getcols['Labels'] != 'index':
        C_RowLabels = np.array(col_ents)[:, np.newaxis]
        CCorner = C_label

    # Subset of col keys as rowlabels
    elif np.any(tools.getkeys(st, 'Col', 'Core', 'Auto', None) != np.array(col_ents)):
        C_dict = tools.damon_dicts(st['coredata'], st['rowlabels'], st['nheaders4rows'],
                                   st['key4rows'], st['rowkeytype'], st['collabels'],
                                   st['nheaders4cols'], st['key4cols'], st['colkeytype'],
                                   range4labels = 'Core',
                                   strip_labkeys = None,
                                   whole = None)['cl_col']
        C_RowLabels = np.zeros((len(col_ents), st['nheaders4cols']), dtype='S60')
        for i, key in enumerate(col_ents):
            C_RowLabels[i, :] = C_dict[key]

        CCorner = st['collabels'][:, st['key4rows']][np.newaxis, :]
        CCorner[:,st['key4cols']] = C_label[0, 0]
            
    # All row keys as rowlabels
    else:
        #orig_C_RowLabels = np.array(col_ents)[:,np.newaxis]
        C_RowLabels = np.transpose(st['collabels'][:, st['nheaders4rows']:])   
        CCorner = st['collabels'][:, st['key4rows']][np.newaxis, :]
        CCorner[:,st['key4cols']] = C_label[0, 0]
        
    C_RowLabels = np.append(CCorner,C_RowLabels,axis=0)
    C_ColLabels = np.append(CCorner,StatColLabels[np.newaxis,:],axis=1)


    #################
    ##   Build     ##
    ##  DataDicts  ##
    #################

    if RECore is not None:
        RowEntDict = {'rowlabels':R_RowLabels,
                        'collabels':R_ColLabels,
                        'coredata':RECore,
                        'key4rows':StdObs['key4rows'],
                        'rowkeytype':DType,
                        'key4cols':0,
                        'colkeytype':DType,
                        'nanval':nanval,
                        'validchars':['All',['All'],'Num'],
                        'dtype':self.dtype
                        }
        RowEntObj = dmn.core.Damon(RowEntDict,'datadict','RCD_dicts_whole',verbose=None)
        r_data_out = RowEntObj
    else:
        r_data_out = None

    if CECore is not None:
        ColEntDict = {'rowlabels':C_RowLabels,
                        'collabels':C_ColLabels,
                        'coredata':CECore,
                        'key4rows':StdObs['key4cols'],
                        'rowkeytype':DType,
                        'key4cols':0,
                        'colkeytype':DType,
                        'nanval':nanval,
                        'validchars':['All',['All'],'Num'],
                        'dtype':self.dtype
                        }

        ColEntObj = dmn.core.Damon(ColEntDict,'datadict','RCD_dicts_whole',verbose=None)
        c_data_out = ColEntObj
    else:
        c_data_out = None

    # Assign EntDicts to object (for exporting)
    if outname is not None:
        r_name = 'row_ents_out'+'_'+outname
        c_name = 'col_ents_out'+'_'+outname
    else:
        r_name = 'row_ents_out'
        c_name = 'col_ents_out'

    self.__dict__[r_name] = r_data_out
    self.__dict__[c_name] = c_data_out

    # Get descriptors
    descriptors = {'row_ents':R_descriptor,
                   'col_ents':C_descriptor}

    # Get objperdim
    try:
        objperdim = self.objperdim
    except AttributeError:
        objperdim = None

    # Get objectivity
    try:
        objectivity = self.objectivity
    except AttributeError:
        objectivity = None

    # Get stability
    try:
        stability = self.seed
    except AttributeError:
        stability = None

    # outputs
    outputs = {'row_ents':r_data_out,
               'col_ents':c_data_out,
               'descriptors':descriptors,
               'getrows':getrows,
               'getcols':getcols,
               'objperdim':objperdim,
               'stability':stability,
               'objectivity':objectivity
               }

    return outputs



######################################################################

def _merge_summstat(_locals):
    "Basis of the merge_summstat() method."

    self = _locals['self']
    merge_by = _locals['merge_by']

    try:
        summstat_out = self.summstat_out
    except AttributeError:
        exc = 'Unable to find my_obj.summstat_out.  Run summstat().'
        raise merge_summstat_Error(exc)


    ################
    ##  Merge by  ##
    ##    Row     ##
    ################

    if merge_by == 'row':

        try:
            outnames = self.outnames
        except AttributeError:
            exc = "Could not find outnames.  When merge_by is 'row', run summstat() multiple times."
            raise merge_summstat_Error(exc)
        
        # Figure out the ent_type
        outname = outnames[0]

        for et in ['row_ents', 'col_ents']:
            if et == 'row_ents':
                re = np.array([summstat_out[name]['getrows']['Rows'][0] for name in outnames])
            if et == 'col_ents':
                ce = np.array([summstat_out[name]['getcols']['Cols'][0] for name in outnames])

        if np.all(re == outnames):
            ent_type = 'col_ents'
        elif np.all(ce == outnames):
            ent_type = 'row_ents'
        else:
            exc = 'Unable to match outnames to summstat output.  Check outnames.'
            raise merge_summstat_Error(exc)

        # Build a merged Damon object for each ent_type
        merged = targ = summstat_out[outnames[0]][ent_type]
        merged.check_dups = None

        if len(outnames) > 1:
            for source in outnames[1:]:
                merged.merge(summstat_out[source][ent_type],
                               axis = {'target':0, 'source':0},
                               targ_data = True,
                               targ_labels = None,
                               source_ids = None,
                               nanval = merged.nanval)

                merged = dmn.core.Damon(merged.merge_out, 'datadict', 'RCD', verbose=None)

        # Validchars (merge() doesn't return helpful results)
        merged.validchars = merged.data_out['validchars'] = None

        # Rebuild column labels
        collabels = np.zeros(np.shape(merged.collabels), dtype='S60')
        cl_stats = tools.getkeys(merged, 'Col', 'Core', 'Auto', None)
        cl_names = np.repeat(outnames, len(np.unique(cl_stats)))
        collabels[merged.key4cols, merged.nheaders4rows:] = [cl_stats[i]+'_'+cl_names[i]
                                                             for i in range(len(cl_stats))]
        corner = targ.collabels[:, :targ.nheaders4rows]
        collabels[merged.key4cols, :merged.nheaders4rows] = corner
        merged.collabels = merged.data_out['collabels'] = collabels

        # Rebuild row labels
        rowlabels = np.copy(merged.rowlabels)
        rowlabels[0, :] = corner
        merged.rowlabels = merged.data_out['rowlabels'] = rowlabels

        # Create merged Damon object
        merged = dmn.core.Damon(merged.data_out, 'datadict',
                                workformat='RCD_dicts_whole', verbose=None)

        # Assign to Damon object
        name = ent_type+'_out'
        self.__dict__[name] = merged


    #####################
    ##  Merge row_ents ##
    ##   to col_ents   ##
    #####################

    elif merge_by == 'row2col':
        
        try:
            self.outnames
            exc = "When using merge_by = 'row2col', summstat() should only be run once. The merge is between the row and column entities in that run."
            raise merge_summstat_Error(exc)
        except AttributeError:
            pass

        # Get ent objects
        e = summstat_out['row_ents']
        f = summstat_out['col_ents']

        # Merge cores
        merge_core = np.append(e.coredata, f.coredata, axis=0)

        # Build collabels
        cl = np.zeros((1, 2 + np.size(e.coredata, axis=1)), dtype='S60')
        cl[0, :2] = np.array(['id', 'Ent_Type'])
        cl[0, 2:] = e.collabels[0, e.nheaders4rows:]

        # Build rowlabels
        e_rkeys = tools.getkeys(e, 'Row', 'Core', 'Auto', None).astype('S60')
        f_rkeys = tools.getkeys(f, 'Row', 'Core', 'Auto', None).astype('S60')
        rl = np.zeros((1 + len(e_rkeys) + len(f_rkeys), 2), dtype='S60')
        rl[0, :] = np.array(['id', 'Ent_Type'])
        merged_keys = np.append(e_rkeys, f_rkeys)
        rl[1:, 0] = merged_keys

        # Add ent labels
        e_ent_label = e.rowlabels[e.key4cols, e.key4rows]
        f_ent_label = f.rowlabels[f.key4cols, f.key4rows]
        rl[1:len(e_rkeys) + 1, 1] = e_ent_label
        rl[1 + len(e_rkeys):, 1] = f_ent_label

        datadict = {'rowlabels':rl, 'collabels':cl, 'coredata':merge_core,
                    'nheaders4rows':2, 'key4rows':0, 'rowkeytype':'S60',
                    'nheaders4cols':1, 'key4cols':0, 'colkeytype':'S60',
                    'nanval':e.nanval, 'validchars':['All', ['All'], 'Num'],
                    'check_dups':None}

        merged = dmn.core.Damon(datadict, 'datadict',
                                workformat='RCD_dicts_whole', verbose=None)


    return merged


######################################################################

def _plot_two_vars(_locals):
    "Basis of the plot_two_vars() method."

    self = _locals['self']
    xy_data_ = _locals['xy_data']
    x_name = _locals['x_name']
    y_name = _locals['y_name']
    err_data_ = _locals['err_data']
    x_err = _locals['x_err']
    y_err = _locals['y_err']
    color_by = _locals['color_by']
    ent_axis = _locals['ent_axis']
    cosine_correct = _locals['cosine_correct']
    max_cos = _locals['max_cos']
    plot_ = _locals['plot']
    savefig = _locals['savefig']

    # Overwrite plotting defaults
    plot = {}
    if isinstance(plot_, dict):
        for key in plot_.keys():
            plot[key] = plot_[key]

    # Define need plot parameters
    plot['savefig'] = savefig

    if 'xy_nchars' not in plot.keys():
        plot['xy_nchars'] = 5

    if 'y_line_ncuts' not in plot.keys():
        plot['y_line_ncuts'] = 4
        

    # Buffer against undefined solutions
    cos_theta_buffer = np.abs(1.0 - np.abs(max_cos))

    # Prepare Damon objects
    d = tools.check_datadict(self, xy_data_, [x_name, y_name], ent_axis, 'Core')

    if err_data_ == xy_data_ or err_data_ is None:
        e = d
    else:
        e = tools.check_datadict(self, err_data_, [x_err, y_err], ent_axis, 'Core')

    # Flag group colors
    group_colors = False
    if isinstance(color_by, list) and len(color_by) == 2:
        group_colors = True


    ##############
    ##  Cosine  ##
    ##  Correct ##
    ##############


    # WARNING:  There is a confusion between ent_axis and ent_type.  Will need additional work.



    # Handle cosine_correct
    if cosine_correct == 'coord':
        exc = None
        if ent_axis == 'row':
            exc = "When 'coord' is used, ent_axis needs to be 'col'."
        try:
            lookup = tools.lookup_coords(self, x_name, y_name, 'col_ents')
        except tools.lookup_coords_Error:
            exc = "Unable to use cosine_correct = 'coord' for some reason (see docs). Consider cosine_correct = 'coord|corr'.\n"

        if exc is not None:
            raise plot_two_vars_Error(exc)

    elif cosine_correct in ['coord', 'coord|corr']:
        try:
            lookup = tools.lookup_coords(self, x_name, y_name, 'col_ents')
        except tools.lookup_coords_Error:
            lookup = None

    # Error lookup function
    def get_errs(var, err, ent_axis):
        "Lookup or interpret err keys"

        if err is None:
            err_ = np.zeros(np.shape(var))
        elif isinstance(err, float):
            err_ = np.zeros(np.shape(var)) + err
        else:
            if ent_axis == 'col':
                err_ = e.core_col[err]
            elif ent_axis == 'row':
                err_ = e.core_row[err]
        return err_

    # Get values and cosine of B vector
    if ent_axis == 'col':
        fac = 'Row'
        a = d.core_col[x_name]
        b = d.core_col[y_name]

    elif ent_axis == 'row':
        fac = 'Col'
        coord_flag = False
        a = d.core_row[x_name]
        b = d.core_row[y_name]

    # Get errors
    a_err = get_errs(a, x_err, ent_axis)
    b_err = get_errs(b, y_err, ent_axis)

    # Clean out nanvals
    clean =(a != d.nanval) & (b != d.nanval)
    a = a[clean]
    b = b[clean]
    a_err = a_err[clean]
    b_err = b_err[clean]

    # Assign colors
    if group_colors is True:
        plotted_ent_axis = 'row' if ent_axis == 'col' else 'col'
        colors = tools.lookup_group_colors(d, color_by, plotted_ent_axis)
        plot['ellipse_color'] = [colors[i] for i in range(len(clean)) if clean[i] == 1]

    id_type = 'S'+str(plot['xy_nchars'])
    ids = tools.getkeys(d, fac, 'Core', 'Auto', None)[clean].astype(id_type)

    # Calculate cosine(theta)
    if cosine_correct is not None:
        plot['cosine_corrected'] = True
        if ent_axis == 'col':
            if cosine_correct == 'coord':
                x = lookup['x']
                y = lookup['y']
                coord_flag = True
            elif cosine_correct in ['coord', 'coord|corr']:
                if lookup is not None:
                    x = lookup['x']
                    y = lookup['y']
                    coord_flag = True
                else:
                    x = a
                    y = b
                    coord_flag = False
            else:
                x = a
                y = b
                coord_flag = False

            # Get the cosine
            cos_theta = tools.cosine(x, y, coord_flag, d.nanval)

        elif ent_axis == 'row':
            cos_theta = cosine(a, b, coord_flag, d.nanval)


    ##################
    ##  x, y coords ##
    ##################

    # Get x, y coordinates of
    if cosine_correct is not None:

        if np.abs(cos_theta) < cos_theta_buffer:
            Ax = a
            Ay = b
            exc = 'x and y variables are sufficiently uncorrelated that you should set cosine_correct = None.'
            raise plot_two_vars_Error(exc)
        elif np.abs(np.abs(cos_theta) - 1.0) < cos_theta_buffer:
            print 'cos_theta= ', cos_theta
            print 'buffer= ', cos_theta_buffer
            exc = 'x and y variables are too highly correlated. Unable to calculate y-coordinate.'
            raise plot_two_vars_Error(exc)
        else:
            theta = np.arccos(cos_theta)
            alpha = np.pi/2. - theta
            tan_alpha = np.tan(alpha)
            cos_alpha = np.cos(alpha)

            # Calculate A|y (see formula in docs)
            Ax = a
            Ay = ((b - a*cos_theta) / cos_theta) * tan_alpha

        # Get points on vector b
        uniques = np.unique(b)
        if len(uniques) < 0.10 * len(b):
            b_vals = uniques
        else:
            b_min = np.amin(b)
            b_max = np.amax(b)
            ncuts = plot['y_line_ncuts']
            increment = (b_max - b_min) / ncuts

            if increment < 1.0:
                increment = max(0.10, round(increment, 1))
                b_start = round(b_min - 3 * increment, 1)
            else:
                increment = round(increment, 0)
                b_start = round(b_min, 0)
                
            b_vals = [b_start + n * increment for n in range(int(ncuts + 6))]

        b_vals_ = np.around(np.array(b_vals), 1)

        
        Bx = [b_val * cos_theta for b_val in b_vals]
        By = [b_val * cos_alpha for b_val in b_vals]
        Bid = np.around([b_vals_[i] for i in range(len(b_vals_))], 1)

    # Do not correct for cosine
    else:
        Ax = a
        Ay = b
        Bx = By = None
        Bid = None
        theta = np.pi/2.

        # Get errors
        a_err = get_errs(a, x_err, ent_axis)
        b_err = get_errs(b, y_err, ent_axis)


    ##################
    ##  Plot Points ##
    ##################

    # Prepare colors
    if group_colors is False:
        nAx = len(Ax)

        if color_by is None:
            c_list = ['w' for i in range(nAx)]
        elif color_by is 'rand':
            c_list = [npr.rand(3) for i in range(nAx)]
        else:
            c_list = [color_by for i in range(nAx)]

        if tools.check_colors(c_list) is True:
            plot['ellipse_color'] = c_list
        else:
            exc = "Could not figure out colors in 'color_by'."
            raise plot_two_vars_Error(exc)

    # Plot points
    if savefig is None:
        fig = None
    else:
        import matplotlib.pyplot as plt

        fig = tools.plot(ids, Ax, Ay, Bid, Bx, By, a_err, b_err, x_name, y_name,
                         theta, plot['ellipse_color'], plot)
        if savefig == 'show':
            plt.show()
        else:
            plt.savefig(savefig)

    return {'figure':fig,
            'A_ids':ids,
            'x':Ax,
            'y':Ay,
            'B_ids':Bid,
            'Bx':Bx,
            'By':By,
            'a_err':a_err,
            'b_err':b_err,
            'x_name':x_name,
            'y_name':y_name,
            'theta':theta,
            'colors':plot['ellipse_color'],
            'plot_params':plot
            }




######################################################################

def _wright_map(_locals):
    "Basis of the wright_map() method"

    # Variables
    self = _locals['self']
    x_name = _locals['x_name']
    y_name = _locals['y_name']
    y_err = _locals['y_err']
    row_ents_color_by = _locals['row_ents_color_by']
    col_ents_color_by = _locals['col_ents_color_by']
    plot_ = _locals['plot']
    savefig = _locals['savefig']
    
    # Get primary Damon object
    names = [x_name, y_name, y_err]

    if x_name is None:
        names.remove(x_name)
        
    if x_name is 'rand':
        names.remove(x_name)

    if y_err is None:
        names.remove(y_err)

    if isinstance(y_err, (int, float)):
        names.remove(y_err)
        
    d = tools.check_datadict(self, 'merge_summstat_out',
                             names, 'col', 'Core')

    if x_name is None:
        x_name = 'rand'
    core = d.coredata

    if x_name is not 'rand':
        
        # Reverse the sign of row entities on x
        core_keys = tools.getkeys(d, 'Col', 'Core', 'Auto', None)
        col_ind = np.where(core_keys == x_name)
        fac_var = d.rl_col['Ent_Type']
        fac0_name = fac_var[0]
        fac1_name = fac_var[-1]
        row_ind = np.where(fac_var == fac0_name)

        # Check that all are positive
        var = core[row_ind, col_ind][0]
        if any(var < 0):            #[var != d.nanval] < 0):
            exc = 'The variable associated with x_name is not allowed to have negative values, including nanval.'
            raise wright_map_Error(exc)

        # Overwrite second facet with negative values
        core[row_ind, col_ind] *= -1

    else:
        fac_var = d.rl_col['Ent_Type']
        fac0_name = fac_var[0]
        fac1_name = fac_var[-1]       
        row_ind = np.where(fac_var == fac0_name)

    # Rebuild Damon object
    d.coredata = d.data_out['coredata'] = core
    d = dmn.core.Damon(d.data_out, 'datadict', 'RCD_dicts', verbose=None)

    # Get variables
    A_ids = tools.getkeys(d, 'Row', 'Core', 'Auto', None).astype('S60')

    if x_name is not 'rand':
        Ax = d.core_col[x_name]
    else:
        Ax = npr.rand(len(A_ids)) + 0.50
        Ax[row_ind] *= -1
    
    Ay = d.core_col[y_name]
    try:
        b_err = d.core_col[y_err]
    except KeyError:
        if y_err is None:
            y_err = 0.25
        b_err = np.zeros((len(A_ids))) + y_err
        
    #a_err done later
    
    # Clean the variables of nanvals
    clean = (Ax != d.nanval) & (Ay != d.nanval)
    A_ids = A_ids[clean]
    Ax = Ax[clean]
    Ay = Ay[clean]
    #a_err = a_err[clean]

    # Assign colors
    row_ent_colors = tools.lookup_group_colors(self.row_ents_out,
                                       row_ents_color_by,
                                       'row')
    col_ent_colors = tools.lookup_group_colors(self.col_ents_out,
                                               col_ents_color_by,
                                               'row')
    row_ent_colors.extend(col_ent_colors)
    
    # Clean colors
    colors = [row_ent_colors[i] for i in range(len(clean)) if clean[i] == 1]

    # Refine plot paramaters.  Buffers are added inside plot().    
    x_max = np.amax(np.abs(Ax))
    x_min = np.amin(np.abs(Ax))
    x_lim = (-1 * x_max, x_max)

    y_min = np.amin(Ay)
    y_max = np.amax(Ay)
    y_lim = (y_min, y_max)

    x_buff = 0.05
    y_buff = 0.05

    # The following is not used.  x-width is handled using buffer inside plot().
    a_err = np.copy(b_err) * 1 

    # Plot the chart
    B_ids = None
    Bx = None
    By = None
    theta = None

    # Set x-label
    space = '                   '
    x_lab = fac0_name+space+x_name+space+fac1_name

    # Prepare plot dictionary
    if plot_ is None:
        plot_ = {}

    # Defaults for this method
    plot = {}
    plot['savefig'] = savefig
    plot['aspect'] = 'auto'
    plot['xlim'] = x_lim
    plot['ylim'] = y_lim
    plot['x_buffer'] = x_buff
    plot['y_buffer'] = y_buff
    plot['xlabel'] = x_lab
    plot['shape'] = 'rectangle'

    # Overwrite with user specs
    for key in plot_.keys():
        plot[key] = plot_[key]

    # Plot points and edit them
    if savefig is not None:
        
        import matplotlib.pyplot as plt
        
        fig = tools.plot(A_ids, Ax, Ay, B_ids, Bx, By, a_err, b_err, x_name, y_name,
                   theta, colors, plot)

        y_buff_ = y_buff * (y_max - y_min)

        # Add vertical divider line
        plt.plot([0, 0], [y_min - y_buff_, y_max + y_buff_], 'k-')

        if savefig == 'show':
            plt.show()
        else:
            plt.savefig(savefig)

    else:
        fig = None

    
    return {'figure':fig,
            'A_ids':A_ids,
            'x':Ax,
            'y':Ay,
            'B_ids':None,
            'Bx':None,
            'By':None,
            'a_err':a_err,
            'b_err':b_err,
            'x_name':x_name,
            'y_name':y_name,
            'theta':theta,
            'colors':colors,
            'plot_params':plot
            }




######################################################################

def _equate(_locals):
    "Basis of the equate() method."

    # Get self
    self = _locals['self']

    # Get _locals
    construct_ents = _locals['construct_ents']
    label = _locals['label']
    subscales = _locals['subscales']
    facet = _locals['facet']
    correlated = _locals['correlated']
    logits_ = _locals['logits']
    rescale = _locals['rescale']
    refresh = _locals['refresh']
    cuts = _locals['cuts']
    stats = _locals['stats']
    group_se = _locals['group_se']

    ##############
    ##   Get    ##
    ##  Coords  ##
    ##############

    # Get opposing entity coordinates
    Coord = {}
    try:
        Coord[0] = self.coord_out['fac0coord']
        Coord[1] = self.coord_out['fac1coord']
        nanval = self.coord_out['fac0coord']['nanval']
    except AttributeError:
        exc = ('Unable to find coord_out.  Run coord().  Use anchors if '
               'building off an existing bank (see docs).\n')
        raise equate_Error(exc)

    # Get EAR coordinates
    ear_coord = {}
    try:
        ear_coord[0] = self.base_ear_out['ear_coord']['fac0coord']
        ear_coord[1] = self.base_ear_out['ear_coord']['fac1coord']
    except AttributeError:
        ear_coord = None

    # Get SE coordinates
    se_coord = {}
    try:
        se_coord[0] = self.base_se_out['se_coord']['fac0coord']
        se_coord[1] = self.base_se_out['se_coord']['fac1coord']
    except AttributeError:
        se_coord = None

    # Get refit params
    refit_params = {}
    try:
        refit_params['ear'] = self.base_ear_out['refit']
    except AttributeError:
        refit_params['ear'] = None
    
    try:
        refit_params['se'] = self.base_se_out['refit']
    except AttributeError:
        refit_params['se'] = None
        

    ##############
    ##   Get    ##
    ##   Bank   ##
    ##############

    bankflag = False
    try:
        BankFile = self.coord_out['anchors']['Bank']
        Facet = self.coord_out['anchors']['Facet']
        OppFacet = list(set([0,1]) - set([Facet]))[0]
    except (KeyError,TypeError,AttributeError):
        bankflag = True

    # Build a new bank
    if bankflag is True:
        self.bank(filename = None,
                  bankf0 = {'Remove':[None],'Add':['All']},
                  bankf1 = {'Remove':[None],'Add':['All']}
                  )
        BankFile = self.bank_out

    # Get the bank
    if isinstance(BankFile,dict):
        Bank = BankFile
        if facet is None:
            exc = ('In the absence of anchoring to a bank in coord(), you '
                   'need to specify the facet.\n')
            raise equate_Error(exc)
        else:
            Facet = facet
            OppFacet = list(set([0,1]) - set([Facet]))[0]
    else:
        try:
            file_ = open(BankFile,'rb')
            Bank = cPickle.load(file_)
            file_.close()
        except IOError:
            exc = 'Unable to find bank pickle file.\n'
            raise equate_Error(exc)

    # Build equate_params dict
    equate_params = {}
    for key in _locals:
        if key == 'self':
            pass
        else:
            equate_params[key] = _locals[key]

    # Load equate() parameters TO empty bank
    params = ['construct_ents', 'label', 'subscales', 'facet',
              'logits', 'rescale', 'refresh', 'cuts', 'stats', 'group_se']
    if Bank['equate_params'] is None:
        for k in params:
            
            # No 'Bank' allowed if no stored params
            if equate_params[k] in ['Bank', 'Auto']:
                exc = ("Unable to get equate() parameters using 'Bank'. "
                       "Specify construct_ents and/or subscales.\n")
                raise equate_Error(exc)
                
        # Assign existing params to bank
        else:
            Bank['equate_params'] = equate_params

    else:
        if ('Bank' in equate_params.values() or
            'Auto' in equate_params.values()
            ):
            equate_params = Bank['equate_params']
            construct_ents = Bank['equate_params']['construct_ents']
            label = Bank['equate_params']['label']
            subscales = Bank['equate_params']['subscales']
            facet = Bank['equate_params']['facet']
            correlated = Bank['equate_params']['correlated']
            logits_ = Bank['equate_params']['logits']
            rescale = Bank['equate_params']['rescale']
            refresh = Bank['equate_params']['refresh']
            cuts = Bank['equate_params']['cuts']
            stats = Bank['equate_params']['stats']
            group_se = Bank['equate_params']['group_se']
        else:
            for k in params:
                Bank['equate_params'][k] = equate_params[k]

    # Extract entity info
    if Facet == 0:
        Fac = 'facet0'
    elif Facet == 1:
        Fac = 'facet1'

    # Other variables
    ndim = Bank['ndim']
    if refresh is None:
        refresh = []


    #################
    ##   Define    ##
    ##  Construct  ##
    ##  function   ##
    #################

    # TODO: Ents with nanvals are not allowed in the bank, but may
    # appear in fac0coord, fac1coord, causing a mismatch. Currently,
    # coords with nanvals will be omitted from constructs, but there is
    # no guarantee of consistency in nanvals between ent_coords, ear_coords,
    # and se_coords.


    def construct(bank, fac, coordtype, ndim, ents, mnsq, correlated, nanval):
        "Average coordinates to get construct"
        
        ent_coord = []
        for ent in ents:
            try:
                ent_coord.append(bank[fac][coordtype][ent])
            except KeyError:
                pass
        ent_coord = np.array(ent_coord)
        ix = np.unique(np.where(ent_coord != nanval)[0])
        
        # Is this necessary any more?
        if mnsq:
                
            # This calculates construct EAR and SE
            try:
                construct = np.mean(ent_coord[ix, :], axis=0)
            except:
                print '\nEntities in bank=\n', sorted(bank[fac][coordtype].keys())
                print '\nEntities in scale=\n', sorted(ents)
                exc = 'Unable to find coordinates for the required entities.'
                raise equate_Error(exc)
        else:
            construct = np.mean(ent_coord[ix, :], axis=0)
        
        return construct
        

    #################
    ##    Ents     ##
    ##  Construct  ##
    #################

    nconstr = 0
    
    if construct_ents is not None:
        nconstr += 1

        # Get list of construct entities
        AllEnts = Bank[Fac]['ent_coord'].keys()

        if construct_ents['Get'] == 'AllExcept':
            Ents = list(set(AllEnts) - set(construct_ents['Ents']))
        elif construct_ents['Get'] == 'NoneExcept':
            if construct_ents['Ents'] == ['All']:
                Ents = AllEnts
            else:
                Ents = construct_ents['Ents']

        # Get construct coordinates
        if label in refresh or label not in Bank[Fac]['ent_coord']:
            ent_coord_ = construct(Bank, Fac, 'ent_coord', ndim,
                                   Ents, None, correlated, nanval)
            Bank[Fac]['ent_coord'][label] = ent_coord_
        else:
            ent_coord_ = Bank[Fac]['ent_coord'][label]
        
        ent_coord = ent_coord_[:, np.newaxis]

        # Get EAR and SE coordinates
        if Bank[Fac]['ear_coord'] != {}:
            if label in refresh or label not in Bank[Fac]['ear_coord']:
                EntEARCoord_ = construct(Bank, Fac, 'ear_coord', 1,
                                         Ents, True, correlated, nanval)
                Bank[Fac]['ear_coord'][label] = EntEARCoord_
            else:
                EntEARCoord_ = Bank[Fac]['ear_coord'][label]
            
            EntEARCoord = EntEARCoord_[:, np.newaxis]
            
        if Bank[Fac]['se_coord'] != {}:
            if label in refresh or label not in Bank[Fac]['se_coord']:
                EntSECoord_ = construct(Bank, Fac, 'se_coord', 1,
                                        Ents, True, correlated, nanval)
                Bank[Fac]['se_coord'][label] = EntSECoord_
            else:
                EntSECoord_ = Bank[Fac]['se_coord'][label]
            
            EntSECoord = EntSECoord_[:, np.newaxis]


    #################
    ##    Subs     ##
    ##  Constructs ##
    #################

    if subscales is not None:
        
        # Figure out what the subscales are

        # Convert labels and keys into arrays
        Keys = Bank[Fac]['ent_coord'].keys()
        KeyArr = np.array([key for key in Keys if key in Bank[Fac]['labels']])
        ValArr = np.array([Bank[Fac]['labels'][key] for key in Keys
                           if key in Bank[Fac]['labels']])

        # Get row of entity attributes
        ind = subscales['Labels']
        try:
            SubsRow = ValArr[:,ind]
        except (IndexError,ValueError):
            exc = "Could not figure out subscales['Labels'] parameter."
            raise equate_Error(exc)

        # These are the available subscales
        AllSubs = np.unique(SubsRow)

        # Filter out nanvals among subs
        if isinstance(AllSubs[0],str):
            nan = str(nanval)
        else:
            nan = nanval

        AllSubs = AllSubs[AllSubs != nan]

        # Refine list of subs
        if subscales['Get'] == 'AllExcept':
            Subs = list(set(AllSubs) - set(subscales['Subs']))
        elif subscales['Get'] == 'NoneExcept':
            Subs = list(set(subscales['Subs']) & set(AllSubs))
        Subs.sort()

        # Initialize array of coords for all subs
        nsubs = len(Subs)
        nconstr += nsubs
        SubCoord = np.zeros((ndim, nsubs))
        SubEARCoord = np.zeros((2, nsubs))
        SubSECoord = np.zeros((2, nsubs))

        for i, Sub in enumerate(Subs):

            # Entity coordinates
            if Sub in refresh or Sub not in Bank[Fac]['ent_coord']:
                Loc = np.where(SubsRow == Sub)
                SubEnts = KeyArr[Loc]
                SubCoord_ = construct(Bank, Fac, 'ent_coord', ndim, 
                                      SubEnts, None, correlated, nanval)
                Bank[Fac]['ent_coord'][Sub] = SubCoord_
            else:
                SubCoord_ = Bank[Fac]['ent_coord'][Sub]
            
            SubCoord[:, i] = SubCoord_

            # Get EAR and SE coordinates (ignore if dicts are empty)
            if bool(Bank[Fac]['ear_coord']):
                if Sub in refresh or Sub not in Bank[Fac]['ear_coord']:
                    SubEARCoord_ = construct(Bank, Fac, 'ear_coord', 1,
                                             SubEnts, True, correlated, nanval)      
                    Bank[Fac]['ear_coord'][Sub] = SubEARCoord_
                else:
                    SubEARCoord_ = Bank[Fac]['ear_coord'][Sub]

                SubEARCoord[:, i] = SubEARCoord_
               
            if bool(Bank[Fac]['se_coord']):
                if Sub in refresh or Sub not in Bank[Fac]['se_coord']:
                    SubSECoord_ = construct(Bank, Fac, 'se_coord', 1,
                                            SubEnts, True, correlated, nanval)
                    Bank[Fac]['se_coord'][Sub] = SubSECoord_
                else:
                    SubSECoord_ = Bank[Fac]['se_coord'][Sub]
                    
                SubSECoord[:, i] = SubSECoord_


    ######################
    ##  Calc Construct  ##
    ##     Measures     ##
    ######################

    OppEntCoords = Coord[OppFacet]['coredata']
    nOppEnts = np.size(OppEntCoords, axis=0)
    Measures = np.zeros((nOppEnts, nconstr))
    collabels = np.zeros((1, nconstr),dtype='S60')
    substart = 0

    # Construct measures
    if construct_ents is not None:
        EntMeas = np.ravel(np.dot(OppEntCoords, ent_coord))
        Measures[:, 0] = EntMeas 
        collabels[:, 0] = label
        substart += 1

    # Subscale measures
    if subscales is not None:
        SubMeas = np.dot(OppEntCoords, SubCoord)
        Measures[:, substart:] = SubMeas
        collabels[:, substart:] = Subs

    # Capture original mean, sd and store in rescale params
    if len(Measures) > 1:
        if rescale is not None:
            orig_means = tools.mean(Measures, 0, nanval)
            orig_sds = tools.std(Measures, 0, nanval)
            rs_keys = rescale.keys()
            
            # Redo rescale params to be column based
            if ('All' in rs_keys and len(rs_keys) == 1):
                rescale_ = {}
                for col in collabels[0, :]:
                    rescale_[col] = {}
                    for key in rescale['All'].keys(): #rs_keys:
                        rescale_[col]
                         
                        # I'm not sure why I had to redo this.
                        try:
                            rescale_[col][key] = rescale['All'][key]
                        except:
                            print 'rs_keys=', rs_keys
                            print 'rescale=\n', rescale
                            print 'rescale_=\n', rescale_
                            print 'key=', key
                            print 'col=', col
                            rescale_[col][key] = rescale['All'][key]
                rescale = rescale_
            
            # Add original means, stds
            for i, col in enumerate(collabels[0, :]):
                rsc = rescale[col]                    
                if ('mean_sd' in rsc
                    and rsc['mean_sd'] is not None
                    and len(rsc['mean_sd']) == 2):
                    rescale[col]['mean_sd'].extend([orig_means[i], orig_sds[i]])
            
            # Add revised params to bank
            Bank['equate_params']['rescale'] = rescale                                
        
    # Apply NaNVals
    NaNLoc = np.where(OppEntCoords == nanval)[0]
    Measures[NaNLoc] = nanval


    ######################
    ##  Calc Construct  ##
    ##      EARs        ##
    ######################

    if ear_coord is not None:
        OppEARCoords = ear_coord[OppFacet]['coredata']
        EAR = np.zeros((nOppEnts, nconstr))
        substart = 0

        # Entity construct EARs 
        if construct_ents is not None:
            EntEAR = np.exp(np.dot(OppEARCoords, EntEARCoord))
            EntEAR = np.clip(Bank['refit_params']['ear'](EntEAR), 0.0001, np.inf)
            EntEAR = np.ravel(np.sqrt(EntEAR))
            
            if group_se:
                EntEAR = tools.group_se(EntEAR, group_se, ndim, Facet,
                                        nrows=len(OppEARCoords),
                                        ncols=len(Ents))            
            try:
                EAR[:, 0] = EntEAR
            except ValueError:
                print 'EntEAR shape=', np.shape(EntEAR)
                print 'EAR shape=', np.shape(EAR)
                exc = ('Got a mismatch in EAR arrays. Check that all rows/cols '
                       'have valid data and variation or use extract_valid().')
                raise equate_Error(exc)
            substart += 1

        # EARs for subscales
        if subscales is not None:
            SubEAR = np.exp(np.dot(OppEARCoords, SubEARCoord))
            SubEAR = np.clip(Bank['refit_params']['ear'](SubEAR), 0.0001, np.inf)

            # TODO: handle ncols for subscales
            if group_se:
                SubEAR = tools.group_se(SubEAR, group_se, ndim, Facet,
                                        nrows=len(OppEARCoords),
                                        ncols=None)            
            EAR[:, substart:] = SubEAR

        # Apply NaNVals
        NaNLoc = np.where(OppEARCoords == nanval)[0]
        EAR[NaNLoc] = nanval
        

    ######################
    ##  Calc Construct  ##
    ##      SEs        ##
    ######################

    if se_coord is not None:
        OppSECoords = se_coord[OppFacet]['coredata']
        SE = np.zeros((nOppEnts, nconstr))
        substart = 0

        # Entity construct SEs
        if construct_ents is not None:
            EntSE = np.exp(np.dot(OppSECoords, EntSECoord))
            EntSE = np.clip(Bank['refit_params']['se'](EntSE), 0.0001, np.inf)
#            EntSE = np.ravel(np.sqrt(EntSE))
            
            if group_se:
                EntSE = tools.group_se(EntSE, group_se, ndim, Facet,
                                       nrows=len(OppSECoords),
                                       ncols=len(Ents))
            
            SE[:, 0] = EntSE[:, 0]
            substart += 1

        # SEs for subscales
        if subscales is not None:
            SubSE = np.exp(np.dot(OppSECoords, SubSECoord))
            SubSE = np.clip(Bank['refit_params']['se'](SubSE), 0.0001, np.inf)

            if group_se:
                SubSE = tools.group_se(SubSE, group_se, ndim, Facet,
                                       nrows=len(OppSECoords),
                                       ncols=None)            
            SE[:, substart:] = SubSE

        # Apply NaNVals
        NaNLoc = np.where(OppSECoords == nanval)[0]
        SE[NaNLoc] = nanval


    #################
    ##   Rescale   ##
    #################

    # Convert to logits to get revised Measures, EAR, SE
    if logits_:
        d = {'ecut':['All', 0.0],
             'ear':1.0, 
             'missing':False,
             'count_chars':True,
             'max_chars':5,
             'p_items':None,
             'meth':'CombineFacs'
             }
        
        if isinstance(logits_, dict):
            for key in logits_.keys():
                d[key] = logits_[key]

        plog = tools.cumnormprob(Measures, d['ear'], None, d['ecut'], True, 
                                 nanval)
        logits = plog['Logit']
        p = plog['Prob']
        
        # Adjust ears to be consistent with new logits
        nix = (logits == nanval) | (Measures == nanval)
        ear_factor = np.std(logits[~nix]) / np.std(Measures[~nix])
        EAR = EAR * ear_factor
        EAR[nix] = nanval
        
        # Get cell binomial information
        pq = np.where(p == nanval, nanval, np.clip(p * (1 - p), 0.001, 0.999))
        
        # Get obspercell_factor
        try:
            obs = self.standardize_out['coredata']
        except AttributeError:
            try:
                obs = self.extract_valid_out['coredata']
            except:
                obs = self.data_out['coredata']
        
        nfacs = self.coord_out['facs_per_ent']
        n_ents = np.size(obs, axis=Facet)
            
        if Facet == 0:
            by_rows = Bank['shape'][OppFacet]
            by_cols = 'obs' if d['missing'] else Coord[OppFacet]['opp_count']
            fac = 'row'
        elif Facet == 1:
            by_rows = 'obs' if d['missing'] else Coord[OppFacet]['opp_count']
            by_cols = Bank['shape'][OppFacet]
            fac = 'col'

        opc_fact = tools.obspercell(obs, by_rows, by_cols, fac, ndim, nfacs,
                                    d['count_chars'], d['max_chars'],
                                    d['p_items'], d['meth'],
                                    nanval)[:, np.newaxis]

        # Adjust if columns are correlated
        # TODO:  Is correlated needed here?
#        corr_fact = 1.0 / np.sqrt(n_ents) #if not correlated else 1.0
#        SE = opc_fact * corr_fact / np.sqrt(pq)
        SE = opc_fact / np.sqrt(pq)

        if group_se:
            SE = tools.group_se(SE, group_se, ndim, Facet,
                                nrows=len(OppSECoords),
                                ncols=None)
            
        # Catch nanvals
        nix = (opc_fact == nanval) | (pq == nanval)
        SE[nix] = nanval
        Measures = logits
    
    # Apply rescale parameters to Measures, EAR, SE
    if rescale is not None:
    
        # Construct measures
        orig_measures = np.copy(Measures)
        rs_keys = rescale.keys()

        if ('All' in rs_keys
            and len(rs_keys) == 1
            ):
            rs_args = rescale['All'].copy()
            rs_args['nanval'] = nanval
            
            # Apply other rescale options
            for i in range(np.size(Measures, axis=1)):
                rs_args['score'] = Measures[:, i]
                Measures[:, i] = np.ravel(tools.rescale(**rs_args))

        # Do by column
        else:
            struct_dict = rescale
            for key in struct_dict.keys():
                loc = np.where(collabels == str(key))[1]
                if len(loc) == 0:
                    continue
                rs_args = struct_dict[key].copy()
                rs_args['nanval'] = nanval
                
                try:
                    rs_args['score'] = Measures[:,loc]
                    Measures[:,loc] = tools.rescale(**rs_args)
                except (IndexError,SystemExit):
                    pass

    # Rescale EARs
    if rescale is not None and ear_coord is not None:
        if ('All' in rs_keys
            and len(rs_keys) == 1
            ):
            rs_args = rescale['All'].copy()
            rs_args['nanval'] = nanval
            rs_args['clip'] = None # TODO -- Could cause weird EARs at extremes
            
            for i in range(np.size(EAR,axis=1)):
                if 'mean_sd' in rs_args:
                    t_sd = rs_args['mean_sd'][1]
                    om = orig_measures[:,i]
                    s_sd = np.std(om[om != nanval])
                    m = t_sd / s_sd
                elif 'm_b' in rs_args:
                    m = rs_args['m_b'][0]
                else:
                    m = 1
                rs_args_ = {'score':EAR[:,i],'m_b':[m,0],'nanval':nanval}
                EAR[:,i] = tools.rescale(**rs_args_)

        else:
            struct_dict = rescale
            for key in struct_dict.keys():
                loc = np.where(collabels == str(key))[1]
                if len(loc) == 0:
                    continue
                rs_args = struct_dict[key].copy()
                if 'mean_sd' in rs_args:
                    t_sd = rs_args['mean_sd'][1]
                    try:
                        om = orig_measures[:,loc]
                        s_sd = np.std(om[om != nanval])
                        m = t_sd / s_sd
                    except IndexError:
                        pass
                elif 'm_b' in rs_args:
                    m = rs_args['m_b'][0]
                else:
                    m = 1
                try:
                    rs_args_ = {'score':EAR[:,loc],'m_b':[m,0],'nanval':nanval}
                    EAR[:,loc] = tools.rescale(**rs_args_)
                except (IndexError,SystemExit):
                    pass

    # Rescale SEs
    if rescale is not None and se_coord is not None:
        if ('All' in rs_keys
            and len(rs_keys) == 1
            ):
            rs_args = rescale['All'].copy()
            rs_args['nanval'] = nanval
            rs_args['clip'] = None # TODO -- Could cause weird SEs at extremes
            
            for i in range(np.size(SE,axis=1)):
                if 'mean_sd' in rs_args:
                    t_sd = rs_args['mean_sd'][1]
                    om = orig_measures[:, i]
                    s_sd = np.std(om[om != nanval])
                    m = t_sd / s_sd
                elif 'm_b' in rs_args:
                    m = rs_args['m_b'][0]
                else:
                    m = 1
                rs_args_ = {'score':SE[:,i],'m_b':[m,0], 'nanval':nanval}
                SE[:,i] = tools.rescale(**rs_args_)
        else:
            struct_dict = rescale
            for key in struct_dict.keys():
                loc = np.where(collabels == str(key))[1]
                if len(loc) == 0:
                    continue
                rs_args = struct_dict[key].copy()
                if 'mean_sd' in rs_args:
                    t_sd = rs_args['mean_sd'][1]
                    try:
                        om = orig_measures[:, loc]
                        s_sd = np.std(om[om != nanval])
                        m = t_sd / s_sd
                    except IndexError:
                        pass
                elif 'm_b' in rs_args:
                    m = rs_args['m_b'][0]
                else:
                    m = 1
                try:
                    rs_args_ = {'score':SE[:,loc],'m_b':[m,0],'nanval':nanval}
                    SE[:,loc] = tools.rescale(**rs_args_)
                except (IndexError,SystemExit):
                    pass


    ##################
    ##  Performance ##
    ##   Levels     ##
    ##################

    if cuts is not None:
        
        def apply_cuts(arr, cuts_, nanval):
            "Apply cuts to array to convert into performance levels."

            pls = np.zeros(np.shape(arr))
            for i, cut in enumerate(cuts_):
                pls[arr >= cut] = i + 1
            pls[arr == nanval] = nanval
            
            return pls

        # Calculate cut-points if unavailable
        if isinstance(cuts.values()[0], str):
            ckeys = cuts.keys()
            
            def get_ratings(self, cuts, col):
                try:
                    ratings = self.core_col[cuts[col]].astype(float)
                except KeyError:
                    try:
                        ratings = self.rl_col[cuts[col]].astype(float)
                    except KeyError:
                        exc = 'Could not find the ratings column for cuts.'
                        raise equate_Error(exc)
                return ratings
                
            if ('All' in ckeys 
                and len(ckeys) == 1 
                and construct_ents is not None
                ):
                meas = Measures[:, 0]
                ratings = get_ratings(self, cuts, 'All')
                cuts_ = tools.get_cuts(meas, ratings, nanval)
                cuts = {'All':cuts_}
            elif 'All' in ckeys:
                exc = ('construct_ents need to be specified when calculating '
                       'cutpoints using "All".')
                raise equate_Error(exc)
            else:
                cuts_ = {}
                for i, col in enumerate(collabels[0]):
                    if col in ckeys:
                        meas = Measures[:, i]
                        ratings = get_ratings(self, cuts, col)
                        cuts_[col] = tools.get_cuts(meas, ratings, nanval)
                cuts = cuts_
            
            # Store new cuts in bank
            Bank['equate_params']['cuts'] = cuts
                
        # Apply cuts to all columns at once
        ckeys = cuts.keys()
        if 'All' in ckeys and len(ckeys) == 1:
            pls = apply_cuts(Measures, cuts['All'], nanval)
        
        # Apply cuts to one column at a time
        else:
            pls = np.zeros(np.shape(Measures)) + nanval
            for key in ckeys:
                loc = np.where(collabels == str(key))[1]
                if len(loc) == 0 or cuts[key] is None:
                    continue
                pls[:, loc] = apply_cuts(Measures[:, loc], cuts[key], nanval)
                
                
    #################
    ##  Get Stats  ##
    #################
    
    rpt_stats = False
    
    if stats is True and np.size(Measures, axis=0) > 1:
        rpt_stats = True
        stats_ = np.array(['Stats', 'Count', 'Mean', 'SD', 'SE', 'Sep', 
                           'Rel', 'Min', '25Perc', 'Median', 
                           '75Perc', 'Max'])[:, np.newaxis]
        ncols = np.size(Measures, axis=1)
        stats = np.zeros((len(stats_) - 1, ncols))
        
        # Apply stats to row.  "0" refers to axis=0.
        stats[0, :] = tools.count(Measures, 0, nanval)
        stats[1, :] = tools.mean(Measures, 0, nanval)
        stats[2, :] = sd = tools.std(Measures, 0, nanval)
        
        # Do rmse column-wise if SE is available
        if se_coord is not None:
            for i in range(ncols):
                stats[3, i] = tools.rmsr(None, None, SE[:, i], nanval, False)
                
            rmse = stats[3, :]
            stats[4, :] = sep = tools.separation(sd, rmse, None, nanval)
            stats[5, :] = tools.reliability(sep, None, None, None, nanval)
        else:
            stats[3, :] = nanval
            stats[4, :] = nanval
            stats[5, :] = nanval
            
        stats[6, :] = tools.amin(Measures, 0, nanval)
        
        stats[8, :] = med = tools.median(Measures, 0, nanval)
        stats[10, :] = tools.amax(Measures, 0, nanval)
    
        # Do percentiles column-wise
        for i in range(ncols):
            stats[7, i] = tools.percent25(Measures[:, i], med[i], nanval)
            stats[9, i] = tools.percent75(Measures[:, i], med[i], nanval)

        
    #################
    ##   Build     ##
    ##  DataDicts  ##
    #################

    CoordDict = Coord[OppFacet]
    rowlabels = Coord[OppFacet]['rowlabels']
    key4cols = Coord[OppFacet]['key4cols']

    try:
        Subs
    except UnboundLocalError:
        Subs = []

    if construct_ents is not None:
        collabels = np.concatenate(([rowlabels[key4cols,:],[label],Subs]))[np.newaxis,:]
    else:
        collabels = np.concatenate(([rowlabels[key4cols,:],Subs]))[np.newaxis,:]

    # Construct datadict
    ConstructDict = {'rowlabels':rowlabels, 
                     'collabels':collabels,
                     'coredata':Measures, 
                     'nheaders4rows':CoordDict['nheaders4rows'],
                     'key4rows':CoordDict['key4rows'],
                     'rowkeytype':CoordDict['rowkeytype'],
                     'nheaders4cols':1,
                     'key4cols':0,
                     'colkeytype':'S60',
                     'nanval':nanval,
                     'validchars':['All',['All'],'Num']
                     }

    # EAR datadict
    if ear_coord is not None:
        EARDict = {}
        for key in ConstructDict.keys():
            EARDict[key] = ConstructDict[key]
            EARDict['coredata'] = EAR
    else:
        EARDict = None

    # SE datadict
    if se_coord is not None:
        SEDict = {}
        for key in ConstructDict.keys():
            SEDict[key] = ConstructDict[key]
            SEDict['coredata'] = SE
    else:
        SEDict = None

    # Performance level datadict
    if cuts is not None:
        pls_dict = {}
        for key in ConstructDict.keys():
            pls_dict[key] = ConstructDict[key]
            pls_dict['coredata'] = pls
    else:
        pls_dict = None

    # Stats datadict
    if rpt_stats:
        k4r = CoordDict['key4rows']
        nh4r = CoordDict['nheaders4rows']
        cl = np.concatenate((['Stats'], collabels[k4r, nh4r:]))[np.newaxis]
        stats_dict = {'rowlabels':stats_, 'collabels':cl, 'coredata':stats,
                      'nheaders4rows':1, 'key4rows':0,'rowkeytype':'S60',
                      'nheaders4cols':1, 'key4cols':0, 'colkeytype':'S60',
                      'nanval':nanval, 'validchars':['All', ['All'], 'Num']}
    else:
        stats_dict = None
    
    # Attach outputs to Damon object (easier to export)
    self.__dict__['equate_out_construct'] = ConstructDict
    self.__dict__['equate_out_se'] = SEDict
    self.__dict__['equate_out_pl'] = pls_dict
    self.__dict__['equate_out_stats'] = stats_dict

    # Phase in lower case naming
    return {'Construct':ConstructDict,
            'construct':ConstructDict,
            'EAR':EARDict,
            'ear':EARDict,
            'SE':SEDict,
            'se':SEDict,
            'PLs':pls_dict,
            'pls':pls_dict,
            'Stats':stats_dict,
            'stats':stats_dict,
            'cuts':cuts,
            'equate_params':equate_params,
            }



######################################################################

def _bank(_locals):
    "Basis of the bank() method."

    # Get self
    self = _locals['self']

    # Get _locals
    filename = _locals['filename']
    bankf0 = _locals['bankf0']
    bankf1 = _locals['bankf1']
    New = False
    Rasch = False

    ###################
    ##      Get      ##
    ##  Ingredients  ##
    ###################

    # Rasch flag
    try:
        self.rasch_out
        Rasch = True
    except AttributeError:
        pass

    # Get entity coordinates
    try:
        fac0coord = self.coord_out['fac0coord']
        fac1coord = self.coord_out['fac1coord']
        ndim = self.coord_out['ndim'] if self.coord_out['ndim'] != 0 else 1
        shape = (fac1coord['opp_count'], fac0coord['opp_count'])

        # Get Rasch step parameters (if available)
        try:
            step_coord = fac0coord['step_coord']
        except KeyError:
            try:
                step_coord = fac1coord['step_coord']
            except KeyError:
                step_coord = None
    except AttributeError:
        exc = 'Unable to find coord_out. Run coord() or rasch().\n'
        raise bank_Error(exc)

    # Get ear_coord and se_coord (coord() only)
    if Rasch is False:

        # Get EAR coordinates
        try:
            Fac0EARCoord = self.base_ear_out['ear_coord']['fac0coord']
            Fac1EARCoord = self.base_ear_out['ear_coord']['fac1coord']
        except AttributeError:
            Fac0EARCoord = None
            Fac1EARCoord = None

        # Get SE coordinates
        try:
            Fac0SECoord = self.base_se_out['se_coord']['fac0coord']
            Fac1SECoord = self.base_se_out['se_coord']['fac1coord']
        except AttributeError:
            Fac0SECoord = None
            Fac1SECoord = None

        # Get standardize() answer key
        try:
            anskey_param = self.score_mc_out['anskey']
        except AttributeError:
            anskey_param = None

        # Get std_params
        try:
            std_params = self.standardize_out['std_params']
        except AttributeError:
            std_params = None

        # Get parse_params
        try:
            parse_params = self.parse_out['parse_params']
        except AttributeError:
            parse_params = None

        # Get refit params
        refit_params = {'ear':None, 'se':None}
        try:
            refit_params['ear'] = self.base_ear_out['refit']
        except AttributeError:
            pass
        
        try:
            refit_params['se'] = self.base_se_out['refit']
        except AttributeError:
            pass

        # Initialize equate_params
        try:
            equate_params = self.equate_out['equate_params']
        except AttributeError:
            equate_params = None
    else:
        Fac0EARCoord = None
        Fac1EARCoord = None
        Fac0SECoord = None
        Fac1SECoord = None
        anskey_param = None
        std_params = None
        parse_params = None
        refit_params = None
        equate_params = None


    ##################
    ##  Convert to  ##
    ##     Dicts    ##
    ##################

    ObjList = [fac0coord, fac1coord, Fac0EARCoord, Fac1EARCoord, Fac0SECoord,
               Fac1SECoord]
    DictList = ['fac0coord', 'fac1coord', 'Fac0EARCoord', 'Fac1EARCoord',
                'Fac0SECoord', 'Fac1SECoord']
    Dicts = {}

    # Create dicts with workformat = 'RCD_dicts'
    for i in range(len(DictList)):
        try:
            Dicts[DictList[i]] = dmn.core.Damon(data = ObjList[i],    # [<array, file, [file list], datadict, Damon, hd5 file>  => data in format specified by format_=]
                                         format_ = 'datadict',    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','Damon','hd5','pickle'>]
                                         workformat = 'RCD_dicts',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                                         validchars = ['All',['All'],'Num'],   # [<None, ['All',[valid chars]], or ['Cols', {'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'], 'ID4':['0 -- '],...}]>]
                                         nheaders4rows = ObjList[i]['nheaders4rows'],  # [N columns to hold row labels]
                                         key4rows = ObjList[i]['key4rows'],   # [Nth column from left which holds row keys]
                                         rowkeytype = ObjList[i]['rowkeytype'],     # [data type of row keys]
                                         nheaders4cols = ObjList[i]['nheaders4cols'],  # [N rows to hold column labels]
                                         key4cols = ObjList[i]['key4cols'], # [Nth row from top which holds column keys]
                                         colkeytype = ObjList[i]['colkeytype'],     # [data type of column keys]
                                         dtype = [object,8,None], # [[Numpy string type of whole matrix when cast as string, <None, int number of decimals>], e.g. ['S60',8]]
                                         nanval = ObjList[i]['nanval'],    # [Value to which non-numeric/invalid characters should be converted.]
                                         verbose = None,    # [<None, True> => report method calls]
                                         ).data_out
        except TypeError:
            Dicts[DictList[i]] = None


    ################
    ##   Define   ##
    ##  modbank() ##
    ################

    def modbank(FacBank, Dict, Remove, Add, nanval):
        "Modify bank using specified dictionary."

        # Remove entities from bank
        if Remove == [None]:
            pass
        elif Remove == ['All']:
            FacBank.clear()
        else:
            for Ent in Remove:
                del FacBank[Ent]

        # Add entities to bank
        if Add == [None]:
            pass
        elif Add == ['All']:
            for Ent in Dict.keys():
#                FacBank[Ent] = Dict[Ent]
                if isinstance(Dict[Ent][0], float):
                    if np.any(Dict[Ent] == nanval):
                        pass
                    else:
                        FacBank[Ent] = Dict[Ent]
                else:
                    FacBank[Ent] = Dict[Ent]
        else:
            for Ent in Add:
#                FacBank[Ent] = Dict[Ent]
                if isinstance(Dict[Ent][0], float):
                    if np.any(Dict[Ent] == nanval):
                        pass
                    else:
                        FacBank[Ent] = Dict[Ent]
                else:
                    FacBank[Ent] = Dict[Ent]

        return FacBank

    ##############
    ##  Build   ##
    ##  Bank    ##
    ##############

    try:
        Bank = self.bank_out # equate() outputs self.bank_out
    except AttributeError:
        try:
            file_ = open(filename,'rb')
            Bank = cPickle.load(file_)
            file_.close()
        except (IOError, TypeError):
            Bank = {}
            Bank['facet0'] = {'labels':{},'ent_coord':{},'ear_coord':{},'se_coord':{}}
            Bank['facet1'] = {'labels':{},'ent_coord':{},'ear_coord':{},'se_coord':{}}
            Bank['anskey_param'] = anskey_param
            Bank['std_params'] = std_params
            Bank['parse_params'] = parse_params
            Bank['equate_params'] = equate_params
            Bank['refit_params'] = refit_params
            Bank['step_coord'] = step_coord
            Bank['ndim'] = ndim
            Bank['shape'] = shape

    nanval = Dicts['fac0coord']['nanval']

    # Modify Facet 0 (row entity) coordinates
    try:
        Bank['facet0']['labels'] = modbank(Bank['facet0']['labels'],
                                           Dicts['fac0coord']['rl_row'],
                                           bankf0['Remove'],
                                           bankf0['Add'],
                                           nanval)
    except TypeError:
        Bank['facet0']['labels'] = {}

    try:
        Bank['facet0']['ent_coord'] = modbank(Bank['facet0']['ent_coord'],
                                              Dicts['fac0coord']['core_row'],
                                              bankf0['Remove'],
                                              bankf0['Add'],
                                              nanval)
    except TypeError:
        Bank['facet0']['ent_coord'] = {}

    try:
        Bank['facet0']['ear_coord'] = modbank(Bank['facet0']['ear_coord'],
                                              Dicts['Fac0EARCoord']['core_row'],
                                              bankf0['Remove'],
                                              bankf0['Add'],
                                              nanval)
    except TypeError:
        Bank['facet0']['ear_coord'] = {}

    try:
        Bank['facet0']['se_coord'] = modbank(Bank['facet0']['se_coord'],
                                             Dicts['Fac0SECoord']['core_row'],
                                             bankf0['Remove'],
                                             bankf0['Add'],
                                             nanval)
    except TypeError:
        Bank['facet0']['se_coord'] = {}


    # Modify Facet 1 (col entity) coordinates
    try:
        Bank['facet1']['labels'] = modbank(Bank['facet1']['labels'],
                                           Dicts['fac1coord']['rl_row'],
                                           bankf1['Remove'],
                                           bankf1['Add'],
                                           nanval)
    except TypeError:
        Bank['facet1']['labels'] = {}

    try:
        Bank['facet1']['ent_coord'] = modbank(Bank['facet1']['ent_coord'],
                                              Dicts['fac1coord']['core_row'],
                                              bankf1['Remove'],
                                              bankf1['Add'],
                                              nanval)
                                              
    except TypeError:
        Bank['facet1']['ent_coord'] = {}

    try:
        Bank['facet1']['ear_coord'] = modbank(Bank['facet1']['ear_coord'],
                                              Dicts['Fac1EARCoord']['core_row'],
                                              bankf1['Remove'],
                                              bankf1['Add'],
                                              nanval)
    except TypeError:
        Bank['facet1']['ear_coord'] = {}

    try:
        Bank['facet1']['se_coord'] = modbank(Bank['facet1']['se_coord'],
                                             Dicts['Fac1SECoord']['core_row'],
                                             bankf1['Remove'],
                                             bankf1['Add'],
                                             nanval)
    except TypeError:
        Bank['facet1']['se_coord'] = {}


    ###############
    ##  Save as  ##
    ##  pickle   ##
    ###############

    self.bank_out = Bank

    if filename is not None:
        file_ = open(filename,'wb')
        cPickle.dump(Bank,file_)
        file_.close()

        if self.verbose is True:
            print '\n',filename,'has been saved.\n'

    return Bank




######################################################################

def _restore_invalid(_locals):
    "Basis of the restore_invalid() method."

    # Get self
    self = _locals['self']
    outputs = _locals['outputs']
    getrows = _locals['getrows']
    getcols = _locals['getcols']

    if not isinstance(outputs, list):
        outputs = [outputs]

    # Available outputs
    avail = self.__dict__.copy()

    if 'coord_out' in outputs:
        avail['fac0coord'] = self.coord_out['fac0coord']
        avail['fac1coord'] = self.coord_out['fac1coord']
        del avail['coord_out']
        outputs.remove('coord_out')
        outputs += ['fac0coord','fac1coord']

## Haven't got it to work yet.  Check whether getcols needs to be skipped, like coords
##    if 'score_mc_out' in outputs:
##        avail['RowScore'] = self.score_mc_out['RowScore']
##        avail['ColScore'] = self.score_mc_out['ColScore']
##        outputs += ['RowScore','ColScore']

    # Merge source output to original IDs
    for output in outputs:
        try:
            source = avail[output]
        except KeyError:
            exc = 'Unable to find '+output+' .\n'
            raise restore_invalid_Error(exc)

        # Merge outputs to original row IDs. Special handling with 'fac1coord'
        # because coordinates are transposed.
        if getrows is True:
            if output != 'fac1coord':
                try:
                    self.merge(source, {'target':0,'source':0}, None, None, None,
                               self.nanval)
                except:
                    print ('Warning in restore_invalid():  Unable to restore '
                           'invalid rows to', output, '.\n')
            else:
                try:
                    self.merge(source, {'target':1,'source':0}, None, None, None,
                               self.nanval)
                    self.merge_out = self.transpose(self.merge_out)
                except:
                    print ('Warning in restore_invalid():  Unable to restore '
                           'invalid rows to', output, '.\n')                

        # Merge estimates to original col IDs. Not needed for 'coord_out'
        if getcols is True and output not in ['fac0coord', 'fac1coord']:
            if getrows is True:
                try:
                    self.merge(self.merge_out, {'target':1,'source':1}, None,
                               None, None, self.nanval)
                except:
                    print ('Warning in restore_invalid(): Unable to restore '
                           'invalid columns to', output, '.\n')
            else:
                try:
                    self.merge(source, {'target':1,'source':1}, None, None,
                               None, self.nanval)
                except:
                    print ('Warning in restore_invalid(): Unable to restore '
                           'invalid columns to', output, '.\n')

        if getrows is not True and getcols is not True:
            exc = 'At least one of getrows or getcols needs to be True.\n'
            raise restore_invalid_Error(exc)

        # Restore all source keys
        merged = self.merge_out        
        merged_keys = merged.keys()
        for key in source.keys():
            if key not in merged_keys:
                merged[key] = source[key]

#       It is not safe to overwrite self.coord_out, so coords are 
#       not nested in 'coord_out'
        self.__dict__[output] = merged
        self.merge_out = None

    return None




######################################################################

def _export(_locals):
    "Basis of the export() method."

    # Get self
    self = _locals['self']
    pytables = self.pytables
    fileh = self.fileh

    # Get _locals
    outputs = _locals['outputs']
    output_as = _locals['output_as']
    outprefix = _locals['outprefix']
    outsuffix = _locals['outsuffix']
    delimiter = _locals['delimiter']
    format_ = _locals['format_']
    obj_params = _locals['obj_params']

    # Available outputs
    AvailDict = self.__dict__

    # Export __init__ parameters as pickle
    if obj_params is True:
        ObjParamsDict = {}
        DataKeys = ['rowlabels','collabels','coredata','pytables','fileh']
        for key in self.data_out.keys():
            if key not in DataKeys:
                ObjParamsDict[key] = self.data_out[key]
            else:
                pass

        # pickle it
        ParamFileName = outprefix+'_ObjParams'
        outfile = open(ParamFileName,'wb')
        cPickle.dump(ObjParamsDict,outfile)
        outfile.close()

        if self.verbose is True:
            print ParamFileName,'has been saved as a pickle file.\n'

    # Open new pytables file if necessary
    if output_as == 'hd5':
        if pytables is not None and isinstance(pytables,str):
            TabFileName = pytables.replace('_temp','')
        else:
            TabFileName = outprefix+'_PyTable.hd5'
        NewFileh = tab.openFile(TabFileName,'w')

    # Export output
    for Output in outputs:
        filename = outprefix+'_'+Output+outsuffix
        try:
            OutVal = AvailDict[Output]
        except KeyError:
            exc = 'Unable to find '+Output+' for exporting.\n'
            raise export_Error(exc)

        if isinstance(OutVal,dmn.core.Damon):
            OutVal = OutVal.data_out

        # pickle format
        if output_as == 'pickle':
            outfile = open(filename,'wb')
            cPickle.dump(OutVal,outfile)
            outfile.close()

            if self.verbose is True:
                print filename,'has been saved as a pickle file.\n'

        # textfile format
        elif output_as == 'textfile':

            if not isinstance(OutVal,dict):
                try:
                    np.savetxt(filename,OutVal,format_,delimiter)
                except:
                    print 'OutVal=\n', OutVal
                    exc = 'Unable to figure out how to export '+Output+' .\n'
                    raise export_Error(exc)

                if self.verbose is True:
                    print filename,'has been saved as a text file.\n'

            elif (isinstance(OutVal,dict)
                  and 'coredata' in OutVal.keys()
                  ):

                try:
                    dtype = OutVal['dtype']
                except KeyError:
                    dtype = [object, 8, '']
                
                Whole = tools.addlabels(OutVal['coredata'],
                                        OutVal['rowlabels'],
                                        OutVal['collabels'],
                                        dtype = dtype,
                                        nanval = OutVal['nanval']
                                        )['whole']

                np.savetxt(filename,Whole,format_,delimiter)

                if self.verbose is True:
                    print filename,'has been saved as a text file.\n'

            elif (isinstance(OutVal,dict)
                  and 'coredata' not in OutVal.keys()
                  ):
                outfile = open(filename,'wb')
                cPickle.dump(OutVal,outfile)
                outfile.close()
                print 'Warning in export(): Unable to save',Output,'dictionary as a text file.  Saving as pickle instead.\n'

            else:
                exc = 'Unable to figure out how to export '+Output+' .\n'
                raise export_Error(exc)

        elif (output_as == 'hd5'
              and pytables is not None
              ):
            if (isinstance(OutVal,dict)
                and 'coredata' in OutVal
                ):
                try:
                    A = tools.pytables_(data = OutVal,   # [<None, array, file,[files],datadict, hd5 file, or data generating function {'chunkdict':{...},'ArgDict':{...}}> ]
                                 format_ = 'hd5',  # [<'array','textfile',['textfiles'],'datadict','hd5','init_earray','chunkfunc'>]
                                 putinfile = NewFileh,    # [<None,'MyFileName.hd5',MyPyTable['fileh']>  => name of file in which to put groups and arrays> ]
                                 filemode = None,    # [<None,'w','r','a','r+'>  => mode in which to open file (write, read, append, read+) ]
                                 ingroup = Output,   # [<None,GroupName>  => e.g., 'Group1' ]
                                 array_names = ['rowlabels','collabels','coredata'],   # [ ['ArrayName0','ArrayName1']>  => list of one or more arrays to be created and/or read]
                                 )['arrays']
                except TypeError:
                    A = tools.pytables_(data = OutVal,   # [<None, array, file,[files],datadict, hd5 file, or data generating function {'chunkdict':{...},'ArgDict':{...}}> ]
                                 format_ = 'datadict',  # [<'array','textfile',['textfiles'],'datadict','hd5','init_earray','chunkfunc'>]
                                 putinfile = NewFileh,    # [<None,'MyFileName.hd5',MyPyTable['fileh']>  => name of file in which to put groups and arrays> ]
                                 filemode = None,    # [<None,'w','r','a','r+'>  => mode in which to open file (write, read, append, read+) ]
                                 ingroup = Output,   # [<None,GroupName>  => e.g., 'Group1' ]
                                 array_names = ['rowlabels','collabels','coredata'],   # [ ['ArrayName0','ArrayName1']>  => list of one or more arrays to be created and/or read]
                                 )['arrays']

                if self.verbose is True:
                    print filename,'has been saved as an hd5 file.\n'

            else:
                exc = 'Unable to export '+Output+' as an hd5 file.  Consider using tools.pytables() directly.\n'
                raise export_Error(exc)

        elif (output_as == 'hd5'
              and pytables is None
              ):
            if (isinstance(OutVal,dict)
                and 'coredata' in OutVal
                ):
                A = tools.pytables_(data = OutVal,   # [<None, array, file,[files],datadict, hd5 file, or data generating function {'chunkdict':{...},'ArgDict':{...}}> ]
                             format_ = 'datadict',  # [<'array','textfile',['textfiles'],'datadict','hd5','init_earray','chunkfunc'>]
                             putinfile = NewFileh,    # [<None,'MyFileName.hd5',MyPyTable['fileh']>  => name of file in which to put groups and arrays> ]
                             filemode = None,    # [<None,'w','r','a','r+'>  => mode in which to open file (write, read, append, read+) ]
                             ingroup = Output,   # [<None,GroupName>  => e.g., 'Group1' ]
                             array_names = ['rowlabels','collabels','coredata'],   # [ ['ArrayName0','ArrayName1']>  => list of one or more arrays to be created and/or read]
                             )['arrays']

                if self.verbose is True:
                    print filename,'has been saved as an hd5 file.\n'

            else:
                exc = 'export() is unable to export '+Output+' as an hd5 file.  Consider using tools.pytables() directly.\n'
                raise export_Error(exc)

        else:
            exc = 'Unable to figure out output_as parameter.\n'
            raise export_Error(exc)

    # Close new fileh
    if output_as == 'hd5':
        NewFileh.close()

    # Remove working pytables files
    if pytables is not None:

        # Close old fileh
        FilehName = fileh.filename
        fileh.close()

        # Delete temp PyTable files
        AllFiles = os.listdir(os.getcwd())
        for file_ in AllFiles:
            if 'temp_' in file_:
                os.remove(file_)
        try:
            os.remove(FilehName)
        except OSError:
            pass

    return None



######################################################################

def _flag(_locals):
    "Basis of the flag() method."
    
    # parameters
    self = _locals['self']
    d_ = _locals['datadict']
    flag_rows = _locals['flag_rows']
    flag_cols = _locals['flag_cols']
    extract = _locals['extract']
    
    # Convert to Damon obj
    if isinstance(d_, str):
        d_ = vars(self)[d_]

    if isinstance(d_, dict):
        d = dmn.core.Damon(d_, 'datadict', verbose=None)
    elif isinstance(d_, dmn.core.Damon):
        d = d_
    else:
        exc = 'Could not figure out datadict parameter.'
        raise ValueError(exc)

    # Flag rows
    if flag_rows:
        if isinstance(flag_rows[0], str):
            rows = vars(dmn.tools)[flag_rows[0]](d, **flag_rows[1])
        elif callable(flag_rows[0]):
            rows = flag_rows[0](d, **flag_rows[1])
        else:
            exc = 'flag_rows needs to be a function or lambda. See docs.'
            raise ValueError(exc)
    else:
        rows = ['All']

    # Flag cols
    if flag_cols:
        if isinstance(flag_cols[0], str):
            cols = vars(dmn.tools)[flag_cols[0]](d, **flag_cols[1])
        elif callable(flag_cols[0]):
            cols = flag_cols[0](d, **flag_cols[1])
        else:
            exc = 'flag_cols needs to be a function or lambda. See docs.'
            raise ValueError(exc)
    else:
        cols = ['All']

    # Extract from Damon object
    if extract is not None:
        try:
            x = d.extract(d,
                          getrows={'Get':extract['rows'], 'Labels':'key', 'Rows':rows},
                          getcols={'Get':extract['cols'], 'Labels':'key', 'Cols':cols}
                          )
            d_x = dmn.core.Damon(x, 'datadict', 'RCD_dicts_whole', verbose=None)
        except Damon_Error:
            d_x = None
    else:
        d_x = None
    
    
    return {'datadict':d_, 'rows':rows, 'cols':cols, 'extract':d_x}



######################################################################

def _extract(_locals):
    "Basis of the extract() method."

    # Get self
    self = _locals['self']

    # Get _locals
    datadict = _locals['datadict']
    getrows = _locals['getrows'].copy()
    getcols = _locals['getcols'].copy()
    labels_only = _locals['labels_only']

    # Overwrite getrows and getcols if using flag_out
    if datadict == 'flag_out':
        datadict = self.flag_out['datadict']
        getrows['Rows'] = self.flag_out['rows']
        getcols['Cols'] = self.flag_out['cols']
    
    # Convert object to datadict
    if isinstance(datadict, str):
        datadict = vars(self)[datadict]

    if isinstance(datadict, dmn.core.Damon):
        datadict = datadict.data_out

    # Convert NoneExcept All to AllExcept None
    if (getrows['Get'] == 'NoneExcept'
        and len(getrows['Rows']) == 1
        and 'All' in getrows['Rows']
        ):
        getrows['Get'] = 'AllExcept'
        getrows['Rows'] = [None]

    if (getcols['Get'] == 'NoneExcept'
        and len(getcols['Cols']) == 1
        and 'All' in getcols['Cols']
        ):
        getcols['Get'] = 'AllExcept'
        getcols['Rows'] = [None]

    # Convert Labels to 'key'
    if (isinstance(getrows['Labels'],int)
        and getrows['Rows'] == [None]
        ):
        getrows['Labels'] = 'key'

    if (isinstance(getcols['Labels'],int)
        and getcols['Cols'] == [None]
        ):
        getcols['Labels'] = 'key'

    # Define variables
    coredata = datadict['coredata'][:,:] if labels_only is not True else None
    rowlabels = datadict['rowlabels'][:,:]
    collabels = datadict['collabels'][:,:]

    # Get nheaders4rows if necessary
    try:
        nheaders4rows = datadict['nheaders4rows']
    except KeyError:
        nheaders4rows = np.size(rowlabels,axis=1)

    # Get nheaders4rows if necessary
    try:
        nheaders4cols = datadict['nheaders4cols']
    except KeyError:
        nheaders4cols = np.size(collabels,axis=0)

    key4rows = datadict['key4rows']
    key4cols = datadict['key4cols']
    rowkeytype = datadict['rowkeytype']
    colkeytype = datadict['colkeytype']
    nrows = np.size(rowlabels[:,:],axis=0)
    ncols = np.size(collabels[:,:],axis=1)
#    nRowsCore = nrows - nheaders4cols
#    nColsCore = ncols - nheaders4rows
    DType = 'S60'
#    RowErrMsg = False
#    ColErrMsg = False


    ####################
    ##    Define      ##
    ##  buildindex()  ##
    ####################

    def buildindex(Fac,Key4OppFac):

        # Get generic ent variables from facet
        if Fac == 'Rows':
            nEnts = nrows
            FlagEnts = np.zeros((nrows))
            FlagEntsCore = np.zeros((nrows - nheaders4cols))
            GetEntsLabels = getrows['Labels']
            GetEnts = getrows['Rows']
            EntLabels = rowlabels
            Core = coredata
            nHeaders = nheaders4rows
            nHeaders4Opp = nheaders4cols
            Key4Fac = key4rows
            EntKeyType = rowkeytype
            OppEnts = tools.getkeys(datadict,'Col','All','Auto',None)  #collabels[key4cols,:].astype(DType)
            #FacEnts = tools.getkeys(datadict,'Row','All','Auto',None)

        elif Fac == 'Cols':
            nEnts = ncols
            FlagEnts = np.zeros((ncols))
#            FlagEntsCore = np.zeros((ncols - nheaders4rows))
            GetEntsLabels = getcols['Labels']
            GetEnts = getcols['Cols']
            EntLabels = np.transpose(collabels)
            Core = np.transpose(coredata)
            nHeaders = nheaders4cols
            nHeaders4Opp = nheaders4rows
            Key4Fac = key4cols
            EntKeyType = colkeytype
            OppEnts = tools.getkeys(datadict,'Row','All','Auto',None)  #rowlabels[:,key4rows].astype(DType)
            #FacEnts = tools.getkeys(datadict,'Col','All','Auto',None)      # Will need to redo

        # Index desired rows by position
        if (GetEntsLabels == 'index'
            and 'index' not in OppEnts
            ):
            Index = np.array(GetEnts)
            try:
                FlagEnts[Index] = 1
            except IndexError:
                exc = 'Index went out of range.\n'
                print 'Row or column flags =\n',FlagEnts
                print 'Index =\n',Index
                raise extract_Error(exc)

            NonIndex = np.where(FlagEnts == 0)[0]

        # Index desired rows by desired entity
        else:
            if (GetEntsLabels == 'key'
                and 'key' not in OppEnts
                ):
                EntLabelInd = Key4Fac
            else:
                if isinstance(GetEntsLabels,int):
                    EntLabelInd = GetEntsLabels
                else:
                    if (GetEntsLabels == 'key'
                        or GetEntsLabels == 'index'
                        ):
                        print "Warning in extract(): Special extract() words 'key' and 'index' are doubling as key names.  Will use as key names.\n"
                    try:
                        EntLabelInd = np.where(OppEnts == str(GetEntsLabels))[0][0]
                    except IndexError:
                        exc = 'Unable to find label: '+str(GetEntsLabels)+' .\n'
                        raise extract_Error(exc)

            # Deal with no target rows
            try:
                GetEnts[0]
            except IndexError:
                GetEnts = [None]
#                exc = 'Got an empty list of entities.  Check your "getrows" and "getcols" parameters.'
#                raise extract_Error(exc)
            
            if GetEnts[0] is None:
                Index = np.array([])
                NonIndex = np.where(FlagEnts == 0)[0]
            else:
                try:
                    TargEnts = np.array(GetEnts,dtype=EntKeyType)
                    try:
                        FacEnts = EntLabels[:,EntLabelInd].astype(EntKeyType)
                    except IndexError:
                        FacEnts = np.append(np.zeros((nHeaders4Opp)),Core[:,EntLabelInd - nHeaders],axis=0).astype(EntKeyType)
                except:
                    TargEnts = np.array(GetEnts,dtype=DType)
                    try:
                        FacEnts = EntLabels[:,EntLabelInd].astype(DType)
                    except IndexError:
                        FacEnts = np.append(np.zeros((nHeaders4Opp)),Core[:,EntLabelInd - nHeaders],axis=0).astype(DType)

                # Index each row individually
                for i in xrange(nEnts):
                    if FacEnts[i] in TargEnts:
                        FlagEnts[i] = 1
                    else:
                        pass

                Index = np.where(FlagEnts == 1)[0]
                NonIndex = np.where(FlagEnts == 0)[0]

        # Restore deleted keys
        Index_L = list(Index)
        NonIndex_L = list(NonIndex)

        if Key4OppFac not in Index_L:
            Index_L.insert(0,Key4OppFac)
            Index = np.sort(np.array(Index_L))

        if Key4OppFac not in NonIndex_L:
            NonIndex_L.insert(0,Key4OppFac)
            NonIndex = np.sort(np.array(NonIndex_L))
            
        # Modify index for coredata
        AdjIndex = Index - nHeaders4Opp
        IndexCore = np.sort(np.delete(AdjIndex,np.where(AdjIndex < 0)))

        AdjNonIndex = NonIndex - nHeaders4Opp
        NonIndexCore = np.sort(np.delete(AdjNonIndex,np.where(AdjNonIndex < 0)))

        return {'Index':Index,
                'NonIndex':NonIndex,
                'IndexCore':IndexCore,
                'NonIndexCore':NonIndexCore
                }


    ################
    ##  Extract   ##
    ##  Entities  ##
    ################

    R = buildindex('Rows',key4cols)
    C = buildindex('Cols',key4rows)

    #   HEAD-SCRATCHER:  PROCEED WITH CARE
    ########################################
    # Extract rows to prep for columns
    if getrows['Get'] == 'NoneExcept':
        NewCore = np.take(coredata,R['IndexCore'],axis=0) if coredata is not None else None
        NewRowLabels = np.take(rowlabels,R['Index'],axis=0)

        if getcols['Get'] == 'NoneExcept':
            NewKey4Rows = key4rows - np.sum(C['NonIndex'] < key4rows)
        elif getcols['Get'] == 'AllExcept':
            NewKey4Rows = key4rows - np.sum(C['Index'] < key4rows)

        TruncInd = R['Index'][np.where(R['Index'] < nheaders4cols)]
        NewColLabels = collabels[TruncInd,:]

    if getrows['Get'] == 'AllExcept':
        NewCore = np.take(coredata,R['NonIndexCore'],axis=0) if coredata is not None else None
        NewRowLabels = np.take(rowlabels,R['NonIndex'],axis=0)

        if getcols['Get'] == 'NoneExcept':
            NewKey4Rows = key4rows - np.sum(C['NonIndex'] < key4rows)
        elif getcols['Get'] == 'AllExcept':
            NewKey4Rows = key4rows - np.sum(C['Index'] < key4rows)

        TruncInd = R['NonIndex'][np.where(R['NonIndex'] < nheaders4cols)]
        NewColLabels = collabels[TruncInd,:]

    ########################################
    # Extract columns from row-modified data
    if getcols['Get'] == 'NoneExcept':
        try:
            NewCore = np.take(NewCore,C['IndexCore'],axis=1) if coredata is not None else None
        except:
            print "np.shape(NewCore)", np.shape(NewCore)
            print "C[IndexCore]=\n", C['IndexCore']
            NewCore = np.take(NewCore,C['IndexCore'],axis=1) if coredata is not None else None
        NewColLabels = np.take(NewColLabels,C['Index'],axis=1)

        if getrows['Get'] == 'NoneExcept':
            NewKey4Cols = key4cols - np.sum(R['NonIndex'] < key4cols)
        elif getrows['Get'] == 'AllExcept':
            NewKey4Cols = key4cols - np.sum(R['Index'] < key4cols)

        TruncInd = C['Index'][np.where(C['Index'] < nheaders4rows)]
        NewRowLabels = NewRowLabels[:,TruncInd]

    if getcols['Get'] == 'AllExcept':
        NewCore = np.take(NewCore,C['NonIndexCore'],axis=1) if coredata is not None else None
        NewColLabels = np.take(NewColLabels,C['NonIndex'],axis=1)

        if getrows['Get'] == 'NoneExcept':
            NewKey4Cols = key4cols - np.sum(R['NonIndex'] < key4cols)
        elif getrows['Get'] == 'AllExcept':
            NewKey4Cols = key4cols - np.sum(R['Index'] < key4cols)

        TruncInd = C['NonIndex'][np.where(C['NonIndex'] < nheaders4rows)]
        NewRowLabels = NewRowLabels[:,TruncInd]


    ################
    ##   Build    ##
    ##  datadict  ##
    ################

    if datadict['validchars'] is None:
        validchars = None
    else:
        validchars = datadict['validchars'][:]
        vcdict = validchars[1]
        
        if isinstance(vcdict, dict):
            for k in vcdict.keys():
                if k not in NewColLabels:
                    del vcdict[k]
            validchars[1] = vcdict
    
    Extract = {'rowlabels':NewRowLabels,
               'collabels':NewColLabels,
               'coredata':NewCore,
               'nheaders4rows':np.size(NewRowLabels,axis=1),
               'key4rows':NewKey4Rows,
               'rowkeytype':rowkeytype,
               'nheaders4cols':np.size(NewColLabels,axis=0),
               'key4cols':NewKey4Cols,
               'colkeytype':colkeytype,
               'nanval':datadict['nanval'],
               'validchars':validchars
               }

    return Extract





######################################################################

def _merge(_locals):
    "Basis of the merge() method."

    # Get variables
    self = _locals['self']
    targ_dict = self.data_out
    source_dict = _locals['source']
    tkeys= tools.getkeys(targ_dict,'Row','All','Auto',None)

    # Add nheaders4rows if necessary
    try:
        source_dict['nheaders4rows']
    except KeyError:
        source_dict['nheaders4rows'] = np.size(source_dict['rowlabels'],axis=1)

    # Add nheaders4cols if necessary
    try:
        source_dict['nheaders4cols']
    except KeyError:
        source_dict['nheaders4cols'] = np.size(source_dict['collabels'],axis=0)

    axis = _locals['axis']
    targ_data = _locals['targ_data']
    targ_labels = _locals['targ_labels']
    source_ids = _locals['source_ids']
    nanval = _locals['nanval']

    # Create source dicts
    arg_list = ['coredata','rowlabels','nheaders4rows','key4rows','rowkeytype',
                'collabels','nheaders4cols','key4cols','colkeytype']
    s_args = {}
    for key in arg_list:
        s_args[key] = source_dict[key]
    s_args['range4labels'] = 'All'
    s_args['strip_labkeys'] = True
    s_args['whole'] = None
    s_dicts = tools.damon_dicts(**s_args)

    # Create target dicts
    t_args = {}
    for key in arg_list:
        t_args[key] = targ_dict[key]
    t_args['range4labels'] = 'All'
    t_args['strip_labkeys'] = None
    t_args['whole'] = None
    t_dicts = tools.damon_dicts(**t_args)

    # Get source dicts by axis
    if axis['source'] == 0:
        s_keys = tools.getkeys(source_dict,'Row','Core','Auto',None)
        s = s_dicts['core_row']
        s_labkeys = tools.getkeys(source_dict,'Row','All','Auto',None)
        s_lab = s_dicts['rl_row']
        s_labarr = source_dict['rowlabels'][:,:]
        s_oppkeys = source_dict['collabels'][:,source_dict['key4rows']].astype('S60')
        s_oppkey4 = s_oppkeys[source_dict['key4cols']]
        s_opp = s_dicts['cl_row']
        s_opp_nheads = source_dict['nheaders4rows']
        s_core = source_dict['coredata'][:,:]

    elif axis['source'] == 1:
        s_keys = tools.getkeys(source_dict,'Col','Core','Auto',None)
        s = s_dicts['core_col']
        s_labkeys = tools.getkeys(source_dict,'Col','All','Auto',None)
        s_lab = s_dicts['cl_col']
        s_labarr = np.transpose(source_dict['collabels'][:,:])
        s_oppkeys = source_dict['rowlabels'][source_dict['key4cols']].astype('S60')
        s_oppkey4 = s_oppkeys[source_dict['key4rows']]
        s_opp = s_dicts['rl_col']
        s_opp_nheads = source_dict['nheaders4cols']
        s_core = np.transpose(source_dict['coredata'][:,:])

    # Get target dicts by axis
    if axis['target'] == 0:
        t_keys = tools.getkeys(targ_dict,'Row','Core','Auto',None)
        t = t_dicts['core_row']
        t_labkeys = tools.getkeys(targ_dict,'Row','All','Auto',None)
        t_lab = t_dicts['rl_row']
        t_labarr = targ_dict['rowlabels'][:,:]
        t_labkey4 = targ_dict['key4rows']
        t_oppkeys = targ_dict['collabels'][:,targ_dict['key4rows']].astype('S60')
        t_oppkey4 = t_oppkeys[targ_dict['key4cols']]
        t_opp = t_dicts['cl_row']
        t_opp_nheads = targ_dict['nheaders4rows']
        t_core = targ_dict['coredata'][:,:]

    elif axis['target'] == 1:
        t_keys = tools.getkeys(targ_dict,'Col','Core','Auto',None)
        t = t_dicts['core_col']
        t_labkeys = tools.getkeys(targ_dict,'Col','All','Auto',None)
        t_lab = t_dicts['cl_col']
        t_labarr = np.transpose(targ_dict['collabels'][:,:])
        t_labkey4 = targ_dict['key4cols']
        t_oppkeys = targ_dict['rowlabels'][targ_dict['key4cols'],:].astype('S60')
        t_oppkey4 = t_oppkeys[targ_dict['key4rows']]
        t_opp = t_dicts['rl_col']
        t_opp_nheads = targ_dict['nheaders4cols']
        t_core = np.transpose(targ_dict['coredata'][:,:])

    # Add non-overlapping source IDs to target IDs
    if source_ids is True:
        add_keys = np.sort(np.array(list(set(s_keys) - set(t_keys))))
        t_keys = np.append(t_keys,add_keys)

        add_labkeys = np.sort(np.array(list(set(s_labkeys) - set(t_labkeys))))
        t_labkeys = np.append(t_labkeys,add_labkeys)


    #################
    ##   Build     ##
    ##  coredata   ##
    #################

    # Add source data to array, robust to mistyped keys, throw KeyError if no match
    def add_source(s_d,key):
        try:
            out = s_d[key]
        except KeyError:
            try:
                out = s_d[int(float(key))]
            except ValueError:
                out = s_d[str(key)]
        return out

    nrows = np.size(t_keys)
    n_tcol = np.size(t_core,axis=1)
    n_scol = np.size(s_core,axis=1)

    if targ_data is True:
        ncols = n_tcol + n_scol
        s_start = n_tcol
    else:
        ncols = n_scol
        s_start = 0

    # Initialize new coredata
    new_core = np.zeros((nrows,ncols),dtype=object) + int(float(nanval))

    # Fill in new coredata
    for i,key in enumerate(t_keys):
        if targ_data is True:
            try:
                new_core[i,:s_start] = t[key]
            except KeyError:
                pass
        try:
            new_core[i,s_start:] = add_source(s,key)
        except KeyError:
            pass

    try:
        new_core = new_core.astype(float)
    except ValueError:
        pass

    # Transpose if necessary
    if axis['target'] == 1:
        new_core = np.transpose(new_core)


    #################
    ##   Build     ##
    ##  rowlabels  ##
    #################

    nlrows = np.size(t_labkeys)
    n_tlabcol = np.size(t_labarr,axis=1)
    n_slabcol = np.size(s_labarr,axis=1)

    if targ_labels is True:
        nlcols = n_tlabcol + n_slabcol - 1
        s_lstart = n_tlabcol
    else:
        nlcols = n_slabcol # -1 + 1
        s_lstart = 1        # Leave room for target keys, which must be included

    # Initialize new rowlabels
    new_rl = np.zeros((nlrows,nlcols),dtype=object) + int(float(nanval))

    # Fill in new rowlabels
    for i,key in enumerate(t_labkeys):
        if targ_labels is True:
            try:
                new_rl[i,:s_lstart] = t_lab[key]
            except KeyError:
                new_rl[i,t_labkey4] = t_labkeys[i]
        else:
            new_rl[i,:s_lstart] = t_labkeys[i]

        try:
            new_rl[i,s_lstart:] = add_source(s_lab,key)
        except KeyError:
            pass

    # Transpose if necessary
    if axis['target'] == 1:
        new_cl_ = np.transpose(new_rl)
    else:
        new_rl_ = new_rl


    #################
    ##   Build     ##
    ##  collabels  ##
    #################

    # Build the "core" section, refers to core variables
    nrows_ = len(t_opp)
    ncols_ = ncols

    # Initialize new column labels
    new_cl = np.zeros((nrows_,ncols_),dtype=object) + int(float(nanval))

    # Force source colkey label to equal its target equivalent
    # Don't quite understand this!

    s_opp[t_oppkey4] = s_opp[s_oppkey4]

    # Fill in new collabels
    for i,key in enumerate(t_oppkeys):
        if targ_data is True:
            try:
                new_cl[i,:s_start] = t_opp[key][t_opp_nheads:]
            except KeyError:
                try:
                    new_cl[i,:s_start] = s_opp[key][s_opp_nheads:]
                except KeyError:
                    pass
        try:
            new_cl[i,s_start:] = add_source(s_opp,key)[s_opp_nheads:]
        except KeyError:
            pass

    # Add corner from rowlabels
    corner = new_rl[:nrows_,:]
    new_cl = np.append(corner,new_cl,axis=1)

    # Transpose if necessary
    if axis['target'] == 1:
        new_rl_ = np.transpose(new_cl)
    else:
        new_cl_ = new_cl


    ##################
    ##  datadict    ##
    ##  validchars  ##
    ##################

    # Build datadict
    merged = {'rowlabels':new_rl_,
              'collabels':new_cl_,
              'coredata':new_core,
              'nheaders4rows':np.size(new_rl_,axis=1),
              'key4rows':targ_dict['key4rows'],
              'rowkeytype':targ_dict['rowkeytype'],
              'nheaders4cols':np.size(new_cl_,axis=0),
              'key4cols':targ_dict['key4cols'],
              'colkeytype':targ_dict['colkeytype'],
              'nanval':nanval,
              'validchars':None  # temporary
              }
    
    # Check dups and build validchars
    rowkeys = tools.getkeys(merged,'Row','Core','Auto',None)
    colkeys = tools.getkeys(merged,'Col','Core','Auto',None)
    validchars_flag = True

    if self.check_dups is not None:
        row_dups = tools.dups(rowkeys)
        col_dups = tools.dups(colkeys)

        if len(row_dups) > 0:
            print 'Warning in merge():  Found duplicate row keys.\n'
            print 'Duplicates (first 20):\n', dict(row_dups.items()[:20])
            
        if len(col_dups) > 0:
            validchars_flag = False
            print 'Warning in merge():  Found duplicate column keys. Setting validchars = None\n'
            print 'Duplicates (first 20):\n', dict(col_dups.items()[:20])
            # validchars = None by default

    if validchars_flag is True:
        
        # Resolve source validchars
        skeys = tools.getkeys(source_dict,'Col','Core','Auto',None)
        sval_dict = {}
        if source_dict['validchars'] is None:
            for key in skeys:
                sval_dict[key] = ['All']
        elif source_dict['validchars'][0] == 'All':
            for key in skeys:
                sval_dict[key] = source_dict['validchars'][1]
        else:
            for key in skeys:
                sval_dict[key] = source_dict['validchars'][1][key]

        # Resolve target validchars
        tkeys = tools.getkeys(targ_dict,'Col','Core','Auto',None)
        tval_dict = {}
        if targ_dict['validchars'] is None:
            for key in tkeys:
                tval_dict[key] = ['All']
        elif targ_dict['validchars'][0] == 'All':
            for key in tkeys:
                tval_dict[key] = targ_dict['validchars'][1]
        else:
            for key in tkeys:
                tval_dict[key] = targ_dict['validchars'][1][key]

            #val_dict = targ_dict['validchars'][1]

        # Assign appropriate validchar to each column
        mval_dict = {}

        for key in colkeys:
            try:
                mval_dict[key] = sval_dict[key]
            except KeyError:
                try:
                    mval_dict[key] = tval_dict[key]
                except KeyError:
                    mval_dict[key] = ['All']
                    #print "Warning in merge(): Unable to create a validchars spec for",key,". Making it ['All']."

        if ((isinstance(source_dict['validchars'],list)
             and 'Num' not in source_dict['validchars'])
            or (isinstance(targ_dict['validchars'],list)
                and 'Num' not in targ_dict['validchars'])
            ):
                merged['validchars'] = ['Cols',mval_dict]
        else:
            merged['validchars'] = ['Cols',mval_dict,'Num']

    # Some methods refer to obj.merge_out. It was deleted, now restored
    self.merge_out = merged

    return merged





######################################################################
def _transpose(_locals):
    "Basis of the transpose() method."

    # Get and transpose arrays
    data = _locals['datadict']
    if data is None:
        data = _locals['self'].data_out
    
    # Get nheaders, if necessary
    try:
        data['nheaders4rows']
    except KeyError:
        data['nheaders4rows'] = np.size(data['rowlabels'],axis=1)

    try:
        data['nheaders4cols']
    except KeyError:
        data['nheaders4cols'] = np.size(data['collabels'],axis=0)

    T_Data = {'rowlabels':np.ascontiguousarray(np.transpose(data['collabels'])),
             'collabels':np.ascontiguousarray(np.transpose(data['rowlabels'])),
             'coredata':np.ascontiguousarray(np.transpose(data['coredata'])),
             'nheaders4rows':data['nheaders4cols'],
             'key4rows':data['key4cols'],
             'rowkeytype':data['colkeytype'],
             'nheaders4cols':data['nheaders4rows'],
             'key4cols':data['key4rows'],
             'colkeytype':data['rowkeytype'],
             'nanval':data['nanval'],
             'validchars':data['validchars']
              }

    return T_Data



######################################################################

def _to_dataframe(_locals):
    "Basis of the to_dataframe() method."

    import pandas as pd
        
    self = _locals['self']
    d = _locals['datadict']
    
    if isinstance(d, str):
        d = vars(self)[d]
    
    # Prepare dataframe elements
    index = dmn.tools.getkeys(d, 'Row', 'Core')
    columns = dmn.tools.getkeys(d, 'Col', 'Core')
    data = d['coredata']
    data[data == d['nanval']] = np.nan
    ix_name = d['rowlabels'][0, 0]
    
    # Sacrifice descriptive rowlabels and collabels.  To dangerous to
    # pull them into coredata.
    if d['nheaders4rows'] > 1 or d['nheaders4cols'] > 1:
        if self.verbose:
            print ('Warning: Damon.to_dataframe() does not support extra row '
                   'or column headers.  Only the keys will be preserved. ')

    # Rebuild MultiIndex
    if '(' in index[0] and ')' in index[0]:
        tups = [ast.literal_eval(t) for t in index]
        index = pd.MultiIndex.from_tuples(tups)
        ix_name = list(ast.literal_eval(ix_name))

    # Populate dataframe
    df = pd.DataFrame(data, index, columns)
    df.index.name = ix_name
    
    return df




######################################################################

def _combokeys(_locals):
    "Basis of the combokeys() method."

    # Get _locals
    self = _locals['self']
    datadict = self.data_out
    axis = _locals['axis']
    condarr1 = _locals['condarr1']
    condarr2 = _locals['condarr2']
    condarr3 = _locals['condarr3']
    filler = _locals['filler']

    # Define variables
    coredata = datadict['coredata']
    rowlabels = datadict['rowlabels']
    nheaders4rows = datadict['nheaders4rows']
    key4rows = datadict['key4rows']
    rowkeytype = datadict['rowkeytype']
    collabels = datadict['collabels']
    nheaders4cols = datadict['nheaders4cols']
    key4cols = datadict['key4cols']
    colkeytype = datadict['colkeytype']
    nanval = datadict['nanval']
    validchars = datadict['validchars']
    DatRowLabels = rowlabels[nheaders4cols:,:]
    DatColLabels = collabels[:,nheaders4rows:]
    LocDict = locals()
    LocDict.update({'np':np})

    # Create new keys for three conditions and append to row labels
    if axis == 'Row':
        if condarr1 is not None:
            Arr1 = eval(condarr1,{'__builtins__':None},LocDict)[:,np.newaxis]
            DatRowLabels = np.append(DatRowLabels,Arr1,axis=1)

        if condarr2 is not None:
            Arr2 = eval(condarr2,{'__builtins__':None},LocDict)[:,np.newaxis]
            DatRowLabels = np.append(DatRowLabels,Arr2,axis=1)

        if condarr3 is not None:
            Arr3 = eval(condarr3,{'__builtins__':None},LocDict)[:,np.newaxis]
            DatRowLabels = np.append(DatRowLabels,Arr3,axis=1)

        # Add filler to right of top row labels
        nTopFillRows = nheaders4cols
        nTopFillCols = np.size(DatRowLabels,axis=1) - nheaders4rows
        TopFiller = np.zeros((nTopFillRows,nTopFillCols),dtype=int)
        TopFiller[:,:] = filler
        TopRowLabels = np.append(rowlabels[:nheaders4cols,:nheaders4rows],TopFiller,axis=1)
        rowlabels = np.append(TopRowLabels,DatRowLabels,axis=0)
        nheaders4rows = np.size(rowlabels,axis=1)
        nheaders4cols = nheaders4cols

        # Add filler to left of col labels
        collabels = np.append(TopRowLabels,DatColLabels,axis=1)

        # Convert to data() format
        NuDataRCD = {'rowlabels':rowlabels,'collabels':collabels,'coredata':coredata,
                     'nheaders4rows':nheaders4rows,'key4rows':key4rows,'rowkeytype':rowkeytype,
                     'nheaders4cols':nheaders4cols,'key4cols':key4cols,'colkeytype':colkeytype,
                     'nanval':nanval,'validchars':validchars
                     }

    # Create new keys for three conditions and append to col labels
    if axis == 'Col':
        if condarr1 is not None:
            Arr1 = eval(condarr1,{'__builtins__':None},LocDict)[np.newaxis,:]
            DatColLabels = np.append(DatColLabels,Arr1,axis=0)

        if condarr2 is not None:
            Arr2 = eval(condarr2,{'__builtins__':None},LocDict)[np.newaxis,:]
            DatColLabels = np.append(DatColLabels,Arr2,axis=0)

        if condarr3 is not None:
            Arr3 = eval(condarr3,{'__builtins__':None},LocDict)[np.newaxis,:]
            DatColLabels = np.append(DatColLabels,Arr3,axis=0)

        # Add filler to bottom of left col labels
        nLeftFillRows = np.size(DatColLabels,axis=0) - nheaders4cols
        nLeftFillCols = nheaders4rows
        LeftFiller = np.zeros((nLeftFillRows,nLeftFillCols),dtype=int)
        LeftFiller[:,:] = filler
        LeftColLabels = np.append(collabels[:nheaders4cols,:nheaders4rows],LeftFiller,axis=0)
        collabels = np.append(LeftColLabels,DatColLabels,axis=1)
        nheaders4rows = nheaders4rows
        nheaders4cols = np.size(collabels,axis=0)

        # Add filler to top of dat row labels
        rowlabels = np.append(LeftColLabels,DatRowLabels,axis=0)

        # Convert to data() format
        NuDataRCD = {'rowlabels':rowlabels,'collabels':collabels,'coredata':coredata,
                     'nheaders4rows':nheaders4rows,'key4rows':key4rows,'rowkeytype':rowkeytype,
                     'nheaders4cols':nheaders4cols,'key4cols':key4cols,'colkeytype':colkeytype,
                     'nanval':nanval,'validchars':validchars
                     }

    return NuDataRCD


######################################################################
#def _xxxxx(_locals):




























