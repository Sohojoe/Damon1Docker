"""replace.py
Tool for replacing variable names in Damon.

Copyright (c) 2009 - 2011, Mark H. Moulton for Pythias Consulting, LLC.

Purpose:  This tool was developed to replace CamelCase argument
variables with all lower case equivalents across all Damon modules.
It is being retained because it is readily adaptable to other
global search-and-replace problems.

Damon Version:  1.0.15
Damon Release Date:  5/1/2012

Damon is written in Python 2.7.2 and Numpy 1.6.1, as distributed
by Enthought (EPD 7.2-2 (64-bit)).

License
-------
    This program references one or more software modules that are
    under copyright to Pythias Consulting, LLC.  Therefore, it is subject
    to either the Gnu Affero General Public License or the Pythias
    Commercial License, a copy of which is contained in the current
    working directory.

How To Use
----------
    The program is configured to convert CamelCase function/method
    variables (but not regular variables) to lower case.  To adapt
    it to other uses, edit according to the following principles:

    *   Edit the names.extend(...) statement to get the
        correct variables to edit.  If you already know
        the names to replace, you can comment out this
        part of the program and rely on the special
        dictionary.
        
    *   Edit the creation of the replace_ {} dictionary to
        capture the names you are after.  Currently, it
        applies the s.lower() function, but it can be
        anything.

    *   Set mode = 'inspect'.  Obtain and review all names
        to be replaced to make sure no new names will clash
        with reserved Python or other package names or will
        in other ways mangle the program.
        
        The convention is to add a trailing underscore_ where
        a Python clash would happen.

            *   Edit the removeit [] list to specify module
                contents to ignore

            *   Edit the special {} dictionarary to specify
                how to handle names that need special
                handling.
                
    *   This function replaces only complete words -- those
        governed by the regular expression '\b' (consult re
        "regular expressions" module in the standard library).
        Edit the re.sub(...) statement to replace characters
        or other types of patterns.

    *   Make sure to save a backup of the module to be edited.
        It is quite possible that a global search-and-replace
        will result in unintended side-effects that require
        debugging.

    *   Under filenames, list the Python modules in the current
        working directory that you want to edit.

    *   Otherwise, you don't need to do any other file handling.
        The program will automatically open and edit a Python
        module in place.

    *   Set mode = 'replace' and hit F5 to run the program.
        
"""
import os
import sys

import cPickle
import inspect
import fileinput
import re
import glob

import damon1


#############
##  Specs  ##
#############

# Set mode to: <'inspect','replace'>
mode = 'replace'    

# Files to edit
testpath = damon1.__path__[0]+'/tests/'
sys.path.append(testpath)
testfiles = glob.glob(testpath+'test_*.py')
testfiles.extend([testpath+'ut_template.py'])

files2inspect = ['core.py','tools.py','utils.py']
files2edit = files2inspect + testfiles + ['__init__.py','template.py']

print 'files2edit=\n',files2edit

# Module contents to ignore when getting variable names
removeit = ['core','utils','tools','npla','__package__','np','__doc__',
            'core','cPickle','__builtins__','__file__','sys','__name__',
            'npr','npma','os','__module__','__dict__','__weakref__',
            '__doc__','self','npt','tab']

# Names that need special attention

special = {'DamonObj':'Damon',
           'baseResid':'base_resid',
           'RunSpecs':'runspecs',
           'finSE':'fin_se',
           'baseEst':'base_est',
           'RandPercentNaN':'rand_nan',
           'RandRange':'rand_range',
           'Ents2Destd':'ents2restore',
           'finEAR':'fin_ear',
           'FixedRangeEnts':'ents2nan',
           'FixedRangeLoc':'range2nan',
           'AddSourceIDs':'source_ids',
           'AddDataDict':'add_datadict',
           'finEst':'fin_est',
           'restoreInvalid':'restore_invalid',
           'extractValid':'extract_valid',
           'FacCoords':'fac_coords',
           'Fac0Coord':'fac0coord',
           'fac1coord':'fac1coord',
           'finFit':'fin_fit',
           'PredEnts':'pred_ents',
           'Jolt':'jolt_',
           'baseEAR':'base_ear',
           'TabDataRCD':'tab_datadict',
           'MissingLbls':'miss4headers',
           'RecodeRange1':'recode1',
           'RecodeRange2':'recode2',
           'RecodeRange3':'recode3',
           'baseSE':'base_se',
           'baseFit':'base_fit',
           'CondCoord':'condcoord_',
           'ConstructLabel':'construct_label',
           'ConstructEnts':'construct_ents',
           'mergeAnsKey':'merge_anskey',
           'XtraHeadRng':'extra_headers',
           'PercentNaN':'p_nan',
           'ScoreMC':'score_mc',
           'RespCat':'resp_cat',
           'Dtype':'dtype',
           'finResid':'fin_resid',
           'ConstructAtts':'construct_atts',
           'ResidType':'resid_type',
           'TargData':'targ_data',
           'TargLabels':'targ_labels',
           'OrigData':'orig_data',
           'ItemDiff':'itemdiff',
           'itemDiff':'item_diff',
           'ParseParams':'parse_params',
           'Params':'params',
           'scoreMC':'score_mc',
           'ObjEst':'obj_est',
           'TargMeanSD':'mean_sd',
           'BankF0Ents':'bankf0',
           'BankF1Ents':'bankf1',
           'ObjEnts':'obj_ents',
           'OutputAs':'output_as',
           'RespCats':'resp_cats',
           'RLRow':'rl_row',
           'RLCol':'rl_col',
           'CLRow':'cl_row',
           'CLCol':'cl_col',
           'CoreRow':'core_row',
           'CoreCol':'core_col',
           'WholeRow':'whole_row',
           'WholeCol':'whole_col',
           'WholeArray':'whole',
           'Fileh':'fileh',
           'TextFile':'textfile',
           'TextFiles':'textfiles',
           'DataDictLink':'datadict_link',
           'DataDictWhole':'datadict_whole',
           'Pickle':'pickle',
           'RCD_Whole':'RCD_whole',
           'RCD_Dicts':'RCD_dicts',
           'RCD_Dicts_Whole':'RCD_dicts_whole',
           'ChunkFunc':'chunkfunc',
           'ChunkDict':'chunkdict',
           'Model':'model',
           'Num':'num',
           'extractValid_out':'extract_valid_out',
           'pseudoMiss_out':'pseudomiss_out',
           'scoreMC_out':'score_mc_out',
           'baseEst_out':'base_est_out',
           'baseResid_out':'base_resid_out',
           'baseEAR_out':'base_ear_out',
           'baseSE_out':'base_se_out',
           'baseFit_out':'base_fit_out',
           'finEst_out':'fin_est_out',
           'est2Logit_out':'est2logit_out',
           'itemDiff_out':'item_diff_out',
           'fillMiss_out':'fillmiss_out',
           'finResid_out':'fin_resid_out',
           'finFit_out':'fin_fit_out',
           'mergeAnsKey_out':'merge_anskey_out',
           'restoreInvalid_out':'restore_invalid_out',
           'summStat_out':'summstat_out',
           'RowEnts':'row_ents',
           'ColEnts':'col_ents',
           'ObjPerDim':'objperdim',
           'Stability':'stability',
           'Objectivity':'objectivity',
           'BestDim':'bestdim',
           'MaxPosDim':'maxposdim',
           'Accuracy':'accuracy',
           'PsMsResid':'psmsresid',
           'Fac0SE':'fac0_se',
           'Fac1SE':'fac1_se',
           'Fac0Infit':'fac0_infit',
           'Fac1Infit':'fac1_infit',
           'Fac0Outfit':'fac0_outfit',
           'Fac1Outfit':'fac1_outfit',
           'Reliability':'reliability',
           'CellVar':'cellvar',
           'CellFit':'cellfit',
           'MsIndex':'msindex',
           'PsMsIndex':'psmsindex',
           'TrueMsIndex':'true_msindex',
           'ParsedMsIndex':'parsed_msindex',
           'ParsedTrueMsIndex':'parsed_true_msindex',
           'ParsedPsMsIndex':'parsed_psmsindex',
           'ObjEstimates':'obj_estimates',
           'ObjCoords':'obj_coord',
           'EARCoord':'ear_coord',
           'EntCoord':'ent_coord',
           'StepCoord':'step_coord',
           'Facet0':'facet0',
           'Facet1':'facet1',
           'logitEAR_out':'logit_ear_out',
           'logitSE_out':'logit_se_out',
           'ObsPerCellFactor':'obspercell_factor',
           'SECoord':'se_coord',
           'Logit':'Logit',
           'EquateParams':'equate_params',
           'Ratio':'ratio',
           'Interval':'interval',
           'Sigmoid':'sigmoid',
           'ChangeLog':'changelog',
           'ObjParams':'obj_params',
           'PyTable.hd5':'pytable.hd5',
           'seedBank.pkl':'seedbank.pkl',
           'MyDamonObj':'my_DamonObj',
           'MyDmnObj':'my_obj',
           'StdParams':'std_params',
           'EAR':'EAR',
           'Facet':'Facet',
           'InputArray':'input_array',
           
           'Array':'array',
           'Arrays':'arrays',
           'Data':'data',
           'File':'file',
           'U':'U',
           'x':'x',
           'X':'X',
           'R':'R',
           'C':'C',
           'V':'V',
           'E':'E',
           'InitEArray':'init_earray',
           'InvUTU':'invUTU_',
           'invUTU':'invUTU',
           'Range':'range_',
           'Type':'type_',
           'Return':'return_',
           'ArrayNames':'array_names',
           'CondFacet':'cond_facet',
           'DataDict':'datadict',
           'SolveMethod':'solve_meth',
           'SolveMethSpecs':'solve_meth_specs',
           'SourceIDs':'source_ids',
           'TargetIDs':'target_ids',
           'InclTarg':'targ_in_sum',
           'SigmThresh':'sigma_thresh',
           'PredAlpha':'pred_alpha',
           'OrigObs':'orig_obs',
           'BiasedEst':'biased_est',
           'Shape':'shape',
           'MissLeftColLabels':'fill_left',
           'MissTopRowLabels':'fill_top',
           'MinRating':'min_rating',
           'RegRMSE':'rmse_reg',
           'ErrArray':'st_err',
           'SumSqRowPtBis':'row_ptbis',
           'SumSqColPtBis':'col_ptbis',
           'TargDataIndex':'targ_data_ind',
           'TupData':'tup_data',
           'PredKey':'pred_key',
           'MissMethod':'miss_meth',
           'AttRow':'att_row',
           'CountChars':'count_chars',
           'nKeyColHeaders':'nheaders4cols_key',
           'ExtrEst':'extr_est',
           'EARArray':'ear',
           'DataRCD':'datadict',
           'PyTables':'pytables',
           'Format':'format_',
           'MethSpecs':'meth_specs',
           'NearestVal':'nearest_val',
           'Median':'median_',
           'EstShape':'est_shape',
           'Tests':'tests_',
           'Val':'Val',
           'Res':'Res',

           'Locals':'_locals',
           'Locals1':'_locals1',
           '_baseEAR':'_base_ear',
           '_finResid':'_fin_resid',
           '_extractValid':'_extract_valid',
           '_mergeAnsKey':'_merge_anskey',
           '_scoreMC':'_score_mc',
           '_finEst':'_fin_est',
           '_baseFit':'_base_fit',
           '_baseResid':'_base_resid',
           '_baseSE':'_base_se',
           '_finFit':'_fin_fit',
           '_baseEst':'_base_est',
           '_restoreInvalid':'_restore_invalid',
           '_itemdiff':'_item_diff',

           }


#############
##   Get   ##
##  Names  ##
#############

if mode == 'inspect':
    objs = []
    names = []

    # Import module
    for i in range(len(files2inspect)):
        stringmod = files2inspect[i].replace('.py','')
        mod = __import__(stringmod)
        modobjs = mod.__dict__.keys()

        # Remove unneeded objects
        for obj in removeit:
            try:
                modobjs.remove(obj)
            except ValueError:
                pass

        # Include top-level function names in list
        names.extend(modobjs)

        # Get names automatically
        for obj in modobjs:
            try:
                names.extend(inspect.getargspec(mod.__dict__[obj])[0])
            except TypeError:

                try:
                    subobjs = mod.__dict__[obj].__dict__.keys()
                    for subobj in removeit:
                        try:
                            subobjs.remove(subobj)
                        except ValueError:
                            pass

                    names.extend(subobjs)
                    
                    for subobj in subobjs:
                        names.extend(inspect.getargspec(mod.__dict__[obj].__dict__[subobj])[0])

                    for name in removeit:
                        try:
                            names.remove(name)
                        except ValueError:
                            pass
                except:
                    pass


    #####################
    ##      Build      ##
    ##  replace_ dict  ##
    #####################

    replace_ = {}
    for name in names:
        replace_[name] = name.lower()   # replace name with lowercase version

    for specname in special.keys():
        replace_[specname] = special[specname]

    if mode == 'inspect':
        print 'replace_ dictionary:\n',replace_

    # Save as pickle
    dbfile = open('replaceDB.pkl','wb')
    cPickle.dump(replace_,dbfile)
    dbfile.close()
       

###############
##   Edit    ##
##  Modules  ##
###############

if mode == 'replace':

    print 'replace() is working...\n'

    # Use replace dictionary in pickle db
    dbfile = open('replaceDB.pkl','rb')
    replace_ = cPickle.load(dbfile)
    dbfile.close()
    
    for filename in files2edit:

        print 'Working on',filename
        
        # Edit line
        for line in fileinput.input(filename,inplace=True):

            # Replace all specified names in line
            for name in replace_.keys():
                line = re.sub(r'\b'+name+r'\b',replace_[name],line)

            # Replace line with fully edited line
            print line,

    print 'replace() is done.'


##############
##   Run    ##
##  Module  ##
##############

# To run functions that are defined in this module
##if __name__ == "__main__":
##    A = MyFunc(...)
##    print A



















