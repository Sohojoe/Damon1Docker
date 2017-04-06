# -*- coding: utf-8 -*-
"""
The utils_R.py module contains functions which support
Damon() methods in the core_R.py module.

Copyright (C) 2008, 2009, 2010, 2011 Mark H. Moulton for Pythias Consulting, LLC.
1225 Vienna Drive #86
Sunnyvale, CA 94089-1811
pythiasconsulting@gmail.com


License
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
# Import system modules


#from  rpy import *
##import rpy
##r = rpy.r

try:
    import rpy2 as rpy
    import rpy2.robjects as r
except ImportError:
    import rpy
    r = rpy.r

import damon1 as dmn

######################################################################
def _ltm(_locals):
    '''Supports method core_R.ltm()'''

    class LtmError(Exception): pass
    
    # Get self
    self = _locals['damonObj']
    coredata = dmn.tools.get_damon_datadict(self)["coredata"]
    
    #convert all nanvals to nanvals of r environment.
    coredata[coredata == self.nanval] = r.NAN

    #convert coredata into a dictionary with keys equal to items 
    data_dict={str(x):coredata[:,x-1] for x in range(1,len(self["collabels"][0]))}
    
    #this statement is not necessary ,  BASIC_CONVERSION is default..
    #rpy.set_default_mode(rpy.NO_CONVERSION)

    #import ltm library using r object, (to install libraries in R, you have '
    #to go to R console. as much i know, there is no way to do it from python. )
    r.library("ltm")

    #assign a object df to R environment.
    r.assign('df',data_dict)

    #convert df to r data frame to be passed to ltm function
    r("df = data.frame(df)")

    #this is the formula to be passed to ltm function, r("") is used to call 
    #those statements which are not posssible to be called from python environment..
    formula = r("df ~ "+_locals['formula_rightside'])

    #make the input arguments suitable for r.
    if _locals['na_action'] is not None:
        _locals['na_action'] = r[_locals['na_action']]
    
    #call r.ltm
    try:
        ltm_out = r.ltm(formula,
                        na_action=_locals['na_action'],
                        IRT_param=_locals['irt_param'],
                        constraint=_locals['constraint'],
                        start_val=_locals['start_val'],
                        control=_locals['control']
                        )
    except:
        exc = 'Unable to find r.ltm().  Make sure the package resides in the R library.'
        raise LtmError(exc)

    return ltm_out


######################################################################
def _grm(_locals):
    '''Supports method core_R.grm()'''
    
    # Get self
    self = _locals['damonObj']
    coredata = dmn.tools.get_damon_datadict(self)["coredata"]
    
    #convert all nanvals to nanvals of r environment.
    coredata[coredata==self.nanval]=r.NAN

    #convert coredata into a dictionary with keys equal to items 
    data_dict={str(x):coredata[:,x-1]
                       for x in range(1,len(self["collabels"][0]))}
    
    #this statement is not necessary ,  BASIC_CONVERSION is default..
    rpy.set_default_mode(rpy.NO_CONVERSION)

    #import ltm library using r object, (to install libraries in R, you have to go to R console. as much i know, there is no way to do it from python. )
    r.library("ltm")

    #make the input arguments suitable for r.
    if _locals['na_action']!=None:
        _locals['na_action']=r[_locals['na_action']]

    #assign a object df to R environment.
    r.assign('df',data_dict)

    print _locals['control']

    #convert df to r data frame to be passed to ltm function
    r("df = data.frame(df)")

    #get the robj of data frame created in r.
    r_data_frame=r("df")
    
    #call r.ltm
    grm_out=r.grm(r_data_frame,
                  constrained=_locals['constrained'],
                  na_action=_locals['na_action'],
                  IRT_param=_locals['irt_param'],
                  Hessian=_locals['hessian'],
                  start_val=_locals['start_val'],
                  control=_locals['control']
                  )
    
    return grm_out

######################################################################
def _gpcm(_locals):
    '''Supports method core_R.gpcm()'''
    
    # Get self
    self = _locals['damonObj']
    coredata = dmn.tools.get_damon_datadict(self)["coredata"]
    
    #convert all nanvals to nanvals of r environment.
    coredata[coredata==self.nanval]=r.NAN

    #convert coredata into a dictionary with keys equal to items 
    data_dict={str(x):coredata[:,x-1]
                       for x in range(1,len(self["collabels"][0]))}
    
    #this statement is not necessary ,  BASIC_CONVERSION is default..
    rpy.set_default_mode(rpy.NO_CONVERSION)

    #import ltm library using r object, (to install libraries in R,
    #you have to go to R console. as much i know, there is no way to do it
    #from python. )
    r.library("ltm")

    #make the input arguments suitable for r.
    if _locals['na_action']!=None:
        _locals['na_action']=r[_locals['na_action']]

    #assign a object df to R environment.
    r.assign('df',data_dict)

    #convert df to r data frame to be passed to ltm function
    r("df = data.frame(df)")

    #get the robj of data frame created in r.
    r_data_frame=r("df")
    
    #call r.ltm
    gpcm_out=r.gpcm(r_data_frame,
                  constraint=_locals['constraint'],
                  IRT_param=_locals['irt_param'],
                  start_val=_locals['start_val'],
                  na_action=_locals['na_action'],
                  control=_locals['control']
                  )
    
    return gpcm_out

######################################################################
'''
this is a private function used to calculate the estimates generated by ltm
function of r.                                                                
'''
def __get_ltm_estimates(ltm_out,damonObj,coredata):
    """
    __get_ltm_estimates docstring
    """
    
    if not isinstance(ltm_out,r):
        raise dmn.helper.DamonException("ltm_out must be output of ltm function of r.")

    if not isinstance(damonObj,dmn.core.Damon):
        raise dmn.helper.DamonException("ltm_out must be output of ltm function of r.")
    
    #convert the R obj to the ltm model.
    ltm_out = dmn.models_r.RLtmModel(ltm_out)

    #get the ltm object returned by r.
    ltm_out_as_py=ltm_out.__ltm_out__.as_py(BASIC_CONVERSION)

    #get factor scores to caculate person parameters
    factor_scores=ltm_out.factor_scores()

    #get score.dat dictionary of factor scores (it included the response
    #patterns parameters)
    score_dat=fcc["score.dat"]

    #get response patters from score_dat matrix
    responsepat_array=[score_dat["X"+str(x)] for x in
                       range(1,len(damon_obj["collabels"][0]))]

    #convert the response patters into the format of coredata of damon object.
    np_array_responsepatterns=numpy.swapaxes(responsepat_array,0,1)

    #convert coredata to float.
    np_array_coredata=numpy.array(coredata,dtype=float)

    #convert nanvals in response patters to those in damon.
    np_array_responsepatterns[numpy.isnan(np_array_responsepatterns)]=damon_obj.nanval
    
    #convert nanvals in response patters to those in damon.
    np_array_coredata[numpy.isnan(np_array_coredata)]=damon_obj.nanval

    #get dimensions in ltm model returned by r.
    dims=[]
    if(len(ltm_out_as_py["ltst"]["nams"])==2):
        dims=['z1']
    if(len(ltm_out_as_py["ltst"]["nams"])>2):
        dims=['z1','z2']

    #initialise person parameters to all zeros    
    person_parameters = numpy.zeros((len(damon_obj.rowlabels)-1,len(dims)))

    #iterate over all response patters and assign the parameters corresponding
    #to these response patters to the persons corresponding to these response
    #patterns.
    i=-1
    for row in np_array_responsepatterns:
        i=i+1
        pattern_data=[all(row1==row) for row1 in np_array_coredata]
    
        j=-1
        for dim in dims:
	    j=j+1
            person_parameters[numpy.array(pattern_data),j]=score_dat[dim][i]

    #get item_parameters
    item_parameters=numpy.array(ltm_out_as_py["coefficients"])
    
    #if(ltm_out_as_py["IRT.param"]==True and len(ltm_out_as_py["ltst"]["nams"])
       #==2):
        #item_parameters[:,0]=item_parameters[:,0]*item_parameters[:,1]

    #initialise estimates by ltm model.
    ltm_base_est= numpy.zeros((len(person_parameters),len(item_parameters)))

    #calculate estimates
    i=-1
    for item_param in item_parameters:
        i=i+1
        #if we have one dimensional
        if(len(ltm_out_as_py["ltst"]["nams"])==2):
            ltm_base_est[:,i]=1/(1+exp(item_param[0]
                                       +person_parameters[:,0]*item_param[1]))
            
        #if we have two dimensional
        if(len(ltm_out_as_py["ltst"]["nams"])>2):
            ltm_base_est[:,i]=1/(1+exp(item_param[0]
                                       +person_parameters[:,0]*item_param[1]
                                       +person_parameters[:,1]*item_param[2]))

    return ltm_base_est


# Get help for an R method
def get_help(method):
    """Access documentaton for R method.
    
    """
    from r.packags import importr
    utils = importr('utils')
    help_doc = utils.help(method)
    
    return str(help_doc)
    

