'''
This file is made with purpose to contain all validation related things. For now
, it has the custom methods to validate arguments. At later stage it may have
even more.
'''
import damon1 as dmn
#from helper import *
#
#import damon1.tools as tools

'''
This validates the self argument of ltm method of Damon_R class..
'''
def validate_r_ltm_self(self,arg_val):

    #check if the method is called only inside Damon
    if not isinstance(arg_val.__damon_obj__,dmn.core.Damon):
        raise dmn.helper.DamonException("this method can only be called inside the Damon"
                                 +" object")

    #get info regarding the data in damon obj.
    data_info = dmn.tools.get_datatype(arg_val.__damon_obj__)

    if (data_info["isContinuous"] is True or data_info["isDichotomous"] is False):
        raise dmn.helper.DamonException("data passed to Damon must be dichotomous.")

'''
This validates the formula_rightside argument of ltm method of Damon_R class..
'''
def validate_r_ltm_formula_rightside(self,arg_val):

    #firstly check type.
    if not isinstance(arg_val,str):
        raise dmn.helper.DamonException("Type of formula_rightside must be string.")

    #now lets check if right side of formula is valid or not ?
    terms_in_formula_rightside = arg_val.split('+')

    #now trim any white spaces in terms
    terms_in_formula_rightside = list(e.strip() for e in
                                    terms_in_formula_rightside)

    #if there are no terms in formula , raise error
    if (len(terms_in_formula_rightside) == 0):
        raise dmn.helper.DamonException("formula_rightside is not in correct format."+
                             " Please check the documentation of method.")

    #if there is 1 term in formula, and it is not z1 , raise error
    if (len(terms_in_formula_rightside) == 1
        and terms_in_formula_rightside[0].strip()!="z1"):
        raise dmn.helper.DamonException("formula_rightside is not in correct format. "+
                             "Please check the documentation of method.")
    
    #if there are 2 terms in formula, and these are not z1 , z2 , raise error
    if (len(terms_in_formula_rightside) == 2
       and all(list(e in ['z1','z2'] for e in terms_in_formula_rightside)) is
       False):
        raise dmn.helper.DamonException("formula_rightside is not in correct format. "+
                             "Please check the documentation of method.")

    #if there are 3 terms in formula, and these are not z1 , z2 , z1:z2 ,
    #raise error
    if (len(terms_in_formula_rightside) == 3
       and all( list(e in ['z1','z2','z1:z2']
                     for e in terms_in_formula_rightside)) is False):
        raise dmn.helper.DamonException("formula_rightside is not in correct format. "+
                             "Please check the documentation of method.")
    
    return True

'''
This validates the control argument of ltm method of Damon_R class..
'''
def validate_r_ltm_control(self,arg_val):
    
    #firstly check type.
    if not isinstance(arg_val,dict):
        raise dmn.helper.DamonException("Type of control must be a dictionary.")  

    #now lets check if the control dictionary is in right format
    if (all(list(e in ['iter.em','iter.qN','GHk','method','verbose']
             for e in arg_val.keys())) is False):
        raise dmn.helper.DamonException("control argument is not in correct format. "+
                             "Please check the documentation of method.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("iter.em") is True and not isinstance(arg_val['iter.em'],int)):
        raise dmn.helper.DamonException("iter.em key of control must be of type int.")
    
    #now check if values of all keys are in correct format..
    if (arg_val.has_key("iter.qN") is True and not isinstance(arg_val["iter.qN"],int)):
        raise dmn.helper.DamonException("iter.qN key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("GHk") is True and not isinstance(arg_val["GHk"],int)):
        raise dmn.helper.DamonException("GHk key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("method") is True and not isinstance(arg_val["method"],str)):
        raise dmn.helper.DamonException("method key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("verbose") is True and not isinstance(arg_val["verbose"],bool)):
        raise dmn.helper.DamonException("verbose key of control must be of type int.")

    return True

'''
This validates the self argument of gmr method of Damon_R class..
'''
def validate_r_grm_self(self,arg_val):

    #check if the method is called only inside Damon
    if not isinstance(arg_val.__damon_obj__,dmn.core.Damon):
        raise dmn.helper.DamonException("this method can only called inside the Damon"
                                 +" object")

    #get info regarding the data in damon obj.
    data_info = dmn.tools.get_datatype(arg_val.__damon_obj__)

    if (data_info["isContinuous"] is True):
        raise dmn.helper.DamonException("data passed to Damon must be polytomous.")
        

'''
This validates the control argument of grm method of Damon_R class..
'''
def validate_r_grm_control(self,arg_val):
    
    #firstly check type.
    if not isinstance(arg_val,dict):
        raise dmn.helper.DamonException("Type of control must be a dictionary.")  

    #now lets check if the control dictionary is in right format
    if (all(list(e in ['iter.qN','GHk','method','verbose','digits.abbrv']
             for e in arg_val.keys())) is False):
        raise dmn.helper.DamonException("control argument is not in correct format. "+
                             "Please check the documentation of method.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("iter.qN") is True and not isinstance(arg_val["iter.qN"],int)):
        raise dmn.helper.DamonException("iter.qN key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("GHk") is True and not isinstance(arg_val["GHk"],int)):
        raise dmn.helper.DamonException("GHk key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("method") is True and not isinstance(arg_val["method"],str)):
        raise dmn.helper.DamonException("method key of control must be of type str.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("verbose") is True and not isinstance(arg_val["verbose"],bool)):
        raise dmn.helper.DamonException("verbose key of control must be of type bool.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("digits.abbrv") is True
       and not isinstance(arg_val["digits.abbrv"],int)):
        raise dmn.helper.DamonException("digits.abbrv key of control must be of type int.")
    
    return True

'''
This validates the self argument of gpcm method of Damon_R class..
'''
def validate_r_gpcm_self(self,arg_val):

    #check if the method is called only inside Damon
    if not isinstance(arg_val.__damon_obj__,dmn.core.Damon):
        raise dmn.helper.DamonException("this method can only called inside the Damon"
                                 +" object")

    #get info regarding the data in damon obj.
    data_info = dmn.tools.get_datatype(arg_val.__damon_obj__)

    if (data_info["isContinuous"] is True):
        raise dmn.helper.DamonException("data passed to Damon must be polytomous.")
        

'''
This validates the control argument of gpcm method of Damon_R class..
'''
def validate_r_gpcm_control(self,arg_val):
    
    #firstly check type.
    if not isinstance(arg_val,dict):
        raise dmn.helper.DamonException("Type of control must be a dictionary.")  

    #now lets check if the control dictionary is in right format
    if (all(list(e in ['iter.qN','GHk','optimizer','optimMethod','numrDeriv',
                      'epsHes','parscale','verbose']
             for e in arg_val.keys())) is False):
        raise dmn.helper.DamonException("control argument is not in correct format. "+
                             "Please check the documentation of method.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("iter.qN") is True and not isinstance(arg_val["iter.qN"],int)):
        raise dmn.helper.DamonException("iter.qN key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("GHk") is True and not isinstance(arg_val["GHk"],int)):
        raise dmn.helper.DamonException("GHk key of control must be of type int.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("optimizer") is True and (not isinstance(arg_val["optimizer"],str) 
        or (arg_val["optimizer"] in ["optim","nlmnib"]) is False)):
        raise dmn.helper.DamonException("optimizer key of control must be of type str and"
                             +" it must be one of these ['optim','nlmnib']")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("optimMethod") is True and not isinstance(arg_val["optimMethod"],str)):
        raise dmn.helper.DamonException("optimMethod key of control must be of type str.")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("numrDeriv") is True and (not isinstance(arg_val["numrDeriv"],str) 
        or (args_val["numrDeriv"] in ["fd","cd"]) is False)):
        raise dmn.helper.DamonException("numrDeriv key of control must be of type str and"
                             +" it must be one of these ['fd','cd']")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("epsHes") is True and not isinstance(arg_val["epsHes"],int)):
        raise dmn.helper.DamonException("epsHes key of control must be of type int")

    #now check if values of all keys are in correct format..
    if (arg_val.has_key("parscale") is True and not isinstance(arg_val["parscale"],float)):
        raise dmn.helper.DamonException("parscale key of control must be of type float")
        
    #now check if values of all keys are in correct format..
    if (arg_val.has_key("verbose") is True and not isinstance(arg_val["verbose"],bool)):
        raise dmn.helper.DamonException("verbose key of control must be of type bool.")
    
    return True

'''
This validates the constraint argument of ltm method of Damon_R class..
'''
def validate_r_ltm_constraint(self,arg_val):

    return True

'''
This validates the start_val argument of ltm method of Damon_R class..
'''
def validate_r_ltm_start_val(self,arg_val):

    return True

'''
This validates the type argument of margins method of RLtmModel class..
'''
def validate_RLtmModel_margins_type(self,arg_val):

    #firstly check type.
    if not isinstance(arg_val,list):
        raise dmn.helper.DamonException("Type of margins must be a list.")     

    #now check if list has correct items
    if (all(list(e in ['two-way','three-way']
             for e in arg_val)) is False):
        raise dmn.helper.DamonException("type argument is not in correct format. Please "+
                             "check the documentation.")

    return True
  
'''
This validates the range argument of information method of RLtmModel class..
'''
def validate_RLtmModel_information_range(self,arg_val):

    #firstly check type.
    if not isinstance(arg_val,list):
        raise dmn.helper.DamonException("Type of margins must be a list.")     

    #now check that list must have only two entries and both must be integer..
    if (len(arg_val)!=2 or all(list(type(e) == int for e in arg_val)) is False):
        raise dmn.helper.DamonException("range argument is not in correct format. Please "+
                             "check the documentation.")

    return True

