'''
this file contains the models which are returned by different files.
'''
import damon1 as dmn
#import helper
#from validation import *

#from  rpy import *
try:
    import rpy2 as rpy
    import rpy2.robjects as r
except ImportError:
    import rpy
    r = rpy.r

'''
this is model shich will be returned by irt models of R in lt package
such as ltm() , gtm() , rasch() etc.
'''
class RLtmModel(dmn.helper.DamonBase):
    def __init__(self,ltm_out):
        
        if not isinstance(ltm_out,rpy.Robj):
            return None

        self.__ltm_out__=ltm_out

        super(RLtmModel, self).__init__()
    
    def __validate_instance(self):
        if (self.__ltm_out__ is not None):
            return False
        else:
            return True

    @dmn.helper.validate_attributes([{"name":"robust_se","type":bool
                                  ,"default_value":False}])
    def summary(self,robust_se=None):

        if (self.__validate_instance() is False):
            return dmn.validation.GLOBAL_INVALID_INSTANCE_ERR_MSG

        dmn.validation.set_default_mode(dmn.validation.BASIC_CONVERSION)
        fit_summary = dmn.validation.r.summary(self.__ltm_out__,robust_se=robust_se)
        dmn.validation.set_default_mode(dmn.validation.NO_CONVERSION)

        return fit_summary

    @dmn.helper.validate_attributes([{"name":"resp_patterns","type":list
                                  ,"default_value":None},
                                 {"name":"order","type":bool
                                  ,"default_value":False}])
    def residuals(self,resp_patterns=None,order=None):

        if (self.__validate_instance() is False):
            return dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG

        dmn.helper.set_default_mode(dmn.helper.BASIC_CONVERSION)
        fit_residuals= r.residuals(self.__ltm_out__,resp_patterns=resp_patterns
                                   ,order=order)
        rpy.set_default_mode(rpy.NO_CONVERSION)

        return fit_residuals

    @dmn.helper.validate_attributes([{"name":"type_","type":list
                                  ,"validation_method":
                                  dmn.validation.validate_RLtmModel_margins_type
                                  ,"default_value":["two-way", "three-way"]},
                                 {"name":"rule","type":float
                                  ,"default_value":3.5},
                                 {"name":"nprint","type":int
                                  ,"default_value":3}])
    def margins(self,type_=None,rule=None,nprint=None):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        fit_residuals= r.margins(self.__ltm_out__,type=type_,rule=rule
                                 ,nprint=nprint)
        rpy.set_default_mode(rpy.NO_CONVERSION)
    
        return fit_residuals

    @dmn.helper.validate_attributes([{"name":"range_","type":list
                                  ,"validation_method"
                                  :dmn.validation.validate_RLtmModel_information_range,
                                  "default_value":[-10,10]},
                                 {"name":"items","type":list}])
    def information(self,range_=None,items=None):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        fit_information= r.information(self.__ltm_out__,range=range_
                                       ,items=items)
        rpy.set_default_mode(rpy.NO_CONVERSION)
    
        return fit_information

    @dmn.helper.validate_attributes([{"name":"resp_patterns","type":list
                                  ,"default_value":None},
                                 {"name":"method","type":str
                                  ,"oneofthese":["EB","MI","Component","EAP",None]},
                                 {"name":"B","type":int,"default_value":5},
                                 {"name":"robust_se","type":bool
                                  ,"default_value":False},
                                 {"name":"prior","type":bool
                                  ,"default_value":True},
                                 {"name":"return_MIvalues","type":bool
                                  ,"default_value":False}])
    def factor_scores(self,resp_patterns=None,method=None,B=None,robust_se=None
                      ,prior=None,return_MIvalues=None):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        fit_factor_scores= r.factor_scores(self.__ltm_out__
                                           ,resp_patterns=resp_patterns,
                                           method=method,B=B
                                           ,robust_se=robust_se,
                                           prior=prior
                                           ,return_MIvalues=return_MIvalues)
        rpy.set_default_mode(rpy.NO_CONVERSION)
    
        return fit_factor_scores

    @dmn.helper.validate_attributes([{"name":"G","default_value":10.0},
                                 {"name":"FUN","type":str
                                  ,"default_value":"median"},
                                 {"name":"simulate_p_value","type":bool
                                  ,"default_value":False},
                                 {"name":"B","type":int,"default_value":100.0}])
    def item_fit(self,G=None,FUN=None,simulate_p_value=None,B=None):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        fit_item_fit= r.item_fit(self.__ltm_out__,G=G,FUN=FUN,
                                 simulate_p_value=simulate_p_value,B=B)
        rpy.set_default_mode(rpy.NO_CONVERSION)
    
        return fit_item_fit

    @dmn.helper.validate_attributes([{"name":"alternative"
                                  ,"default_value"
                                  :["less", "greater", "two.sided"]},
                                 {"name":"resp_patterns","type":list,
                                  "default_value":None},
                                 {"name":"simulate_p_value","type":bool
                                  ,"default_value":False},
                                 {"name":"FUN","default_value":None},
                                 {"name":"B","type":int,"default_value":1000}])
    def person_fit(self,alternative=None,resp_patterns=None
                   ,simulate_p_value=None,FUN=None,B=None):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        fit_person_fit= r.person_fit(self.__ltm_out__,alternative=alternative
                                     ,resp_patterns=resp_patterns,
                                     simulate_p_value=simulate_p_value,
                                     FUN=FUN,B=B)
        rpy.set_default_mode(rpy.NO_CONVERSION)
    
        return fit_person_fit

    @dmn.helper.validate_attributes([{"name":"none"}])
    def item_characterstic_curve(self):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.NO_CONVERSION)
        r.plot(self.__ltm_out__, lwd = 2, cex = 1.2, legend = True, cx = "left",
               xlab = "Latent Trait", cex_main = 1.5, cex_lab = 1.3
               , cex_axis = 1.1)

    @dmn.helper.validate_attributes([{"name":"none"}])
    def item_information_curve(self):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        
        rpy.set_default_mode(rpy.NO_CONVERSION)
        r.plot(self.__ltm_out__, type = "IIC", lwd = 2, cex = 1.2, legend = True
               ,cx = "topleft", xlab = "Latent Trait", cex_main = 1.5,
               cex_lab = 1.3, cex_axis = 1.1)
        
    @dmn.helper.validate_attributes([{"name":"none"}])
    def test_information_curve(self):

        if (self.__validate_instance() is False):
            return   dmn.helper.GLOBAL_INVALID_INSTANCE_ERR_MSG        

        rpy.set_default_mode(rpy.NO_CONVERSION)
        r.plot(self.__ltm_out__, type = "IIC", items = 0, lwd = 2,
               xlab = "Latent Trait",cex_main = 1.5, cex_lab = 1.3
               , cex_axis = 1.1)

