'''''
This is the partial of core module. It contains all the methods to call R
packages
'''''
import damon1 as dmn


class DamonR(dmn.helper.DamonBase):
    def __init__(self,_damon_obj=None):

        self.__damon_obj__ = _damon_obj
        
        super(DamonR, self).__init__()

    
    @dmn.helper.validate_attributes([
        {"name":"self","validation_method":dmn.validation.validate_r_ltm_self},
        {"name":"formula_rightside","type":str,"isnull":False},
        {"name":"irt_param","type":bool,"default_value":False},
        {"name":"control","type":dict,
             "validation_method":dmn.validation.validate_r_ltm_control},
        {"name":"na_action","oneofthese":["na.omit","na.exclude",
                                         "na.pass","na.fail",None]},
        {"name":"start_val","type":list,
             "validation_method":dmn.validation.validate_r_ltm_start_val},
        {"name":"constraint","type":list,
             "validation_method":dmn.validation.validate_r_ltm_constraint},
        ])

    def ltm(self,formula_rightside,constraint=None,irt_param=None,
            start_val=None,na_action=None,control=None):
        """Apply the R Latent Trait Model to dichotomous data.
        
        """

        if not isinstance(self.__damon_obj__,dmn.core.Damon):
            raise dmn.helper.DamonException("this method can only called inside a Damon object")

        print 'ltm() is working...\n'

        damonObj=self.__damon_obj__
        
        # Run utility, assign attributes
        ltm_out = dmn.models_R.RLtmModel(dmn.utils_R._ltm(locals()))
        damonObj.R.ltm_out = ltm_out

        print 'ltm() is done -- see my_obj.R.ltm_out '
        print 'Contains:\n',damonObj.R.ltm_out.__dict__.keys(),'\n'
            
        return None


    @dmn.helper.validate_attributes([
        {"name":"self","validation_method":dmn.validation.validate_r_grm_self},
        {"name":"irt_param","type":bool,"default_value":True},
        {"name":"control","type":dict,
             "validation_method":dmn.validation.validate_r_grm_control},
        {"name":"na_action",
             "oneofthese":["na.omit","na.exclude","na.pass","na.fail",None]},
        {"name":"start_val","type":list},
        {"name":"constrained","type":bool,"default_value":False},
        {"name":"hessian","type":bool,"default_value":False}
        ])
    
    
    def grm(self,constrained=False,irt_param=None,hessian=None,start_val=None
            ,na_action=None,control=None):
        """
        grm docstring
        """
        
        if not isinstance(self.__damon_obj__,dmn.core.Damon):
            raise dmn.helper.DamonException("this method can only called inside the Damon"
                                 +" object")

        print 'grm() is working...\n'

        damonObj = self.__damon_obj__
        
        # Run utility, assign attributes
        grm_out = dmn.models_r.RLtmModel(dmn.utils_R._grm(locals()))
        damonObj.R.grm_out = grm_out

        print 'grm() is done -- see my_obj.R.ltm_out '
        print 'Contains:\n',damonObj.R.grm_out.__dict__.keys(),'\n'
            
        return None

    @dmn.helper.validate_attributes([
        {"name":"self","validation_method":dmn.validation.validate_r_gpcm_self},
        {"name":"irt_param","type":bool,"default_value":True},
        {"name":"constraint","type":str,"oneofthese":["gpcm","IPL","rasch"],
             "default_value":"gpcm"},
        {"name":"start_val","type":list},
        {"name":"na_action",
             "oneofthese":["na.omit","na.exclude","na.pass","na.fail",None]},
        {"name":"control","type":dict,
             "validation_method":dmn.validation.validate_r_gpcm_control}])
    
    
    def gpcm(self,constraint="gpcm",irt_param=None,start_val=None
            ,na_action=None,control=None):
        """
        gpcm docstring
        """        
        if not isinstance(self.__damon_obj__,dmn.core.Damon):
            raise dmn.helper.DamonException("this method can only called inside the Damon"
                                 +" object")

        print 'gpcm() is working...\n'

        damonObj = self.__damon_obj__
        
        # Run utility, assign attributes
        gpcm_out = dmn.models_r.RLtmModel(dmn.utils_R._gpcm(locals()))
        damonObj.R.gpcm_out = gpcm_out

        print 'gpcm() is done -- see my_obj.R.ltm_out '
        print 'Contains:\n',damonObj.R.gpcm_out.__dict__.keys(),'\n'
            
        return None

