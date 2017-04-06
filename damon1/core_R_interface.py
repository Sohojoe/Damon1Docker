'''''
This is interface class of core_R class.
'''''
import damon1 as dmn

class DamonR(dmn.helper.DamonBase):
    def __init__(self):

        self.__parent__ = None
        
        super(DamonR, self).__init__()

    @dmn.helper.validate_attributes([{"Name":"formula","Type":str},
                         {"Name":"noOfIterations","Type":int}])
    def ltm(self,formula,noOfIterations):
        if not isinstance(self.__parent__,dmn.core.Damon):
            return 'this method can only called inside the Damon object'
        
        dmn.core_R.ltm(self.__parent__)
        #from core_r import ltm as core_r_ltm
        #core_r_ltm(self.__parent__)

