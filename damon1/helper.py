'''
this file has all the base classes and helpers
which are used all over the project..
'''

'''
#This is a wrapper which is added dynamically around all methods in Damon class
'''
import inspect
#from rpy2 import *

################
#some constants#
################
GLOBAL_INVALID_INSTANCE_ERR_MSG = "Invalid instance. Please read documentation."

################
#Method Wrapper#
################
class MethodWrapper():
    def __init(self):
        self.__parent__ is None
        
    def __execute__(self,*args, **kargs):
        raise DamonException("not implemented")

    def execute(self, *args, **kargs):
        return self.__execute__(self=self.__parent__, *args, **kargs)

    #this tells the python to call excute funtion when myobj() is applied..
    __call__ = execute


'''
this applies wrappers to all the methods passed as argument.
If in the dict contains attributes which are not methods , method
simply skip them
'''

def apply_wrappers(damon_class_name,damon_attrs):

    new_attrs={}
    
    #iterate over all methods and put a wrapper around everyone
    for method_name in damon_attrs:
        #here we will only apply wrapper to publicc methods..
        if (inspect.isfunction(damon_attrs[method_name]) is True
            and method_name.find('__') == -1):

            #create instance of method wrapper class , this will be put on the
            #current method
            m_w=MethodWrapper()
            
            #get all events list
            if(EventsList.has_key("$"+damon_class_name+"$"+method_name)):
                methodevents= EventsList["$"+damon_class_name+"$"+method_name]
            else:
                methodevents=[]

            #iterate over all events and assign to the method wrapper
            for event in methodevents:
                m_w.__dict__[event]=Event()

            #now assign the method to exeute function of wrapper.
            m_w.__execute__ = damon_attrs[method_name]

            #replace the method with method wrapper
            new_attrs[method_name] = m_w
        else:
            new_attrs[method_name] = damon_attrs[method_name]

    return new_attrs

################################################################################

'''
this is to be added as meta class of DamonBase class.
'''
class DamonMetaClass(type):
    def __new__(cls, name, bases, dct):
        
        for attr in dct:
             if (inspect.isfunction(dct[attr]) is True
                 and attr.find('__') == -1):
                 if dct[attr].__name__ != "wrapped":
                     raise DamonException("You have not implemented validation attributes on "
                                          +attr+" function")
                    
        if dct.has_key("__isbase__") is not True or dct["__isbase__"] is not True:
             dct = apply_wrappers(name,dct)
    
        return type.__new__(cls, name, bases, dct)
    
'''
This is base class for all classes
'''
class Base():
    def __init__(self):

        #it calls the __init__ method of its base.
        super(Base, self).__init__()
        
        #get all attributes of DamonBase
        base_dct= self.__class__.__dict__
     
        #here we make copy of all method_wrappers from class and assign the
        #fresh copy to object, and then tell the method wrapper copy about
        #the its calling object by setting the __parent__ attribute of method
        #wrapper copy to Damon object.
        for attr in base_dct:
            if base_dct[attr].__class__ is MethodWrapper:
                from copy import deepcopy as cp
                self.__dict__[attr]=cp(base_dct[attr])
                self.__dict__[attr].__parent__=self
        
    def __call_event(self,methodname,eventname):
        self.__dict__[methodname].__dict__[eventname]()
    

    
'''
This is a base class for all Damon classes.
'''
class DamonBase():
    
    #this tells python to use Damonmetaclass as its meta class.
    __metaclass__ =DamonMetaClass

    #this is a flag to show it is a base class.
    __isbase__=True
    
    def __init__(self):

        #it calls the __init__ method of its base.
        super(DamonBase, self).__init__()
        
        #get all attributes of DamonBase
        base_dct= self.__class__.__dict__
     
        #here we make copy of all method_wrappers from class and assign the
        #fresh copy to object, and then tell the method wrapper copy about
        #the its calling object by setting the __parent__ attribute of method
        #wrapper copy to Damon object.
        for attr in base_dct:
            if base_dct[attr].__class__ is MethodWrapper:
                from copy import deepcopy as cp
                self.__dict__[attr]=cp(base_dct[attr])
                self.__dict__[attr].__parent__=self
        
    def __call_event(self,methodname,eventname):
        self.__dict__[methodname].__dict__[eventname]()

        
'''
#this is a dictionary ccontaining the list of all eventhandlers assosiated
#with methods. if you want to use any eventhandler in a method, then its name
#has to be registered in this dictionary.. in beginining we are using
dicctionary to store eventhandlers, soon these will be put in xml file..
#syntax for key is ${classname}${methodname} e.g. $Damon_R$ltm will be key for
#method ltm in damon_R
#value of each key will be a list of eventhandlers for that method..
'''
EventsList={"$DamonR$ltm":['onstart','onend']}


'''
#this is the event class , which is used in whole project to initialise events.
'''
class Event:
    def __init__(self):
        self.handlers = set()

    def handle(self, handler):
        self.handlers.add(handler)
        return self

    def unhandle(self, handler):
        try:
            self.handlers.remove(handler)
        except:
            raise DamonException("Handler is not handling this event, so cannot unhandle it.")
        return self

    def fire(self, *args, **kargs):
        for handler in self.handlers:
            handler(*args, **kargs)

    def get_handler_count(self):
        return len(self.handlers)

    __iadd__ = handle
    __isub__ = unhandle
    __call__ = fire
    __len__  = get_handler_count


##############
# Validations #
##############
'''
Below is a validation decorator.. This will be put on top of each method in
damon class. 
'''
def validate_attributes(attributesValidationsRulesList):
    
    def validation_decorator(func):
        
        if not isinstance(attributesValidationsRulesList, list):
            raise DamonException("Please provide the list of attribute validation " +
                                 "rules" +
                                 "in validate_attributes decorator. See documentation of" +
                                 "validate_attributes.")

        i = 0
        for attributeRule in attributesValidationsRulesList:
            i = i+1
            if(attributeRule.has_key("name") == False or attributeRule["name"] == None or attributeRule["name"] == ""):
                raise DamonException("Please  provide name of attribute "+str(i) + " in validation" +
                                     "rules. See documentation of validate_attributes.")

        '''
        This loop iterates over all arguments of function and then checks if a rule is defined for
        this argument, if no rule is defined, error is shown,
        '''
        args_notfound_error = ""
        for arg in inspect.getargspec(func).args:
             #continue if argument is self..
             if(arg == "self"):
                 continue
                                 
             arg_found = False
             for attributeRule in attributesValidationsRulesList:
                 if(attributeRule["name"] == arg):
                     #raise the found flag..
                     arg_found = True
                     break
             #found flag not raised :( , show error...
             if(arg_found==False):
                 args_notfound_error+= "No validation rule is defined for argument "+arg+" .\n"
        #if any of args is not found in validation attributes list, raise the not found error.
        if(args_notfound_error!=""):
            raise DamonException(args_notfound_error)


        def wrapped(*args, **kwargs) :
            
            '''
            here we check if any custom self is given to method, if so, 
            then remove the self from kwargs and save in args.. this is hack 
            for method-wrapper object
            '''
            if(kwargs.has_key("self")):
                a = list(args)
                a.insert(0, kwargs['self'])
                args=tuple(a)
                del kwargs['self']

            '''
            validates against all rules of arg, and then error is shown to user
            if something does not validate..
            '''
            method_args=inspect.getargspec(func).args

            for attributeRule in attributesValidationsRulesList:

                arg=attributeRule["name"]
                
                #get index of arg in method arguments list..
                i = -1
                if arg in method_args:
                    i=method_args.index(arg)
                else:
                    continue
                
                #get argument's value
                args_val = None
                if(kwargs.has_key(arg)):
                    args_val=kwargs[arg]
                else:
                    #here we check if length of args, is greater than i
                    if(len(args) > i):
                        args_val = args[i]

                arg_null_allowed=True

                #check isnull
                if(attributeRule.has_key("isnull")==True and attributeRule["isnull"]==False):
                    arg_null_allowed = False
                    if args_val==None:
                        raise DamonException(arg +" cannot be null.")                    

                if(args_val!=None):

                    #check if any custom validation method is given in validation rules.
                    if(attributeRule.has_key("validation_method")==True
                       and inspect.isfunction(attributeRule["validation_method"])==True):
                        attributeRule["validation_method"](args[0],args_val)
                       
                    #check type
                    if(attributeRule.has_key("type")==True and type(args_val)
                       !=attributeRule["type"]):
                        raise DamonException("Type for argument "+arg+
                                             " is not correct. Please see documentaion of method.")

                    #check oneofthese
                    if(attributeRule.has_key("oneofthese")==True and type(attributeRule["oneofthese"])==list
                       and (args_val in attributeRule["oneofthese"])==False):
                        raise DamonException(str(arg)+" must be in one of these values-"+
                                             ' , '.join(str(e) for e in attributeRule["oneofthese"]))
                else:
                    #check if any default value is given for argument in validation rules list
                    if(attributeRule.has_key("default_value")==True):
                        if(len(args) > i):
                            args[i]=attributeRule["default_value"]
                        else:
                            kwargs[arg]=attributeRule["default_value"]
                            
            f=func(*args, **kwargs)

            return f
        
        return wrapped
 
    return validation_decorator

'''
This is a custom exception class for Damon object. very simple now. We will
extend it in future.
'''
class DamonException(Exception):
    def __init__(self, value):
        #it calls the __init__ method of its base.
        return super(DamonException, self).__init__(value)
        

'''
following class defines the validation rules for a given attribute..
class AttributeValidationRules(object):
    def __init__(self,_name,_type,_min,_max):
        self.Name=_name
        self.Type=_type
        self.Min=_min
        self.Max=_max
'''


