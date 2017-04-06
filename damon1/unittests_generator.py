# -*- coding: utf-8 -*-
"""
this module creates functionality to generate unit tests at runtime.
For now it will have very basic functionality, but it will be improving with 
time.
"""

import damon1 as dmn
import os
import sys
import inspect
from copy import deepcopy as cp
import numpy as np

class UnitTestsGenerator:
    '''
    this class contains method generate to generate unit tests at runtime.
    '''
    def __init__(self, module_name, class_name, method__name):
        '''
        this class contains methods to generate unit tests at runtime.
        '''
        if module_name is None:
            module_name = ""

        if class_name is None:
            class_name = ""
            
        path = dmn.__path__[0]+'/tests/new/'
        os.chdir(path)
        file_full_name = path + module_name + "_" + class_name + "_" + \
            method__name + ".py"
        
        self.cg_ = dmn.code_generator.CodeGenerator(file_full_name)
        self.codeblocks_list = []
        self.module_name = module_name
        self.class_name = class_name
        self.method_name = method__name
        #super(UnitTestsGenerator, self).__init__()
        
    def generate(self,
                 create_data_scenarios=None,
                 create_data_function_module_name=None,
                 create_data_function_name=None,
                 method_scenarios=None,
                 is_create_data_needed=None):
        '''
        this method will generate the unit tests at runtime based on parameters
        given to method
        '''
        #convert scenarios to parameters
        [create_data_paramters, method_parameters_permutations] = \
              self.convert_scenarios_to_parameters(create_data_scenarios,
              method_scenarios)
        
        #append import statements
        self.append_import_statements()

        self.codeblocks_list.append(self.cg_.create_newline())
        
        if create_data_function_name is not None:
            #get create data method        
            exec("import " + create_data_function_module_name)
            
            create_data_function = getattr(
                  sys.modules[create_data_function_module_name], 
                  create_data_function_name)
                  
            self.codeblocks_list.append(self.cg_.get_object_sourcecode(
                  create_data_function))
        

        self.codeblocks_list.append(self.cg_.get_object_sourcecode(
              get_method_parameters))

        self.codeblocks_list.append(self.cg_.get_object_sourcecode(
              format_method_parameters))
              
        self.codeblocks_list.append(self.cg_.get_object_sourcecode(
              run_method))
              
        self.codeblocks_list.append(self.cg_.get_object_sourcecode(
              generate_reports))

        self.codeblocks_list.append(self.cg_.get_object_sourcecode(
              test_flow))
              
        #create an assignment statement of create_data_parameters
        self.codeblocks_list.append(self.cg_.create_statement(
              "create_data_parameters_list = " + self.
              cg_.convert_object_to_string(create_data_paramters)))           
        
        #create an assignment statement of method_parameters
        self.codeblocks_list.append(self.cg_.create_statement(
            "method_parameters_permutations = " + self.cg_
            .convert_object_to_string(method_parameters_permutations)))
        
        #create an assignment statement of method_parameters
        self.codeblocks_list.append(self.cg_.create_statement(
            "test_flow(create_data_parameters_list, method_parameters_permutations, \"" 
            + self.cg_.convert_object_to_string(self.module_name) + "\", \"" 
            + self.cg_.convert_object_to_string(self.class_name) + "\", \"" 
            + self.cg_.convert_object_to_string(self.method_name) + "\", " 
            + self.cg_.convert_object_to_string(create_data_function_name)
            + " )" ))

        #write whole code to file.    
        self.cg_.write_to_file(self.codeblocks_list)
        
        print "test file generated"
                   
        return None
        
    def append_import_statements(self, import_statemetnts_scenarios=None):
        '''
        it converts scenarios to parameters
        '''
        self.codeblocks_list.append(
              self.cg_.create_statement("import damon1 as dmn"))
        self.codeblocks_list.append(
              self.cg_.create_statement("import sys"))
        self.codeblocks_list.append(
              self.cg_.create_statement("import os"))
        self.codeblocks_list.append(
              self.cg_.create_statement("import inspect"))
        self.codeblocks_list.append(
              self.cg_.create_statement("from copy import deepcopy as cp"))
        self.codeblocks_list.append(
              self.cg_.create_statement("from array import array as array"))
        self.codeblocks_list.append(
              self.cg_.create_statement("import numpy as np"))
        
    def convert_scenarios_to_parameters(self, create_data_scenarios=None,
                                        method_scenarios=None):
        '''
        it converts scenarios to parameters
        '''
        
        return [create_data_scenarios, method_scenarios]
    

def test_flow(create_parameters_list, method_parameters_permutations,
              module_name, class_name, method_name, 
              create_data_function):
    '''
    this method defines the complete flow of test.
    '''
    
    method = None    
    
    #get method from its name
    if module_name is not "" :
        exec("import " + module_name)
        if class_name is not "" :
            cls = getattr(sys.modules[module_name], class_name)
            method = getattr(cls, method_name)
        else :
            method = getattr(sys.modules[module_name], method_name)

    if method == None:
        print "method was not found. Check your module name , class name , method name"
        return None
        

    method_parameters_list = get_method_parameters(method, 
                                 method_parameters_permutations)
    
    method_output_list = []
    method_exception_list = []
    
    for create_parameters in create_parameters_list:
        
        if create_data_function is not None :
            data_object = create_data_function(*create_parameters)

        for method_parameters in method_parameters_list:
            formattted_method_parameters = format_method_parameters( 
                data_object, method_parameters)
            
            [run_method_output, method_exception] = \
                run_method(method, formattted_method_parameters)
            
            method_output_list.append(run_method_output)
            method_exception_list.append(method_exception)
                
    generate_reports(method_output_list , method_exception_list)


def create_Damon_data(nfac0,  # [Number of Facet 0 elements -- rows/persons]
    nfac1,  # [Number of Facet 1 elements -- columns/items]
    ndim,   # [Number of dimensions to create]
    seed = None,  # [None => randomly pick starter coordinates; int => integer of "seed" random coordinates]
    facmetric = [4,-2],  # [[m,b] => rand() * m + b, to set range of facet coordinate values]
    noise = None, # [<None, noise, {'Rows':<noise,{1:noise1,4:noise4}>,'Cols':<noise,{2:noise2,5:noise5}> => add error to rows/cols]
    validchars = None,   # [None; ['All',[valid chars]]; or ['Cols', {1:['a','b','c','d'],2:['All'],3:['1.2 -- 3.5'],4:['0 -- '],...}] ]
    mean_sd = None, # [None; ['All',[Mean,SD]]; or ['Cols', {1:[Mean1,SD1],2:[Mean2,SD2],3:[None],...}] ]
    p_nan = 0.0,  # [Proportion of cells to make missing at random]
    nanval = -999.,  # [Numeric code for designating missing values]
    condcoord_ = None,  # [< None, 'Orthonormal']
    nheaders4rows = 1,  # [Number of header column labels to put before each row]
    nheaders4cols = 1,  # [Number of header row labels to put before each column]
    extra_headers = 0,  # [If headers > 1, range of integer values for header labels, applies to both row and col.]
    input_array = None,   # [<None, name of data array to import>]
    output_as = 'Damon',  # [<'Damon','datadict','array','file','Damon+file','datadict+file','array+file'>]
    outfile = None,    # [<None, name of the output file/path prefix when output_as includes 'file'>]
    delimiter = None,    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '        '>]
    bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]]
    bankf1 = None,  # [<None => no bank,['MyBank.pkl',[<'All', list of F1 (Col) entities>]]> ]
    createbank = None ,#[None, True]
    convert_to_datadict = None , #[<None , True>],
    BankName = None,  # [<None,'MyBank.pkl'>]
    colkeytype = None, #[None, True]
    rowkeytype = None, #[None, True]
    simulate_whole = None, #[None, True]
    extract_test = None,
    create_sample_datadict = None,
    **kwargs
    ):
    '''
    it uses create data parameters to create data to be passed to method.
    '''
    
    '''
    run_fin_est = None
    run_equate = None
    run_summstat = None
    run_est2logit = None
    run_subscale =  None
    run_fin_resid = None
    run_base_fit = None
    run_coord = None
    run_base_est = None
    run_base_resid = None
    run_base_ear = None
    run_base_se = None
    run_fin_resid = None
    run_base_fit = None
    run_parse = None
    run_standardize = None
    run_pseudomiss = None
    
    fin_est_parameters = {}
    equate_parameters = {}
    summstat_parameters = {}
    est2logit_parameters = {}
    subscale_parameters =  {}
    fin_resid_parameters = {}
    base_fit_parameters = {}
    coord_parameters = {"ndim":[[ndim]]}
    base_est_parameters = {}
    base_resid_parameters = {}
    base_ear_parameters = {}
    base_se_parameters = {}
    fin_resid_parameters = {}
    base_fit_parameters = {}
    parse_parameters = {"resp_cat":'Find'}
    standardize_parameters = {"metric":'std_params',"std_params":bankf1[0]}
    pseudomiss_parameters = {"seed":1,"rand_nan" : 0.1}
    
    if(kwargs.has_key("run_fin_est")):
        run_fin_est=kwargs["run_fin_est"]
    if(kwargs.has_key("run_equate")):
        run_equate=kwargs["run_equate"]
    if(kwargs.has_key("run_summstat")):
        run_summstat=kwargs["run_summstat"]
    if(kwargs.has_key("run_est2logit")):
        run_est2logit=kwargs["run_est2logit"]
    if(kwargs.has_key("run_subscale")):
        run_subscale=kwargs["run_subscale"]
    if(kwargs.has_key("run_fin_resid")):
        run_fin_resid=kwargs["run_fin_resid"]
    if(kwargs.has_key("run_base_fit")):
        run_base_fit=kwargs["run_base_fit"]
    if(kwargs.has_key("run_coord")):
        run_coord=kwargs["run_coord"]
    if(kwargs.has_key("run_base_est")):
        run_base_est=kwargs["run_base_est"]
    if(kwargs.has_key("run_base_resid")):
        run_base_resid=kwargs["run_base_resid"]
    if(kwargs.has_key("run_base_ear")):
        run_base_ear=kwargs["run_base_ear"]
    if(kwargs.has_key("run_base_se")):
        run_base_se=kwargs["run_base_se"]
    if(kwargs.has_key("run_fin_resid")):
        run_fin_resid=kwargs["run_fin_resid"]
    if(kwargs.has_key("run_base_fit")):
        run_base_fit=kwargs["run_base_fit"]
    if(kwargs.has_key("run_parse")):
        run_parse=kwargs["run_parse"]
    if(kwargs.has_key("run_standardize")):
        run_standardize=kwargs["run_standardize"]
    if(kwargs.has_key("run_pseudomiss")):
        run_pseudomiss=kwargs["run_pseudomiss"]      
  
    if(kwargs.has_key("fin_est_parameters")):
        fin_est_parameters=kwargs["fin_est_parameters"]
    if(kwargs.has_key("equate_parameters")):
        equate_parameters=kwargs["equate_parameters"]
    if(kwargs.has_key("summstat_parameters")):
        summstat_parameters=kwargs["summstat_parameters"]
    if(kwargs.has_key("est2logit_parameters")):
        est2logit_parameters=kwargs["est2logit_parameters"]
    if(kwargs.has_key("subscale_parameters")):
        subscale_parameters=kwargs["subscale_parameters"]
    if(kwargs.has_key("fin_resid_parameters")):
        fin_resid_parameters=kwargs["fin_resid_parameters"]
    if(kwargs.has_key("base_fit_parameters")):
        base_fit_parameters=kwargs["base_fit_parameters"]
    if(kwargs.has_key("coord_parameters")):
        coord_parameters=kwargs["coord_parameters"]
    if(kwargs.has_key("base_est_parameters")):
        base_est_parameters=kwargs["base_est_parameters"]
    if(kwargs.has_key("base_resid_parameters")):
        base_resid_parameters=kwargs["base_resid_parameters"]
    if(kwargs.has_key("base_ear_parameters")):
        base_ear_parameters=kwargs["base_ear_parameters"]
    if(kwargs.has_key("base_se_parameters")):
        base_se_parameters=kwargs["base_se_parameters"]
    if(kwargs.has_key("fin_resid_parameters")):
        fin_resid_parameters=kwargs["fin_resid_parameters"]
    if(kwargs.has_key("base_fit_parameters")):
        base_fit_parameters=kwargs["base_fit_parameters"]
    if(kwargs.has_key("parse_parameters")):
        parse_parameters=kwargs["parse_parameters"]
    if(kwargs.has_key("standardize_parameters")):
        standardize_parameters=kwargs["standardize_parameters"]
    if(kwargs.has_key("pseudomiss_parameters")):
        pseudomiss_parameters=kwargs["pseudomiss_parameters"]  
      
    '''  
    #setup data to be passed to create_data
    all_args = locals()
    create_data_args = all_args.copy()
    del_args = ['bankf0', 'bankf1','createbank','pPsMiss','convert_to_datadict'
                ,'BankName','convert_colkeytype_tostring','simulate_whole',
                'extract_test', 'create_sample_datadict']
    for arg in del_args:
        del create_data_args[arg]
    
    D = dmn.core.create_data(**create_data_args)
    data = D['data']
    model = D['model']
    anskey = D['anskey'] 
    Keys = data['anskey']['rl_col']['ItemID'][:].astype(int)
    Vals = data['anskey']['core_col']['Correct']
    anskey_2 = ['Cols',dict(zip(Keys,Vals))]
    
    if not isinstance(data,dmn.core.Damon):
        data = dmn.core.Damon(data,'datadict_link',pytables=data['fileh'],verbose=None)
        model = dmn.core.Damon(model,'datadict_link',pytables=model['fileh'],verbose=None)
        anskey = dmn.core.Damon(anskey,'datadict',pytables=None,verbose=None)

    if extract_test is True :
        temp = np.copy(data.data_out['collabels'][2,:])
        data.data_out['collabels'][2,:] = data.data_out['collabels'][0,:]
        data.data_out['collabels'][0,:] = temp
        data.data_out['key4cols'] = 2
        
        temp = np.copy(data.data_out['rowlabels'][:,2])
        data.data_out['rowlabels'][:,2] = data.data_out['rowlabels'][:,0]
        data.data_out['rowlabels'][:,0] = temp
        data.data_out['key4rows'] = 2

    if simulate_whole is True :
        try:
            data = dmn.core.Damon(D.data_out,'datadict','RCD+Whole')
        except AttributeError:
            data = dmn.core.Damon(D,'datadict','RCD+Whole')

    if colkeytype is not None :
        data.colkeytype = colkeytype
        data.data_out['colkeytype'] = colkeytype

    if rowkeytype is not None :
        data.colkeytype = rowkeytype
        data.data_out['colkeytype'] = rowkeytype

    if convert_to_datadict is True :
        data = dmn.core.Damon(data.data_out,'datadict',workformat='RCD_whole')

    if kwargs.has_key("pseudomiss") and kwargs["pseudomiss"] is not None:
        data.pseudomiss(**kwargs["pseudomiss"])
  
    if kwargs.has_key("parse") and kwargs["parse"] is not None:
        data.parse(**kwargs["parse"])
    
    if kwargs.has_key("standardize") and kwargs["standardize"] is not None:
        data.standardize(**kwargs["standardize"])

    if(createbank == True):
        # Needed to test bank specs
        data.coord([[ndim]])    
        data.base_est()
        data.base_resid()
        data.base_ear()
        data.base_se()
        
        if (bankf0 is not None
            or bankf1 is not None
            ):
            try:
                os.remove(bankf1[0])
            except:
                pass

            data.bank(bankf1[0], { 'Remove' : [None], 'Add' : [bankf0] } ,
                      { 'Remove' : [None], 'Add' : bankf1[1] })
        
        # Create new DataObj
        data = dmn.core.Damon(data,'Damon',verbose=None)

        if kwargs.has_key("pseudomiss") and kwargs["pseudomiss"] is not None:
            data.pseudomiss(**kwargs["pseudomiss"])
      
        if kwargs.has_key("parse") and kwargs["parse"] is not None:
            data.parse(**kwargs["parse"])
        
        if kwargs.has_key("standardize") and kwargs["standardize"] is not None:
            data.standardize(**kwargs["standardize"])

    if BankName is not None :
        data.coord([[ndim]])
        data.base_est()
        data.base_resid()
        data.base_ear()
        data.base_se()

        try:
            os.remove(BankName)
        except:
            pass

        data.bank(BankName,{'Remove':[None],'Add':['All']},{'Remove':[None],'Add':['All']})

        np.set_printoptions(precision=2,suppress=True)
        print 'data.data_out=\n',data.data_out['coredata']

        # New DAmonObj
        data = dmn.core.Damon(data.data_out,'datadict',verbose=None)
        kwargs["coord"]= {"ndim":[[ndim]],
                           "anchors":{'Bank':BankName,'Facet':0,
                                    'Coord':'ent_coord',
                                    'Entities':['All'],'Freshen':None}}

    if kwargs.has_key("subscale") and kwargs["subscale"] is not None:
        data.subscale(**kwargs["subscale"])
    if kwargs.has_key("coord") and kwargs["coord"] is not None:
        data.coord(**kwargs["coord"])
    if kwargs.has_key("base_est") and kwargs["base_est"] is not None:
        data.base_est(**kwargs["base_est"])
    if kwargs.has_key("base_resid") and kwargs["base_resid"] is not None:
        data.base_resid(**kwargs["base_resid"])
    if kwargs.has_key("base_ear") and kwargs["base_ear"] is not None:
        data.base_ear(**kwargs["base_ear"])
    if kwargs.has_key("base_se") and kwargs["base_se"] is not None:
        data.base_se(**kwargs["base_se"])
    if kwargs.has_key("base_fit") and kwargs["base_fit"] is not None:
        data.base_fit(**kwargs["base_fit"])    
    if kwargs.has_key("fin_est") and kwargs["fin_est"] is not None:
        data.fin_est(**kwargs["fin_est"])
    if kwargs.has_key("fin_resid") and kwargs["fin_resid"] is not None:
        data.fin_resid(**kwargs["fin_resid"])
    if kwargs.has_key("equate") and kwargs["equate"] is not None:
        data.equate(**kwargs["equate"])   
    if kwargs.has_key("est2logit") and kwargs["est2logit"] is not None:
        data.est2logit(**kwargs["est2logit"])
    if kwargs.has_key("summstat") and kwargs["summstat"] is not None:
        data.summstat(**kwargs["summstat"])  
    
    sample_datadict = None
    if create_sample_datadict is True:
        sample_datadict = dmn.core.create_data(8,4,2,2,nheaders4rows=2,nheaders4cols=2)['data'].data_out    
    
    return {'data' : data, 'model' : model,
            'Args' : all_args, 'anskey' : anskey,
            'anskey_2' : anskey_2, 'sample_datadict' : sample_datadict}
    
def format_method_parameters(data_object, method_parameters):
    '''
    this formats the method parameters to to get values from data object.
    '''
    format_method_parameters=cp(method_parameters)
    
    i = 0
    for mp in format_method_parameters :
        if type(mp) is str and mp[mp.__len__() - 1] == "$" and mp[0] == "$" :
            mp = mp.replace("$","")
            mp = eval("data_object" + mp)

        format_method_parameters[i] = mp        
        i = i + 1

    #format_method_parameters[0]=data_object["data"]
    
    return format_method_parameters
    
def run_method(method, method_parameters):
    '''
    this runs the method corresponding to the parameters list paased.
    '''
    method_output=None
    method_exception= "success"
    try:
        method_output= method(*method_parameters)
    except Exception, inst:
        method_exception = inst
        
    return [method_output , method_exception]
    
def generate_reports(method_output_list , method_exception_list):
    '''
    this generates reports to user.
    '''
    print method_output_list
    print method_exception_list
    
def get_method_parameters(method, args_dict):
    '''
    this iterates over method permutations and returns the parameters list.
    '''
    args_names = inspect.getargspec(method).args
    args_list = [[None] * args_names.__len__()]

    i=0
    for arg_name in args_names :
        
        if(args_dict.has_key(arg_name)):
            j=0

            args_list_prev_len = args_list.__len__()
            prev_args_list=cp(args_list)
            
            for k in range(0,args_dict[arg_name].__len__()-1):
                for l in range(0,prev_args_list.__len__()):
                    args_list.append(cp(prev_args_list[l]))

            for arg_val in args_dict[arg_name] :
                 for m in range(args_list_prev_len * j , args_list_prev_len * (j + 1)):
                    args_list[m][i] = arg_val
                 j=j+1

        i=i+1

    return args_list
