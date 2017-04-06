# -*- coding: utf-8 -*-
"""
this module creates python files , it has the functions to add methods ,
add loops, ad statements to a python file.
"""

import damon1 as dmn
import inspect

class CodeGenerator:
    '''
    this class contains method to generate python files and to add code to 
    python file.
    '''
    def __init__(self, file_full_name):
        '''
        this class contains method to generate python files and to add code to         
        python file.
        '''
        self.file = open(file_full_name, 'w+')
        self.indentcharacter = "    "
        self.newlinecharacter = "\n"
        #super(UnitTestsGenerator, self).__init__()
    
    def create_class(self, class_name, codeblocks_list):
        '''
        this creates a class code block
        '''
        codelines_list = []
        codelines_list.append(self.newlinecharacter)
        codelines_list.append("class " + class_name +":")

        for codeblock in codeblocks_list:
            codelines_list = codelines_list + list(self.indentcharacter + c 
                                                   for c in 
                                                   codeblock.codelines_list)
 
        return dmn.models.CodeBlock(codelines_list)
            
    def cc_(self, class_name, codeblocks_list):
        '''
        this is just a short name for create_class method. It is exactly same 
        to create_class method
        '''
        return self.create_class(class_name, codeblocks_list)
    
    def create_function(self, function_name, function_argumentnames_list, 
                        codeblocks_list):
        '''
        this creates a function code block
        '''
        codelines_list = []
        
        function_argumentnames_string = ' , '.join(function_argumentnames_list)
        
        codelines_list.append(self.newlinecharacter)
        codelines_list.append("def " + function_name + "(" + \
                               function_argumentnames_string + "):")
        
        if(codeblocks_list.__len__() == 0):
            codeblocks_list.append(self.create_statement("pass"))
        
        for codeblock in codeblocks_list:
            codelines_list = codelines_list + list(self.indentcharacter + c \
                                                   for c in 
                                                   codeblock.codelines_list)
            
        return dmn.models.CodeBlock(codelines_list)
    
    def cf_(self, function_name, function_argumentnames_list, codeblocks_list):
        '''
        this is just a short name for create_function method. It is exactly 
        same to create_function method
        '''
        return self.create_function(function_name, function_argumentnames_list,
                                    codeblocks_list)    
    
    def create_ifelse(self, if_or_else, condition_string, codeblocks_list):
        '''
        this creates a if or else code block
        '''
        pass

    def ci_(self, if_or_else, condition_string, codeblocks_list):
        '''
        this is just a short name for create_ifelse method. It is exactly same 
        to create_ifelse method
        '''
        return self.create_ifelse(if_or_else, condition_string, 
                                    codeblocks_list)  

    def create_comment(self, comment):
        '''
        this creates a comment code block
        '''
        pass

    def create_newline(self):
        '''
        this creates a new line statement
        '''
        return self.create_statement(self.newlinecharacter)

    def cco_(self, comment):
        '''
        this is just a short name for create_comment method. It is exactly same
        to create_comment method
        '''
        return self.create_comment(comment)  
    
    def create_statement(self, statement):
        '''
        this creates a statement code block
        '''
        return dmn.models.CodeBlock([statement])

    def cs_(self, statement):
        '''
        this is just a short name for create_statement method. It is exactly 
        same to create_statement method
        '''
        return self.create_statement(statement)
        
    def get_object_sourcecode(self, obj):
        '''
        this gets the cource code of function or class or method.
        '''   
        return self.create_statement(inspect.getsource(obj))
        
    def convert_object_to_string(self, obj):

        return str(obj)
        
        '''
        this converts a list or dictionary to string        
        '''
        if(isinstance(obj, list) == False and isinstance(obj, dict) == False):
            return str(obj)

        #set start and end elements of object        
        start_ele = ""
        end_ele = ""      
        if(isinstance(obj, list)):
            start_ele = "["
            end_ele = "]"
        elif(isinstance(obj, dict)):
            start_ele = "{"
            end_ele = "}"
        
        if(isinstance(obj, list)):
            return start_ele + ' , '.join(self.convert_object_to_string(e) \
                                        for e in obj) + end_ele
            
        if(isinstance(obj, list)):
            return start_ele + ','.join("\"" + self.convert_object_to_string(e)
                                        + "\" : " + "\"" + 
                                        self.convert_object_to_string(obj[e]) \
                                        + "\"" for e in obj.iterkeys()) \
                                        + end_ele
    
    def check_object_type(self, obj):
        '''
        this returns type of object
        '''
        if(isinstance(obj, list)):
            return type(list)
        
        if(isinstance(obj, dict)):
            return type(dict)
            
    def write_to_file(self, codeblocks_list):
        '''
        this writes the code blocks to file
        '''
        total_code_string = ""

        for codeblock in codeblocks_list:

            if(total_code_string.__len__>0):
                total_code_string = total_code_string + self.newlinecharacter
                
            total_code_string = total_code_string +\
                self.newlinecharacter.join(codeblock.codelines_list)
 
        self.file.write(total_code_string)   

    def wf_(self, codeblocks_list):
        '''
        this is just a short name for write_to_file method. It is exactly same 
        to write_to_file method
        '''
        return self.write_to_file(codeblocks_list) 
    