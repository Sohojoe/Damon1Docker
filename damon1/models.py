# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:06:25 2013

@author: harrak
"""

class CodeBlock(object):
    def __init__(self,codelines_list):
        '''
        code block is the model of all methods of CodeGenerator class.
        '''
        self.codelines_list=codelines_list
    