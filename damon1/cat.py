"""cat.py
Created on Sun Aug 28, 2016

@author: Mark Moulton, for Educational Data Systems

cat.py implements computer adaptive testing for the Rasch model
and Damon's multidimensional model.

"""
#234567890123456789012345678901234567890123456789012345678901234567890123456789
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (
                bytes, dict, int, list, object, range, str,
                ascii, chr, hex, input, next, oct, open,
                pow, round, super,
                filter, map, zip)
import sys
import numpy as np
import pandas as pd
import damon1
import damon1.core as core




class Session(object):
    """Keep track of student progress through a test."""

    def __init__(self, bank, cat_funcs, stop, validchars, start=-0.5,
                 targ_p=0.65, engine=(core.Damon.rasch, {})):
        self.bank = bank
        self.cat_funcs = cat_funcs
        self.stop = stop
        self.validchars = validchars
        self.start = start
        self.targ_p = targ_p
        self.engine = engine
        self.persons = {}

    def get_next_item(self, resps):
        """Get next item(s) to deliver to student."""

        resps_ = pd.read_json(resps)
        stud = resps_.columns.values[0]

              

        if stud in self.persons:
            df = self.persons[stud]
            resps_['iter'] = df.iloc[-1, 1] + 1
            
            df = df.append(resps_)
        else:
            df = resps_
            df['iter'] = 0
            self.persons[stud] = df

        data = df.loc[:, stud].to_frame(stud).transpose()
        d = core.Damon(data, 'dataframe', 'RCD_dicts_whole',
                       validchars=self.validchars,
                       verbose=None)

        # Score multiple choice if available
        try:
            d.score_mc(anskey=self.bank)
        except damon1.utils.score_mc_Error:
            pass

        # Standardize if available
        try:
            d.standardize(std_params=self.bank)
        except damon1.utils.standardize_Error:
            pass

        print('\ndf=\n', df)
        print('\nd=\n', d)



        eng = self.engine
        getattr(d, eng[0].__name__)(**eng[1])


        print('d.rasch_out=\n', d.rasch_out)
        sys.exit()


        
