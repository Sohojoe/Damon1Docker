# -*- coding: utf-8 -*-
# opts.py

"""blank.py

"""

import sys
import warnings
warnings.simplefilter('ignore')  # matplotlib produces annoying warnings


# If damon1 is not stored in your Python's site-packages folder, specify path
damon1_path = '/Users/markhmoulton/Google Drive/my_python'
if damon1_path:
    sys.path.append(damon1_path)

import damon1 as damon1
import damon1.core as dmn
import damon1.tools as dmnt

dmnt.test_damon(tests=['All'], check='run', printout=False)









