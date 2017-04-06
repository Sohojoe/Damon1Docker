# -*- coding: utf-8 -*-
"""tester.py
Created on Fri Jul 22 16:23:52 2016

@author: markhmoulton

`tester.py` is intended to reduce unit testing to a single
easy-to-use function, usable across packages, that can be run
on the fly to check a function quickly or executed as part of
a series of unit tests to validate a package.  tester runs
all scenarios (combinations of function parameters), compares
them to expected outputs, and is useful for debugging.  It is
intended to make test-driven programming fast and practical.

"""
#1234567890123456789012345678901234567890123456789012345678901234567890123456789

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

### future module not necessary here
##from future.builtins import (
##                bytes, dict, int, list, object, range, str,
##                ascii, chr, hex, input, next, oct, open,
##                pow, round, super,
##                filter, map, zip)

import sys
import os
import itertools as it
import timeit


try: import cPickle as pickle
except: import pickle

import numpy as np
import damon1

  
class Setup(object):
    """Setup objects hold user-defined functions for building parameters
    on the fly.
    """

    def __init__(self, name, func, args):
        self.name = name
        self.func = func
        self.args = args
    
    def __repr__(self):
        return ('Setup({0}, func={1}, '
               'args={2})'.format(self.name, str(self.func), str(self.args)))
    
    def __str__(self):
        return ('Setup({0}, func={1}, '
               'args={2})'.format(self.name, str(self.func), str(self.args)))
        

def test(func, args, check='run', asserts=None, suffix=None, printout=True):
    r"""Unit test a function or method.
    
    test() is intended to reduce unit testing to a single function
    that can be run with minimal hassle and setup.  It is not
    restricted to Damon unit tests, so is an all-purpose utility.
    In its 'run' mode, test() checks whether the each scenario can
    be run without triggering an exception.  In its 'outputs' mode,
    it supports inspection and validation of outputs, with persistence
    for future comparisons.  It also supports a 'debug' mode.
    
    Parameters
    ----------
    func : function
        The function to be tested. It is assumed that func is a
        function and that it outputs some value.  As this is
        frequently not the case, e.g., class methods that alter
        the state of an object as a side effect, or functions that
        do nothing but export files, you will sometimes need to
        wrap your function or method inside another and test that.
        
    args : dict of parameter lists
        A dictionary of arguments giving a list of possible parameters
        for each argument.  Each arg should work in combination with others
        to produce a result without exceptions. Parameters that are complex
        objects with possible side-effects can be listed as Setup
        objects, which pass a function for building the desired
        parameter.

            `{'arg0':[1, 2], 'arg1':[True, False, 'arg2':[Setup obj]}`

            Test func with parameters 1, 2 for 'arg0', True and
            False for 'arg1'.  Pass a Setup() object for 'arg2'.
                
    check : {'run', 'debug', 'outputs'} (optional)
        This is the type of test check to perform.
        
        'run'
            Check each scenario. Return '.' if the test passes, 'E'
            if it fails.  If test() can find the pickle file of
            expected outputs, "success" means that the scenario ran
            and produced outputs that match those stored in the pickle
            file. Otherwise, it means only that the scenario ran without
            an exception.
        
        'debug'
            Check scenarios until an exception occurs, print out the
            traceback.
        
        'outputs'
            For each scenario, export the output of the function in
            'my_func.txt' for visual inspection. Also, store the
            {scenario:output} dict in  a function-specific pickle file.
            Every time test() is run with `check` = 'outputs', the
            relevant text and pickle files are overwritten.
            
    asserts : {None, function} (optional)
        `asserts` specifies an assert-like function by which to 
        check, assert the equivalence between, the actual and expected 
        outputs of the function under test. When passing
        an assert-like function, the only requirement is that it
        return True (the new output and the reference output are equal)
        or False (they are not equal).  If the asserts function throws
        an exception, False is returned.  If asserts is None, the
        outputs are not checked against expected outputs, even if
        the pickled outputs file exists.

        `tester.py` includes two assert-like functions:
        
            `tester.obj_equal` wraps Python's `assert` expression
            `tester.damons_equal` compares two Damon objects

        Also consider:

            `numpy.array_equal` to compare arrays
            `pandas.DataFrame.equals` to compare Pandas dataframes

    suffix : {None, str} (optional)
        If specified, suffix adds a string to the output pickle and
        text files so that you can differentiate tests applied to
        the same function.

    printout : bool (optional)
        `printout` = True causes '.' and 'E' to be printed to the
        console for each unit test.  As this can be time-consuming,
        False results in a more accurate time report.  When `check`
        equals 'outputs', `printout` causes the outputs to be
        printed to the console screen.  when `check` equals 'debug',
        `printout` lists the scenarios that work until an exception
        is encountered.
        
    Returns
    -------
    out : {dict, None}
        `check` = 'run' or 'ouptuts' returns {'time', 'n_tests',
        'exceptions', 'n_exceptions'} with console printouts.  If
        `check` = 'outputs', function-specific text and pickle files
        are created; the text file is for your visual inspection.
        For `check` = 'debug', there is only a console printout with
        traceback.

    See Also
    --------
    The '...damon1/tests.py' module provides a large collection of
    unit tests written specifically for Damon.  It shows pretty much
    all the various unit test coding patterns, though they are
    probably more complicated than you will need.
        
    Notes
    -----
    The first time test() is run, it does not check whether the output
    of each scenario is valid, just whether the scenario runs without
    throwing an exception.  As with other unit testing frameworks, '.'
    means success and 'E' means an exception (which can be suppressed
    for speed by setting `printout` = False).
    
    When `check` = 'outputs', test() writes a file with all outputs for
    visual inspection. It also exports a function-specific pickle file of 
    {scenarios:outputs}.  To inspect the outputs, either look at the
    console printout (if you specified `printout` = True) or look up
    the outputs file at the indicated location.  Look through the outputs
    for each scenario and make sure they are what you expect.  If you
    approve the outputs, run test() again, specifying `check` = 'run'.
    This will automatically add assertion checks for the outputs of each
    scenario, evaluated with the assert-like function you specify or define.
    Each assertion compares the current scenario output with the output
    you found to be "correct" and stored in the function pickle file.  If 
    you do not approve the outputs, fix the program and keep running
    `check` = 'outputs' until they look right.
    
    test() internally manages the creation of a 'test_asserts' directory 
    that resides inside the same directory that contains tester.py.  Inside 
    'test_asserts' is a 'temp' directory to store file inputs and outputs 
    that are part of the function being tested.  'temp' is emptied before
    each new scenario is run.  When testing functions that create files,
    or where the setup function creates files, send those files to the
    '.../damon1/test_asserts/temp/' directory.
    
    When `check` = 'debug', test() will stop and return a traceback the
    first time it encounters an exception.  Fix the program and run it
    again until it passes without exception.
    
    test() prioritizes testing all or a large sample of combinations of 
    parameters.  Functions whose arguments are not orthogonal, where the
    parameter assigned to one argument requires that another parameter be
    set a certain way, will trigger exceptions. To get around this, either
    rewrite the function so the arguments are orthorgonal (which is good
    practice if you can manage it) or call test() multiple times, each
    time with a restricted set of parameters that are mutually compatible.
    If calling test() multiple times for the same function, specify
    different filename suffixes to avoid overwriting the output text
    and pickle files.
    
    Most argument parameters have pretty simple types and don't require
    special treatment in the `args` argument, but some are more
    complex and may have side-effects.  For these, you have the option
    (but are not required) to specify a Setup() object, in which you
    specify a function for building the parameter in question dynamically.
    For example, Damon uses Setup() to build Damon objects on the fly
    to test a particular method.  Setup() avoids issues raised by
    side-effects or changes in state that may persist across scenarios.

    This bit about the function you feed into Setup() is quite important.
    After all, many functions require data, and data requires container
    objects of some kind, and you will want to be able to build these
    container objects dynamically for testing.  So it's worth taking
    some time to write a function for Setup() that builds the data
    containers you work with, something you can customize and tweak
    without a lot of effort. Damon's `tests` module has an all-purpose
    function specifically for testing Damon methods called `setup_damon()`
    that creates artificial datasets and outputs them as Damon objects
    or text files.

    As mentioned under Arguments, you may want to test a class method or
    function that does not return a value, or returns only None.  For instance,
    most Damon methods return None but attach outputs to the Damon object.
    Python `list` methods just change the state of the list. Or a function's 
    purpose is just to output files, or change something.  In these cases,
    wrap the function or method inside another function and have the wrapper
    function return some value that reflects what the function did. It might
    return the modified object for instance, or the name of an output file
    and some of its values. You specify `func = the_wrapper_function`.  This
    makes it possible to determine whether your function is doing what
    it's supposed to and gives you power to control the outputs.

    The `asserts` parameter is helpful to make sure your outputs are valid,
    but it is quite possible that your function outputs will include objects,
    or objects wrapped inside of other objects, for which Python's `assert`
    expression and the numpy and damon assert equal methods won't work.  Don't
    let that be a problem.  Just define an `asserts` function that works with
    your outputs.  All it requires is two arguments, one for the current
    output, one for the expected output, and it must return True or False
    where True means the current and expected outputs are equivalent.
    Then pass the function to test() via the `asserts` argument.  Here is an
    example where the output is a dict that could either be an array or
    a Damon object:

    # "obj" is the current object. "ref" or reference is the expected object.
    
        def dfunc_assert(obj, ref):
            eq = True
            for i in range(len(obj)):
                try:
                    eq = ut.damon_equal(obj[i], ref[i])
                except:
                    eq = ut.obj_equal(obj[i], ref[i])
                if not eq:
                    break
            return eq

    Examples
    --------
    >>> import damon1.tester as ut
    >>> def g(a, b=1, c=2):
            return (a + b) / c
    >>> ut.test(func=g,
                args=[('a', [0, 1, 2]),
                      ('b', [10, 11, 12]),
                      ('c', [0, 101, 102])],
                 check='run',
                 asserts=None)
    E..E..E..E..E..E..E..E..E..
    Performed 27 tests in 0.000221 seconds. Got 9 exceptions.

    Paste Function
    --------------
    test(func, # [<function object>]
         args, # [<{'a':[1, 2], 'b':[3, 4]}> => dict of parameter:[possible values]]
         check='run', # [<'run', 'outputs', 'debug'> => test mode]
         asserts=None,  # [<None, func> => assertion function, e.g. tester.obj_equal]
         suffix=None,  # [<None, str> => suffix to add to output pickle/text files]
         printout=True  # [<True, False> => printout mode]
         )
    """
    
    # Set up test_asserts and temp directories
    ta_path = sys.modules[__name__].__file__
    ix = ta_path.find('tester.py')
    ta_path = ta_path[:ix] + 'test_asserts/'
    ta_path_temp = ta_path + '/temp/'

    if not os.path.exists(ta_path):
        os.makedirs(ta_path)

    if not os.path.exists(ta_path_temp):
        os.makedirs(ta_path_temp)
    tear_down(ta_path_temp)

    # Load {scen:outputs} pickle file if possible
    assertions = False
    if not suffix:
        suffix = ''
    else:
        suffix = '_' + suffix
    pick_name = ta_path + func.__name__ + suffix + '.pkl'
    fname = ta_path + func.__name__ + suffix + '.txt'
    try:
        ref = from_pickle(pick_name)
        assertions = True
    except IOError:
        pass
        
    # Get all permutations of parameters (scenarios)
    tup_args = args.items()
    keys = [t[0] for t in tup_args]
    params = [t[1] for t in tup_args]
    scens = it.product(*params)
    
    exc = {}
    out = {}
    scen_outputs = {}
    i = 0

    start = timeit.default_timer()
    for i, scen_ in enumerate(scens):
        s = setup(keys, scen_)
        scen, scen_id = s['scen'], s['scen_id']

        if check == 'run':
            try:
                output = func(**scen)
                if assertions and asserts is not None:
                    eq = True
                    try:
                        eq = asserts(output, ref[scen_id])
                    except:
                        eq = False
                    if not eq:
                        raise AssertionError(exc)
                else:
                    assertions = False
                if printout:
                    print('.', end='')
            except BaseException, e:
                if printout:
                    print('E', end='')
                exc[i] = e

        elif check == 'outputs':               
            try:
                output = func(**scen)
                scen_outputs[scen_id] = output
            except BaseException, e:
                scen_outputs[scen_id] = 'FAILED'
                exc[i] = e

        elif check == 'debug':
            if printout:
                print('Scenario', scen_id)
            output = func(**scen)
            if assertions and asserts is not None:
                eq = True
                try:
                    eq = asserts(output, ref[scen_id])
                except:
                    eq = False
                if not eq:
                    if not printout:
                        print('Scenario:\n', scen_id)
                    exc = ('Scenario:\n{0}\noutput:\n{1}\n'
                           'expected:\n{2}'.format(scen_id, output,
                                                   ref[scen_id]))
                    raise AssertionError(exc)
            if printout:
                print(': OK')
        else:
            exc = "`check` parameter should be 'run', 'outputs', or 'debug'"
            raise ValueError(exc)
        
        i += 1
        elapsed = timeit.default_timer()

    # Pickle outputs
    if check == 'outputs':
        to_pickle(scen_outputs, pick_name)
        to_file(scen_outputs, fname)

    out['time'] = round(elapsed - start, ndigits=3)
    out['n_tests'] = i
    out['exceptions'] = exc
    out['n_exceptions'] = len(exc.keys())
    
    print('\nRan {0} tests in {1}s. Got {2} '
          'exceptions.'.format(out['n_tests'], out['time'], 
                              out['n_exceptions']))
    if assertions:
        if check == 'run':
            print('Tests included assertions.')
    else:
        if check == 'run':
            print('Tests did not include assertions.')

    if check == 'outputs':
        if printout:
            print(scen_outputs)
        print('Function outputs exported to "{}" for inspection.'.format(fname))
        
    if out['n_exceptions'] == 0:
        print('OK\n\n')
    
    return out


def setup(keys, scen_):
    """Build parameters and an id from Setup objects for a
    scenario as needed.
    """
    scen = []
    scen_id = []
    for s in scen_:
        if isinstance(s, Setup):
            sid = s.name
            if isinstance(s.args, dict):
                obj = s.func(**s.args)
            else:
                obj = s.func(*s.args)
        else:
            sid = str(s)
            obj = s
            
        scen.append(obj)
        scen_id.append(sid)

    # Scenario id is a tuple
    scen_dict = dict(zip(keys, scen))
    scen_id = str(dict(zip(keys, scen_id)))

    return {'scen':scen_dict, 'scen_id':scen_id}
                             
def tear_down(path):
    "Delete files in test_asserts/temp directory"
    files = os.listdir(path)
    for f in files:
        os.remove(path + f)

def to_file(scens, filename):
    "Save scen:output information to a text file."
    with open(filename, 'w') as f:
        for sid in scens:
            f.write('\n\n' + str(sid) + '\n')
            f.write(str(scens[sid]))
        
def to_pickle(scens, filename):
    "Save a {scen:output} dict to a pickle file."

    with open(filename, 'wb') as f:
        pickle.dump(scens, f, protocol=2)

def from_pickle(filename):
    "Load a {scen:output} dict from a pickle file."

    with open(filename, 'rb') as f:
        return pickle.load(f)

def obj_equal(obs, exp):
    "Test whether two objects are equal using Python's assert."
    eq = True
    try:
        assert(obs == exp)
    except AssertionError:
        eq = False
    return eq

def allclose(obs, exp, tol=0.1):
    "Checks if any absolute difference exceeds the tolerance."
    eq = True
    diff = np.abs(obs - exp)
    if np.amax(diff) > tol:
        eq = False
    return eq
     
def damon_equal(obs, exp):
    "Test whether two Damon objects or datadicts (just the arrays) are equal."
    eq = True
    d1, d2 = obs, exp
    
    # Check that keys are equal
    if (isinstance(d1, damon1.core.Damon) and
        isinstance(d2, damon1.core.Damon)
        ):
        d1_keys = sorted(vars(d1).keys())
        d2_keys = sorted(vars(d2).keys())
    elif (isinstance(d1, dict) and
          isinstance(d2, dict)
          ):
        d1_keys = sorted(d1.keys())
        d2_keys = sorted(d2.keys())
    else:
        exc = 'Unable to figure out the input types.'
        raise ValueError(exc)
        
    try:
        assert(d1_keys == d2_keys)
    except AssertionError:
        eq = False

    # Check that values and arrays are equal
    if eq is True:
        for k in d1_keys:
            if isinstance(d1, damon1.core.Damon):
                a1, a2 = vars(d1)[k], vars(d2)[k]
            else:
                a1, a2 = d1[k], d2[k]
            if isinstance(a1, np.ndarray):
                if not np.array_equal(a1, a2):
                    eq = False
                    break
    return eq
            






    
