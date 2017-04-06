# -*- coding: utf-8 -*-
# opts.py
"""
Summary
-------
core defines a class for creating Damon objects.

"""
import sys
import numpy as np

# Import Damon utilities
import damon1 as dmn







################################################################################

class Damon(object):
    """Load, parse, format data, obtain Damon estimates and standard errors.

    Returns
    -------
        Returns an instance of the Damon class, or "Damon object',
        which is a package of observed data and other inputs to which
        Damon methods can be applied to produce cell estimates,
        standard errors, and other statistics.

        Most generally, Damon computes an array of estimates
        corresponding to an array of observations.  These estimates
        compute the "most likely" value for each cell, including
        those for which observations are randomly or nonrandomly
        missing.

        Damon is used in psychometrics and data mining and has
        significant potential in many fields as a tool for calculating
        sample-independent or "objective" measures and predictions.

    Comments
    --------
        Both Damon and TopDamon can be used to create Damon objects.
        Damon is more low-level and generic (TopDamon depends on it).
        TopDamon is optimized for dealing with real-world, messy
        datasets.  For information about TopDamon, type:

            >>>  input damon1.core as dmn
            >>>  help(dmn.TopDamon)

        To create or "initialize" a Damon object, type:

            >>>  input damon1.core as dmn
            >>>  my_object = dmn.Damon(...)

        where (...) is a set of input parameters and my_object
        is a new Damon object.  (Incidentally, the selection of
        "dmn" to represent damon1.core is purely arbitrary.  You
        can use your own abbreviation.)

        To get a list of Damon's input parameters, or "arguments',
        consult the documentation for the Damon initialization
        function, called Damon.__init__:

            >>>  input damon1.core as dmn       (if not already done)
            >>>  help(dmn.Damon.__init__)

        The __init__ method "initializes" and formats the input
        variables that are specified in dmn.Damon(...).  These are
        made automatically available to all the Damon methods.

        The most important of these parameters is called "data";
        it is the name of the data file or array to be analyzed.
        Damon() automatically loads this data and makes it part of
        the Damon object.  There is no data loading utility other than
        Damon().

        To apply methods to my_damon_obj, type the name of the object,
        a dot, and the name of the method along with any special parameters
        the method requires:

            >>>  my_damon_obj.standardize(...)  Standardizes the columns
                                                to have a common metric

            >>>  my_damon_obj.coord(...)        Calculates row entity and
                                                column entity coordinates

            >>>  my_damon_obj.base_est(...)     Multiplies the coordinates
                                                to get cell estimates


        For a list of available Damon methods, type:

            >>>  dir(dmn.Damon)

        Messy data
        ----------
        Damon() requires data to be arranged with rowlabels
        to the left, collabels to the top, and coredata for
        the rest of the array.  No duplicate row or column IDs
        are allowed.  When, as often happens, data do not
        fit this format, use the core.TopDamon() function.
        TopDamon() converts messy datasets into preliminary
        Damon objects that can be formatted further using Damon().

        Getting outputs
        ---------------
        To get the outputs of each method, say for base_est(), you
        may be tempted to type something like:

            >>>  MyOutput = my_damon_obj.base_est(...)

        This won't work in Damon.  Instead type:

            >>>  my_damon_obj.base_est(...) =>  does the calculation and
                                                assigns attribute called
                                                my_damon_obj.base_est_out

            >>>  my_ests = my_damon_obj.base_est_out

        The reason for this convention is that each output is a
        so called "datadict" (a python dictionary describing the data)
        and returning the whole datadict every time a method is
        invoked would clog up the screen.  So the method actually
        returns "None" and assigns the datadict to the Damon object
        where it can be accessed using the my_damon_obj.method_out
        syntax (the '_out' suffix is added to the method name).
        This convention holds for most of the Damon methods (for
        some, such as transpose() and extract(), it doesn't make
        sense).

        What is nice is that every time you apply a method, its
        outputs are automatically added to your Damon object for
        other methods to refer to, so a lot of the logistics of
        the analysis are handled behind the scenes.  Each method's
        output can be accessed as what is called (in Damon) a
        "datadict" -- a data dictionary.  For instance, to get the
        core array of estimates from the commands above, you would
        type:

            >>>  my_damon_obj.base_est_out['coredata']

        To get just the column labels, you would type:

            >>>  my_damon_obj.base_est_out['collabels']

        To get the complete base_est_out data dictionary, type:

            >>>  my_damon_obj.base_est_out

        Or get just the dictionary keys:

            >>>  my_damon_obj.base_est_out.keys()

        You will either be running Damon from the Python shell
        (such as IDLE) or from your own Python script, which is
        often easiest.  To create a DamonObj script, open the Damon
        template by going to the shell menu, under file/Open_Module,
        and type Damon.template.  This will bring up a pre-formatted
        script with the necessary import statements already in
        place.  Save the template under a different name in a
        directory of your choice, and make that the basis of
        your Damon application.

    Methods
    -------
        For a complete list of Damon methods, type

            >>>  dir(damon1.core.Damon)

        Each method has a set of "parameters" or "arguments" which
        are described in its accompanying documentation:

            >>>  help(dmn.Damon.my_method)

        They are summarized at the end under the "Paste Method"
        section (which is handy for pasting into your Python script).
        The options available for each argument are indicated inside
        angular brackets:  < option1, option2 > .  Think of them
        as menu options.


    """

    #############################################################################

    def __init__(self,
                 data,    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                 format_,    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                 workformat = 'RCD_dicts',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                 validchars = None,   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                 nheaders4rows = 0,  # [number of columns to hold row labels]
                 key4rows = 0,   # [<None, nth column from left which holds row keys>]
                 rowkeytype = 'S60',     # [<None, type of row keys>]
                 nheaders4cols = 0,  # [number of rows to hold column labels]
                 key4cols = 0, # [<None, nth row from top which holds column keys>]
                 colkeytype = 'S60',     # [<None, type of column keys>]
                 check_dups = 'warn',   # [<None,'warn','stop'> => response to duplicate row/col keys]
                 dtype = [object, 3, ''], # [[type of 'whole' matrix, <None, int decimals>, <None, display nanval], e.g. ['S60', 8, ' '],[object, None, None] ]
                 nanval = -999,    # [Value to which non-numeric/invalid characters should be converted.]
                 missingchars = None,  # [<None, [list of elements to make missing]>]
                 miss4headers = None, # [<None, [[list of elements to make missing in headers]>]
                 recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
                 cols2left = None,    # [<None, [ordered list of col keys, to shift to left and use as rowlabels]>]
                 selectrange = None,   # [<None,[slice(StartRow,EndRow),slice(StartCol,EndCol)]>]
                 delimiter = ',',  # [<None, character to delimit input file columns (e.g. ',' for .csv and '\t' for .txt tab-delimited files)]
                 pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                 verbose = True,    # [<None, True> => report method calls]
                 ):

        """Initialize a Damon object.  Load, parse, clean, index row/column data.

        Returns
        -------
            The method returns None but assigns attributes to the initialized
            Damon object.  It also assign a "data_out" attribute to offer
            an alternative way to access Damon attributes.  Thus, to access
            Damon's rowlabels attribute, type:

                >>>  my_damon_obj.data_out['rowlabels'] or
                >>>  my_damon_obj.rowlabels

            Workflow
            --------

                my_damon_obj = TopDamon(...)    # useful alternative to Damon()
            or
                my_damon_obj = Damon(...)       # most general

            Then run Damon methods, accessed using dots (no equal signs):

                my_damon_obj.standardize(...)
                my_damon_obj.coord(...)
                my_damon_obj.base_est(...)

            Each method automatically finds the outputs of previously run methods.

            Outputs are obtained by accessing the method's output data dictionary
            (called a datadict), which has been assigned to the Damon object
            as an attribute and can be accessed using dot notation:

                std_results = my_damon_obj.standardize_out
                coord_results = my_damon_obj.coord_out
                estimates = my_damon_obj.base_est_out

        Comments
        --------
            The Damon() __init__ method prepares tabular data for analysis
            by Damon methods by parsing out row labels and column labels
            from the core data to be analyzed and returning python dictionaries
            for accessing these subtables and their entities.  It also can
            recombine the labels and data to form a single string array for
            printing.  This function is Damon's workhorse and most functions
            rely on Damon() formatted inputs.

            __init__() only loads or reads existing data.  To create new
            data, use the create_data() function in the core module.

            __init__() returns attributes (variables) that travel with every
            Damon object, including arrays called "rowlabels", "collabels",
            and "coredata".  It assumes that incoming data is tabular and
            fits the following scheme:

                            _________________________
                            |rl/cl|     collabels    |
                            |_____|__________________|
                            |     |                  |
                            |row  |      core        |
                            |lab- |      data        |
                            |els  |                  |
                            |     |                  |
                            |_____|__________________|

            It assumes that every row and column has a unique row
            and column key, or ID, and that they reside somewhere in the
            rowlabels and collabels sections of the array.  Damon does
            not permit duplicate keys, and unless you specify differently
            it will rename any duplicates it finds.

            If your dataset does not have any labels at all, Damon will
            automatically assign integer row and column labels.

            Dealing with messy data formats
            -------------------------------
            If your data is not in this format, e.g, your labels are
            intermixed with the core data or you have duplicate row
            or column identifiers, use TopDamon() to build the Damon
            object.  TopDamon() is a little more complicated, but it
            can Damonize just about any tabular data array.

            For documentation, type:

                >>>  import damon1.core as dmn      (if not done already)
                >>>  help(dmn.TopDamon)

            "datadict" vs "Damon object"
            ---------------------------
            In the documentation, you will see reference to the "datadict"
            format or data dictionaries.  These are Python dictionaries
            that contain keys called 'rowlabels,'collabels','coredata',
            etc., more or less equivalent to the attributes of a Damon
            object.  All of the Damon() methods output datadicts, which
            are simpler and have less overhead than full Damon objects.

            It is easy to convert a datadict into a DamonObj by
            loading it into Damon.  The simplest way is:

                >>>   my_obj = Damon(data = mydatadict,
                                     format_='datadict'
                                     )

            There are alternative ways to Damonize datadicts describe
            below, as when you want to modify the parameters for example.

            To build your own datadict, just make sure it contains
            at least the following keys:
                {'rowlabels',
                'collabels',
                'coredata',
                'key4rows',
                'rowkeytype',
                'key4cols',
                'colkeytype',
                'nanval',
                'validchars',
                }

            Dot vs Bracket Notation
            -----------------------
            In Damon, attributes are accessed in one (or both) of two
            ways -- as object attributes using dot "." notation or as
            Python dictionary keys using bracket "[]" notation.

            For instance, the coredata array can be accessed either as:

                my_damon_obj.coredata, or

                my_damon_obj.data_out['coredata']

            In this case, the "data_out" attribute stores all the Damon
            object attributes as a datadict.

            Whenever a Damon method is run, the outputs are assigned
            to the Damon object as a datadict with a '_out' extension.
            The datadict itself is accessed using dot notation.

                my_obj.standardize(...)
                standardized_values = my_obj.standardize_out['coredata']


            Note Regarding Speed
            --------------------
            To maximize speed, set as many options as possible to None, especially
            those like missingchars which check each cell individually.  Use the
            'SkipCheck' option in validchars.  Avoid using workformat = 'whole',
            reserving it for final printouts.  Using 'RCD' is fastest, and
            'RCD_dicts' is fast, too.  If data refers to a data dictionary,
            use format_ = 'datadict_link' instead of 'datadict_whole', which
            takes the extra step of merging labels and core data into a single
            array, which is computationally expensive.


            Damon Object Attributes
            -----------------------
            Every time Damon(...) is run, the following attributes are
            created and assigned to the new Damon object, accessible via
            dot notation (e.g., my_obj.data_out).  More attributes are
            added as Damon methods are run.

            data_out            =>  This might just as well be called
                                    Damon.__init__out, as it is a
                                    dictionary of all the attributes
                                    created for Damon() and described
                                    below.  To access attributes using
                                    Python's dictionary format, use

                                        my_obj.data_out['coredata'],

                                    which is the same as

                                        my_obj.coredata

            rowlabels           =>  An nRows x nFields array of row labels
                                    that describes every row in the original
                                    array.

                                    The array type is assigned automatically
                                    by Numpy as the least general type needed
                                    to describe all the elements in the array.

            collabels           =>  An nFields x nColumns array of column
                                    labels that describes every column in
                                    the original array.  Note that rowlabels
                                    and collabels overlap in the top-left
                                    corner of the original array.  These corner
                                    values show up in both arrays.

            coredata            =>  This is the array of values that will
                                    be parsed, standardized, and otherwise
                                    analyzed using Damon methods.  The validchars
                                    parameter is used to filter out invalid
                                    values and force the array to be numerical,
                                    if appropriate.

            whole               =>  This is the complete cleaned and formatted array,
                                    including rowlabels, collabels, and coredata.

                                    Because Numpy arrays can only be of one type
                                    the whole array is cast to a single type, generally
                                    string ('S60') or object (the most general)
                                    using the dtype parameter.

            nheaders4rows       =>  The number of columns in rowlabels, the width
                                    of the rowlabels array.

            key4rows            =>  The column in rowlabels that contains unique
                                    keys for every row entity. Count from the left
                                    starting with 0.

            rowkeytype          =>  The type to which the row keys should be cast
                                    (to the maximum extent possible) when they are
                                    pulled from rowlabels.  The best way to get
                                    these keys is using:

                                        my_keys = tools.getkeys(...)

                                    For documentation, see:

                                        >>>  import damon1.tools as dmnt
                                        >>>  help(dmnt.getkeys)

            nheaders4cols       =>  The number of rows in collabels, the vertical width
                                    of the collabels array.

            key4cols            =>  The row in collabels that contains unique keys for
                                    every column entity.  Count from the top starting
                                    with 0.

            colkeytype          =>  The type to which the column keys should be cast
                                    (to the maximum extent possible) when they are
                                    pulled from collabels.

            nanval              =>  The user-defined "Not-a-Number" value (a designated
                                    float or integer like -999) that will be assigned to all
                                    values in the coredata array that are not a float or
                                    integer, or are missing, or for some other reason should
                                    not be included in the analysis.  nanval can be string
                                    if coredata is string, but it must be castable
                                    to a float or integer (e.g., '-999').

            validchars          =>  A list of valid characters or numerical ranges for
                                    each column entity or for the whole array.

            The following are key-accessible dictionaries that may or may not be
            created, depending on how you set the "workformat" parameter.

            rl_row              =>  A rowlabels Python {key:Value,...} dictionary where the
                                    keys are the unique row IDs and the values are the
                                    row arrays for each row in rowlabels.

            rl_col              =>  A rowlabels Python {key:Value,...} dictionary where the
                                    keys are unique column IDs and the values are the
                                    column arrays of each column in rowlabels, minus the
                                    values in the top-left corner of the array.

            cl_row              =>  A collabels Python {key:Value,...} dictionary where the
                                    keys are unique row IDs and the values are the
                                    row arrays for each row in collabels, minus the values
                                    in the top-left corner of the array.

            cl_col              =>  A collabels Python {key:Value,...} dictionary where the
                                    keys are the unique column IDs and the values are the
                                    column arrays for each column in collabels.

            core_row            =>  A coredata Python {key:Value,...} dictionary where the
                                    keys are the unique row IDs and the values are the
                                    row arrays for each row in coredata.

            core_col            =>  A coredata Python {key:Value,...} dictionary where the
                                    keys are the unique column IDs and the values are the
                                    column arrays for each column in coredata.

            whole_row           =>  A whole Python {key:Value,...} dictionary where the
                                    keys are the unique row IDs and the values are the
                                    row arrays for each row in the whole array (including
                                    rowlabels and collabels).

            whole_col           =>  A whole Python {key:Value,...} dictionary where the
                                    keys are the unique column IDs and the values are the
                                    column arrays for each column in the whole array.


        Arguments
        ---------
            "data" is the source data to be formatted, such as a
            data file.  However, various data formats are supported:

                data = 'My/file/Path/Name.csv'  (format_ = 'textfile')
                            =>  The data resides as a text file at
                                the address given by the path string.  If
                                the file is in the current working
                                directory, data = 'Name.csv' has the
                                same effect.

                                The file name cannot be numerical.

                            IMPORTANT:  Unlike Unix-based machines, Windows
                            machines use backward slashes in pathnames.
                            This can confuse Python, since backslash has
                            other uses.  The suggested workarounds are
                            either to use forward slashes for both types
                            of operating systems, or preface the path
                            name with a letter r (meaning 'raw string'):

                            data = r'My\file\Path\Name.csv'

                data = ['File1.csv','File2.csv'](format_ = ['textfiles'])
                            =>  data resides across a series of text
                                files, each formatted the same with
                                the same number of columns.  Only the
                                first file should have column headers.

                data = MyArray                  (format_ = 'array')
                            =>  The data resides in a numpy array
                                in memory.  It may or may not contain
                                row and column labels.  If not, integer
                                labels are assigned automatically.

                data = MyDataDict               (format_ = 'datadict')
                            =>  The data is a dictionary that contains
                                Damon-like output arrays, and you want
                                to convert these to a proper Damon object.

                        To use this format, "required" arrays/variables are:

                        ['rowlabels','collabels','coredata','key4rows',
                        'rowkeytype','key4cols','colkeytype','nanval',
                        'validchars']

                        where all arrays are 2-dimensional.

                        Damon will automatically set these parameters according
                        to how they appear in your datadict, and will ignore
                        the corresponding Damon arguments.

                        All "non-required" parameters (not in the "required"
                        list above) will be as you specify them using Damon's
                        arguments.  However, the cols2left, selectrange, and
                        recode arguments are ignored.  (If you need these,
                        use the format_ = 'datadict_whole' option.)

                        Any new variables/arrays that you may have added to
                        MyDataDict will automatically be carried forward to the
                        new Damon object's datadict.

                data = MyDataDict               (format_ = 'datadict_whole')
                            =>  The data is a dictionary that contains
                                Damon-like output arrays, and you want
                                to convert these to a proper DamonObj.

                        format_ = 'datadict_whole' works like format_ = 'datadict'
                        except that it does not ignore any of the Damon
                        object arguments.  It basically rebuilds the datadict
                        into a single "whole" array and applies the Damon
                        parameters to this whole array, as if it were a
                        totally new dataset.

                        This approach allows maximum flexibility in building
                        a new Damon object from an existing datadict.

                data = MyDataDict               (format_ = 'datadict_link')
                            =>  Damon objects can also be directly linked
                                to a pre-existing datadict to save time and
                                memory.  No cleaning, checking, or reformatting
                                is performed, with the exception that a
                                datadict can be converted to 'hd5' (pytables)
                                format using the pytables parameter.

                                In addition to the required arrays
                                and variables (see above), links are built to
                                any other keys residing in the dictionary.

                                'datadict_link' is the fastest and most
                                lightweight way to convert datadicts into
                                Damon objects, at the price of minimum
                                flexibility in redefining the parameters.

                                Remember that any changes you make to the
                                output arrays and variables will automatically
                                ripple back to the source datadict.

                data = my_dmnobj               (format_ = 'Damon')
                            =>  This option essentially copies an existing
                                Damon object to a new name.  It ignores
                                all of new Damon arguments except "data"
                                and "format_".

                data = my_pandas_dataframe          (format_ = 'dataframe')
                            =>  If you have pandas, you can feed Damon a
                                dataframe.  dataframe.index becomes the
                                rowlabels.  dataframe.columns becomes the
                                collabeels.  dataframe.values becomes the
                                coredata.  MultiIndex dataframe indices become
                                single column str(tuple(...)) identifiers.

                data = MyFile.hd5                (format_ = 'hd5')
                            =>  Data can be in hd5 format, the format used
                                by pytables.  It is assumed, in this case,
                                that the data consists of either a single array
                                or that it is in datadict format where
                                'coredata', 'rowlabels', and 'collabels'
                                are in hd5 format (reside in a .hd5 file).

                                This format is only readable if you have
                                pytables and the pytables parameter specifies
                                the name of the hd5 file.  All subsequent
                                PyTable outputs for the Damon object will be stored
                                under the same file name.                

            ------------
            "format_" describes the format of the data in the data=
            parameter above.  (Don't forget the trailing underscore; this
            differentiates it from Python's format function.)

            The data and format_ parameters must be consistent with each
            other; "format_" must accurately describe the format of "data".
            The format options are:

                [<'textfile',
                ['textfiles'],
                'array',
                'datadict',
                'datadict_link',
                'datadict_whole',
                'Damon',
                'dataframe'         # Damon must be able to import Pandas
                'hd5'>
                ]

            If format_ = 'textfile', make sure to fill out the 'delimiter'
            parameter.

            The ['textfiles'] format assumes that only the first file
            in the list contains column headers and that all files are
            otherwise formatted identically.

            'array' assumes that data is a single array that includes
            (or omits) row and column labels.

            'datadict', 'datadict_link', and 'datadict_whole' all mean
            that the "data" parameter lists a "data dictonary" (datadict).
            "datadict_link" maximizes speed.  "datadict_whole" maximizes
            flexibility in changing the datadict.  See the "data"
            documentation above for more information about these formats.

            'Damon' means that "data" lists a Damon object and you simply
            want to return a copy of it under another name.  Any parameters
            besides "data" and "format_" are ignored.  (If you want to change
            it, call it as a datadict using my_damon_obj.data_out, with
            format_ = 'datadict' or 'datadict_whole', and edit to your
            heart's content.
            
            'dataframe' means that "data" refers to a Pandas dataframe.
            Make sure to specify the validchars parameter and any others
            you may need.  Pandas doesn't do that.

            'hd5' means either that the data consists of a single pytables
            ('hd5') array or that it is a dictionary in which rowlabels,
            collabels, and coredata are stored as pytables arrays.  The
            'hd5' format is only readable if you have pytables installed
            on your computer and if you correctly specify its "object name"
            in the "pytables" parameter (see "pytables" below).

            'pickle' is a format used for storing Python dictionaries and is
            used in Damon to store row and column coordinates in banks.  When
            this format is specified, __init__ just opens the pickle file
            named in "data" and returns a dictionary in whatever format,
            with whatever keys, it was formatted.  It does not return a
            "datadict" with keys for rowlabels, collabels, coredata, etc.
            unless it was pickled that way.

            ------------
            "workformat" is used to specify the outputs/attributes of the
            new Damon object, the elements to be included as Damon attributes
            and in the my_obj.data_out datadict.  In the options below, 'RCD'
            stands for the "Rowlabels", "ColLabels", and "Data" arrays.

            workformat = 'RCD'
                {'rowlabels','collabels','coredata','nheaders4rows','key4rows',
                'rowkeytype','nheaders4cols','key4cols','colkeytype','nanval',
                'validchars'}

            workformat = 'whole'
                {'whole'}
                This is the whole data array with row labels and column labels
                affixed to the data.

            workformat = 'RCD_whole'
                The full RCD dictionary (see above) plus the 'whole' array.

            workformat = 'RCD_dicts'
                The full RCD dictionary above, plus additional dictionaries
                for accessing each row and column in the rowlabels, collabels,
                and coredata arrays:  ('rl' stands for rowlabels, 'cl for
                collabels.)

                {'rowlabels','collabels','coredata','whole','nheaders4rows',
                'key4rows','rowkeytype','nheaders4cols','key4cols','colkeytype',
                'nanval','validchars','rl_row','rl_col','cl_row','cl_col',
                'core_row','core_col'}

            workformat = 'RCD_dicts_whole'
                Same as above, but with the two more dictionaries for accessing
                rows and columns in the whole array:

                {'rowlabels','collabels','coredata','whole','nheaders4rows',
                'key4rows','rowkeytype','nheaders4cols','key4cols','colkeytype',
                'nanval','validchars','rl_row','rl_col','cl_row','cl_col',
                'core_row','core_col','whole_row','whole_col'}

                Examples:
                    my_obj.core_col['Item1'] returns the data in the Item1 column

                    my_obj.cl_col['Item1'] returns just the Item1 column labels

                    my_obj.rl_col['ID'] returns the list of persons in the
                        rowlabels array, minus the values at the top in the
                        collabels section.

                    my_obj.whole_row['Mary'] returns all data and labels in Mary's
                        row.

                    np.array([my_obj.cl_row['ID'],my_obj.core_row['Mary']])
                        returns an array whose first row consists of item IDs
                        and whose second row consists of Mary's scores on those
                        items.

            Gotcha Alert:  When workformat = 'RCD_dicts', dictionaries are created
            in addition to rowlabels, collabels, and coredata.  If you
            CHANGE any values in the rowlabels, collabels, or coredata arrays,
            those values will not automatically change in the corresponding
            entity lookup dictionaries.  Nor will they automatically change
            in the 'whole' output.  To avoid confusion, if you plan to
            change any data in a Damon object, such as replace values, specify
            workformat = 'RCD'.  Only add dictionaries later, when the data has
            stabilized.

            ------------
            "validchars" specifies a list or range of valid characters for
            the whole coredata array or for each of the individual columns
            in the coredata array.  (It does not evaluate by row in coredata,
            nor does it evaluate values in rowlabels or collabels.)  All non-valid
            characters are converted to nanval.  It is also possible to ask
            Damon to "guess" the valid characters for each column and construct
            a validchars parameter automatically, though it is usually better
            to specify it explicitly.

            The validchars specification has a lot going on and is referred
            to throughout Damon.  It sets the range of valid values, determines
            the metric of the data (nominal, ordinal, sigmoid, interval, ratio),
            and indicates whether the data are discrete or continuous.

            validchars contains a 'Num' switch to specify that all values
            in the coredata array are to be converted to numbers where possible.
            When it encounters a value that cannot be converted to a number (e.g., 'a')
            it will be changed to nanval (the number assigned to represent
            not-a-number values).

            validchars contains a 'Guess' switch to specify that Damon
            should look at the first 500 rows and try to guess the valid
            characters for each column, disregarding values already converted
            to nanval for whatever reason.

            Because checking the validity of each cell can be time-consuming with
            large datasets, validchars also offers a 'SkipCheck' switch.

            The syntax is:

                ['All',[list/range of possible responses],<'Num','Guess','SkipCheck',omitted>] or
                ['Cols',{'Col1':[list/range of responses],'Col2':[list/range of responses],...},<'Num','Guess','SkipCheck',omitted>]

            The {} braces signify a Python dictionary.

            The first two elements in the list need to be in order.  The "switches"
            ('Num', 'Guess', and 'SkipCheck') do not.

            Examples:

                validchars = ['All',['a','b','c','1.0','2.0','3.0'],'Guess']

            means try to figure out the valid characters for each column
            automatically based on an inspection of the first 500 rows.  It
            is understood that all values in the array that are not listed
            should first be converted to nanval.  This procedure is good
            when you have a mix of item types (e.g., multiple choice and ratings)
            but don't want to be bothered specifying the correct range of
            validchars for each column.

            Note in this case that we listed validchars as including '1.0',
            '2.0', and '3.0', even though the original numerical data might
            have been integers.  This is because when strings are mixed with
            integers, all numerical values are converted to float, then to
            string.  If validchars does not take this into account, the
            numerical values will convert into nanval.

            To gain greater control over the process, you can use the
            tools.guess_validchars() function. The function requires that all
            invalid values have already been changed to nanval, either by
            being excluded in the validchars list, specified in the missingchars
            list, or being a string type when 'Num' is specified.

                validchars = ['All',['All'],'Guess']

            means that all characters in the array are valid and 'Guess'
            will assign valid characters to each column based on all the unique
            values in that column.

                validchars = ['All',['a','b','c','d']]

            means across the whole range of the coredata array the only
            valid values are 'a', 'b', 'c', and 'd'.  All other values should
            be converted to nanval (in quotes, e.g., '-999').

                validchars = ['All',[0,1,2,3]]

            means across the coredata array the only valid values are the
            integers 0, 1, 2, and 3.  ['All',['0 -- 3']] would be equivalent.

            Left to its own devices, Numpy will read numbers as numbers and
            alphabetical characters as string.  But just to make sure that
            all values in coredata are truly numerical and can be put through
            mathematical functions, specify:

                validchars = ['All',[0,1,2,3],'Num']

            To indicate that all NUMERICAL values in coredata (from negative
            to positive infinity) are valid:

                validchars = ['All',['All'],'Num']

            or with dash notation

                validchars = ['All',['. -- .'],'Num']

            To indicate that ALL values are allowed (except any listed
            in "missingchars"), specify either:

                validchars = None
            or
                validchars = ['All',['All']]

            or (to make sure you don't waste time checking values)

                validchars = ['All',['All'],'SkipCheck']

            You can also specify individual columns and, as mentioned,
            use a "dash" or double hyphen (space, hyphen, hyphen, space)
            to describe valid ranges.  The double hyphens should have a space
            character on each side (' -- ') whenever this notation is used:

            validchars = ['Cols',{'Item1':['a','b'],
                                  'Item2':['All'],
                                  'Item3':['1.5 -- 3.5'],
                                  'Item4':['1 -- 5'],
                                  'Item5':[' -- '],
                                  'Item6':['. -- .']
                                  'Item7':['0 -- ']
                                  'Item8':['0.0 -- ']
                                  }
                          ]

            This means for the column labeled Item1, the valid values are 'a' and
            'b'.  For the column labeled Item2, all values in the column are
            valid.  For the column labeled Item3, all continuous values between
            1.5 and 3.5 are valid, this type of specification requiring a ' -- '
            dash and decimal points to indicate all continuous values between
            (and including) 1.5 and 3.5.

            For Item4, all integers from 1 up to and including 5 are valid.  The
            absence of decimal points means these are to be treated as integers.

            For Item5, all INTEGERS from -infinity to +infinity are valid (no
            decimal points), whereas for Item6 all DECIMALS from -infinity to
            +infinity are valid (note the decimal points and don't forget the
            spaces on either side of the dash.

            For Item7, all integers from 0 to +infinity are valid, whereas for
            Item8 all decimals from 0 to +infinity are valid.  Note the right
            hand space before the quote.

            All invalid values are converted to nanval, but there is one exception.
            If a range of integers is specified (['1 -- 5']), and a non-integer is
            encountered in the data (3.1), the non-integer is TRUNCATED to the
            next lowest integer value and treated as valid rather than invalid.
            This only applies when the DASH range notation is used.  A LIST of
            valid values [1,2,3,4,5] will cause any value not on that list to be
            flagged as invalid and converted to nanval.

            It is completely possible for coredata to contain a mix of alpha
            and numerical values (e.g., ['a','b'] for Item1 and ['1 -- 5']
            for 'Item4'.  However, if you don't want to convert the alpha
            values into nanval, do not add the 'Num' switch.  What will happen
            is that Numpy will store all these values at the least general
            type needed to include all values (generally object or string).
            But before performing numerical analysis, you need to convert all
            the alpha values to numerical values using the score_mc() or
            parse() methods.

            The validchars argument is important in other Damon functions.
            For instance, in computing residuals, when fin_resid() sees that
            validchars is not equal to None, e.g., that it contains a list
            of valid integers [0,1,2,3], it allows rounding (true rounding, not
            truncating) all the Damon estimates to those integers before
            computing the residuals.  That way, the residuals and the resulting
            standard errors reveal the prediction accuracy of the model without
            complications caused by comparing integers to decimals.  The
            information in the validchars specification is also used by the parse(),
            standardize(), and fin_est() methods.

            When validchars = None, all values are considered valid except
            those specified by the missingchars argument.  ("None" just means
            "null" here -- no specification necessary.)

            In the standardize() and fin_est() methods, validchars is used to
            flag automatically whether data are nominal (non-numeric),
            ordinal (rating scale), sigmoid (with a minimum and maximum like
            ordinal, but continuous and non-linear), interval (continuous from
            - to + infinity, or ratio (continuous or integer from 0 to +infinity).
            Note that to be flagged as a ratio scale validchars must have the
            form ['0 -- '] or ['0.0 -- '].

            validchars is also used to determine the output metric.  When ordinals
            are specified as valid, without decimal points, Damon final estimates
            are automatically rounded.  When decimal points are included, they
            are continuous:

                validchars = ['All',[0,1],'Num']

            means that when Damon.fin_est() will automatically report estimates
            (predictions, really) as 0 or 1.

                validchars = ['All',[0.0,1.0],'Num'] = ['All',[0.,1.],'Num']

            means the Damon.fin_est() estimates will be continuous, ranging
            from 0.0 to 1.0.

            Storing Valid Chars in a File
            -----------------------------
            To save yourself the trouble of writing out the validchars
            column dictionary in your program, you can store all item
            information in a separate items x attributes textfile, including
            a column for specifying valid response options for each item.
            Let's say the file is called 'item_info.txt'.  In the valid
            characters column you would place the list of valid characters
            corresponding to each item (remember, items are rows here),
            putting quotes around the list:

                "MyValidChars"
                "['a','b','c']"
                "[0,1,2,3]"
                "[0 -- 10]"
                etc.

            Because there are commas, you will want to save 'item_info.txt'
            as a tab-delimited file, not comma-delimited.

            Then, to access the information, you load 'item_info.txt' and
            format it as the "item_info" Damon object.  There are several
            ways to access and use the validchars info.  The best is as
            follows:

            When building your current Damon object (the one containing
            actual data), set validchars = None.  Then, apply the
            merge_info() method while specifying get_validchars = 'MyValidChars'
            (your field name):

                    my_obj.merge_info(item_info,
                                      ...
                                      get_validchars = 'MyValidChars'
                                      )

            merge_info will automatically load the validchars info and apply
            it to your Damon object to clean out invalid values.

            ------------
            "nheaders4rows" is the number of columns, or "row headers", in
            the rowlabels array after cols2left and selectrange have
            done their thing.  Default nheaders4rows = 0 assumes an
            array with no designated row labels.  In this case integer
            labels are appended as the first column (the integer column
            header is usually 0 or 1 but may be a negative integer
            chosen not to clash with other labels).

                nheaders4rows = 3       =>  The first three columns of
                                            the data array are row headers.

            ------------
            "key4rows" is the column in the rowlabels array, counting from
            zero, that contains the identifying key for each row entity.
            Keys should be unique identifiers that always label their
            entity, though Damon has no prohibition against row entities
            and column entities having the same key.

                key4rows = None         =>  The array has no row headers

                key4rows = 3            =>  Row keys are located in the
                                            fourth column from the left,
                                            counting from 0 (0,1,2,3).

            ------------
            "rowkeytype" is the Python or Numpy type you want for the row
            keys.  As of version 1.2.0, rowkeytype is forced to be string,
            default being 'S60' (maximum of 60 characters).
            
                rowkeytype = 'S20'      =>  Cast row keys to be 20-character
                                            strings.

            See help(dmnt.getkeys) for more information.

            ------------
            "nheaders4cols" is the number of rows, or "column headers", in
            the collabels array after cols2left and selectrange have
            done their thing.  Default nheaders4cols = 0 assumes an
            array with no designated column labels, in which case integer
            labels are appended as the first row.

                nheaders4cols = 3       =>  The first three rows of
                                            the data array are column
                                            headers.

            ------------
            "key4cols" is the row in the collabels array, counting from
            zero, that contains the identifying key for each column entity.
            Keys should be unique identifiers that always label their
            entity.

                key4cols = None         =>  The array has no column headers

                key4cols = 3            =>  Column keys are located in the
                                            fourth column from the left,
                                            counting from 0 (0,1,2,3).

            ------------
            "colkeytype" is the Python or Numpy type you want for the column
            keys.  As of version 1.2.0, colkeytype is forced to be string,
            default being 'S60' (maximum of 60 characters).
            
                colkeytype = 'S20'      =>  Cast col keys to be 20-character
                                            strings.

            See help(dmnt.getkeys) for more information.

            ------------
            "check_dups" is used to check row and column keys for duplicates,
            which are not allowed.

                None    =>  Do not check for duplicates.  Either row and
                            column keys are known to be unique within each
                            facet or your particular usage does not require
                            unique ids.

                'warn'  =>  When duplicates are found, rename them so that
                            they are unique and issue a warning.

                'stop'  =>  When duplicates are found, throw an exception
                            and stop the program.  You will rename
                            or otherwise deal with the duplicates yourself.

            ------------
            "dtype" controls how "whole" arrays are printed to the screen
            and to text files.  It specifies the type to which all cells
            should be cast when the core data is merged with the row and column
            labels, as well as the integer number of decimals according
            to which the core data should be formatted and how missing
            values should appear.  dtype is a list with three fields:

                dtype = ['whole' type,  =>  type of 'whole' array
                          str type/decimals, => how many chars per cell
                          nanval replacment  => character to replace nanval
                          ]

            
            [Note: 'object' dtype has been abandoned for the moment.  If
            you specify object, it will come out 'S60'.]

            The other main 'whole' type is string, with a number of characters
            specified to be large enough to handle all possible string
            characters in a label cell, such as 'S60', giving a maximum
            of 60 string characters per cell in rowlabels and collabels.
            Cells with more than 60 characters will be truncated.

            Examples:

                dtype = [object, None, None]
                                    =>  Cast whole array to object,
                                        the most general dtype.  No
                                        specific number of decimals
                                        is specified.  The nanval will
                                        be as specified for the Damon
                                        object, not changed for printing
                                        the whole array.

                dtype = ['S60', 8, '.']
                                    =>  format_ each number in core data so
                                        that it prints out with 8 decimal
                                        places.  format_ row/column labels
                                        so they are capped at 60 characters.
                                        The 'S60' is referred to whenever
                                        there is a need to cast the array
                                        to string.  Wherever there is a
                                        missing (nanval) cell, fill it
                                        with '.'.

            When casting the whole array as string, it is a good
            idea to specify an integer decimal parameter (e.g., 8).  This
            avoids the awkwardness of accidentally casting scientific notation
            (which can pop up unexpectedly when numbers are small) to string,
            which may cause an unfortunate truncation.  For instance, one
            might have a cell value of 3.1234567891234e-17 which is
            accidentally truncated to 3.1234567891234 when the array
            is converted to ('S15'), even though the true number is almost
            zero.  (In this case, a larger string size such as 'S20' would
            avoid the problem by leaving room for the e-17 notation, although
            this consumes more memory.)

            Bear in mind, however, that specifying the decimal parameter
            can create a speed bottleneck in tools.addlabels()
            for very large arrays, which is used internally by Damon in
            several places.  Specifying as "object" avoids these issues.

            ------------
            "nanval" (short for not-a-number value) is a numerical value
            that YOU specify (integer or float -- default is -999)
            to which all missing or invalid core data should be coded.  If
            coredata contains alpha string values, nanval will automatically
            converts to string ('-999').  Make sure nanval is a number that
            is not likely to occur naturally in your dataset.

            nanval is Damon's somewhat old-fashioned way of dealing with
            missing and invalid values.  For several reasons, Damon has avoided
            other Numpy tools for dealing with missing values (masked arrays
            and np.nan).  Numpy's relatively recent (version 1.7) np.NA object
            may end up being the right way to go, but that hasn't been explored
            yet.  At least by specifying a nanval, you know what you're
            getting and that it will behave like other numbers and that it
            can be cast to any type.

            Note:  It is tempting to specify nanval = np.nan, i.e.,
            to use numpy's nan definition.  This will not work, so Damon
            converts np.nan to -999.

            ------------
            "missingchars" is a list of characters, e.g., [10, '*'] to be made
            missing and converted to NaNVals in the core data.

            ------------
            "miss4headers" is a list of values/words, e.g., ['bogusID','another']
            to be made missing in rowlabels and collabels.  These values will
            be replaced with nanval, cast to string if necessary.

            ---------------
            "recode" is used to recode any number of values in one or more
            ranges that you specify.  It consists of a dictionary of ranges,
            where each range is described using Python's slice notation and
            each contains a recode dictionary.  The syntax is:

            recode = {0:[[slice(StartRow,EndRow),slice(StartCol,EndCol),{From:To,...}],
                      1:[[slice(StartRow,EndRow),slice(StartCol,EndCol),{From:To,...}],
                      ... any number of ranges ...
                      }

            recode = {0:[[slice(3,5),slice(None,None)],{'A':'a','B':'b'}],
                      1:[[slice(None,10),slice(5,6)],{'D':'d','E':'e'}]
                      }

            This means, the 0th range goes from row 3 up to (but not including)
            row 5, and from the left-most column to the right-most column ("None"
            means "go to the end").  Within that range, recode 'A' to 'a' and
            'B' to 'b'.

            Range 1 runs from all rows up to (but not including) row 10 and
            from column 5 up to (but not including) column 6 (in other words,
            just column 5).  Within range 1, recode 'D' to 'd' and 'E' to 'e'.

            Every range must be labeled by an integer, starting with 0.  They
            are recoded in that order.

            Be careful when recoding integers that are read as strings.  A
            1 may be loaded as a '1.0' and not be recognized.

            The recode procedure is run BEFORE the cols2left and select range
            procedures are run.

            ---------------
            "cols2left" addresses the situation where columns of data you
            want to analyze are interspersed with columns containing only
            labeling information.  cols2left makes it possible to shift
            these columns to the left so they can be joined into
            a single rowlabels array.

            cols2left = None means the row labels are already arranged
            along the left side of the data array and the data to be analyzed are
            contiguous.  Otherwise, cols2left is the list of column keys
            that identify each column of potential row labels.  The list of
            elements should be in the desired order.  For instance,

                cols2left = ['A','B'] causes...

                    [['0','A','1','B'],
                     ['3','x','7','y']]

                to become:
                    [['A','B','0','1'],
                     ['x','y','3','7']]

            Note 1: The nheaders4rows and related arguments should be based
            on how the data matrix looks AFTER cols2left has been applied.
            Also, the list of column keys should include those for any label
            columns that are already all the way to the left, that you want
            to have stay there.

            Note 2: cols2left is run after recode and before selectrange.
            Therefore, recode ranges are defined in terms of the original
            array, whereas the selectrange procedure is defined in terms
            of the reshuffled array.

            ---------------
            "selectrange" makes it possible to define the range of data from the
            incoming data file or array that contains the numeric data, row labels,
            and column labels of interest.  It is entered using slice notation.
            selectrange is performed AFTER cols2left has been performed, so
            it applies to the rearranged matrix.  Example:

                selectrange = [slice(0,None),slice(10,30)]

            means select the range that comprises all the rows from row 0 (the
            first row) to the last row ('None' means 'to the end') and  the
            columns from column 10 up to (but not including) column 30.
            Remember that Python starts counting at 0, not 1.

            ------------
            "delimiter" is the character that delimits fields in the file. Use
            ',' for comma delimited files (with a .csv extension) and '\t' for
            text tab delimited files (with a .txt extension).  If you specify
            a tab delimiter with a .csv file Numpy gets confused.
            outputs of most mathematical programs, such as Excel, support
            export as a text file with some specified type of delimiter.

            ------------
            NOTE:  The "pytables" option has been deprecated.  Avoid using.
            
            "pytables" calls Damon's tools.pytables() function to store and
            access large arrays on disk rather than in memory.  It was designed
            to get around situations where your accumulated arrays are
            too large to store in memory and pretty much empowers Damon
            to handle "enterprise-sized" datasets -- a million rows and
            a thousand columns per analysis, say, though even this is small
            by modern standards.  Note that pytables itself can handle
            much larger datasets for data storage and access, but Damon uses
            it for much more complicated purposes that can be memory
            intensive and computationally demanding, even with pytables.

            PyTables is an independent open-source scientific computing
            package (www.pytables.org), which is not included in
            Python and Numpy.  To install pytables, it is recommended
            that you purchase the Enthought distribution (www.enthought.com),
            an comprehensive suite of scientific, graphic, and numeric
            packages built on Python and Numpy that includes pytables.  It
            takes care of all dependencies and makes sure versions are
            compatible, in addition to loading you up with some excellent
            scientific computing and visualization software.

            pytables stores one or more arrays in a series of virtual
            directories ("groups") inside a single large file with a .hd5
            suffix in HDF5 format.  HDF stands for Hierarchical data format_
            and is used for storing very large amounts of data in a tree-like
            hierarchical data structure (nested directories). It is a
            hierarchical, rather than relational, database.

            If pytables is not None, Damon() automatically stores
            all arrays produced by the various Damon methods for
            a given Damon instance inside a 'hd5' file whose name
            is specified by the user.  That means the user
            only needs to specify pytables once, during the Damon
            initialization.  There should be only one 'hd5' file per
            Damon analysis -- though some of the methods create their
            own 'hd5' files for internal use which are automatically
            deleted by the export() method.

            Options:

                pytables = None     =>  pytables is not used and all Damon
                                        arrays are stored and accessed in memory.

                pytables = 'MyFileName.hd5'
                                    =>  All Damon outputs are to be stored inside
                                        directories ("groups") inside a file
                                        specified by the user with a .hd5 extension,
                                        e.g., 'MyFileName.hd5'.  Although the
                                        tools.pytables() function accepts
                                        pathnames ('/Me/Scripts/MyFileName.hd5'),
                                        __init__() will get confused if you
                                        specify a pathname.  It prefers to
                                        assign the file to the current
                                        working directory or to
                                        '/site-packages/damon1/tests/play' if
                                        Damon is run from IDLE.  So don't
                                        use a pathname here, just a file name.

                pytables = other_damon_obj['fileh']
                                    =>  .hd5 files have a pathname, but they
                                        also correspond to a file object
                                        created by pytables.  You have the
                                        option of entering this file
                                        object instead of a path name
                                        in order to refer to an existing
                                        PyTable instead of creating a new
                                        one.  You can find this file object
                                        by looking in an existing Damon
                                        object under the key name
                                        'fileh'.  The 'fileh' attribute
                                        is added automatically to a DamonObj
                                        anytime you use the pytables option.

            As mentioned, when using the pytables option a 'fileh'
            attribute is added to the Damon object and referred to for
            all methods applied to that Damon.  This only works
            if the file is left "open".  Therefore, it is up to the user
            (you) to tell the program to close a pytables file
            when you are done working with the Damon object.  This is done
            by typing:

                my_dmnobj.fileh.close()                or
                my_dmnobj.data_out['fileh'].close()

            Since it is used at the end, Damon's report() method
            automatically closes fileh.  pytables will also automatically
            close fileh in many situations, so you probably won't
            suffer much if you forget to close it.  However, you will
            get an error message if you try to create a new pytables
            file when an existing one with the same name is still open.

            When using pytables, the output of each Damon method
            is a data dictionary containing pytables objects, not
            array objects.  These have the nice property of residing
            on the disk, not in memory, which is why they are output
            that way.  But what do you do when you want to access
            it as a numpy array?  Use the .read() method:

                MyCoreDataArray = MyDataDict['coredata'].read()
                MyRowLabelsArray = MyDataDict['rowlabels'].read()

            This also works:

                MyColLabelsArray = MyDataDict['collabels'][:,:]

            Anytime you apply slice notation to a PyTable, it converts
            it into a readable array:

                MySlice = MyDataDict['coredata'][3:,5:]

            The price of the .read() method is that it opens an array
            in memory.  However, you can often avoid this.  A lot of
            Numpy functions work when applied directly to the pytables
            object instead of the array:

                Result = np.some_func(my_dmnobj.coredata)

            which uses much less memory than

                Result = np.some_func(my_dmnobj.coredata.read())

            But try it out first.  Sometimes the function doesn't work.
            Sometimes it does, but slowly.  Optimization takes a lot
            of work.

            pytables is a wonderful program but it has a steep learning
            curve and its documentation is targeted at scientist
            programmers.  In lieu of using pytables directly, you can
            use Damon's tools.pytables() wrapper function which will
            allow you to handle most situations involving the conversion
            of numpy arrays to and from the pytables 'hd5' format.  It
            is a LOT easier to learn.  See the tools.pytables() docs.
            You will find the tool is easily applied to use outside of
            Damon.

            When to Use pytables
            --------------------
            If your datasets are small (say, less than 10,000 x 1000), you
            don't need pytables, but it doesn't hurt to use it.  It may
            even speed things up in some cases.

            If your datasets are large, you have a trade-off to consider
            that depends on the RAM available on your machine.  It is generally
            faster not to use pytables and to just run in memory -- if you
            can get away with it.  But if you are consuming so much RAM
            that everything slows to a crawl, or you get a memory error,
            you will want to switch to pytables.

            I wish I could say that switching to pytables will allow you
            to run ANY size data set, but that is not true.  To maximize
            speed, Damon strikes a balance between memory usage and working
            in chunks.  Sometimes, it holds the entire dataset in memory
            to speed up a given operation.  Therefore, even using pytables,
            it is possible to exceed your available memory at any given
            moment.

            In practice, on my machine, I find that I can crunch
            a 100,000 x 100 array easily and a 1,000,000 x 100 array
            with difficulty.

            ---------------
            "verbose", when equal to True, causes the interpreter to report all
            Damon methods that are called.  Options:  <None, True>


        Examples
        --------

            [under construction]


        Paste Class
        -----------
            Damon(data,    # [<array, file, [file list], datadict, Damon object, hd5 file>  => data in format specified by format_=]
                  format_,    # [<'textfile', ['textfiles'],'array','datadict','datadict_link','datadict_whole','Damon','hd5','pickle'>]
                  workformat = 'RCD_dicts',   # [<'RCD','whole','RCD_whole','RCD_dicts','RCD_dicts_whole'>]
                  validchars = None,   # [<None,['All',[valid chars],<'Num','Guess','SkipCheck',omitted>],['Cols',{'ID1':['a','b'],'ID2':['All'],'ID3':['1.2 -- 3.5'],'ID4':['0 -- '],...}]>]
                  nheaders4rows = 0,  # [number of columns to hold row labels]
                  key4rows = 0,   # [<None, nth column from left which holds row keys>]
                  rowkeytype = 'S60',     # [<None, type of row keys>]
                  nheaders4cols = 0,  # [number of rows to hold column labels]
                  key4cols = 0, # [<None, nth row from top which holds column keys>]
                  colkeytype = 'S60',     # [<None, type of column keys>]
                  check_dups = 'warn',   # [<None,'warn','stop'> => response to duplicate row/col keys]
                  dtype = [object, 3, ''], # [[type of 'whole' matrix, <None, int decimals>, <None, display nanval], e.g. ['S60', 8, ' '],[object, None, None] ]
                  nanval = -999,    # [Value to which non-numeric/invalid characters should be converted.]
                  missingchars = None,  # [<None, [list of elements to make missing]>]
                  miss4headers = None, # [<None, [[list of elements to make missing in headers]>]
                  recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
                  cols2left = None,    # [<None, [ordered list of col keys, to shift to left and use as rowlabels]>]
                  selectrange = None,   # [<None,[slice(StartRow,EndRow),slice(StartCol,EndCol)]>]
                  delimiter = ',',  # [<None, character to delimit input file columns (e.g. ',' for .csv and '\t' for .txt tab-delimited files)]
                  pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                  verbose = True,    # [<None, True> => report method calls]
                  )

        """
        if verbose is True:
            print 'Building Damon object...\n'

        # Get data outputs, add to self
        data_out = dmn.utils._data(locals())
        self.data_out = data_out

        # Initialize input variables
        self.format_ = format_
        self.workformat = workformat
        self.validchars = validchars
        self.check_dups = check_dups
        self.dtype = dtype
        self.nanval = nanval
        self.missingchars = missingchars
        self.miss4headers = miss4headers
        self.recode = recode
        self.cols2left = cols2left
        self.selectrange = selectrange
        self.delimiter = delimiter
        self.pytables = pytables
        self.verbose = verbose

        # Attributes assigned after running _data (overwrites allowed)
        self.rowlabels = data_out['rowlabels']
        self.collabels = data_out['collabels']
        self.coredata = data_out['coredata']
        self.nheaders4rows = data_out['nheaders4rows']
        self.key4rows = data_out['key4rows']
        self.rowkeytype = data_out['rowkeytype']
        self.nheaders4cols = data_out['nheaders4cols']
        self.key4cols = data_out['key4cols']
        self.colkeytype = data_out['colkeytype']
        self.validchars = data_out['validchars']
        self.nanval = data_out['nanval']
        self.rl_row = data_out['rl_row']
        self.rl_col = data_out['rl_col']
        self.cl_row = data_out['cl_row']
        self.cl_col = data_out['cl_col']
        self.core_row = data_out['core_row']
        self.core_col = data_out['core_col']
        self.whole_row = data_out['whole_row']
        self.whole_col = data_out['whole_col']
        self.whole = data_out['whole']
        self.fileh = data_out['fileh']

        if dmn.R_flag is True:
            self.R = dmn.core_R.DamonR(self)

        if self.verbose is True:
            print 'Damon object has been built.'
            print 'Contains:\n',self.__dict__.keys(),'\n'

    #############################################################################

##    # This attribute acts as an access point for R package methods
    if dmn.R_flag is True:
        R = dmn.core_R.DamonR()

    ##############################################################################

    def __str__(self):
        "Print representation of Damon object."

        class Damon_str_Error(Exception): pass

        try:
            if self.whole is not None:
                return 'Damon object\n %s' % self.whole
            else:
                np.set_printoptions(precision=3, suppress=True)
                return 'Damon object (coredata)\n %s' % self.coredata
        except AttributeError:
            exc = 'Unable to output Damon as string.  Try >>> print my_dmnobj.data_out.\n'
            raise Damon_str_Error(exc)

    ##############################################################################
        
    def __repr__(self):
        "Print representation of Damon object."

        class Damon_str_Error(Exception): pass

        try:
            if self.whole is not None:
                return 'Damon object\n %s' % self.whole
            else:
                np.set_printoptions(precision=3, suppress=True)
                return 'Damon object (coredata)\n %s' % self.coredata
        except AttributeError:
            exc = 'Unable to output Damon as string.  Try >>> print my_dmnobj.data_out.\n'
            raise Damon_str_Error(exc)

    ##############################################################################

    def __getitem__(self,key):
        "Access attributes using dictionary notation."

        return self.__dict__[key]

    ##########################################################################

    def merge_info(self,
                   info,  # [Damon object or datadict of entities x attributes]
                   target_axis = 'Col', # [<'Row','Col'> => i.e., merge to row or column entities]
                   get_validchars = None,   # [<None,'ValRespField'> => get and apply validchar info to dataset]
                   ):
        """Merge item or person information into Damon object labels.

        Returns
        -------
            merge_info() returns None but assigns a merge_info_out datadict
            to the Damon object.  It merges item or person information
            into the row or column labels.

            Workflow:
                my_test = dmn.Damon(testdata,...)
                my_item_info = dmn.Damon(iteminfo,...).data_out
                my_test.merge_info(my_item_info,...)
                merged = my_test.merge_info_out

        Comments
        --------
            merge_info() loads information about persons or entities
            into the labels of the current Damon object.  Such information
            is often stored in a text file or database that is separate
            from the data file.  merge_info() brings it into the dataset
            where it is useful.  Use cases:

                *   We want to compute statistics (using summstat()) for
                    individual item groups or clusters, but the item/cluster
                    information is stored in a separate file.

                *   We want to have access to the text of the items
                    so that it is easier to diagnose item misfit, but
                    the text is stored in a separate file.

                *   We want to compute statistics (using summstat()) for
                    person demographic groups, but the demographic
                    information is stored in a separate file.

                *   We want label-based access to correct responses,
                    i.e., an answer key.

                    (Note, however, that Damon methods such as score_mc(),
                    parse(), and fin_est() which use answer key information
                    do not require correct responses to be included in the
                    column labels.  merge_info() is not needed for
                    that purpose.)

            All of these cases require a simple way to merge outside
            information with the entities in the current dataset.  Here
            is how to do it:

                1)  You start with the information file.  Let's say it
                    is an answer key file containing information about
                    items.  Make sure that it is formatted such that
                    the rows are items and the columns are item attributes.
                    (The convention is that the entities are rows and
                    the entity attributes are columns, regardless of
                    whether the entities are persons or items.)

                2)  Load and format the information file as a Damon
                    object.  Call it, say, "my_item_info".

                3)  Load and format the main person x item dataset
                    as a second Damon object called "my_test".  Note
                    that if the item info file contains valid response
                    information (range of valid responses) you can
                    pull that information from my_item_info when
                    defining the validchars parameter in my_test.  Call
                    up help(dmn.Damon.__init__) and find the validchars
                    docs for a simple way to do this.

                4)  Merge the item information into my_test:

                    my_test.merge_info(my_item_info.data_out,...)

                    (Remember that the "data_out" attribute simply
                    restates the my_item_info object as a datadict.)

            Now the item information is available for all Damon methods
            that make use of row or column labels.

            Note, by the way, that there is no requirement that the
            information array have the same number of entities as
            the dataset.  Only information for entities that exist in
            the dataset will be merged.

        Arguments
        ---------
            "info" is a Damon object or a datadict, generally obtained
            by reading in an entities x attributes textfile as a Damon
            object.  It can be specified as either:

                info = my_info_obj          or
                info = my_inf_obj.data_out


            When creating my_info_obj using Damon(), you will need
            to specify:

                work_format = 'RCD_dicts_whole'

            A typical item info file (e.g., an answer key) will include
            columns labeled something like:

                ['ItemID','Cluster','Correct','ValidResp']

            However, there is no hard requirement regarding the label
            names.

            ---------------
            "target_axis" refers to the axis in the current dataset
            that holds the type of entity described in the info
            datadict.  (In this context, "target" means the current
            Damon object, the one you are merging TO.)

                target_axis = 'Row'     =>  The entities in the info
                                            datadict correspond to
                                            row entities in the current
                                            Damon object (e.g., persons).

                target_axis = 'Col'     =>  The entities in the info
                                            datadict correspond to
                                            column entities in the current
                                            Damon object (e.g., items).

            ---------------
            "get_validchars" is used to find the column in "info" that
            contains validchars information.  It is used to overwrite the
            validchars parameter of the current Damon object and is
            applied to the data to convert all non-valid responses
            to nanval.

                get_validchars = None   =>  Do not look for or use the
                                            the field containing validchar
                                            info.

                get_validchars = 'MyValidResp'
                                        =>  Look for the column in info
                                            labeled 'MyValidResp'.  Convert
                                            this into a validchars parameter
                                            and overwrite the existing validchars
                                            parameter.  Apply the validchars
                                            info to the dataset to ensure
                                            that all data is valid.

        Examples
        --------

            [under construction]

        Paste Method
        ------------
            merge_info(info,  # [Damon object or datadict of entities x attributes]
                       target_axis = 'Col', # [<'Row','Col'> => i.e., merge to row or column entities]
                       get_validchars = None,   # [<None,'ValRespField'> => get and apply validchar info to dataset]
                       )


        """
        if self.verbose is True:
            print 'merge_info() is working...\n'

        # Run the damon utility
        merge_info_out = dmn.utils._merge_info(locals())
        self.merge_info_out = merge_info_out

        if self.verbose is True:
            print 'merge_info() is done -- see my_obj.merge_info_out'
            print 'Contains:\n',self.merge_info_out.keys(),'\n'

        return None






    ##################################################################################################

    def extract_valid(self,
                      minperrow = 10, # [<int,proportion> => minimum observations per row]
                      minpercol = 10, # [<int,proportion> => minimum observations per column]
                      minsd = None,  # [<None,minimum row/col allowed standard deviation>]
                      rem_rows = None, # [<None, [row keys], str, dict>] => row entities to remove]
                      rem_cols = None,   # [<None, [col keys], str, dict> => col entities to remove]
                      iterate = False,  # [<bool> => keep cycling until all rows/cols valid]
                      ):
        """Extract only rows and columns with sufficient data.

        Returns
        -------
            extract_valid() returns None but assigns an output
            datadict attribute to the Damon object accessed using:

                my_obj.extract_valid_out
            
            my_obj.extract_valid_out['iterations'] reports how many
            iterations were necessary to clean the data if iterate = True.
            
            Workflow:

                d = Damon(...)
                d.extract_valid(...)
                d.score_mc(...)
                d.standardize(...)

        Comments
        --------
            To produce valid estimates, Damon requires at least
            as many observations per row and column as there are
            dimensions.  This is a basic mathematical requirement.
            In fact, when exploring optimal dimensionality,
            the coord() method will not even attempt dimensionalities
            that exceed this requirement.  Ideally, there should be
            be many more observations than dimensions, depending
            on the noisiness and imprecision of the data.  extract_valid()
            extracts only those rows and columns that have a specified
            minimum number or proportion of observations.

            In addition, it is frequently necessary to flag rows
            or columns that have no variation in their data values,
            i.e., a standard deviation of 0.0.  This is especially
            true of dichotomous data, where extreme scores (all 0's
            or all 1's) imply infinity and are mathematically
            intractable.  extract_valid() also flags for no variation
            in string arrays, making it possible to use prior to
            score_mc().
            
            extract_valid() can also be used to check that all entities
            in a given dataset reside in a given person or item bank.
            This is important when scoring individual data vectors using
            bank parameters.  It is done by specifying the filename of
            a bank in the rem_rows or rem_cols parameters.

            extract_valid() does not delete the rows and columns marked
            invalid; they continue to reside in the obj.data_out()
            datadict.  It merely removes them from consideration
            for subsequent methods.

            To restore invalid rows and columns to desired outputs
            after running Damon methods, apply the restore_invalid()
            method.  All estimates and statistics for those rows and
            columns will consist of NaNVals.

        Arguments
        ---------
            "minperrow" is either an integer or a percentage
            used to specify the minimum number of observations
            per row.  All other rows will be filtered out (though
            not deleted).

                minperrow = 10      =>  Each row must have a
                                        minimum of 10 observations.

                minperrow = 0.05    =>  The number of observations
                                        in each row must be at least
                                        5% of the total number of
                                        columns.

            -------------
            "minpercol" is either an integer or a percentage
            used to specify the minimum number of observations
            per column.  All other columns will be filtered out
            (though not deleted).

                minpercol = 10      =>  Each column must have a
                                        minimum of 10 observations.

                minpercol = 0.05    =>  The number of observations
                                        in each column must be at least
                                        5% of the total number of
                                        rows.

            -------------
            "minsd" is a float indicating a minimum required standard
            deviation for both the rows and columns.  All rows and
            columns not meeting this minimum will be filtered out.  If
            the array values are string, the method converts them to
            integers before calculating a standard deviation.

                minsd = 0.001       =>  Each row and column must have
                                        a standard deviation of at
                                        least 0.001.

            -------------
            "rem_rows" <None, [rows], str, dict> is a list of row entities to
            remove from the Damon object.  Typically, these will have
            been flagged for having issues by the flag() method. If rem_rows
            is a string or dict, it is assumed that it refers to a bank,
            in which case entities are automatically removed that don't
            exist in the bank.

                rem_rows = ['mark', 'mary', 'sheila']
                                    =>  Remove 'mark', 'mary', and 'sheila'
                                        from the analysis.
                
                rem_rows = 'my_bank.pkl'
                                    =>  Remove all row entities that don't
                                        also reside in 'my_bank.pkl'
            
            -------------
            "rem_cols" <None, [cols], str, dict> is a list of column entities 
            to remove from the Damon object. Like rem_rows, if rem_cols
            is a string or dict, it is assumed that it refers to a bank,
            in which case entities are automatically removed that don't
            exist in the bank.
            
            -------------
            "iterate" <True, False> instructs the method to remove entities,
            check the array again for invalid rows or columns, and repeat
            until no further invalid rows or columns are found.  This is
            for the fairly rare case where removal of some rows or columns
            causes others to become invalid.

        Example of Iterative Item Flagging
        ----------------------------------
            extract_valid() can be used effectively in combination with
            flag() to iteratively clean a dataset.  See the flag() docs
            for an example.

        Paste method
        ------------
            extract_valid(minperrow = 10, # [<int,proportion> => minimum observations per row]
                          minpercol = 10, # [<int,proportion> => minimum observations per column]
                          minsd = None,  # [<None,minimum row/col allowed standard deviation>]
                          rem_rows = None, # [<None, [row keys]>] => row entities to remove]
                          rem_cols = None,   # [<None, [col keys]> => col entities to remove]
                          iterate = False,  # [<bool> => keep cycling until all rows/cols valid]
                          )
        """

        if self.verbose is True:
            print 'extract_valid() is working...\n'

        # Run utility
        extract_valid_out = dmn.utils._extract_valid(locals())
        self.extract_valid_out = extract_valid_out

        if self.verbose is True:
            print 'extract_valid() is done -- see my_obj.extract_valid_out.'
            print 'Contains:\n',self.extract_valid_out.keys(),'\n'

        return None




    ##################################################################################################

    def pseudomiss(self,
                   rand_range = 'All',     # [<None, 'All', [<'Rows','Cols'>,['ID1','ID2',...]]> ]
                   rand_nan = 0.10,     # [<None, proportion to make missing>]
                   ents2nan = None,     # [<None, [('Person1','Item3'),('Person5','Item2')] => list of entity tuple pairs>]
                   range2nan = None,      # [<None,([1,1,5],[1,2,1]),[slice(0,2),slice(0,2)] => row/col indices from np.where() function> ]
                   seed = 1,  # [<None => any random selection; int => integer of "seed" random coordinates>]
                   ):
        """return_ indices for cells deliberately made missing.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The outputs are accessed using:

                my_dmnobj.pseudomiss_out

            pseudomiss_out is a Python dictionary containing six indices of
            missing cells based on row and column position:

            'msindex'           =>  np.where() index of ALL missing
                                    cells (pseudo- and original)

            'true_msindex'       =>  np.where() index of true
                                    missing cells

            'psmsindex'         =>  np.where() index of only
                                    pseudo-missing cells

            'parsed_msindex'     =>  "None" placeholder until parse()
                                    is run, which expands msindex
                                    to describe the expanded array

            'parsed_true_msindex' =>  "None", see above

            'parsed_psmsindex'   =>  "none", see above

            'seed'              =>  Reports the seed specification

            Workflow:
                my_dmnobj.pseudomiss(...)


        Comments
        --------
            To determine "best" dimensionality as well as to produce
            unbiased estimates and errors, it is helpful to make
            a sample of observations missing and measure how well
            Damon predicts them.  The resulting "accuracy" statistic
            is an element in Damon's objectivity formula.  pseudomiss()
            generates indices describing a sample of cells to make
            pseudo-missing for purposes of calculating this statistic.

            Damon's coord() method automatically runs pseudomiss()
            internally setting rand_range = 'All' and seed = 1, to
            compute accuracy --- running pseudomiss() yourself is purely
            optional.  Do so if you want greater control over the
            cells to be made pseudo-missing, i.e., you want to use
            a different random range or seed.

            Important (but subtle) Notes
            ----------------------------
            The parse() and standardize() methods are always applied to
            the original data, not to the pseudo-missing data.  The
            coord() method, run after standardize(), applies the
            pseudo-missing index to the standardized values.  Unfortunately,
            this can introduce subtle statistical biases in the standardized
            values.  Although, this is highly unlikely to be a problem with
            real-world data, you should be aware of it.

            If you need to be absolutely rigorous and apply standardize
            to the pseudo-missing array, a simple approach is to create
            a separate Damon based on a pseudo-missing data array.  This
            will force all methods to refer only to that array.  You can
            then compare its outputs with a DamonObj based on complete data.

            Although no parsed or standardized data are actually made
            pseudo-missing, when parse() is run it does generate
            new pseudo-missing indices to describe the pseudo-missing
            locations in the expanded, parsed array.  These are stored
            in the pseudomiss_out dictionary with the other indices and
            used by coord() if necessary.

            If you do not run pseudomiss(), but run parse(), coord()
            will automatically make cells pseudo-missing in the parsed
            matrix, but in a random way without taking into account
            that when one response in a parsed column is missing
            they all are.  Running pseudomiss() manually avoids any
            statistical biases this might cause.

        Arguments
        ---------
            "rand_range" specifies where to make cells randomly missing --
            across the whole array or for specified row or column entities
            in a list.

                rand_range = None    => don't make any cells randomly missing,
                                       or use the "Fixed__" range arguments
                                       instead.

                rand_range = 'All'   => make pseudo-missing a random sample
                                       of all cells in array

                rand_range = ['Rows',['P1','P2']]
                                    => make pseudo-missing a random sample
                                       of cells in the rows corresponding to
                                       row entities 'P1' and 'P2'

                rand_range = ['Cols',['It1','It2']]
                                    => make pseudo-missing a random sample
                                       of cells in the columns corresponding to
                                       col entities 'It1' and 'It2'

            ----------------
            "rand_nan" is the proportion of cells in rand_range
            to make pseudo-missing.  Options are: <None, decimal>

                None        =>  The rand_range parameter is None, so
                                this argument does not apply.

                0.10        =>  Make 10% of the cells in rand_range
                                missing at random.

                                Note:  Specifying 10% missing cells
                                can create a speed bottleneck with
                                large datasets.  Consider shrinking
                                the percentage as a function of
                                data size.

            ents2nan and range2nan are alternative ways
            to specify cells to be made pseudo-missing, one in terms of
            entity pairs (tuples), the other in terms of row and column
            position.

            ----------------
            "ents2nan" = <None,[('Fac0Ent','Fac1Ent'),...]> is
            a list of entity pairs in tuple format, each of which indicates
            a cell to be made pseudo-missing.

                ents2nan = [('P1','I3'),('P5','I7')]

            means make the cell corresponding to the 'P1' row entity
            and 'I3' column entity pseudo-missing.  Do the same with
            the 'P5' row entity and the 'I7' column entity.

            ----------------
            "range2nan" = <None,([Rows],[Cols])> is a tuple containing
            a list of row indices followed by a corresponding list of
            column indices.  Alternative it is a list containing slice
            objects to select contiguous ranges of cells.

                range2nan = ([1,1,5],[1,2,1])

            means make cells pseudo-missing that correspond to Row 1 and Col 1,
            Row 1 and Col 2, and Row 5 and Col 1.  This is the row/col output
            format used by numpy's where() function.

                range2nan = [slice(0,2),slice(1,3)]

            means make cells pseudo-missing that are in the first two rows
            (row 0 up to but NOT including 2) and in columns 2 and 3.

            WARNING:  Any true missing cells within this range will be
            treated as pseudo-missing.

            ----------------
            "seed" controls which cells to make randomly missing whenever
            Numpy's random() functionality is invoked.

                seed = None         =>  Every time the function is run, a
                                        different set of random numbers is
                                        used.

                seed = 1            =>  Every time the function is run, the
                                        same set of random numbers -- the '1'
                                        set -- is used.

                seed =  2            =>  Every time the function is run, the
                                        '2' set of random numbers is used.
                                        And so on, a different set or "seed"
                                        for each integer.


        Examples
        --------



        Paste method
        ------------
            pseudomiss(rand_range = 'All',     # [<None, 'All', [<'Rows','Cols'>,['ID1','ID2',...]]> ]
                       rand_nan = 0.10,     # [<None, proportion to make missing>]
                       ents2nan = None,     # [<None, [('Person1','Item3'),('Person5','Item2')] => list of entity tuple pairs>]
                       range2nan = None,      # [<None; ([1,1,5],[1,2,1]),[slice(0,2),slice(0,2)]) => row/col indices from np.where() function> ]
                       seed = 1,  # [<None => any random selection; int => integer of "seed" random coordinates>]
                       )

        """
        if self.verbose is True:
            print 'pseudomiss() is working...\n'

        # Run utility
        pseudomiss_out = dmn.utils._pseudomiss(locals())
        self.pseudomiss_out = pseudomiss_out

        if self.verbose is True:
            print 'pseudomiss() is done -- see my_obj.pseudomiss_out.'
            print 'Contains:\n',self.pseudomiss_out.keys(),'\n'

        return None




    ##########################################################################

    def score_mc(self,
                 anskey = None,    # [<None, 'bank.pkl', ['All',['A']],['Cols',{'ID1':['A'],'ID2':['C'],'ID3':[None],'ID4':[4],...}]> ]
                 report = None,   # [<None,[list reports]>  => ['All','RowScore','ColScore','RowFreq','ColFreq','MostFreq','AnsKey','MatchKey','PtBis'] ]
                 getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                 usecols = {'Scores':'All','Freqs':'Scored'},    # [<{'Scores':<'All','Scored'>,'Freqs':<'All','Scored'>} => columns to use when calculating scoring or frequency stats]
                 score_denom = 'All',    # [<'All','NonMissing'> => how to count total observations for denominator]
                 nanval = -999. # [Numerical Not-a-Number value to replace object nanval]
                 ):
        """Score multiple choice responses according to an answer key.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The score_mc() outputs are accessed using:

                my_dmnobj.score_mc_out

            score_mc_out is a datadict whose coredata consists of a
            dichotomous (0,1) scored array for those columns
            designated using the anskey parameter, plus the original
            data values for those columns not designated for scoring.
            score_mc_out includes datadicts for statistics commonly
            reported for multiple choice data.  These statistics
            are calculated only from the scored multiple-choice
            section of the dataset:

                 'RowScore'         =>  % correct by row
                 'ColScore'         =>  % correct by column
                 'RowFreq'          =>  Category response frequencies by row
                 'ColFreq'          =>  Category response frequences by column
                 'MostFreq'         =>  Most frequent response per column
                 'AnsKey'           =>  Correct response
                 'MatchKey'         =>  Is most frequent response same as key
                 'PtBis'            =>  Point-biserial correlations by column
                 'anskey'           =>  The "anskey" parameter is also saved

            Workflow:
                MyScoredObj = my_obj.score_mc(...)

                ScoredArray = MyScoreObj.score_mc_out['coredata']
                PtBis = MyScoreObj.score_mc_out['PtBis']['coredata']
                RowScore = MyScoreObj.score_mc_out['RowScore']['coredata']

        Comments
        --------
            When data include alpha characters ('A','B','C',...), Damon
            allows you to parse each alpha column into a separate
            column per response category using the parse() method.
            The resulting dichotomous matrix can be analyzed, though
            the mathematical consequences of violation of the item
            independence assumption are significant and remain to be
            studied.

            Usually, however, if you know the "correct" answer it is
            desirable to score each column instead, where 1 means
            the response is correct and 0 means it is incorrect.  This
            dichotomous matrix (preferably its standardized equivalent)
            can be Damon-analyzed and is preferable to analyzing a parsed
            array since it does not violate the item independence requirement.
            score_mc() performs this scoring.

            Not all items need to be multiple-choice in order for
            score_mc() to work, nor need the correct response be alpha.
            An answer key value of None or nanval means the item will not
            be scored.  Such "unscored" columns can be included when
            calculating row and column scores and point biserial correlations.

            The score_mc() output array is a datadict whose core array
            is the same size as the input array, even if not all columns
            are multiple choice.  Non-multiple-choice columns are simply
            left unchanged.

            score_mc() (like most of the Damon methods) assumes that
            the metric of a cell is a column characteristic, not a row
            characteristic.  A row may contain values from different
            metrics (different response options, different numerical scales),
            but the cells within each column can have only one metric.

            score_mc() supports multiple "correct" answers per column.

            With the "getrows" argument, score_mc() provides the ability
            to specify a subset of rows from which statistics should be
            calculated, such as for a particular test form within a merged
            file.  In this event, each row should have a form ID of some kind.
            Regardless of how getrows is set, the entire scored array is
            output as coredata.  Only the statistics are affected.

            With the "usecols" argument, score_mc() can be used to compute
            frequencies-based statistics or scores and point biserials using
            all columns or just those that are scored using the answer key.

            With the "score_denom" argument, it is possible to specify when
            calculating scores whether to divide by the maximum possible score
            or to take into account missing scores.

        Arguments
        ---------
            "anskey" is a dictionary relating specified items to a
            "correct" response.  Note that the correct response can be
            either the raw correct response or a list containing one
            or more correct raw responses.

            Options:

                anskey = ['All',['a']] = ['All','a']
                                    =>  For each column entity, score each
                                        response as 1 that is an 'a'.

                anskey = ['All',['a','c']]
                                    =>  For each column entity, score each
                                        response as 1 that is either
                                        'a' or 'c'.

                anskey = ['Cols',{'It1':['a'],'It2':['b','d'],'It3':None,'It4':[-999],...}]
                                    =>  Score 'It1' responses as 1 where they
                                        are 'a'.  Score 'It2' responses
                                        as 1 where they are 'b' or 'd'.
                                        Do not score either 'It3' or 'It4'.

                                        'It4':-999 has the same effect as
                                        'It3':None so long as -999 is the
                                        nanval (Not-a-Number value).

                anskey = ['Cols',{'It1':'a','It2':'b','It3':None,'It4':-999,...}]
                                    =>  Here, the correct responses are not
                                        in a list, but the spec still works.

                anskey = 'my_bank.pkl'
                                    =>  The answer key parameter is brought
                                        in from the indicated item bank.
                                        
            The answer key is stored in the output datadict under the key
            "anskey" and is also stored by Damon.bank().

            Answer key
            ----------
            What is the easiest way to prepare an answer key dictionary
            for this method?  If 'All' is specified and the "correct"
            response is the same for each item, there is no need to build
            a dictionary.

            Otherwise, you can use the merge_info() method as follows.  Let's
            assume the answer key information is stored in a separate
            file and contains information about valid responses:

                # Load answer key
                ak = dmn.Damon('anskey.txt', 'textfile', 'RCD_dicts_whole',
                               nheaders4rows=1, nheaders4cols=1, delimiter='\t')

                # Load data file.  Leave validchars=None.
                d_ = dmn.Damon('mydata.txt', 'textfile', nheaders4rows=1,
                               nheaders4cols=1, delimiter='\t', validchars=None)

                # Merge answer key into data file and create a new Damon object.
                d_.merge_info(ak, 'Col', 'ValidResp')
                d = dmn.Damon(d_.merge_info_out, 'datadict', verbose=None)

                # This has the nice side-effect of transferring the validchars
                #  information from the answer key to the data object.  It also
                #  adds all the item information as rows in the column headers.

                # Build the answer key
                akey = ['Cols', dict(zip(d.cl_row['ItemID'], d.cl_row['Correct']))]

                # Score the multiple choice columns
                d.score_mc(akey, ...)
            
            The above procedure is not the only way to build the answer key,
            but it's useful when you want valid response information included
            in the Damon object. If you just need the the answer key, you can also
            do:

                akey = ['Cols', dict(zip(ak.rl_col['ItemID'], ak.core_col['Correct']))]            

            When creating artificial response data using core.create_data(),
            an answer key Damon object or file is automatically created.

            ---------------
            "report" allows you to specify a list of common reports
            to be output as datadicts within scoreMC_Out:

                report = None           =>  return_ just the scored array
                                            (0,1) as a datadict.  Non-MC
                                            items (as flagged using the
                                            answer key) are left unchanged.

                report = ['All']        =>  report all statistics.

                'RowScore'              =>  The percent correct across all
                                            or scored columns for each row (if
                                            data is dichotomous) or more
                                            generally the mean row score.

                'ColScore'              =>  The percent correct across all
                                            rows for each column (if
                                            data is dichotomous) or more
                                            generally the mean column score.

                'RowFreq'               =>  Response frequencies across all
                                            MC columns for each row.

                'ColFreq'               =>  Response frequencies across all
                                            rows for each column.

                'MostFreq'              =>  Most frequent response for each
                                            column.

                'AnsKey'                =>  "Correct" response.  If there are
                                            multiple correct responses, just
                                            the first one is reported in order
                                            to simplify building tabular reports.
                                            To retain multiple keys, use the
                                            specified anskey dictionary.

                'MatchKey'              =>  Is the most frequent response for
                                            each column the same as the "correct"
                                            response for that column?

                'PtBis'                 =>  The correlation of each column
                                            with the test as a whole, less
                                            that column.

                report = ['ColScore','ColFreq']
                                        =>  report only ColScore and ColFreq.

            To get full key access to these datadicts, convert them to Damon
            objects when you need to.

            ---------------
            "getrows" specifies the row entities or attributes that should
            be extracted from an array from which scores and frequencies
            should be compiled.  Its primary role in the score_mc() function
            is to make sure statistics are computed from one test form in
            the event that the input array is the result of merging multiple
            forms.  Its syntax is the same as specified for the extract()
            method -- >>> help(dmn.Damon.extract):

                getrows = {'Get':<'AllExcept','NoneExcept'>,
                           'Labels':<'key',int,'index'>,
                           'Rows':<KeyIDs,Attributes,Indices>
                           }

            To select rows for a given test form, the specification should
            look something like:

                getrows = {'Get':'NoneExcept','Labels':'TestForm','Rows':['Form1']}
                                        =>  This extracts just those rows/
                                            persons who took 'Form1'.

            Regardless of the getrows specification, score_mc() outputs the
            entire scored array as its main output.

            ---------------
            "usecols" specifies whether to compute statistics using all columns
            or just those scored using the answer key.  This decision is made
            separately for the frequency-based statistics ('RowFreq','ColFreq',
            'MostFreq','AnsKey','MatchKey') and the score-based statistics
            ('RowScore', 'ColScore', 'PtBis').  The syntax is:

                usecols =  {'Scores':<'All','Scored'>,
                            'Freqs':<'All','Scored'
                            }

                usecols = {'Scores':'All','Freqs':'Scored'}
                                        =>  When calculating row and column
                                            scores and point biserials, use
                                            all columns.  When calculating
                                            row and column frequencies and
                                            reporting the key and matchkey,
                                            use only those columns that were
                                            scored using the answer key.

                usecols = {'Scores':'Scored','Freqs':'All'}
                                        =>  When calculating row and column
                                            scores and point biserials, use
                                            only those columns that were scored
                                            using the answer key.  When
                                            calculating frequencies, use all
                                            columns, including those that were
                                            not scored.  This is useful, for
                                            instance, when you want frequencies
                                            for rating scale categories.  It's
                                            a bad idea if any columns contain
                                            continuous values.

            ---------------
            "score_denom" specifies how to define the denominator used to
            calculate 'RowScore' and 'ColScore' statistics -- the total
            number of observations.

                score_denom = 'All'     =>  When calculating 'RowScore', the
                                            denominator equals the total number
                                            of columns.  When calculating
                                            'ColScore', the denominator equals
                                            the total number of rows (as defined
                                            by getrows). Any missing values are
                                            counted as "incorrect".

                score_denom = 'NonMissing'
                                        =>  When calculating scores, the
                                            denominator is the total number
                                            of non-missing observations, i.e.,
                                            those not equal to -999, the value
                                            specified in score_mc's nanval argument.

            ---------------
            "nanval" is the new not-a-number value to replace the incoming
            string nanval of the Damon object.  It must be a number and is
            generally equal to float(string_nanval).  It should be consistent
            with the score_denom parameter, if applicable.

        Examples
        --------

            [under construction]

        Paste method
        ------------
            score_mc(anskey = None,    # [<None, 'bank.pkl', ['All',['A']],['Cols',{'ID1':['A'],'ID2':['C'],'ID3':[None],'ID4':[4],...}]> ]
                     report = None,   # [<None,[list reports]>  => ['All','RowScore','ColScore','RowFreq','ColFreq','MostFreq','AnsKey','MatchKey','PtBis'] ]
                     getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                     usecols = {'Scores':'All','Freqs':'Scored'},    # [<{'Scores':<'All','Scored'>,'Freqs':<'All','Scored'>} => columns to use when calculating scoring or frequency stats]
                     score_denom = 'All',    # [<'All','NonMissing'> => how to count total observations for use in denominator]
                     nanval = -999., # [Numerical Not-a-Number value to replace object nanval]
                     )

        """
        if self.verbose is True:
            print 'score_mc() is working...\n'

        # Run the damon utility
        score_mc_out = dmn.utils._score_mc(locals())
        self.score_mc_out = score_mc_out

        if self.verbose is True:
            print 'score_mc() is done -- see my_obj.score_mc_out'
            print 'Contains:\n',self.score_mc_out.keys(),'\n'

        return None


    #############################################################################

    def subscale(self,
                 data = 'data_out', # [<'data_out','merge_info_out','score_mc_out',...> => data from which to get subscales]
                 subscales = {'Get':'AllExcept','Labels':1,'Cols':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<ind,header,'key'>,'Cols':[<subscales,items>]} => desired subscales]
                 method = {'coord':{'ndim':[range(1,5)]}},    # [<{'coord':{'ndim':[range(1,5)]...}}, {'rasch':{...}}, {'mean':{'missing':<'ignore','row2nan'>}},{'filter':{'lo_hi':[lo,hi]}}> => method to calc subscale(s)]
                 rescale = None,  # [<None,{<'All','Sub1','Sub2',...>:{'straighten':<True,None,'Percentile'>,'mean_sd':[mean,sd],'m_b':[m,b]}}> => nested dict of rescale params]
                 ):
        """Build subscales and merge with data.

        Returns
        -------
            subscale() returns None but assigns the subscale_out
            datadict as a new attribute of the Damon object.  subscale_out
            consists of the input core data array, appended to which are
            mean scores computed for specified subsets of items.

            subscale_out also includes a datadict called 'subscales',
            which contains just the subscales and nothing else.

            Workflow:

            d = dmn.Damon(...)
            d.extract_valid(...)
            d.score_mc(...)
            d.subscale(...)
            d.standardize(...)
            d.objectify(...)    # Use instead of coord() in this context
            d.base_ear(...)
            d.base_se(...)

        Comments
        --------
            subscale() is used to collapse groups of items into single
            columns of scores and append these columns to the dataset
            as if they were additional "items".  This makes it possible
            to analyze subscales within coord() rather than computing
            them after the fact using summstat().

            Standard errors and other statistics are automatically computed
            for subscales when base_resid(), base_ear(), and base_se() are
            run.

            The method is only useful when all the component items of a
            subscale are positively correlated.

            Warning
            -------
            subscale() automatically assigns subscale names ('sub_MyScale')
            to collabels.  This forces the type of the column keys to be
            string, even if it was originally integer.  This may lead to
            unexpected behavior.  For instance, an attempt to extract columns
            from datadicts downstream from subscale using integers will
            fail.  To avoid this confusion, set the colkeytype of your Damon
            object to string (e.g., 'S60').

        Arguments
        ---------
            "data" is used to identify a data output datadict assigned
            to the Damon object, generally 'data_out' or 'score_mc_out' (Approach
            2 above) or 'base_est_out' (Approach 3).  It must contain numerical data.

            ------------
            "subscales" is used to specify the names of the subscales
            for which raw scores should be calculated using the extract()
            syntax.  It is also possible to define a single subscale using
            'Labels':'key' and listing items or 'Labels':'index' and listing
            column index numbers.

                subscales = {'Get':'AllExcept','Labels':1,'Cols':[None]}
                            =>  Build subscales using all the attributes
                                listed in row 1 of the column labels.

                subscales = {'Get':'NoneExcept','Labels':'Contents','Cols':['MA','ELA']}
                            =>  Build subscales using just the items labeled
                                'MA' and 'ELA' in the row of the column labels
                                labeled 'Contents'.

                subscales = {'Get':'NoneExcept','Labels':'key','Cols':['Item1','Item2']}
                            =>  Build a single subscale composed of items
                                called 'Item1' and 'Item2'.

                subscales = {'Get':'AllExcept','Labels':'key','Cols':[None]}
                            =>  Build a single scale composed of all items.

                subscales = {'Get':'NoneExcept','Labels':'index','Cols':[1,2]}
                            =>  Build a single subscale composed of items
                                corresponding to columns 1 and 2.

            ------------
            "method" is the method by which subscale scores should be calculated
            for each row.  To support complex methods, a nested dictionary syntax
            is used.

            Here are the options:

                method = {'coord':{ndim:[[2]]}}
                            =>  Calculate the subscale using the coord() method
                                with a dimensionality of 2.  To further control
                                the coord() run, specify coord() method arguments
                                in the braces (see coord() docs).  If the data
                                need to be standardized prior to running coord(),
                                you will need to do so outside of the current
                                Damon object.

                method = {'rasch':{}}
                            =>  Calculate Rasch measures (missing doesn't matter).
                                The empty braces {} mean use the default rasch() args.
                                To control the Rasch run, specify rasch() method
                                arguments in the braces (see rasch() docs).  'rasch'
                                only works if the data are dichotomous or polytomous.

                method = {'mean':{'missing':'ignore'}}
                            =>  Calculate the mean of the data in each row.  Ignore
                                missing cells.  If 'missing':'row2nan' is specified,
                                the subscale score is made missing if any cell in
                                the subscale is missing.

                method = {'filter':{'lo_hi':[-3,3]}}
                            =>  This is a somewhat experimental method that calls the
                                tools.subscale_filter() function (see docs) and should
                                by applied only if data = 'base_est_out' (after running
                                coord() and base_est()).

                                It works by defining two artificial "reference persons" who
                                are defined to differ only in terms of the subscale of
                                interest and in no other respect.  Actual persons are
                                "projected" onto the line defined by these two reference
                                persons to get the desired subscale.  In this case,
                                we define the "low" reference person to have scores
                                of -3 and the "high" reference person to have scores
                                of 3.  Other subscale_filter() arguments can be included
                                in the braces.

            ------------
            "rescale" is a nested dictionary of rescale parameters used to
            rescale the subscale row scores in any of several ways, after the
            "method" has been applied.  Any rescale parameters not specified are
            assigned a default.  For further information, see the tools.rescale()
            docs.

                rescale = None      =>  Do not rescale.  Each subscale score is
                                        as output by the specified method.

                If you specify 'All', all calculated subscales are rescaled
                using the same set of parameters:

                rescale = {'All':{'straighten':True,    =>  linearize using log((1 - rank) / rank)
                                  'mean_sd':[300,30],   =>  adjust to have Mean of 300 and SD of 30
                                  'm_b':None            =>  Do multiply by m or add b
                                  'clip':None}          =>  Do not clip values.
                           }

                Otherwise, you can set different parameters for each subscale:

                rescale = {'MA':{'straighten':None,    =>  Do not linearize
                                 'mean_sd':[300,30],   =>  adjust to have Mean of 300 and SD of 30
                                 'm_b':None            =>  Do not multiply by m or add b
                                 },
                           'ELA':{'m_b':[10,100]       =>  Multiply by 10, add 100
                                  }                    =>  Other params set at None by default
                           }

        Examples
        --------

            [Under construction]

        Paste Method
        ------------
            subscale(data = 'data_out', # [<'base_est_out','data_out','merge_info_out','score_mc_out',...> => data from which to get subscales]
                     subscales = {'Get':'AllExcept','Labels':1,'Cols':[None]}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<ind,header,'key'>,'Cols':[<subscales,items>]} => desired subscales]
                     method = {'coord':{'ndim':[range(1,5)]}},    # [<{'coord':{'ndim':[range(1,5)]...}}, {'rasch':{...}}, {'mean':{'missing':<'ignore','row2nan'>}},{'filter':{'lo_hi':[lo,hi]}}> => method to calc subscale(s)]
                     rescale = None,  # [<None,{<'All','Sub1','Sub2',...>:{'straighten':<True,None,'Percentile'>,'mean_sd':[mean,sd],'m_b':[m,b]}}> => nested dict of rescale params]
                     )

        """
        if self.verbose is True:
            print 'subscale() is working...\n'

        # Run the damon utility
        subscale_out = dmn.utils._subscale(locals())
        self.subscale_out = subscale_out

        if self.verbose is True:
            print 'subscale() is done -- see my_obj.subscale_out'
            print 'Contains:\n',self.subscale_out.keys(),'\n'

        return None





    #############################################################################

    def parse(self,
              items2parse = ['AllExcept',[None]],   # [<['AllExcept',['I1','I2']], ['AllExcept','continuous'], ['NoneExcept',['I1','I2']]>]
              resp_cat = 'Find',   # [<'Auto', 'Find', ['All',[valid chars]], ['Cols', {'ID1':['a','b'],'ID2':['All'],'ID3':[1,2,3,4]...}]> ]
              extractkey = None,    # [<None,['All',['A']],['Cols',{'ID1':['A'],'ID2':['C'],'ID3':[None],...}]> ]
              ordinal = None,  # [<None, True => when responses are numeric, treat as ordinal rather than nominal>]
              nanval = -999.,    # [nanval to which incoming nanval's should be converted in parsed array]
              parse_params = None    # [<None,'MyBank.pkl'> => parse() params in bank are used]
              ):
        """Return parsed categorical response data as dichotomous values.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The parse() outputs are accessed using:

                my_dmnobj.parse_out

            parse_out is a datadict in which categorical response data
            has been parsed into an array of dichotomous values.

            Added to the datadict are:

                'MethDict'      =>  described below, the method by which
                                    parsed columns will eventually be
                                    recombined.

                'OrdDict'       =>  a dictionary describing for each
                                    original entity whether its response
                                    options are ordinal (True) or
                                    not (False).

                'KeyDict'       =>  a {ParseKey:EntKey} lookup dictionary
                                    relating the column keys for the parsed
                                    array to the keys of the column entities
                                    in the original data array.

                'EntRow'        =>  the row in the parsed array collabels
                                    containing the original entity IDs.
                                    Counting starts at 1 instead of 0,
                                    like Key4Col.

                'RespRow'       =>  the row in the parsed array collabels
                                    containing the possible responses.
                                    Counting starts at 1.

                'parse_params'   =>  passes through the parameters used in
                                    running parse().  These are used by
                                    bank() and fin_est().

            If pseudomiss() was run, parse() updates its dictionary
            of pseudo-missing cells with parsed array equivalents.

            Workflow:
                my_dmnobj.parse(...)

            MethDict
            --------
            In addition to the usual Damon attributes, parse() creates
            an attribute called 'MethDict' (methods dictionary) which
            captures information for other Damon methods (mainly
            fin_est()) regarding how the parsed columns should be
            recombined, if at all:

            {'It1':['Extr',['a','c']],'It2':['Exp'],'It3':[None],'It4':['Pred'],...}

            'Extr' means "extract" and instructs fin_est() to
            report only the column corresponding to one of the nominal
            response options associated with a given item as specified
            by extractkey.

            'Exp' means "get the expected value" found by multiplying
            each ordinal response category of a given item by its
            probability and summing the products.

            'Pred' means "predict the most likely response", i.e.,
            return the response option that has the highest
            probability.

            'None' (without quotes) means the column was not
            parsed and should be read in as is.

            The decision about which "meth" to assign to an entity is
            made internally by the program based on the type of characters
            in the column (alpha or numerical), the extractkey (which
            columns to extract), and the ordinal specification (compute
            an expected value or a predicted value).  If you enter
            the parameters that make sense, the MethDict should just
            take care of itself and, later when running fin_est(), return
            the right kind of estimates.  But, to be explicit, here
            are the decision rules on a per column basis:

                1.  If extractkey is not None, MethDict for that
                    column is 'Extr'.

                    Otherwise:

                2.  If ordinal is True and the response characters
                    for the column ARE integers, MethDict codes 'Exp'.

                3.  If ordinal is True and the response characters
                    for the column are NOT integers, MethDict codes 'Pred'.

                4.  If ordinal is NOT True and extractkey is None,
                    MethDict codes 'Pred'.

            Note that MethDict stores information about the
            column entities using their unique IDs
            BEFORE parsing.  This information is used later by
            fin_est() to build an array of estimates corresponding
            to the original data array.

        Comments
        --------
            [Note:  You probably shouldn't be using parse().  The 
            mathematical consequences of analyzing parsed data are not 
            well understood and you may end up with unsound results. 
            In addition, running parse() introduces a lot of complications 
            internally and it may not work in all scenarios.  To deal 
            with nominal categorical data, either use score_mc() or 
            make each category its own item if you can do so without 
            compromising local item independence.]

            parse() converts nominal or ordinal categorical data,
            alpha or numeric, into a form analyzable by coord().  It
            does this by converting non-numerical values into extra
            column entities, which are then coded as 0 or 1.  Non-
            numerical data become numerical.

            Say the Item 1 column contains response data values 'a',
            'b', and 'c'.  parse() breaks the Item 1 column into
            three columns ('1_a','1_b','1_c'), each of which records a 1 if
            the original column contains that response category, otherwise
            a zero (or nanval if missing).  So if the first entry under Item
            1 is an 'a', the value under '1_a' becomes a 1 while '1_b' and
            '1_c' record zeros.  This is the scenario under which "nominal"
            data are parsed.

            Here is another scenario.  Item 2 contains response data values
            '1', '2', and '3' and the ordinal argument is set at True.  Again,
            the column is broken into three columns ('2_0','2_1','2_2'), but
            this time each column records a 1 if the value in the original
            column is less than or equal to the response category.  So if
            the first entry under Item 2 is a '1', then the value for '2_0'
            becomes a 1, the value for '2_1' becomes a 1, but the value for
            '2_2' becomes a zero.  This tells coord() that because the person
            scored a '2' it can be assumed he would have scored a '1'
            as well (the scores being ordinal), but not a '3'.  A salmon that
            jumps to the third rung of a fish ladder can safely be assumed to
            have jumped to the second.

            This is the scenario under which "ordinal" data are parsed, but the
            description is not complete.  parse() will automatically DELETE the
            first parsed column, the one called '2_0'.  Why? Because following
            the logic stated above, the lowest category will always
            consist entirely of 1's.  Such columns cannot be analyzed (they
            serve no mathematical function, anyway), so the lowest response
            category is dropped when the data are ordinal.

            [Note: the following paragraph is on hold pending re-engineering
            of the est2logit() method.]

            The dichotomous data output by parse() is generally standardized
            in the "PreLogit" metric before going into coord(). When the
            response categories are ordinal rating scale values, the resulting
            coord() and base_est() estimates are logits convertible into
            the probability of meeting or exceeding a given category.  When
            the response categories are nominal values ('a','b',...), the
            logit estimates are convertible into the probability of obtaining
            the given response.  It is possible to use these probabilities
            to get an "expected" rating which makes fewer metric assumptions
            than if the rating scale data were analyzed directly.  It is also
            possible to make prediction regarding which category is most likely.

            By converting nominal data into dichotomous data, parse() makes it
            possible to analyze virtually any kind of dataset whose values can
            be broken into categories. However, it is important to remember
            that forced-choice responses mean that the resulting parsed columns
            are not statistically independent, causing the standard error of
            each estimate to be artificially low.  This is controlled using
            the 'nIndepItem=' argument in coord().

            IMPORTANT NOTE:  parse() is only applied to the original complete
            data array, not the data with pseudomissing cells made missing
            (pseudomiss output).  However, if pseudomiss() was run, it updates
            the pseudo-missing indices with their parsed array equivalents.
            (See pseudomiss() docs.)

        Arguments
        ---------
            "items2parse" specifies which items to parse.  The options are:

            items2parse =
                ['AllExcept',[None]]    =>  parse all items (including
                                            RowLabel cols)

                ['AllExcept','continuous']
                                        =>  parse all items that do not
                                            contain continuous data, i.e.,
                                            where validchars does not specify
                                            a range of ['All'] or include
                                            a dash ('1 -- 10').

                ['AllExcept',['I1','I2']]
                                        => parse all items except 'I1' and 'I2'

                ['NoneExcept'],['I1','I2']]
                                        =>  parse none of the items except 'I1'
                                            and 'I2'.
                
                Note: parse() will automatically cast integer item identifiers
                to string.
                
            ---------------
            "resp_cat" specifies valid response categories (labels).  When 'Auto',
            it refers directly to the validchars output from data(), or it can
            list specific responses that are valid across the whole array, or it
            can list responses that are valid for individual columns.  It can
            also allow the program to draw a list of valid responses from the
            data ('Find').  The options are:

            resp_cat =
                'Auto'              =>  Response categories will be captured
                                        automatically from the validchars
                                        specification of the Damon object.

                'Find'              =>  Find unique responses in each column
                                        and define them as valid response
                                        categories for that column.

                ['All',['A','B','C']]
                                    =>  Make the response categories consist
                                        of 'A', 'B', and 'C' for all items
                                        specified in items2parse.

                ['Cols',{'I1':['a','b'],'I2':['All'],'I3':[1,2,3,4],...}]
                                    =>  For Item 1, valid response are 'a'
                                        and 'b'.

                                        For Item 2, all responses
                                        are valid, which is interpreted to mean
                                        that the data are continuous and not
                                        categorical.  This will cause parse()
                                        to throw an error, unless items2parse
                                        is ['AllExcept','Continuous'].

                                        The same thing happens with
                                        'I2':['1.2 -- 3.0']; the dash
                                        means continuous.

                                        NOTE: To prevent errors, make sure
                                        the items2parse parameter does not
                                        contain items with continuous data.

                                        For Item 3, valid responses are the
                                        integers 1, 2, 3, and 4, which can be
                                        in string or numerical format.  If
                                        ordinal = 1, these are treated as ordinal.
                                        Otherwise, they are treated as nominal
                                        labels, with no order.


            ---------------
            "extractkey" is used in two ways:  1) it tells the function
            which responses should be "extracted" for those items coded 'Extr'
            in the MethDict output; 2) when resp_cat = 'Auto' it ensures
            that the extracted response is not inadvertently omitted should
            the response not occur in the actual data.  Options:

                extractkey = None   =>  No individual responses are flagged
                                        for extraction from any of the column
                                        entities.

                extractkey = ['All',['a']]
                                    =>  For each column entity, extract the
                                        probability of an 'a' response.  If an
                                        entity does not have an 'a', code that
                                        entity as None.

                extractkey = ['All',['a','b']]
                                    =>  For each column entity, extract the
                                        probabilities of 'a' and 'b'.

                extractkey = ['Cols',{'It1':['a'],'It2':['b','d'],'It3':None,'It4':['c'],...}]
                                    =>  For each column entity, extract the probability
                                        of response 'a' for Item1, of responses
                                        'b' and 'd' for Item2, of response
                                        'c' for Item4.

                                        For Item3, do not use the Extract
                                        method.  'It3':-999 , where -999 is
                                        the defined nanval, has the same
                                        effect as 'It3':None .

            When creating nominal data using core.create_data(), an answer
            key object is automatically created and formatted for use in this
            method.  To use this answer key in parse(), the specification
            should look something like:

                Predefine:
                    AnsKeyObj = core.create_data(...)['anskey']
                    Keys = dmnt.getkeys(AnsKeyObj.data_out,'Row','Core')
                    Vals = AnsKeyObj.core_col['Correct']

                Specify:
                    extractkey = ['Cols',dict(zip(Keys,Vals))]

            Note that parse() doesn't actually do anything with extractkey
            specifications except store them in the the MethDict
            (method dictionary) output for use by other data methods.
            It is used mainly by fin_est().

            MethDict provides information needed to convert estimates
            from a parsed format back into a format compatible with the
            pre-parsed array.  These conversion functions are: 'Extr'
            (extract probability of a response), 'Exp' (calculate an
            expected value), and 'Pred' (predict the most likely value).
            None means none of the parse()/deparse() conversion functions
            should be applied.

            Example: you want to apply an answer key to multiple choice data
            to obtain the probability of the "correct" response.  extractkey
            is where you would enter the answer key.

            ---------------
            "ordinal = True" instructs parse() to try to interpret the responses in
            each column as ordinal if possible.  If it encounters letter responses,
            it treats them as nominal.  If it encounters numbers (floats
            are rounded to integers), it treats them as ordinal -- i.e., as having
            a specific order.  If the valid responses are [1,2,3,4] item X will be
            parsed into four response columns -- 'X_1','X_2','X_3','X_4'.  If a
            person's response is 3, the resulting data for that person for that
            set of four columns will be [1,1,1,0].  If ordinal = None, the resulting
            data will be [0,0,1,0].

            ---------------
            "nanval" is important in this context.  Since categorical
            data is often alphabetical, incoming NaNVals may not be numerical, and
            may be quite different from the output NaNVals.  Parse() obtains the
            incoming nanval from the Damon dictionary.  When creating the
            new dichotomous matrix, it replaces the incoming nanval (generally
            string) with the new nanval specified by this argument, which needs
            to be numerical.

            ---------------
            "parse_params" is a set of parameters for running parse() that
            overwrites all other arguments.  It is used for automatically passing
            parameters to parse(), especially when they are contained in
            a "bank" pickle file.  A bank (see bank() docs) contains a Python
            dictionary for looking up entity coordinates.  If an item has
            been parsed, coordinates are assigned to the parsed item IDs,
            not the regular item IDs.  That means, when anchoring a new
            dataset to an old one, the parameters used to parse (and
            standardize) the original dataset must be applied to the
            new dataset as well.  That is why the parse() and standardize()
            parameters are saved in the bank, as well as various
            coordinates.  They are accessed using the parse_params parameter
            (and StdParam parameter in standardize()).  Options:

                parse_params = None  =>  Do not apply parse() parameters
                                        from a previous dataset.

                parse_params = 'MyBank.pkl'
                                    =>  Overwrite all parse() parameters
                                        with those contained in the bank
                                        pickle file called 'MyBank.pkl'.
                                        (All existing parameters/defaults
                                        are ignored.)

            NOTE:  If you keep getting weird errors when the parse_params
            option is used, it may be because you are reading an
            obsolete bank file.  Go back to the original dataset and
            rebuild the bank.

        Examples
        --------



        Paste method
        ------------
            parse(items2parse = ['AllExcept',[None]],   # [<['AllExcept',['I1','I2']], ['AllExcept','continuous'], ['NoneExcept',['I1','I2']]>]
                  resp_cat = 'Find',   # [<'Auto', 'Find' ['All',[valid chars]], ['Cols', {'ID1':['a','b'],'ID2':['All'],'ID3':[1,2,3,4]...}]> ]
                  extractkey = None,    # [<None,['All',['A']],['Cols',{'ID1':['A'],'ID2':['C'],'ID3':[None],...}]> ]
                  ordinal = None,  # [<None, True => when responses are numeric, treat as ordinal rather than nominal>]
                  nanval = -999.,    # [nanval to which incoming nanval's should be converted in parsed array]
                  parse_params = None    # [<None,'MyBank.pkl'> => parse() params in bank are used]
                  )

            """
            
        if self.verbose is True:
            print 'parse() is working...\n'
            print ("\nWarning:  You probably shouldn't be using parse().  The "
                   "mathematical consequences of analyzing parsed data are "
                   "not well understood and you may end up with unsound results. "
                   "In addition, running parse() introduces a lot of complications "
                   "internally and it may not work in all scenarios.  To deal "
                   "with nominal categorical data either use score_mc(), or "
                   "make each category its own item, if you can do so "
                   "without compromising local item independence.")

        # Run the damon utility
        parse_out = dmn.utils._parse(locals())
        self.parse_out = parse_out

        if self.verbose is True:
            print 'parse() is done -- see my_obj.parse_out'
            print 'Contains:\n',self.parse_out.keys(),'\n'

        return None




    ##################################################################################################
    def standardize(self,
                    metric = 'PreLogit',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                    referto = 'Cols',   # [<None,'Whole','Cols'>]
                    rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                    std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                    add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                    ):
        """Standardize data to a specified metric.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon object.  The standardize() outputs are accessed using:

                my_dmnobj.standardize_out

            standardize_out is a datadict whose coredata array is a
            standardized version of data in the current Damon object.
            The datadict includes a 'std_params' key which stores
            the parameters used to do the standardization.  This
            information is used to do banking (so that data from
            a subsequent analysis can be prepared for coord()
            in the same way as for the intial analysis).  It is
            also used by the fin_est() method.

            Workflow:
                my_dmnobj.standardize(...)

                This standardizes the Damon object.

        Comments
        --------
            Matrix decomposition requires that all cells be
            in the same metric -- a requirement of a common space.
            Preconditioning the data by standardizing is a common and
            useful way to deal with data whose columns may be in
            different metrics, but it does introduce a lot of
            options and complications.  One of these complications
            is that the resulting coordinate system is in a different
            metric than the original observations.  This becomes an
            issue when we want to equate two tests (see the docs for
            coord() and bank()).  We equate the data in one test
            by anchoring to the coordinates of another.  How
            does this work when the data and the coordinates are not
            in the same metric?  The solution is to standardize the
            data in the second test using the standardization parameters
            generated for the first test.  These standardization parameters
            include statistics like mean, standardization, minimum,
            maximum, etc.  Once the raw data from the second test
            is standardized in the same way as the data from the
            first test, we can apply the coordinates and do anchoring
            and equating.  All perfectly doable, but a little
            complicated.  Therefore, whenever a standardization is
            performed, the standardization parameters are saved off
            as part so that they can be referenced later, especially by
            fin_est() and bank().

            Whenever making predictions based on standardized values,
            test which metric works best in reproducing the original
            values.  Each has its strengths and weaknesses; none is
            perfect.  The most appropriate metric for one sample may not
            be the most appropriate for another sample, reinforcing the
            fact that standardization is sample-dependent and lacks to a
            certain degree the sample-independent properties of Damon
            when the data are all in the same metric and generated the same
            way.

            standardize() does not report row or column statistics, just
            standardized values for each cell.  These may be calculated
            from just the cell value itself or from the cell value in
            relation to the rest of the values in that row, column, or
            array.

            Sometimes, incoming columns are on a "ratio" scale such
            as "counts", which range from 0 to infinity.  Interval scales,
            by contrast, are symmetric and range from -infinity to +infinity.
            ordinal (integer) and sigmoid (continuous) scales have a finite
            minimum and maximum.  Rating scales (0,1,2,3) are examples of
            ordinal metrics.  Percentages are an example of a sigmoid
            metric.

            The ALS decomposition, as embodied in the coord() method,
            mathematically requires all input data to be in the same metric.
            Therefore, when the column metrics vary, it is first necessary
            to standardize them.  That's what standardize() does.  It offers
            a variety of standardization metrics from which to choose,
            described below.

            When the data are already in a common metric, are interval
            (proceed continuously on a scale that theoretically stretches
            from negative to positive infinity), and when the errors (the
            residuals between the estimates and observations) can be assumed
            to be homoscedastic (the same size at all locations of the scale),
            standardize() is not necessary and you can run Damon fairly
            blindly.  The results will be well-behaved and easy to interpret.

            When these conditions are not met and you need to standardize
            or parse the data to make it digestible, it is important
            not to approach the analysis slavishly.  Experiment with different
            metrics and different dimensionalities, look for non-linearities,
            graph observations against estimates, check that the standard errors
            make sense, check predictive validity.  Heterogeneous datasets are
            challenging and need to be approached with eyes open.

            What is the "best" standardization metric?  The most general
            and the one that best preserves the objectivity properties
            of Damon is definitely 'PreLogit'.  The 'PreLogit' metric was
            designed to convert a mix of data types -- dichotomous, polytomous,
            interval, ratio -- into a common well-behaved interval metric.
            You can safely use it as the default for pretty much any dataset.
            
            Nonetheless, each standardization metric has its strengths
            and weaknesses depending on your data.  In general, the
            'interval' metrics ('SD','LogDat','PreLogit','PLogit') are
            to be preferred, as the ordinary least squares engine at the
            center of Damon assumes interval data.  But the 0-to-1 metrics
            may be useful for problems where the emphasis is on prediction
            rather than measurement.

            As a rule of thumb, 'PreLogit' and 'PLogit' can standardize
            any type of incoming (numerical) data, but PLogit pays a price in
            being sample dependent and hard to convert back to the original
            metric.

            'PMinMax' works reasonably with ordinal and sigmoid
            data where there is a hard minimum and maximum.  '0-1'
            is like 'PMinMax' but automatically handles all metrics. 'SD'
            works best with interval data, poorly with ratio data and ordinal
            or sigmoid data.

            However, as mentioned, 'PreLogit' handles all the metrics, 
            approximates an interval logit scale, and is easy to work with.  
            Techically, estimates calculated from PreLogits are not "true"
            logits -- they mimic logits but are not based on probabilities.
            That's why down-stream methods like est2logit() and equate()
            provide options for converting PreLogit estimates to true logits
            by taking into account standard errors.  In practice, however,
            the Prelogit estimates are not that different from true logits
            and can be used as if they really are true logits.  The main
            obvious difference is that true logits yield smoothed binomial 
            standard errors that are more in line with the outputs of most
            IRT models.

            When 'PreLogit' or '0-1' is specified, standardize() 
            automatically adjusts for the incoming metric of each 
            column separately.  For other standardization metrics, the 
            user needs to select the metric that is appropriate
            for the whole array.
            
            For example, when the incoming column is on a "ratio" scale, 
            such as with counts, its data is automatically converted to an 
            interval scale by taking its log, then converting to an interval
            z-score, then converting to a logit-like scale. Thus, "count" 
            is automatically converted to "log(count)" before going into
            the specified metric.  This has the effect of converting what is
            essentially a multiplicative scale to an additive one and
            causes it to range from -infinity (when count = 0) to +infinity
            (when count is +infinity).  standardize() determines whether
            the data are on a ratio scale by evaluating the validchars
            specification attached to the incoming dataset.  When validchars
            contains a '0.0 -- ' specification, it is assumed the data in that
            column are on a ratio scale.  In such cases, the minimum value
            has to be zero; a negative number does not exist for a ratio scale.
            For example,

                MyDataRCD['validchars'] = [Cols,{'ID1':['0 -- '],'ID2':['0.0 -- ']}]
                                    =>  the columns for 'ID1' and 'ID2' data are
                                        on a ratio scale and will be put on a log
                                        scale before being standardized.

            As in this example with ratio scales, standardize() relies 
            heavily on the Damon.validchars attribute to interpret the data, 
            so make sure to specify validchars correctly.

        Arguments
        ---------
            "metric" specifies what standardized metric the current
            dataset should be converted to.  The most appropriate
            for use by the coord() method are those that produce
            interval or quasi-interval metrics (without a hard floor or
            ceiling):  'SD','LogDat','Logit', and 'PLogit'.  The remaining
            metrics all produce numbers from 0.0 to 1.0 on a sigmoid
            scale (hard floor and ceiling).  These can also be run
            through coord(), though the resulting estimates will tend to
            have a nonlinear relationship to the underlying latent
            variable.  These may be fine for producing cell PREDICTIONS
            (predicting a 0 or 1), as opposed to cell MEASURES
            (interval values that can be added or subtracted).

            There is also the option to apply parameters from another
            data object using 'std_params'.

            Here are the options:

                None means the standardization parameters are
                to be taken from the std_params argument (described below).

                'SD' standardizes each cell according to the mean
                and standard deviation of the range (the whole array or
                individual columns, depending on referto).  It assumes the data
                are continuous on an interval scale.

                'LogDat' takes the log of the value in each cell.  It
                is useful for ratio or count data where values below
                zero are not allowed.  The resulting metric runs from
                negative to positive infinity, where negative numbers
                correspond to ratio values between 0 and 1.

                'PreLogit' converts any kind of data into logits or a
                reasonable facsimile thereof.  It is called 'PreLogit'
                to distinguish it from the 'logit' reporting option
                in est2logit(), which represents a logit form of the
                probability of exceeding some defined threshold.  'PreLogit',
                on the other hand, does not correspond to a probability
                (though it could) but rather to any number constrained
                to be between 0 and 1 that follows a sigmoid metric.
                PreLogit is meant to be used as an all-purpose standardization
                function for arrays which contain a mix of ordinal, sigmoid,
                interval, and ratio data.  The formula, roughly, is:

                    Logit[ni] = log(P[ni] / (1 - P[ni])

                where P[ni] is the "probability" of a given cell for row n
                and column i, though it would be more accurate to call it
                simply a number that ranges from 0.0 to 1.0 in a nonlinear
                (sigmoid) fashion.

                logits are good for use in Damon because they are linear
                and behave well with the ordinary least squares procedure
                at the heart of the coord() method.  Running 0 - 1 numbers
                through coord() creates a sigmoid relationship between
                the estimates and the true model values.

                The 'PreLogit' option handles each column metric differently.
                Interval data are standardized by subtracting the mean and
                dividing by the standard deviation, then multiplied by
                pi * sqrt(3) = 1.8138, a factor used to convert standard
                deviations to logits.  Ratio data are similarly standardized,
                but only after being linearized by taking the log.  Ordinal
                data (dichotomous and polytomous integers with an upper and
                lower bound) are handled by being compressed between 0 and 1
                such that the upper and lower extremes are not 0 or 1 but
                the midpoints of the top and bottom categories, and then
                converted into logits using the formula.  Dichotomous
                data is transformed from [0,1] to [0.25,0.75], and from there
                to logit values [-1.1,1.1].  The compression of categories is
                necessary because the logit formula can't handle 1.0 or 0.0.

                It should be noted that ordinal data presents several challenging
                mathematical issues, related especially to their discreteness.
                This simple standardization procedure does not solve them,
                but it helps.

                'PLogit' converts percentiles (see below) to a linear scale using
                the logit formula.  Whenever 'PLogit' is specified, the function
                first calculates percentiles.  If for some reason all
                the cells in a range have the same value, the percentile value
                is set to 0.50 (all entities have in effect performed equally)
                and the PLogit is set to 0.0.  'PLogit' differs from 'PreLogit'
                in that it borrows information from the distribution of values
                in each column.  Say the values are all 0 and 1.  The formula
                is:

                    PLogit[x = 1] = log (Count0 + 0.5 * Count1) / (0.5 * Count1)
                    PLogit[x = 0] = log (0.5 * Count0) / (Count1 + 0.5 * Count0)

                where Count0 and Count1 are the count of zeros and ones in
                the column and x is the cell value.  Basically, the numerator
                is an estimate of the number of "persons" scoring less than x
                while the denominator estimates the number scoring greater
                than x.  The corresponding 'PreLogit' values would always be
                1.1 and -1.1, regardless of the counts.

                When data are dichotomous, the PLogit formula underestimates
                the true variance of the logits and their corresponding
                probabilities.  To correct for this, the fin_est() function
                assumes in this case that the minimum and maximum "true" probability
                are 0.0 and 1.0, and forces the PLogits to fill this range.  This
                works fine except when the true probabilities do not span the
                full range, in which case their variance is overestimated.

                One advantage of PLogits over PreLogits is that they are robust
                to data arrays where the raw metric is highly nonlinear and hard to
                work with, so long as each higher number means "more" of the
                underlying variable (are 'monotonic').  By reducing all values
                to a rank order, a lot of metric weirdness is removed.

                The chief disadvantage of PLogits is that the price of its
                flexibility is loss of sample independence; results may differ
                across testing samples, and may degrade if there's too
                much missing data.  Another disadvantage is that by making
                no assumptions about the incoming metric (except for monotonicity),
                it may be hard to recover the original metric.

                PLogits can be applied to any numeric metric,
                continuous (2.34, 3,7), polytomous (2,3,4), or
                dichotomous (0,1).  They can be defined using the column
                or the whole array, as specified by referto.

                '0-1' is specified to convert all array values, regardless
                of whether they belong to ratio, interval, sigmoid, or
                ordinal scales, into a single sigmoid scale that goes from
                zero to one.  It is similar to 'PreLogit', but without the
                log function and without the category compression.

                'Percentile' is the percentile rank of the cell
                value relative to the other values in the array or column
                range and is basically the same as 'PLogit', without the
                conversion to a log metric.

                'PMinMax' is the cell's proportional distance between
                the range minimum and maximum and is expressed as a number
                from zero to one.  It is appropriate for ordinal data, less
                so for interval data, and poor with ratio data since it can lead
                to nonlinearities.

            IMPORTANT:  When data are ordinal, it is assumed that the raw
            data are expressed as unit integers, e.g. 0,1 or 1,2,3,4.  The
            categories should not be more than one integer apart.

            -------------------
            "referto=" controls whether standardized values are referenced
            to column statistics (referto = 'Cols') or to statistics drawn from
            the whole array (referto = 'Whole').  For instance, when
            converting all values to standard deviation units, 'Cols' tells
            the method to subtract from each value its column mean and
            divide by the column standard deviation.  'Whole' tells
            it to subtract from each value the mean of the whole array and
            divide by the standard deviation of the whole array.  When
            referto = None, the parameter is drawn from the std_params
            argument.

            -------------------
            "rescale" multiplies the standardized values for each column
            by m and adds b.  This makes it possible to return standardized
            values in a user-friendly metric on a per-column basis.  It
            is only used if 'metric' is 'SD', 'PreLogit', 'PLogit', or
            'LogDat'.   Options:

            rescale =       =>  if 'referto' == 'Whole':
                                    {'All':[m,b]} (m = multiplier, b = intersept)

                                elif 'referto' == 'Cols':
                                    {'It1':[m1,b1],'It2':[m2,b2],...}

                                else:
                                    None

            -------------------
            "std_params" is a dictionary of standardization parameters
            calculated from another Damon object -- usually stored in a bank.
            The std_params dictionary is created automatically.  When
            it is not None, it essentially overwrites the remaining
            standardize() parameters.

            std_params also accepts entity coordinate banks (see
            bank()) -- just enter the name of the Python pickle file
            containing the bank and standardize() will figure out the
            std_params.

            NOTE:  If you keep getting weird errors when the std_params
            option is used, it may be because you are reading an
            obsolete bank file.  Go back to the original dataset and
            rebuild the bank.

            The std_params dictionary has several uses:

                1.  As an argument to the standardize() method, it
                    standardizes the data in Damon in a way that
                    is consistent with the standardization of another
                    data set, essentially over-writing the relevant
                    arguments in the standardize() expression.  This is
                    important in Damon anchoring designs where we
                    are trying to get the estimates for specified
                    entities onto the same metric.

                2.  Alternatively, std_params can be used in fin_est()
                    to convert standardized data to any target metric when
                    there is no original data set.  It is used in this way
                    by the core.create_data() function.

            When 'stdmetric' is 'Percentile' or 'PLogit', std_params['params']
            consists of the whole original dataset (needed to get
            percentiles from two datasets into the same metric). When figuring
            out the original metric ('ratio', 'interval', etc.), standardize()
            ordinarily refers to the validchars specification of the current
            data set, not that of the previous data set (as presumably
            the two are consistent) UNLESS the validchars option in std_params
            is exercised.  In that case, the information in std_params['validchars']
            takes precedence.  Options:

                std_params = None    =>  Do not apply standardize() parameters
                                        from another dataset.

                std_params = 'MyBank.pkl'
                                    =>  The relevant standardize() parameters
                                        will be pulled automatically from the
                                        'MyBank.pkl' pickle file, containing
                                        coordinates and other information from
                                        a prior analysis (see bank() docs).

                std_params = {...}
                                    =>  See options below.  Note that std_params
                                        is not simply a repeat of the standardize()
                                        arguments.  It contains a lot of extra
                                        information calculated internally by
                                        standardize().

            Here are the parameters:

            {'stdmetric':   =>  <'SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>

            'validchars':   =>  <None, the validchars specification from the original
                                data set or for any target dataset>

            'referto':      =>  <'Cols','Whole'>

            'params':       =>  if 'stdmetric' == 'SD':
                                    if 'referto' == 'Whole':
                                        {'All':[ArrayMean,ArraySD]}
                                    elif 'referto' == 'Cols':
                                        {'It1':[Mean1,SD1],'It2':[Mean2,SD2],...}

                            =>  elif 'stdmetric' == 'LogDat':
                                    None (no parameters are necessary; data are log ratios/counts)

                            =>  elif 'stdmetric' == 'PreLogit':
                                    if 'referto' == 'Whole':
                                        {'All':<'VCMinMax' or [ArrayMean,ArraySD]>}
                                    elif 'referto' == 'Cols':
                                        {'It1':'VCMinMax, ...} or {'It1':[col_mean, col_sd]}

            Note:  The 'PreLogit' metric can standardize datasets that are a
            mix of interval/ratio data and ordinal/sigmoid data,
            so it can also destandardize back to the same
            metrics using std_params.  To destandardize back to interval/
            ratio data, it allows you to specify a desired mean and
            standard deviation.  To destandardize back to ordinal/sigmoid
            data, it allows you to specify a desired minimum and maximum.
            However, the specification of minimum and maximum is not done
            under the 'params' key, but under the 'validchars' key, as
            validchars contains the information necessary to infer a
            minimum and maximum.  Therefore, when filling out the 'params'
            values, specify a mean and standard deviation for each entity
            that is originally in an interval/ratio metric.  Specify 'VCMinMax'
            (validchars minimum and maximum) for each entity that is originally
            in an ordinal/sigmoid metric.

            It is important to note that the decision to treat a given
            entity as interval/ratio or as ordinal/sigmoid is based
            entirely on information in the validchars parameter.  If
            std_params['params'] specifies a mean and standard deviation for
            an entity that is actually ordinal/sigmoid (based on whether it
            has a bounded range in std_params['validchars']), it will over-write
            the mean and standard deviation with the appropriate min and max.
            If std_params['params'] specifies 'VCMinMax' for an entity that
            is actually ratio/interval, it will arbitrarily impose a mean of
            0 and a standard deviation of 1.
                    {'It1':'VCMinMax','It2':[ColMean,ColSD],...}

                            =>  elif 'stdmetric' == 'PLogit':
                                    if 'referto' == 'Whole' or 'referto' = 'Cols':
                                        None (no parameters are necessary; data are probabilities)

                            =>  elif 'stdmetric' == '0-1':
                                    if 'referto' == 'Whole':
                                        {'All':<'VCMinMax' or [ArrayMean,ArraySD]>}
                                    elif 'referto' == 'Cols':
                                        {'It1':'VCMinMax','It2':[ColMean,ColSD],...}

                                Note:  The '0-1' metric, like the 'PreLogit' metric can
                                standardize datasets that are a mix of interval/ratio data
                                and ordinal/sigmoid data.  See the 'PreLogit' note for
                                details.

                            =>  elif 'stdmetric' == 'Percentile' or 'PLogit':
                                    if 'referto' == 'Whole' or 'referto' = 'Cols':
                                        whole original dataset as datadict

                            =>  elif 'stdmetric' == 'PMinMax':
                                    if 'referto' == 'Whole':
                                        {'All':[ArrayMin,ArrayMax]}
                                    elif 'referto' == 'Cols':
                                        {'It1':[ColMin,ColMax],'It2':[ColMin,ColMax],...}


            'rescale':      =>  if 'stdmetric' == 'SD','PreLogit','PLogit' or 'LogDat':
                                if 'referto' == 'Whole':
                                    {'All':[m,b]} (m = multiplier, b = intersept)
                                elif 'referto' == 'Cols':
                                    {'It1':[m1,b1],[m2,b2],...}
                                else:
                                    None

            'orig_data':     =>  orig_data':datadict
                                   (previous data set is stored under 'orig_data' as
                                    a 'datadict')
                                else:
                                    None
            }

            -------------------
            "add_datadict" specifies, additionally, whether to add the current
            data set as a 'datadict' to the std_params dictionary under the
            'orig_data' key.  This means the current datadict can be transmitted
            to other data objects through the std_params argument under two keys,
            params and orig_data.  (Since the datadict is the same object in
            two places, this does not increase the size of the std_params
            dictionary.)


        Examples
        --------

            [under construction]

        Paste method
        ------------
            standardize(metric = 'PreLogit',   # [<None,'std_params','SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>]
                        referto = 'Cols',   # [<None,'Whole','Cols'>]
                        rescale = None,   # [<None,{'All':[m,b]},{'It1':[m1,b1],'It2':[m2,b2],...}>]
                        std_params = None,   # [<None, 'MyBank.pkl', {'stdmetric','validchars','referto','params','rescale','orig_data'}>]
                        add_datadict = None,  # [<None, True> => store current datadict in std_params as 'orig_data':]
                        )
        """
        if self.verbose is True:
            print 'standardize() is working...\n'

        # Run utility
        standardize_out = dmn.utils._standardize(locals())
        self.standardize_out = standardize_out

        if self.verbose is True:
            print 'standardize() is done -- see my_obj.standardize_out'
            print 'Contains:\n',self.standardize_out.keys(),'\n'

        return None




    ##################################################################################################

    def rasch(self,
              groups = None,    # [<None, {'row':int row of group labels}, ['key', {'group0':['i1', i2'],...}], ['index', {'group0':[0, 1],...}]> => identify groups]
              anchors = None,   # [<None, {'Bank':<pickle file>, 'row_ents':[<None,'All',row entity list>], 'col_ents':[<None,'All',col entity list>]}> ]
              runspecs = [0.0001,20],  # [<[stop_when_change, max_iteration]> => iteration stopping conditions ]
              minvar = 0.001,  # [<decimal> => minimum row/col variance allowed during iteration]
              maxchange = 10,  # [<+num> => maximum change allowed per iteration]
              labels = {'row_ents':'Person', 'col_ents':'Item'},   # [<None, {'row_ents':<None, 'person',...>, 'col_ents':<None, 'item',...>}> => to describe summarized entities]
              extreme = [0.50, 0.50]  # [<[float,float]> => row, col max score adjustment]
              ):
        """Returns Rasch Joint Maximum Likelihood Estimate statistics for a given array.

        Returns
        -------
            The method returns None but assigns outputs to the Damon
            object.  It returns a complete set of Rasch statistics
            in my_dmnobj.rasch_out, but it also assigns some
            statistics to other relevant Damon attributes.  For
            instance, coordinates (Rasch measures) are automatically
            assigned to my_dmnobj.coord_out and estimates are assigned
            to my_dmnobj.base_est_out.  This means you don't need to
            run these methods for your Damon object -- the outputs have
            already been calculated.

            In particular, rasch() outputs the following:

                MyDaObj.rasch_out (the following are datadicts):

                   {fac0coord,      =>  person measures
                    fac1coord,      =>  item difficulties, and includes 'step_coord'
                    observed,       =>  observed values
                    estimates,      =>  cell expected values, or estimates
                    residuals,      =>  observed - estimates
                    cell_var,       =>  cell variance
                    cell_fit,       =>  cell fit, a standardized residual
                    fac0_se,        =>  person measure standard errors
                    fac1_se,        =>  item difficulty standard errors
                    fac0_infit,     =>  person infit, sensitive to structural misfit (e.g., > 1.3)
                    fac1_infit,     =>  item infit, sensitive to structural misfit (e.g., > 1.3)
                    fac0_outfit,    =>  person outfit, sensitive to outlier misfit
                    fac1_outfit,    =>  item outfit, sensitive to outlier misfit
                    reliability     =>  gives the summary "Separation" and "reliability" for both facets.
                    summstat        =>  summary statistics merged in one table
                                        contains: {'row_ents',
                                                   'col_ents',
                                                   'reliability'}
                    }

            In addition, rasch() assigns Damon attributes:
                my_obj.coord_out      => {'fac0coord', 'fac1coord', 'step_coord'}
                my_obj.base_est_out   =>  estimates
                my_obj.base_resid_out =>  residuals
                my_obj.base_ear_out   =>  cell_var
                my_obj.base_fit_out   =>  cell_fit
                my_obj.summstat_out   =>  Person/Item level summary stats.
                                          Because rasch() outputs are somewhat
                                          different from most Damon outputs,
                                          the summstat_out results created
                                          by rasch() do not parallel those
                                          created by coord() and associated
                                          methods.
                my_obj.row_ents_out   =>  Row/person summary statistics
                my_obj.col_ents_out   =>  Column/item summary statistics
                my_obj.reliability_out
                                      =>  Reliability statistics

                Separately assigning row_ents_out and col_ents_out
                makes it possible to export them using the export()
                method.

            Workflow:
                MyRasch = Damon.rasch(...)

            You will not need to run any other Damon() methods.

            "Facet"
            -------
            The abbreviation 'Fac' stands for 'facet' and indicates
            one of the edges of the data matrix.  By convention, the
            row facet, often called 'persons' is called Facet 0, the
            column facet, often called 'items', is called Facet 1.

            The term 'facet' comes from generalizability theory and
            from Michael Linacre's Many-Facet Rasch model, in which the
            Rasch model is extended to handle not just persons and items,
            but persons, items, raters, and any number of other types
            of entities that impact the probability of the data in a
            consistent way.  The equivalent in linear algebra is
            'tensor'.  Damon currently handles only 2-facet designs.

        Comments
        --------
            rasch() implements Andrich's Rasch Rating Scale Model using
            the Joint Maximum Likelihood Estimation algorithm
            described in "Rating Scale Analysis" by Wright and
            Masters.  It is modeled on the routine used by Michael
            Linacre's Winsteps program, but much simplified.  However, any
            problems that may arise are my fault, not Mike's.

            rasch() supports mixed dichotomous and polytomous data and
            allows items to be grouped into different rating scale response
            formats.  When each item is its own group, it becomes
            a Partial Credit Model, where items are allowed their own
            step structure.

            For a full-featured Rasch tool, consider using one of
            the commercial Rasch software packages, such as those
            listed in http://www.rasch.org/software.htm .  The benefits
            of Damon's rasch() method are that:

                1.  It requires objectivity as a condition of fit and
                    is the inspiration behind coord() in this regard
                2.  It can access Damon's full suite of methods
                2.  It gives access to all Python/Numpy methods and packages
                3.  It's easy to expand and integrate with other software
                4.  It's free

            The Rasch model applies the specific objectivity criterion
            to non-negative 1-dimensional dichotomous and polytomous
            datasets.  One might expect it to yield results approximating
            those of Damon's 1-dimensional model, but there are some
            important differences.

                rasch:  * Estimates parameters using maximum likelihood
                        * Assumes dichotomous or ordinal data
                        * Assumes all items are 1-dimensional and positively
                          correlated

                coord:  * Estimates parameters using alternating least squares
                        * Assumes interval data (but works on dich/ordinal data)
                        * Does not assume unidimensionality and does not
                          required items to be positively correlated

            coord() is the more general algorithm and can be used on any
            dataset that rasch() can be used on, plus many that rasch() cannot.
            However, rasch() may provide better estimates when data is
            unidimensional, dichotomous, and positively correlated. The reason
            is that rasch() enforces a positive coordinate system whereas
            coord() allows negative coordinates.  When the coord() dimensionality
            is set to "1" and there is a preponderance of zeros or ones in a
            given row/column, a hook forms in the estimates vs true values curve
            at the tails, with various other distortions.  This is caused by
            coordinates that should be positive spilling into negative territory.
            However, coord() can be run effectively with such data using the
            following strategies:

                1)  Let coord() find the most "objective" dimensionality.  It
                    will use extra dimensions to model the tails more accurately.
                    However, the cell estimates have to be interpreted as 
                    PREDICTIONS of observed values, not as continuous 
                    probabilities.

                2)  Use coord()'s 'NonNeg_1D' condcoord option.  This forces
                    the coordinates to be positive, avoiding issues caused by
                    negative coordinates.

                Prior to running coord(), use the standardize() method to convert
                the data to "PreLogits".

            ...or you can use rasch().

            Anchoring
            ---------
            The "anchors" parameters make it possible to apply
            Rasch row (person) and column (item) parameters (known as
            "coordinates" in Damon) from one dataset analysis
            to another, so long as there are common row or column entities.
            It is like the "anchors" specification in the coord() method.
            These coordinates are stored in a "bank" using the bank() method
            (banks are Python pickle files).  Say you have a bank of anchor
            values for Items 1 - 10 and the present analysis contains
            items 5 - 15.  Using anchors, you can anchor the
            item difficulties in the present analysis to the old analysis
            (used to calibrate the bank in the first place) in terms of common
            items 5 - 10.  This makes the two tests essentially equivalent
            and puts their respective persons on the same scale.

            (You can obtain a similar result by merging the old and new datasets
            into a single dataset and running rasch() without any anchors.)

            Generally, you anchor EITHER rows or columns, not both at the
            same time.  However, if you do anchor both at the same time
            rasch() will use those coordinates to build cell estimates.

            Step anchors
            ------------
            When anchoring polytomous data (more than two possible responses)
            rasch() requires "step anchors" in addition to row or column
            anchors.  These are the relative difficulties of each subsequent
            "step" on the rating scale.  The step anchor file is automatically
            saved with the row or column anchor files, so you shouldn't have
            to worry about it.  However, if you are creating person or items
            from scratch, run without anchoring first in order to get the
            needed step anchors and save the person or item measures to a
            bank using Damon.bank().  Refer to this bank for the anchored
            run.

            Keep in mind that item difficulties and step parameters are
            only valid relative to a specific rating scale and response set.
            An item difficulty calculated with three response options is not
            comparable to the same item calculated with two response options.
            The step and item difficulties travel together in the bank.

            Purging Banks
            -------------
            Damon's bank() method stores a persistent bank file in your
            current working directory.  It supports modifications such as
            adding and removing bank entities.  However, if you do something
            like build a bank with all one group (groups = None), then try
            to read it for a multi-group dataset, Damon will return an
            exception because it can't find the relevant groups in the bank.
            This is a case of an obsolete bank.  This type of problem is
            generally solved by deleting the relevant bank files in the
            current working directory and building a new bank from scratch.

            Simulations
            -----------
            When using create_data() to build artificial datasets to feed into
            Damon.rasch(), set facmetric[1] to zero, plus a small number:

                facmetric = [1, 0.0001]

            The 0.0001 forces the coordinates to be positive (see create_data()
            docs), which is required by the Rasch model.  facmetric = [4,-1]
            allows negative coordinates, which violates the model.

            Note that create_data() is coded so that a perfect zero in the
            right position ([1, 0]) will force the underlying calculations
            to work in a ratio scale, good for exploring certain data
            designs.  To keep the underlying metric on an interval scale,
            add a bit of decimal dust to keep the right-hand value from
            being a perfect zero.

            Invalid Rows and Columns
            ------------------------
            rasch() will return an exception if there are any missing response
            categories. Damon expects to see at least one of each of the
            responses specified in my_obj.validchars.  Address this by
            modifying the Damon() validchars argument to reflect the actual
            data available. The validchars "Guess" option will do this 
            automatically.

            Extreme Scores
            --------------
            Extreme scores (all maximum or all minimum for a person or
            item) are handled internally by imposing boundaries on the logit
            scale. You can control these boundaries using the "extreme"
            parameter.  You may find that a test is giving suspiciously
            low person or item reliability.  Check the row_ents_out and
            col_ents_out outputs.  You may find that the low reliabilities
            are caused by persons or items with extreme measures and
            consequently standard errors that are MUCH larger than those
            for non-extreme values.  Since these are somewhat artifactual
            and don't reflect the true quality of the test, you may want
            to adjust the "extreme" parameter upwards until the extreme
            measures are not too far from the main distribution.
            
            However, you will also want to run the extract_valid()
            method prior to running rasch().  This filters out rows and columns
            with all missing data or all the same score, and is particularly
            useful during the item calibration phase (if you are building
            a bank and doing item anchoring).  Run restore_invalid()
            afterward to add the invalid rows and columns back in.  The only
            issue with this approach is it assigns nanval to persons
            with extreme high or low scores.  An alternative approach is to
            filter out only the invalid columns and adjust the "extreme"
            parameter so that extreme persons are not too far above or
            below the rest of the person distribution.

        Arguments
        ---------
            "groups" is used to group items according to their rating
            scale in accordance with the Andrich Rating Scale Model (1978).
            Each item group gets its own step parameters.  Items that
            have a different number of response options certainly must
            be grouped, but it is possible to group items that have the
            same number of response options if you think the relative
            difficulties of the steps is likely to be different.  You can
            also let each item be its own group, which is equivalent to the
            Partial Credit Model.

                groups = None       =>  All items have the same dichotomous
                                        or polytomous rating scale and should
                                        be treated, in effect, as one big
                                        group.  rasch() automatically assigns
                                        all items to a group called "All" and
                                        stores them in the bank accordingly.

                groups = {'row':1}  =>  Item group labels are given in row
                                        1 (counting from 0) of the column
                                        labels.  These will be used to assign
                                        items to groups.  If your columns are
                                        not pre-labeled, use the merge_info()
                                        method to suck item information into
                                        the column labels.

                groups = ['key', {'group0':['i1', 'i3'], 'groupX':['i4', 'i2']}
                                    =>  Instead of using column labels, you just
                                        assign items to groups.  Here we assign
                                        items 'i1' and 'i3' to 'group0' (or any
                                        name you specify) and items 'i4' and 'i2'
                                        to 'groupX'.  The items are labeled by
                                        their unique key.

                groups = ['index', {'group0':[0,1,2,3], 'groupX':[range(4, 8)]}
                                    =>  Instead of identify items by key, you can
                                        identify them by index or column number,
                                        where 0 refers to the first column in the
                                        coredata array.  Here the items for columns
                                        0 through 3 are assigned to 'group0' and
                                        the items for columns 4 up to (but not
                                        including) 8 are assigned to 'groupX'.

            -------------
            "anchors" specifies an entity bank and a set of row entities
            or column entities to whose coordinates the current analysis
            should be anchored.

                anchors = None      =>  Do not anchor the analysis to
                                        pre-existing bank coordinates.

                anchors = {'Bank':'MyBank.pkl','row_ents':[None],'col_ents':['All']}
                                    =>  Open and load the MyBank.pkl
                                        pickle file.  Do not use any
                                        of its row entities (Facet 0)
                                        to anchor the analysis.  Use
                                        all of its column (Facet 1) entities
                                        (those it has in common with the
                                        current dataset) as anchors.

                anchors = {'Bank':'MyBank.pkl','row_ents':[1,5,10],'col_ents':[None]}
                                    =>  Open and load the MyBank.pkl
                                        pickle file.  Use row entities
                                        1, 5, and 10 to anchor the analysis.
                                        Do not use any column entities
                                        as anchors.

            Important
            ---------
            Note that anchor entities refer to entities in the bank
            (as these are, by definition, anchored), not entities in
            the current dataset.  However, at least some of the anchored
            entities must appear in the current dataset.
            Entitities listed as anchors that do not exist in the
            current dataset are ignored.

            -------------
            "runspecs" specifies the stopping conditions:

                runspecs = [StopWhenClose,MaxIteration]

                where the "StopWhenClose" number is how close the
                sum of estimates should be to the sum of observations
                (in accord with maximum likelihood) before stopping.
                The "MaxIteration" number is the maximum number of
                iterations that are allowed.

                runspecs = [0.001,20]

                means run until the maximum discrepancy between
                the observed and expected sums is less than 0.001
                or until 20 iterations have elapsed, whichever occurs
                first.

            -------------
            "minvar" specifies the minimum allowable variance
            allowed during iteration.  The larger the number, the
            smaller the relative sizes of changes in variance per
            iteration.

            -------------
            "maxchange" specifies the maximum amount of change
            allowed per iteration.  minvar and maxchange are internal
            mechanisms for controlling the iterative process and can
            safely be left at their defaults.

            -------------
            "labels" gives a name to the row and column entities.
            This shows up in some of the output reports and is
            important when using merge_summstat() and wright_map().

                labels = None   =>  Use the existing  corner id for
                                    both row and column labels.

                labels = {'row_ents':'Person', 'col_ents':'Item'}

            -------------
            "extreme" <row float, col float> is used to put an artificial 
            bound on extreme high or low scores for rows and columns.
            The logit formula, applied to row scores, is:
            
                logit = logn(p / (1 - p))
            
            where p = n_correct / max_possible.  You can see that when
            p becomes 0.0 or 1.0, the logit goes to negative or positive
            infinity.  To prevent this, p is constrained to fall between,
            but not including, 0.0 or 1.0.  This is done by putting a
            bound on n_correct.  "extreme" is how many points less than
            max_possible we choose to define a perfect score.  A similar
            bound can be put on column scores.
            
                extreme = [0.50, 3.0]
                                    =>  The maximum possible row score will
                                        equal (max_possible - 0.50), half
                                        a raw score point.  If a kid
                                        gets 100 out of 100 questions
                                        correct, we assign her a raw score
                                        of 99.5.  The maximum possible
                                        column score (maximally easy item) 
                                        will be (max_possible - 3.0).
                            
            As discussed above, how you set this parameter determines how
            far out on the scale the maximum/minimum persons and items
            will fall, and how high their standard errors will be.  To get
            meaningful reliabilities and other summary stats, you will
            want to adjust these upward until the logit measures look like
            they belong to the main person or item distribution.  The
            price of setting them too high is that persons (or items) with 
            different high scores will get the same score.
            
        Examples
        --------

            [under construction]

        Paste Method
        ------------
            rasch(groups = None,    # [<None, {'row':int row of group labels}, ['key', {'group0':['i1', i2'],...}], ['index', {'group0':[0, 1],...}]> => identify groups]
                  anchors = None,   # [<None, {'Bank':<pickle file>, 'row_ents':[<None,'All',row entity list>], 'col_ents':[<None,'All',col entity list>]}> ]
                  runspecs = [0.0001,20],  # [<[stop_when_change, max_iteration]> => iteration stopping conditions ]
                  minvar = 0.001,  # [<decimal> => minimum row/col variance allowed during iteration]
                  maxchange = 10,  # [<+num> => maximum change allowed per iteration]
                  labels = {'row_ents':'Person', 'col_ents':'Item'},   # [<None, {'row_ents':<None, 'person',...>, 'col_ents':<None, 'item',...>}> => to describe summarized entities]
                  extreme = [0.50, 0.50]  # [<[float,float]> => row, col max score adjustment]
                  )
        """
        if self.verbose is True:
            print 'rasch() is working...\n'

        # Run utility, assign attributes
        rasch_out = dmn.utils._rasch(locals())
        self.rasch_out = rasch_out
        self.coord_out = {}
        self.coord_out['fac0coord'] = rasch_out['fac0coord']
        self.coord_out['fac1coord'] = rasch_out['fac1coord']
        self.coord_out['ndim'] = 1
        self.coord_out['anchors'] = anchors
        self.base_est_out = rasch_out['estimates']
        self.base_resid_out = rasch_out['residuals']
        # No base_se_out -- Rasch doesn't give cell level SE's
        self.base_ear_out = rasch_out['cell_var']
        self.base_fit_out = rasch_out['cell_fit']
        self.summstat_out = rasch_out['summstat']
        self.row_ents_out = rasch_out['summstat']['row_ents']
        self.col_ents_out = rasch_out['summstat']['col_ents']
        self.reliability = rasch_out['summstat']['reliability']

        if self.verbose is True:
            print 'rasch() is done -- see my_obj.rasch_out and other my_obj.*_out attributes.'
            print 'Contains:\n',self.rasch_out.keys(),'\n'

        return None



##################################################################################################

    def coord(self,
              ndim = None,      # [<None,[[dim list],'search','homogenize']> => set dimensionality or search range, possibly homogenized]
              runspecs = [0.0001,10],  # [<[StopWhenChange,MaxIteration]>]
              seed = 'Auto',  #[<None,int,'Auto',{'MinR':0.90,'MaxIt':<10,[3,10]>,'Facet':<0,1>,'Stats':[<'Stab','Acc','Obj','PsMsResid','NonDegen'>],'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[...]},'Group2':{'Get':'AllExcept','Labels':'index','Entities':[...]}}>]
              homogenize = None,    # [<None,{'ApplyAncs':<True,False>,'Facet':1,'Max':500,'Form':'Cov'} => homogenize params]
              anchors = None,    # [<None,{'Bank':<bank,pickle file>,'Facet':<0,1>,'Entities':<['All',list entities]>,'Refresh_All':<bool>}> ]
              quickancs = None,  # [<None,[<0,1>,ent x ndim array]> => facet, anchor array]
              startercoord = None,    # [<None,[<0,1>,ent x ndim array]> => facet, starter array]
              pseudomiss = None,    # [<None,True> => make cells pseudo-missing for "official" run]
              miss_meth = 'IgnoreCells', # [<'ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells>]
              solve_meth = 'LstSq', # [<'LstSq','IRLS'> => method for solving equations]
              solve_meth_specs = None,    # [<None, spec dictionary> => specs for solve_meth (cf. solve2() docs), e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]
              condcoord = {'Fac0':'Orthonormal','Fac1':None},  # [<None,{'Fac0':<'a func',myfunc>,'Fac1':<'a func',myfunc>}> ]
              weightcoord = True,   # [<None,True> => downweight influential coordinates]
              jolt_ = None,  # [<None,[sigma,jolt_]> e.g., [20,1.5] => Apply 1.5 noise factor if sigma exceeds 20]
              feather = None,     # [<None,float> => add small amount of randomness to the data]
              condcoord_ = None     # [<None, condcoord args> => deprecated, only for backward compatibility]
              ):
        """Calculate facet coordinates for the ndim dimension.

        Returns
        -------
            The method returns None but assigns outputs to the Damon object.
            coord() outputs are accessed using:

                my_dmnobj.coord_out

            coord_out contains results of applying Alternating Least
            Squares decomposition to the Damon object for one or more
            dimensionalities in the form of row and column entity
            coordinates.  The coordinates arrays are output as datadicts.
            Additional variables are included:

            {'fac0coord',   =>  row entity x dimension datadict
             'fac1coord',   =>  column entity x dimension datadict
             'ndim',        =>  number of dimensions
             'changelog',   =>  Change in coordinates per iteration
             'anchors',     =>  Passes on the "anchors" argument
             'facs_per_ent' =>  Number of unanchored facets per entity
                                [facet,n unanchored facets per entity]
                                (See tools.obspercell() docs)
             }

            In addition, new variables are assigned as attributes
            to the Damon object:

            fac0coord       =>  same as coord_out['fac0coord'] but made
                                accessible directly as a Damon object attribute.

            fac1coord       =>  same as coord_out['fac1coord'] but made
                                accessible directly as a Damon object attribute.

            bestdim         =>  the number of dimensions that best predicts
                                the values in pseudo-missing cells from the
                                specified range of dimensionalities.

            objperdim       =>  A datadict reporting root mean squared residual
                                and correlation between Damon estimates and
                                predictions of cells made pseudo-missing, as
                                well as objectivity, convergence speed, and
                                a combined "overall objectivity" for
                                each dimensionality within the specified
                                range.

            maxposdim       =>  The maximum possible dimensionality supportable
                                by the dataset given its missing data.

            seed            =>  Statistics relating to the selection of initial
                                "seed" coordinates for calculating coordinates.

                                my_obj.seed = {'BestSeed',
                                               'R',
                                               'MinR',
                                               'Attempts',
                                               'RPerSeed',
                                               'StabDict',
                                               'AccDict',
                                               'ObjDict',
                                               'ErrDict',
                                               'StatsPerSeed'   => table of stats
                                                                   per seed, most
                                                                   useful for an
                                                                   overview
                                               }

            homogenized     =>  array of homogenized data (a covariance matrix).

            facs_per_ent    =>  Number of unanchored facets.
                                Syntax:  [facet,n unanchored facets per entity]
                                (See tools.obspercell() docs)

            Workflow:
                D = dmn.Damon(...)
                D.parse(...)
                D.standardize(...)
                D.coord(...)
                row_coords = D.coord_out['fac0coord']['coredata']

        Comments
        --------
            coord() is the heart of Damon.  It calculates row coordinates 
            from column coordinates and data, then column coordinates from 
            row coordinates and data, back and forth until both sets converge 
            on a stable solution.
            
            The algorithm is a version of "Alternating Least Squares" (ALS)
            worked out by Howard Silsdorf.  Its cell estimates are nearly the
            same as those found using other decomposition methods such as
            Singular Value Decomposition, but the matrix is analyzed piece-meal.
            This makes it more tractable for anchoring entities across runs and
            dealing with missing data -- essential in the field of psychometrics.

            When only one dimensionality is specified by ndim, coord() runs ALS
            at that dimensionality and no effort is made to determine "best"
            dimensionality.  This is fastest.  When a range of dimensionalities
            is specified, coord() uses an "objectivity" statistic combining 
            prediction accuracy and coordinate stability to get "best" 
            dimensionality. When the "best" dimensionality is found, coord() 
            runs one last time with no pseudo-missing cells (unless 
            pseudomiss = True). This run yields the "official" coordinates.

            When coordinates are anchored, the dimensionality is set
            automatically according to the dimensionality of the anchored
            persons or items regardless of what may be specified in ndim.

            When assessing dimensionality, if ndim includes the 'homogenize' flag
            the dataset is replaced by a covariance matrix which has the important
            property of being fairly homogeneous dimensionally (same number of
            dimensions per row/col).  This makes it possible to get the correct
            number of dimensions even when the items don't participate in a
            common space.  If desired, the coordinates of the homogenized array
            can be applied to the observed data as anchors to compute a complete
            set of coordinates.  These options make it easier to address
            datasets with strong "between item" multidimensionality.


            Getting "Best Dimensionality":  A Primer
            --------------------------------------------
            Selection of correct dimensionality (ndim = number of dimensions)
            is a BIG deal in Damon.  Set it too low and the model "underfits"
            the data.  Set it too high and the model "overfits" -- often
            a more serious problem.  "Overfit" means that even though
            the model's estimates and observations seem to match, the model
            lacks predictive validity.  Predictions of missing cells may
            be way off.  This is because as unnecessary dimensions are added,
            the model loses sight of the underlying structure of the data and
            models noise with the extra dimensions.  The estimates matrix
            mimics the observed matrix without having any true predictive power.

            Fortunately, Damon contains straightforward and effective tests to
            determine the correct dimensionality, noted above.  The pseudo-missing
            test makes a bunch of cells randomly (or nonrandomly) missing and
            tries to predict their values.  It can be proven that the dimensionality
            (the number of dimensions used by the model to fit the data) that
            best predicts MISSING cells is also the dimensionality that best models
            the "True" matrix -- the observations with all error stripped away.
            (Bear in mind that the "observed" matrix is not the same as the
            "True" matrix.  Observations can, quite simply, be wrong.)

            The result of the pseudo-missing test is a U-shaped prediction error
            curve (x-axis is number of dimensions, y-axis is missing cell 
            prediction error), or alternatively a hill-shaped "accuracy" curve 
            that "plateaus" (with large datasets) at the optimal dimensionality.
            "Accuracy" is defined as the correlation between observed values
            (made pseudo-missing) and the corresponding cell estimates.

            A second test computes row coordinates from two different
            samples of columns.  The degree to which the row coordinates
            agree is a measure of sample invariance, or stability.  This
            also is expressed as a correlation and called the "stability"
            statistic.  This curve tends to start high and climb slowly,
            then drop quickly after the optimal dimensionality.  For a more
            detailed description, see the tools.stability() docs.

            A third test, the convergence speed test, relies on the
            property that the optimal dimensionality will often -- when
            there is very little missing data -- be the dimensionality
            for which coord() converges onto a solution the fastest.  In
            Damon, "convergence speed" is defined as the percentage
            of total change in the most recently computed coordinates
            that occurs after the thirditeration.  In a noiseless complete
            dataset, convergence to approximately zero occurs by the
            third iteration.  (Defined in this sense, in terms of change
            in computed coordinates, this means there is no guarantee that ALS
            will converge at all if the dimensionality is wrong.  However,
            when convergence is defined as "change in the derived estimates",
            it can be proven that convergence is guaranteed.)

            Objectivity
            -----------
            Damon combines the accuracy and stability statistics to
            calculate a third statistic -- "objectivity".

                objectivity = (stability * accuracy)^(1/2)

            The objectivity curve exploits the complementary properties
            of the stability and accuracy curves to build a curve with
            a clear and distinct peak at the optimal dimensionality.  This
            is the curve used to set "best" dimensionality by default.

            "Best" Dimensionality Methods
            -----------------------------
            When the verbose option is used, these statistics and a couple
            others are reported by the Python interpreter.

            Damon offers three methods for finding the maximum of the
            objectivity curve:

                "brute force" method=>  Run the model for each of
                                        a range of dimensionalities and
                                        pick as "best" the one with the
                                        highest objectivity. This is most
                                        reliable method.

                "search" method     =>  To save time, instead of running
                                        every dimensionality in a range,
                                        we go to the middle of the range and
                                        calculate whether the objectivity
                                        maximum is lower or higher.
                                        If lower, repeat the process with
                                        the lower half of the range; if
                                        higher, repeat with the upper half
                                        of the range.  Keep repeating and
                                        subdividing until we can't go
                                        any further. We pick the dimensionality
                                        with the highest overall objectivity,
                                        which should be near where the
                                        'search' algorithm ended up.

                "homogenize"        =>  This is not a separate method so
                                        much as a way of speeding up
                                        either the brute force or the
                                        search method.  When the data
                                        are "homogenized" (converted to a
                                        covariance matrix), and the homogenizing
                                        facet is relatively small (e.g., a small
                                        number of columns relative to rows),
                                        the homogenized data array will be
                                        much smaller than the original array,
                                        yet (if the original data have homogeneous
                                        dimensionality) yield the same
                                        dimensionality. Thus, a 1000 x 100 data
                                        array becomes a 100 x 100 array, which
                                        is much faster to crunch.

                                        This approach only yields speed benefits if
                                        one of the array's sides is much smaller
                                        than the other.  However, it can be
                                        extremely valuable for other reasons.
                                        When there is lots of between-item
                                        multidimensionality, "homogenize" reveals
                                        the true underlying dimensionality in
                                        a way that the raw data cannot.

            'search' is not infallible.  If the overall objectivity curve does not
            have a smooth mountain-shape, as sometimes happens, 'search' can get
            trapped in the wrong range of dimensions.  (When simulating data,
            add some noise; 'search' does not work with "perfect" data.)

            The "homogenize" and "brute force" methods are also fallible.
            Homogenizing can be thrown off by large blocks of missing data.
            However, it is good at getting the correct dimensionality, even when
            there is lots of between-item multidimensionality, and it's fast.

            The brute force method will be thrown off when the items do not
            participate in a common space (lack within-item multidimensionality).  
            The objectivity curve will then tend to peak at an artifactually
            low dimensionality (e.g., "1", if there is one common dimension).
            In this case, run "homogenize" to get the true number of dimensions.

            There are a variety of forces that complicate the effort to find 
            "best dimensionality":

            1.  No Common Space.  You may find, for instance, that the item facet
                contains elements that do not have the same dimensions as the
                others.  A test with word problems contains a mix of reading and
                math, called "within-item" multidimensionality.  Damon
                likes that kind of multidimensionality.  It gets harder
                when some items contain only math and no reading, or only reading
                and no math.  This is "between-item" multidimensionality. 

                To address between-item multidimensionality, there are several
                strategies:

                *   Enforce a common space using Damon fit statistics.  Remove
                    all items that do not participate in the common space.
                    This is known as "analysis of fit".  This is nice because
                    it forces the analyst to understand and clarify the space
                    of the test.  However, it may require breaking the test
                    into smaller sections and analyzing each separately, which
                    results in loss of precision.  This strategy is best when
                    there are a few items that violate the overall space of the
                    test for unknown reasons.

                *   Assess "best" dimensionality using the 'homogenize' option
                    and run coord() at that dimensionality (coord() will do this
                    automatically).  The results will appear to have lower objectivity
                    than running without homogenize (the "objectivity" stats in this
                    case favor the "common" dimensionality, not the "full"
                    dimensionality), but the results will nonetheless be "truer"
                    since they capture the full dimensional range.  This strategy
                    is often sufficient for computing reasonable estimates, but
                    it should be used with caution as it involves an overt
                    violation of the within-item multidimensional conditions of
                    the model.

                *   Identify groups of items that you believe likely to contain
                    somewhat different dimensions (e.g., grammar, vocabulary,
                    reading) and use the sub_coord() method -- instead of 
                    coord() -- to compute subscale measures that draw 
                    information from the whole test without being 
                    "contaminated" or biased by the other dimensionalities.
                    This is an excellent method to use when you have a clear
                    theoretical basis for identifying items that get at unique
                    skills.  It offers greater precision than the first method
                    without the theoretical muddiness of the second. But in
                    practice, it is a bit unwieldy.
                    
                In general, I prefer the first method.  It forces the analyst
                to understand the common space of the test and to select
                items that belong to that space, and this is good science.
                Once the common space is established, analysis is easy,
                equating is easy, banking is easy.  One can always analyze
                different subsets of the test separately.

                Occasionally, a dataset will contain items that bear no relation
                to the rest of the items on the test, e.g., eye color or the price
                of tea in China on a math test.  They are easily identified by
                their fit statistics, and there is no recourse but to remove them
                from the test.  They offer nothing but noise and are intrinsically
                unpredictable when analyzed out of context.

                Bear in mind that you only need one of the facets (items) to define
                a common space.  The persons may be in a 50-dimensional space,
                but so long as all the items defin the SAME 4-dimensional
                space, the extra 46 dimensions will be filtered out and the 
                dataset meets the conditions for a common space.

            2.  Poor Choice of Missing Cells.  Cells can be correctly predicted
                by accident and incorrectly predicted through no fault of the
                model.  The sample of pseudom-missing cells makes a difference
                in the accuracy statistic, which impacts objectivity.

                The pseudomiss() method allows you through the "seed" parameter
                to re-use the same sample of pseudo-missing cells across
                analyses or to generate a completely different sample.  In
                determining best dimensionality, it is useful to do several
                analyses with different random samples to get a sense for how
                stable the objectivity curve is.

            3.  High Error.  As the average level of error of the observations 
                increases, the dimensionality that best predicts missing cells 
                will become less than the "true" dimensionality.  At the extreme,
                where the data matrix is nothing but noise, the objectivity curve
                flattens out and the "best" dimensionality approaches "1" or "0"
                with no convergent solution.  This is because a high error
                dataset is essentially a dataset with high "between-item" 
                multidimensiality; coord() will default to the "common 
                dimension", to the degree there is one.  If the 'homogenize' 
                option does not suggest a higher dimensionality, it really is 
                just a lot of random error and there's not much you can do.  
                Just accept the dimensionality that Damon wants.

                The loss of dimensionality becomes an issue when you want to combine
                a high error dataset with a low error one, e.g., two datasets 
                with the same items but different testing accuracy.  Each
                dataset will want its own dimensionality.  One strategy is 
                to analyze only the low error dataset and get its dimensionality.
                Anchor its items at the derived calibrations and put them in an
                item bank.  Then anchor the high error dataset to those item
                coordinates, forcing them into a common space.

            4.  Cell Misfit.  This is a variant of the "no common space"
                problem.  When the data do not fit the model, the objectivity
                curve is degraded.  One solution is to perform analysis of fit,
                which is squarely in the tradition of Rasch Analysis.  Pick a
                preliminary dimensionality, preferably on the low side.  This
                dimensionality should have  a reasonable theoretical basis.
                Perform multiple Damon analyses deleting row and column entities
                that report high misfit statistics and for which there is a
                theoretical reason for their departure from the common space.
                Do this until the misfit statistics do not exceed some threshold.
                In this way, you retain only those entities that fit the specified
                model.

            5.  Non-Linear data.  This is data that is simply not rigorously
                definable in terms of a linear model, i.e., as the dot product
                of linearly independent coordinates.  (In this usage, "linear"
                has nothing to do with whether the data have "straight-line"
                relationships to each other; it is used in the "linear algebra"
                sense -- decomposable into independent components/coordinates.)

                An example of non-linear data is a matrix of "distances" between
                points.  With distances, as with most datasets, Damon's linear
                estimates will often be more than close enough, but they won't
                exactly match the "true" distances.  This is because distances
                do not decompose into a sum of independent components in the
                desired way.  However, if you understand the function that
                generated the dataset, it is sometimes possible using coord()'s
                condcoord specification to build a non-linear model that will
                describe the "true" values correctly.

            6.  Non-Negative data.  This is data that can be conceived as the
                product of coordinate arrays that do not contain any negative
                numbers.  (The literature calls it "non-negative" rather than
                "positive" for some reason.)  This type of data poses a challenge
                and is discussed below under "limitations of the model".

            7.  Categorical data.  When data are dichotomous (0,1) or polytomous
                (0,1,2,...), a new kind of error is introduced.  The
                observed values do not line up linearly with the latent continuous
                probabilities that produced them.  What's worse, the error is
                not homoscedastic -- cells with high or low probabilities of
                "success" have a different average discrepancy from 0 and 1
                than cells with a medium probability of success. Cell estimates
                of dichotomous data (even standardized) have different 
                mathematical properties than cell estimates of continuous 
                interval data, especially for entities with a very high or 
                low probability of "success".  
                
                Yet another complicating factor, which is quite important,
                is whether the "generating coordinates" (the 
                real-world latent forces that shape each data point) should
                be assumed to be "non-negative" or a mix of positive and
                negative (which is what Damon assumes).  These conditions 
                can be simulated using create_data(), and experiments show
                that non-negative generating coordinates can seriously
                undermine the accuracy of the estimates with dichotomous,
                low-dimensional data, especially when the data are 
                1-dimensional (see discussion of undimensional datasets below). 
                Multidimensional data is much more robust to these issues.
                There is no procedure for determining which assumption is 
                correct ("positive only" or "mixed positive and negative") 
                in real-world datasets.
                
                Because Damon is often applied to categorical data that is
                also 1-dimensional, it has evolved effective strategies
                for getting around these issues.
                
                1)  Standardize() using the 'PreLogit' metric.
                
                2)  Trust the objectivity statistic to yield the dimensionality
                    that will best counteract artifacts caused by non-negative
                    generating coordinates.  You may know that a test with 
                    dichotomous responses is unidimensional, but Damon may
                    find that the best way to analyze it (to avoid artifacts)
                    is to use two or three dimensions.  Let it.  The only
                    downside is that the cell estimates may work better as
                    PREDICTIONS of success or failure on a particular item
                    rather than as continuous MEASURES.  Damon knows how to
                    handle this situation internally.  
                    
                    Besides, the most common and sensible use-case is not to
                    use individual item estimates as measures, but rather to 
                    aggregate them to produce measures (constructs), which
                    is done using the equate() method.  In this case,
                    it doesn't matter whether the individual cell estimates 
                    are more like predictions than measures.
                
                3)  Alternatively, if you know the data are unidimensional
                    dichotomous, and that the coordinates can be assumed to
                    be all positive, you can use coord()'s condcoord parameter
                    (see below) to force the coordinates to be non-negative.
                    (You will also need to use the base_est() refit option
                    to get the estimates back into the correct metric.)
                    
                        condcoord = {'Fac0':'NonNeg_1D', 
                                      'Fac1':'NonNeg_1D'}
                                    
                4)  Experience and simulations suggest that with most
                    tests that are a mix of dichotomous and polytomous
                    items and whose columns are positively correlated, the
                    best dimensionality is 2.  Even when the underlying
                    dimensionality is higher, the fact that the items are
                    positively correlated makes ndim = 2 the best choice.
                    
                    The persistent 2-dimensionality is mostly an effect of the 
                    "generating" coordinates (the coordinates generating the
                    data) being all positive (non-negative).  However, when
                    the generating coordinates are evenly positive and
                    negative, Damon's "best" dimensionality will correspond
                    with the true dimensionality.
                    
                    The downside of all positive generating coordinates is
                    that the true dimensional structure of the data gets
                    obscured in the imprecision of dichotomous data and the
                    fact the columns are all correlated.  The upside is that
                    the objectivity statistics will be much higher, an
                    effect of the high cross-correlations.
            
            In conclusion, Damon's coord() function uses objectivity to find
            best dimensionality.  It is effective enough that if you have a 
            fair understanding of the data and how it behaves, you can trust
            coord() to find the correct dimensionality on its own.

            That said, when dealing with new data, never trust Damon's
            recommended dimensionality blindly.  Come up with a theoretical
            dimensionality, compare it with actual results.  Check for a
            smooth mountain-shaped overall objectivity curve.  Run with multiple
            pseudo-missing random samples by setting seed = None.
            Perform analysis of fit to enforce a common item space.  Use
            the "Brute Force" method if possible, reserving 'search' for
            situations requiring speed.

            Although this is a bit of work up front, once the dimensionality
            for a certain type of dataset is established, it can be reused
            for comparable datasets.  The dimensionality becomes a permanent
            feature of the dataset and travels around with its elements.

            The Curse of Unidmensionality
            -----------------------------
            
            [Note:  This section is under construction and changes a lot.
            Do not believe anything it says.]
            
            It sounds weird, but Damon actually has the most trouble
            with 1-dimensional datasets.  Its estimates do not always match the
            "true" model values during simulation experiments as closely
            as when the data are multidimensional.  The reason is that
            Damon assumes that the "generating coordinates" (the true 
            numerical forces that are behind the data) are a mix of
            positive and negative numbers, and during the iterative process 
            it is possible for coordinates that are close to zero to tip into 
            the wrong sign, to yield row or column estimates that are negatively
            correlated to the "true" values.  In addition, while on a column 
            by column basis the estimates will tend to have a strong linear 
            relationship with the true values (sufficient for many purposes), 
            the two will often not fall on the identity line, each column
            requiring a different slope factor to match the "true" value.  And 
            some columns may show an ogival relationship.
            
            In addition to the distortions this causes in the estimates
            for 1-(or 2- or 3-)dimensional datasets, it will cause the
            standard errors to be underestimated for those items.  This is
            because the estimates will be to some degree compressed for
            that item relative to the true values.  Were the cell values
            expanded to match the spread of the true values (which is not
            known unfortunately), the standard errors would be more or
            less accurate.
            
            These issues go away as the number of dimensions increases.  
            At two dimensions, a plot of the estimates vs true 
            values per column will show a strong linear relationship but may 
            have a somewhat different slope.  At four dimensions and higher,
            even these deviations go away.  The extra dimensions provide
            sufficiently flexibility to compensate for problems in getting
            the sign right.
            
            If the generating coordinates are all non-negative, and are
            well above zero (e.g., create_data()'s facmetric = [4, 1] 
            instead of [4, 0.0001]) the estimates for all dimensionalities
            converge toward the estimates vs true identity line.  In this
            case, Damon exacts no price for assuming mixed-sign coordinates.
            The same applies with mixed-sign coordinates, so long as the
            generating coordinates are well away from zero.
            
            The effects caused by accidentally reversed signs relative 
            to the generating coordinates on the ability to estimate true 
            values provides a rationale for the methods called 
            Non-negative Matrix Factorization (NMF).  By  forcing all 
            coordinates to be positive, NMF skirts problems caused by 
            reversed signs.  I chose not to go this route with 
            Damon because:
                
                a)  I considered the mixed sign solutions (with special
                    workarounds for unidimensional datasets) to be 
                    "good enough" for most purposes.
                
                b)  The problem mainly comes up as generating coordinates
                    approach zero.  Because coordinates of zero (for some items
                    and not others) violate the common space requirement,
                    the normal process of writing, editing, and collecting
                    items for Damon will naturally incline researchers
                    to avoid those that approach the zero condition for 
                    some dimensions.  As mentioned above, to the degree the
                    generating coordinates are above (or below) zero the
                    accuracy of the estimates increases and the various
                    artifacts caused by incorrect signs go away.
                
                c)  Most important, I did not want to assume that the 
                    generating coordinates are always positive.  The 
                    consequences of assuming non-negative
                    coordinates when they are in fact mixed are just as 
                    problematic as assuming the reverse.  What is not
                    known is the true mix of signs "in nature".  Are
                    mixed coordinates a common feature of the natural world
                    or a rarity?
            
            A definitive solution to the problem of mixed signs requires
            an algorithm that determines what the true signs are and
            imposes these as a condition.  This is easy when there is only
            one dimension (the sign of the coordinate drives the sign of
            the correlation of the column data with the other columns) but
            much more challenging as dimensionality increases.
            
            So what options does Damon offer for handling unidimensional
            data?  When data are dichotomous or polytomous, undimensionality 
            is handled using the steps discussed above regarding categorical
            data -- Damon will, on its own, find a higher dimensional solution 
            that deals with the both the categorical nature of the data and
            its unidimensionality.  
            
            When the data are continuous and precise, and also unidimensional, 
            Damon offers a couple other strategies.
            
                1)  Force all columns to be positively correlated and use
                    the 'NonNeg_1D' condcoord option.  Damon does not yet
                    have a dedicated function for reversing the sign of
                    a column, but it's not hard to do.  Find columns that
                    are negatively correlated to the others and reverse
                    the sign of their data values.  Then Damon-analyze.  See
                    the condcoord parameter discussion below for how
                    to specify 'NonNeg_1D'.  Note that you will need to
                    use the base_est() refit parameter to get the estimates
                    in the correct metric.
                    
                    This strategy, when correctly implemented, yields highly
                    accurate estimates and is the basis of Damon's
                    calculation of standard errors, which involves a 
                    1-dimensional analysis of absolute residuals.
                
                2)  Use the obj.rasch() method.  The Rasch model is built for
                    unidimensional data.  However, it is ususally applied
                    to dichotomous and polytomous data and it assumes,
                    like 'NonNeg_1D', that the coordinates are all positive.
                
                3)  Or don't sweat it.  In a regular Damon analysis with
                    a mix of positive and negative coordinates, most of
                    the columns will have reasonable estimates and tend to
                    differ from the "true" values by a fairly moderate 
                    constant factor.  Aberrant items should show up 
                    in the fit statistics, and since they are caused 
                    by coordinates that are too close to zero and thus
                    close to breaching the "common space" requirement, they
                    should probably be removed from the analysis anyway.
                    They can also be flagged using the point-biserial
                    correlation.  Just bear in mind that the standard
                    errors, hence the reliabilities, will be too low
                    for the problematic items.
                    
            So far, Damon does not attempt to address all unidimensionality 
            issues automatically.  A future version may attempt to force all 
            unidimensional analyses to reverse signs where necessary and 
            use the 'NonNeg_1D' method behind the scenes.

            To explore these issues, use create_data() to generate datasets
            whose estimates can be compared to "true" values, playing
            around especially with the 'ndim', 'facmetric', and 'noise'
            parameters.  Here's a code snippet to get started.  Note that 
            you have to rescale the true values to get them into the 
            standardized metric before you can compare them to the 
            estimates.
    
            import damon1.core as dmn
            import damon1.tools as dmnt
            
            # Parameters
            nrows, ncols = 200, 100
            ndim = 1
            seed = None
            ns_seed = seed + 1 if seed is not None else None
            
            # Non-negative data: [4, 0.01]. Mixed sign: [4, -2]. Ratio: [4, 0]
            facmetric = [4, 0.001]  
            noise = {'Rows':1.0, 
                     'Cols':dict(zip(range(ncols),
                                  np.random.RandomState(ns_seed).rand(ncols) * 5))}
    
            # For continuous interval data: ['. -- .']
            # For ratio: ['0.0 -- ']. For sigmoid: ['0.0 -- 1.0']
            vc = ['. -- .'] 
    
            # Create the dataset
            d = dmn.create_data(nrows, ncols, ndim, seed, facmetric, 
                                noise=noise,
                                validchars=['All', vc], verbose=False)
            m_ = d['model']  # This is "true".
            d = d['data']
            data = d.data_out['coredata']
            true_ = m_.coredata
            
            # Do the analysis
            d.standardize()
            data = d.standardize_out['coredata']
            
            # Rescale "true" to the standardized metric
            true = np.zeros((nrows, ncols))
            for i in range(ncols):
                mnsd = [np.mean(data[:, i]), np.std(data[:, i])]
                true[:, i] = dmnt.rescale(true_[:, i], mean_sd=mnsd)
    
            # Get coordinates                
            d.coord([[ndim]], 
            
                    # For non-negative 1-dimensional analysis
                    condcoord_={'Fac0':'NonNeg_1D', 'Fac1':'NonNeg_1D'},
                    
                    # For default analysis
    #                condcoord_={'Fac0':'Orthonormal', 'Fac1':None},
                    )
            
            # Get estimates and refit
            # Default: None, ratio:'PreLogit', interval':'LogDat', sigmoid:'0-1'
            d.base_est(refit='LogDat') 
            est = d.base_est_out['coredata']
            
            # To look at one column at a time, or all columns together in one plot
    #        i = [np.random.RandomState(None).randint(0, ncols, (1,))[0]]
            i = np.arange(100)  
            
            # Plot estimates against data and against true
            dmnt.plot_identity(data[:, i], est[:, i], None, 
                               'vs Observed', 'data', 'est')
            
            dmnt.plot_identity(true[:, i], est[:, i],
                               None, 'vs True', 'true', 'est')


            Getting "Best Starting Coordinates" (seed)
            ------------------------------------------
            Almost as important as finding the correct dimensionality is
            feeding coord() the right starter coordinates.  The coord()
            alternating least squares algorithm picks row and column
            coordinates at random, then refines them until fit between
            the model estimates and the observations is optimized.

            When the data has little noise, the choice of starter values
            does not matter.  They all converge to the same solution.
            However, as noise and other disturbances are added, a certain
            proportion of starter values leads to degenerate solutions.  The
            precise reason for this is a matter of great interest and still
            not fully understood, but it seems to involve:  a) variations
            in the relative "influence" of each observation and its coordinates;
            and b) the fact that coordinates can be positive and negative and
            sometimes end up with the wrong sign when certain boundary
            conditions are met.  (Non-negative matrix decomposition tries to
            avoid negative coordinates for this reason among others.)

            In Numpy's random.py module, each set of random numbers is
            represented by an integer called a "seed".  A seed of "1"
            corresponds to one set of random numbers, a seed of "2" to
            another, and so on.  In a typically noisy dataset, perhaps
            5% of the seeds lead to degenerate solutions.  The
            coord() method calls on a function called _bestseed(), accessed
            through the 'seed' parameter, to return a seed from the
            95% or so that lead to good solutions.

            So, what is a "good" solution?  As when finding best
            dimensionality, Damon optionally uses Stability, Accuracy,
            and Objectivity to assess the "goodness" of each seed, controlled
            using the 'Stats' parameter in the 'seed' argument.  (By default,
            _bestseed() uses Objectivity, which is a combination of Stability
            and Accuracy, but it can also choose best seed using only Stability
            or only Accuracy.  If Damon is unable to calculate one of these
            statistics for some reason, it will automatically fall back on
            the other.)

            This procedure is controlled by the 'seed' parameter and is
            sufficiently automated that it will generally take care of itself.
            However, with noisy datasets you may get warnings that the dataset may
            not be capable of yielding an "objective" solution.  This means
            that Damon was unable to find a seed within the designated range
            of seeds that leads to a specified R (set at 0.80 by default).
            Damon will automatically run the best seed, and this is often quite
            adequate, but take the warning seriously.  Not all datasets yield
            objective statistics.

            Some data sets, such as dichotomous, produce generally lower
            stability statistics.  It takes trial and error, and comparing
            predictive accuracy with stability, to determine a minimum acceptable
            stability.

            Convergence
            -----------
            Convergence can be defined in two senses:

                1) convergence of the coordinates to their final value
                2) converence of the estimates to their final value

            The first type of convergence is not guaranteed, which makes it
            useful in confirming "best dimensionality".  The second type
            is guaranteed, so long as weightcoord = None and the data meets
            the Gauss-Markov requirements (mainly homoscedasticity of errors).

            When my_dmnobj.verbose is True, coord() issues a run-time
            iteration report which includes a "Change" column.  This reports
            the first type of convergence, convergence of coordinates.
            You can get a report of the second type of convergence by
            adding an 'EstConverge' flag to runspecs, but this is computationally
            expensive and useful only for research into the properties
            of the algorithm.

            Limitations of the model -- Non-Negative Coordinates
            ----------------------------------------------------
            An important limitation of the Alternating Least Squares
            algorithm used by Damon is that it assumes coordinates can
            legitimately range across both positive and negative numbers.
            There are important classes of datasets for which this is not
            true and that are best modeled with only positive numbers as
            coordinates.  Trying to analyze such datasets with Damon at
            the "true underlying dimensionality" will result in low accuracy,
            stability, and objectivity.

            This phenomenon is particularly evident with dichotomous data
            at one dimension.  There are three good strategies in this case.
            One is to use the 'NonNeg_1D' condcoord parameter (see
            "condcoord" below).

            The second strategy is to use the rasch() method instead of
            coord().  The Rasch model is built for this type of data.

            The third strategy is to let Damon be Damon.  Let it find
            the optimal dimensionality on its own, even if you believe
            the dataset to be 1-dimensional.  Damon will add extra
            dimensions to approximate the dichotomous values but stops
            when the stability drops too far.  The resulting estimates
            are usually excellent as predictions, though somewhat clustered 
            around 0 and 1.

            At higher dimensionalities, efforts to force coordinates to
            be positive using condcoord have (so far) been ineffective.
            (That said, the field of Non-Negative Matrix Factorization has
            effective algorithms for exactly this type of problem, though
            so far none are considered to be "optimal", i.e., able to find
            global minima.)

            However, in the multidimensional as with the unidimensional
            case, if Damon is allowed to find its own optimal dimensionality,
            the resulting estimates are often fine as predictions.

            The problem is mitigated by precision in the data.  Having 5 or
            more rating scale categories greatly improves the accuracy of
            the estimates.  Continuous data can be analyzed without any
            problem regardless of the non-negativity of the data.


        Arguments
        ----------
            "ndim" specifies an integer number of dimensions ("dimensionality"),
            or it specifies a range of dimensions to be analyzed.  It supports
            efficient searching ("search") for the "best" dimensionality, dimensional
            analysis of the covariance matrix ("homogenize"), and a choice of
            statistical criteria for deciding the best dimensionality.

            Dimensions are integers ranging from 1 up to some user-defined
            maximum. A dimensionality of 0, describing a matrix of random numbers,
            is also supported.

                ndim = [[int list of dimensionalities],
                         'search',
                         'homogenize',
                         <'Stab','Acc','Obj','Speed','Err','NonDegen'>
                             =>  Statistic to use for determining "best"
                                 dimensionality.  'Obj' is the default if blank.
                        ]

                ndim = None         =>  No coordinates will be computed.  They
                                        will be drawn from the bank designated
                                        in the anchors parameter (so "anchors"
                                        must be filled out).  In any event, if
                                        you specify anchors, whatever dimensionality
                                        you specify in ndim will be overwritten.

                ndim = [[3]]        =>  Run coord() at three dimensions, i.e.,
                                        locate each row and column entity in a
                                        3-dimensional space.  If pseudomiss = True,
                                        the analysis will report how well the
                                        3-dimensional analysis predicted the
                                        values in the pseudo-missing cells.

                ndim = [[1,2,3,4,5]]=>  Run coord() successively across these five
                                        dimensionalities and report how well each
                                        dimensionality predicted the values of
                                        the pseudo-missing cells.  Pick the "best"
                                        dimensionality and return the results of
                                        that run.  The final run will include
                                        make cells pseudo-missing only if
                                        pseudomiss = True.  Because this method
                                        doesn't take any short-cuts, it is called
                                        the "Brute-Force" method.

                ndim = [[0,1,2,3,4,5]], ndim = [[0]]
                                    =>  Same as above, but include a dimensionality
                                        of zero.  The 0-dimensional analysis is
                                        like a 1-dimensional analysis except that
                                        the coordinates for each facet are constrained
                                        to be equal for all entities.  The resulting
                                        estimates array (from base_est()) will yield
                                        the same estimate for each cell, the result of
                                        analyzing an array of random numbers.  When
                                        reviewing dimensionsionality using the
                                        damon_obj.objperdim output, the Objectivity of the
                                        dim = 0 column will always be zero because
                                        Accuracy, correlating against non-varying estimates,
                                        will also be zero.  However, the "Err" column
                                        (similar to Accuracy, but a root mean squared
                                        error rather than a correlation) is helpful.
                                        If the zero dimension has the smallest error,
                                        the data can be considered random and lacking any
                                        dimensional structure, which is valuable information.

                ndim = [range(1,6)] =>  Identical to the specification above, it
                                        uses the Python range() function to produce
                                        the list of dimensionalities.  Read it as
                                        a list of integers from 1 up to, but not
                                        including, 6.

                                        In the two examples above, best dimensionality
                                        is by default determined using the "objectivity"
                                        criterion.  However, another criterion can be
                                        specified.

                ndim = [range(1,6),'Acc']
                                    =>  Select best dimensionality based on predictive
                                        accuracy rather than the default objectivity, which
                                        is based on combined predictive accuracy and coordinate
                                        stability.

                                        Stability ('Stab'), convergence speed ('Speed'),
                                        prediction error ('Err', like accuracy but inverse
                                        and based on mean residuals rather than correlation),
                                        and non-degeneracy ('NonDegen', the tendency of
                                        predictions of missing cells to be in the same
                                        range as those for non-missing cells) can also be
                                        specified as best dimensionality criteria.

                                        'NonDegen' only works if you specify the 'NonDegen'
                                        option in the seed['Stats'] argument described
                                        below.

                ndim = [range(20,50),'search']

                                    =>  coord() will "search" for the best
                                        dimensionality using a method more efficient
                                        than simply running through each in a range of
                                        dimensionalities.  Here, it will search within
                                        the 20 - 50 dimensionality range.

                ndim = [range(1,6),'homogenize']
                                    =>  coord() will step through dimensions
                                        1, 2, 3, 4, and 5, but instead of analyzing
                                        the whole data array it will analyze the
                                        homogenized array, which is the square
                                        generally column-based covariance array.
                                        To specify the facet and other 'homogenize'
                                        parameters, use the "homogenize" argument
                                        (see below).  If homogenize = None, coord()
                                        will apply its own defaults.

                ndim = [range(20,50),'homogenize','search']
                                    =>  coord() will "search" across the 20 - 50 range
                                        of dimensionalities using the homogenized data.
                                        If there are few columns than rows, this is the
                                        fastest method.


            The "search" algorithm is a variation of the binary search algorithm
            and assumes a smooth U-shaped Dim (x-axis) x Objectivity (y-axis)
            curve.  When the data is more or less compliant with the model (all
            items belong to the same space), and the errors are randomly distributed,
            this is a reasonable assumption.  However, if the objectivity curve gets
            choppy, it is possible for 'search' to get stuck in local minima.
            If this is a problem, try having pseudomiss() make a larger
            proportion of cells randomly missing (default is 0.10).
            'search' should not be used blindly without cross-checking.  The
            objperdim output, printed in the Python shell, provides a log
            of dimensionalities and their prediction errors and correlations
            in the order they were attempted.

            The way 'search' works (over-simplifying a bit) is that for the
            first iteration two dimensionalities are chosen from the middle of
            the specified range of dimensionalities and estimates are calculated
            for each.  Their relative objectivity (slope) is calculated.  If the
            slope is negative, coord() guesses that the true dimensionality
            is further to the right, at a higher dimensionality.  If positive,
            coord() guesses that the true dimensionality is to the left, at a
            lower dimensionality.  It then defines a new range between
            the first dimensionality pair and the upper or lower boundary, and
            tests a new pair of dimensionalities at the middle of this new range.
            Again, their "slope" is calculated, the range further subdivided, and
            a new dimensionality pair drawn from the center of the subdivided range.
            This continues until the dimensionality stabilizes.  The dimensionality
            with the largest objectivity is declared "best" In theory, this should
            be near the dimensionality where 'search' ended up, but this does not
            necessarily happen.

            The "homogenize" option is fast and has mathematically desirable
            properties, but may run into problems with sparse matrices or
            arrays with lots of non-randomly missing data.

            ---------------
            "runspecs" specifies the conditions under which coord() will
            cease iterating back and forth between the row and column
            coordinates.  The syntax is:

                runspecs = [StopWhenChange,MaxIteration]

            For instance,

                runspecs = [0.0001,25]

            means keep iterating until either the percent change in the combined
            row and column coordinates between successive iterations is
            less than 0.0001 or until 25 iterations have occurred, whichever
            happens first.  When exploring a range of dimensionalities, a
            lot of your runs will run through the maximum number of iterations,
            so you may want to set this number as low as you can to save time.
            Currently, the same set of runspecs is used for both the exploratory
            "best dimensionality" runs and the final run.

            When my_dmnobj.verbose is True, coord() issues a run-time
            iteration report which includes a "Change" column.  This reports
            the percent change in the values of the coordinates.  Although
            it generally converges, it is not guaranteed to converge, nor
            does it have to.  If you add the flag 'EstConverge' to runspecs,
            you can get a run-time report of the change in the the RMSR
            (root mean squared residual) for each iteration.  This shows
            the convergence of the estimates to their closest possible
            approximation of the observations matrix.  So long as you
            specify that weightcoord = None, this quantity is guaranteed
            to converge so long as the conditions of Gauss-Markov are met.
            However, use 'EstConverge' only for checking the theory; it is
            computationally expensive.  Here's the syntax:

                runspecs = [0.0001,25,'EstConverge']

            ---------------
            "seed" (see discussion above) controls the selection of random
            starter coordinates.  Numpy allows you to select a set of random
            numbers using an integer.  Each integer corresponds to a different
            set of random numbers.  These integers are called "seeds".  A
            seed of "None" picks a seed at random.  In coord(), you can
            either specify the seed or tell coord() to look for a seed that
            meets an "objectivity" criterion that you specify.

                seed = None     =>  Each time you run coord(), pulls a
                                    different set of random starter coordinates.

                seed = 1        =>  Each time you run coord(), use a particular
                                    set of random starter coordinates which is
                                    the same every time you do the run.  This
                                    ensures the same coordinate solution.

                seed = 2, 3, ... N
                                =>  Each seed integer labels a different "seed",
                                    or unique set of random starter coordinates.

                seed = 'Auto'   =>  coord() will try out a range of 3 seeds and
                                    return the first that leads to coordinates
                                    that exceeds an objectivity correlation
                                    R > 0.80 (see comments above).

                seed = {'MinR':0.90,    =>  minimum R correlation
                        'MaxIt':[3,10], =>  maximum number of seeds to try:
                                            [for finding best dimensionality, for running at best dimensionality]

                                            Alternatively,
                                            'MaxIt':10  =>  maximum number of seeds is 10

                        'Facet':1,      =>  facet (0=rows,1=cols) to split into Group1 and Group2
                        'Stats':['Stab']=>  <'Stab','Acc','Obj','NonDegen','Err', or'Speed'>
                        'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[1,3,5,...]},
                        'Group2':{'Get':'AllExcept','Labels':'index','Entities':[1,3,5,...]}
                        }
                                =>  For the "best" dimensionality, find the first
                                    seed from 1 to 10 that yields a 'Stability'
                                    correlation greater than MinR = 0.90 (or pick
                                    the best of the 10).  While searching for best
                                    dimensionality, find the first seed from 1 to 3
                                    that yields correlation > 0.90 (or pick the best
                                    of the three). This R correlation is calculated
                                    using the 'Stab' or stability statistic.  Thus
                                    it's the correlation between the row coordinates
                                    calculated using Group1 items(cols) and Group2 items.

                                    In the case of 'Stats':['Acc'], R is the correlation
                                    between pseudo-missing values and estimates.  In
                                    the case of 'Stats':['Stab','Acc'] or 'Stats':['Obj'],
                                    R = (Stab * Acc)^(1/2).  The other statistics are included
                                    to provide extra information, but Damon doesn't use
                                    them to compute best seed.

                                    The Groups are defined using the 'Group1' and 'Group2'
                                    keys.  This syntax mimics that used in Damon's
                                    extract() method.  Go to help(core.Damon.extract)
                                    for a detailed description of this syntax.  It
                                    allows you to select entities to assign to each
                                    group.  If 'Facet' is 1, then the groups are groups
                                    of columns.  If 'Facet' is 0, then the groups
                                    are groups of rows.  You can specify entities by
                                    key name, by index, or by attribute.

                                    In this case, Group 1 is defined by index.  It consists
                                    of columns 1,3,5,... (counting from zero from the
                                    left edge of the whole array) -- all the odd-numbered
                                    columns.  Group 2 is 'AllExcept' these columns, in
                                    other words the even-numbered columns.

                                    If 'Labels' is 'key', then the entities would
                                    be key names (['Item1','Item2',...]).  If 'Labels'
                                    is an integer, this refers to a row in collabels
                                    which contains item attribute descriptors.  'Entities'
                                    would then contain the name of one of these attributes.
                                    All items with that attribute would be included in
                                    that Group.

                                    This syntax allows you do just about any 2-way
                                    comparison you might want.  Again, check out
                                    help(core.Damon.extract) to understand the
                                    syntax.  However, for most purposes the 'Auto'
                                    option is quite sufficient.

                Important
                ---------
                Searching for the best seed can be computationally
                time-consuming, especially when also trying to find the best
                dimensionality.  Each of a range of seeds is applied to each
                of a range of dimensionalities, and it multiplies quickly.
                To get the best performance, simply specify "seed = 1" or some
                other integer.  For many datasets this works fine, so long as you
                do it consciously.  But with noisy data, there is a non-zero
                probability of encountering a "bad seed", and you will want to
                do a thorough seed check to get a sense of how your data behaves.
                When automating critical analyses with such data, it is recommended
                that you trade some performance for a seed check.

            ---------------
            "homogenize" serves several functions.  It provides parameters
            to use when 'homogenize' is specified in the ndim argument for purposes
            of calculating best dimensionality.  If ndim contains 'homogenize'
            but the homogenize argument is None, coord() applies defaults.  homogenize
            also determines whether to apply coordinates obtained from analyzing
            the homogenized data to the observed data when calculating the final
            coordinates.  Options:

                homogenize = {'ApplyAncs':<True,False>, =>  Apply anchors calculated
                                                            from homogenized data to
                                                            observed data.  See "anchors"
                                                            below.

                              'Facet':<0,1>,            =>  0 means rows x rows; 1 means
                                                            columns x columns

                              'Max':<None,int>          =>  Number of opposing entities
                                                            to sample when homogenizing
                                                            (e.g., calculating covariance).
                                                            This is useful for tweaking
                                                            performance.  Default is 500.

                              'Form':<'Corr','Cov','SumProd','MeanProd'>
                                                        =>  Formula to use when homogenizing
                                                            data.  Default is 'Cov' for
                                                            covariance.
                             }

                These options are explained further in the tools.homogenize() docs.

                homogenize = None   =>  If 'homogenize' is specified in ndim,
                                        apply defaults (Facet = 1, Max = 500,
                                        Form = 'Cov').  Otherwise, do not
                                        homogenize data.

                homogenize = {'Facet':1,'ApplyAncs':True,'Max':500,'Form':'Cov'}
                                    =>  This means, compute a column x column
                                        covariance matrix.  For performance, sample
                                        no more than 500 rows for each vector
                                        product.  Then calculate coordinates (at
                                        the best or specified dimensionality) for
                                        the homogenized array and apply them as
                                        item anchors to the observed data array
                                        to compute row coordinates.

                If 'ApplyAncs' is False, the homogenize argument is only used
                if 'homogenize' is specified in the ndim argument.

            ---------------
            "anchors" is an extremely important part of Damon and essential
            to test equating.  It allows previously calculated coordinates 
            stored in a bank for one or both facets (as a Python pickle file or
            dictionary in memory) to be applied to the corresponding entities in
            the current data set.  The entity labels in the bank must match
            the key labels in the present data set where applicable. The number
            of matches is reported to ensure finding the appropriate entities
            in the bank.

            Use of "anchors" forces the whole system to align with the
            anchored coordinates.  If only some of the current entities
            are anchored, the remaining entities will be assigned coordinate
            values that maximally fit the observed data while remaining in the
            coordinate system of the anchored entities.  
            
            The 'Refresh_All' option modifies all the listed anchor coordinates 
            to maximally fit the current data set and and to ensure that all 
            entity coordinates are fully consistent.  The cost of 'Refresh_All'
            is that you are no longer applying the same set of coordinates
            across datasets, so you lose a little comparability.  When the
            data perfectly fit the model and there is a lot of data, 'Refresh_All'
            makes no difference.  However, error and misfit will cause coordinates
            calculated from different data sets to be slightly inconsistent.  As
            entities are added to the bank from different datasets, this can
            cause the bank values to be slightly patchy -- not wrong exactly, but
            not perfectly consistent.  In general, it is best to use 'Refresh_All'
            when the current data set is larger and of higher quality than
            the data set originally used to calibrate the anchors and when
            you are prepared to sacrifice a little comparability with previous
            datasets for the sake of improving the quality of the bank.
            
            You will usually want to keep your anchors unchanged and refresh
            only a few items that have started drifting or misfitting for
            some reason.  This is not handled by Refresh_All (which should
            remain False) but by specifying only those items under 'Entities'
            that you want to use as anchors, leaving out the drifting
            items.  coord() (and subsequently bank()) will overwrite those
            item coordinates in the bank with coordinates calculated from the 
            current dataset.  We are saying, in effect, that the item is
            still sufficiently useful to use in scoring but that its
            behavior has changed between when it was originally calculated
            and its appearance in the current dataset, so its coordinates
            need to be updated.
            
            If however you feel the item is just plain flaky, that it's
            prone to misfit, then you should remove it from the analysis 
            entirely as well as from the bank.

            "anchors" options are stored in a python dictionary that has
            the following defaults:
            
                anchors = {'Bank':______    # no default, must be specified
                           'Facet':1        # assumes bank stores column entities
                           'Coord':'ent_coord'  # type of anchor
                           'Entities':['All']   # use all possible anchors
                           'Refresh_All':False  # A total refresh is rare
                           }
            
            Here's the meaning of the options:

                anchors = None      =>  Do not anchor any coordinates.

                anchors =
                    {
                    'Bank':         =>  The filename/path of a pickle file
                                        containing an item/person bank of
                                        coordinates.  This file is built
                                        using Damon.bank().

                                    =>  Bank can also be a dictionary stored
                                        in memory created using Damon.bank().

                    'Facet':        =>  0 means the bank contains anchors for
                                        Facet 0 (row) entities.

                                        1 means the bank contains anchors for
                                        Facet 1 (column) entities

                    'Coord':        =>  This option is only used internally
                                        and should be ignored.

                                        'ent_coord' (default) means that entity
                                        coordinates will be used as anchors.
                                        'ear_coord' and 'se_coord' tell coord()
                                        to anchor on the EAR and SE coordinates
                                        that are stored in the bank for the
                                        purpose of computing standard errors.

                    'Entities':     =>  ['All'] means use anchors for all the
                                        entities in the bank which also appear
                                        in the data file.

                                    =>  ['1', '3', '4'] means use bank entities 
                                        '1', '3', and '4' as anchors.  Once
                                        my_obj.bank() is run, all other
                                        entities in the dateset will either
                                        be assigned coordinates for the first
                                        time or have existing coordinates
                                        overwritten (refreshed).
                                    
                                    =>  {'AllExcept':['1', '3', '4']} means
                                        use all entities EXCEPT '1', '3', '4'
                                        as anchors.  These three entities
                                        will have their coordinates
                                        overwritten/refreshed when bank()
                                        is run.
                                        
                                        Entities should include any that are
                                        not already in the bank.

                    'Refresh_All':  =>  False means do not modify the 
                                        coordinates of the anchored entities
                                        listed under 'Entities'; leave them 
                                        exactly as they are.

                                    =>  True means use the anchored values as
                                        "starter coordinates" to calculate 
                                        coordinates of the opposing 
                                        facet, then use those coordinates to 
                                        recalculate both the anchored entities 
                                        and the unanchored entities.  These
                                        "refreshed" values will replace those
                                        in the bank when bank() is run.
                    }

            Examples:

                anchors = {'Bank':'PersonBank.pkl', 'Facet':0}
                                    =>  Use the coordinates stored in the
                                        "PersonBank" pickle file.  Because
                                        these are person/row coordinates,
                                        specify 'Facet':0.  The default
                                        'Facet':1 would be wrong in this
                                        case.  By default, entities will
                                        be all persons who are both in the
                                        bank and in the current dataset.  No
                                        coordinates will be refreshed.
                
                anchors = {'Bank':'IBank.pkl'}
                                    =>  Use the coordinates stored in the
                                        'IBank.pkl' file.  No 'Facet' is
                                        specified, so by default it interprets
                                        them as item/column coordinates.  
                                        Entities will be all that are both
                                        in the bank and in the current
                                        dataset.  No coordinates will be
                                        refreshed.
                                        
                anchors = {'Bank':'IBank.pkl', 'Entities':['It4', 'It5']}
                                    =>  Use 'IBank.pkl' item coordinates as
                                        above, but only 'It4' and 'It5' as
                                        anchors.  All items on the current
                                        test but not in 'IBank.pkl' will be
                                        assigned coordinates in coord() outputs
                                        and will be added to the bank when 
                                        bank() is run.  All items on the 
                                        current test and in 'IBank.pkl' that 
                                        are NOT 'It4' and 'It5' will get new
                                        coordinates in coord() outputs and in
                                        the bank when bank() is run.
                                    
                anchors = {'Bank':'IBank.pkl', 
                           'Entities':{'AllExcept':['It4', 'It5']}
                                    =>  Use all item coordinates in the bank 
                                        EXCEPT 'It4' and 'It5' as
                                        anchors.  The 'It4' and 'It5'
                                        coordinates will be overwritten/refreshed
                                        when bank() is run.
                
                anchors = {'Bank':'IBank.pkl', 'Refresh_All':True}
                                    =>  Use all item coordinates in the bank
                                        as starter coordinates (where possible).
                                        but return refreshed coordinates.
                                        Overwrite/refresh those in the bank
                                        when bank() is run.
                
            WARNING
            -------
            It is essential that you call the correct bank.  A frequent error 
            is to refer to a bank that has the same name but was created with 
            incompatible data.  When doing an initial calibration, consider
            preceding the initial bank() call with something like:

                try:
                    os.remove('MyBank.pkl')
                except:
                    pass

            ---------------
            "quickancs" imports an array of "anchor" coordinates for a given facet,
            so that new coordinate values are not computed for that facet.  Unlike
            anchors=, the array must have exactly the same number and sequence
            of entities as the data array, formatted as ents x ndim.  It is a little
            faster and simpler than anchors since it does not need to do lookups,
            but mathematically it's no different.

                quickancs = [1, fac1coords array]
                                        =>  Anchor the column coordinates at the
                                            values specified by the ents x ndim
                                            fac1coords array.

                quickancs = [0, fac0coords_array]
                                        =>  Anchor the row coordinates at the
                                            values specified by the ents x ndim
                                            fac1coords array.

            ---------------
            "startercoord" allows you to specify the starting coordinates.  "None"
            means that coord() automatically creates random numbers to start
            with.  Otherwise, specify starter coordinates for Facet 0 (rows) and
            Facet 1 (columns) as [facet number,coordinate array].  The purpose 
            of using non-random starter coordinates is to speed up calculation 
            by providing a reasonable starting point.  All coordinates will 
            adjust to fit the data, however, unlike anchored coordinates.  
            Thus, starter coordinates are not a substitute for "anchoring" 
            the analysis to a set of coordinates.

                startercoord = [1, fac1coords array]
                                        =>  Start off with the ents x ndim
                                            column coordinates array and iterate from
                                            there.

                startercoord = [0, fac0coords array]
                                        =>  Start off with the ents x ndim
                                            row coordinates array.

            ---------------
            "pseudomiss" <None, True>, when True, tells coord() to make
            missing cells indentified by the pseudomissing index (generated by
            the pseudomiss() method).  This allows subsequent methods such
            as base_resid() to measure how successfully the "official"
            coord() outputs predict missing data.

            ---------------
            "miss_meth" specifies how the algorithm should handle missing cells.
            In the 'ImputeCells' method, missing cells are replaced with array means
            that are updated with estimated values at each iteration.  When there
            is a small amount of missing data (< 20%) this saves time because the
            arrays do not have to be resized for each entity/iteration.  However,
            as the data become more sparse (more missing cells) this method has
            increasing trouble converging on the correct values and is slower.
            This method is rarely used anymore.

            The 'IgnoreCells' option ignores missing cells entirely and sizes each
            least squares equation according to the data actually available
            in that row or column.  This makes coord() estimates robust to very
            sparse matrices, even up to 90% missing data, and speeds up such
            analyses considerably.  The 'IgnoreCells' method requires that for
            any given entity there are more observations than dimensions,
            a requirement that is somewhat relaxed in "ImputeCells".  When the
            requirement is not met, coord() issues a warning and sets the
            corresponding coordinates to nanval (the not-a-number value specified
            using data()).

            ---------------
            "solve_meth" specifies the method by which each element's coordinates
            are to be calculated.  Two methods are currently supported:  'LstSq'
            (ordinary least squares, the default) and 'IRLS' (iteratively reweighted
            least squares, for dichotomous data).  It is unclear whether the IRLS
            method actually is an improvement over ordinary least squares for
            dichotomous data in the context of an ALS decomposition.  It
            appears that it is not.  Until the question is decisively resolved, 
            it will remain as an option.

            ---------------
            "solve_meth_specs" is a dictionary of specs specific to the solve_meth
            method given in the previous argument.  It is None if solve_meth = 'LstSq'.
            If solve_meth = 'IRLS', the specifications are, for example:

                solve_meth_specs =
                    {'runspecs':[0.001,10],
                     'ecutmaxpos':[0.5,1.4],
                     'pcut':0.5
                     }

            See the documentation for dmnu.solve2() for information about needed
            input specifications.

            Currently, solve_meth and solve_meth_specs are not particularly
            useful.  They are there to provide Damon the necessary architecture
            for expanding to other solution methods.

            ---------------
            "condcoord" specifies how to condition row or column
            coordinates to meet a pre-defined mathematical requirement.
            It offers several pre-defined conditioning functions
            and provides the user with a syntax for defining his
            own conditioning functions.  While Damon coordinates
            do not require conditioning, there are benefits to doing
            so in many cases, such as making making them orthonormal
            (type {'Fac0':'Orthonormal','Fac1':None}). See "Benefits
            of Orthonormal" below.

            condcoord also provides an option for exploring how to deal
            with data generated (hypothetically) by unidimensional 
            non-negative coordinates.
            
            When it was designed, it was hoped that condcoord would be
            more powerful than it is, e.g., that it would make it 
            easy to impose non-negative multidimensional coordinates
            or to model "non-linear" arrays (where coordinates are not 
            linearly independent).  However, Damon's ALS algorithm  
            will not succeed in many of these circumstances; R and C
            may evolve into arrays that violate the requirements of
            least squares.
            
            condcoord includes a 'first' option that allows you to
            specify which facet coordinates should be created or
            conditioned first. For instance, if you want to set a 
            section of coordinates in one of the facets at a specific value, 
            specify that facet as 'first'.  The default (if unspecified) is:
                
                condcoord['first'] = 1
                
                =>  by default, Damon's ALS routine specifies that a random 
                    set of column coordinates be calculated first

            condcoord also includes a "refit" option which refits the
            estimates to the metric of the input data if necessary.
            This comes into play only when the process of conditioning
            the data forces the resulting estimates into a different
            metric, as wtih 'NonNeg_1D'.

            The options are:

                None        =>  Do not condition coordinates in any way.

                A Python dictionary containing specified functions or
                a user-defined function for imposing conditions on an
                array, with an optional 'refit' key:

                {'Fac0':condition, 'Fac1':condition, 
                 'first':first facet to condition}

                If 'first' is not specified, the row coordinates are
                calculated first.

            The condition functions are as follows:

                'Std'       =>  Adjust the specified coordiantes array so
                                that each column (dimension) has a standard
                                deviation of 1.0.  This option actually adjusts
                                both the target facet and the opposing facet,
                                though the latter is not used by the coord()
                                function.

                'Orthonormal'
                            =>  The coordinates array is transformed
                                to be orthonormal, i.e., so that the
                                dimensions are orthogonal to each other
                                (their dot products are zero) and the
                                root sum of squares (distance) for each axis
                                is 1.0.  This is the default option and has
                                several important benefits, described below.

                                condcoord = {'Fac0':'Orthonormal','Fac1':None}

                                Facet 0 coordinates are constrained to be
                                orthornormal.  Facet 1 coordinates float to
                                maximize fit to the data and to Facet 0.
                                This can be reversed so that the facet 1
                                coordinates are othonormal instead.

                'NonNeg_1D' =>  Force coordinates to be positive for both
                                facets.  This only works if ndim = 1.  This
                                condition creates coordinates that yield
                                estimates in an exponential metric that is
                                different from that of the observed data.
                                To get it back into the observed metric,
                                use the 'refit' option in base_est():

                                condcoord = {'Fac0':'NonNeg_1D',
                                              'Fac1':'NonNeg_1D'}

                                With this option you will want to precondition
                                the data with standardize().  In base_est(),
                                you will want to specify "refit" as follows:
                                    
                                If the original data is on a ratio scale
                                (0.0 to +inf):
                                    
                                    obj.base_est(refit='PreLogit')
                                
                                If the original data is on an interval scale
                                (-inf to +inf):
                                    
                                    obj.base_est(refit='LogDat')
                                    
                                If the original data is ordinal or sigmoid
                                (e.g., '0.0 - 3.0'):
                                    
                                    obj.base_est(refit='0-1')
                                
                                (These estimates tend not to be as accurate,
                                though column by column they will have a
                                strong positive monotonic relationship with
                                the "true" values, when suitably rescaled.)
                                    
                                You can also use Damon.rasch() for this type of
                                analysis.  See "The Curse of Undimensionality"
                                above.

                User-defined function
                            =>  The user can define a function outside of the
                                coord() method and refer to it here.  Examples:

                                def my_func(Fac):
                                    return np.exp(Fac)

                                Then, to apply my_func() to both facets:

                                condcoord = {'Fac0':my_func,
                                             'Fac1':my_func,
                                             'refit':'LstSq'
                                             }
                                
                                --------
                                Here's a fancier function:

                                def my_func2(Fac):
                                    Fac1 = np.zeros(np.shape(Fac))
                                    Fac1[:,0] = Fac[:,0] + 5
                                    Fac1[:,1] = Fac1[:,0]**2
                                    Fac1[:,2] = 1.0
                                    return Fac1

                                condcoord = {'Fac0':my_func2,
                                             'Fac1':None,
                                             }
                                             
                                -------
                                This function forces one dimension of a
                                set of row coordinates to equal the
                                respondent's log(age).
                                
                                # Assuming age is in the rowlabels, specify
                                # this in your code prior to d.coord(...)
                                e = dmn.Damon(d.standardize_out, 'datadict', 
                                              verbose=None)
                                
                                def apply_age(arr):
                                    age = np.log(e.rl_col[AGE].astype(np.float))
                                    arr[:, 0] = age
                                    return arr
                                
                                condcoord = {'Fac0':apply_age,
                                              'Fac1':None,
                                              'first':0}

                            Note: coord() is used internally for 
                            calculating stability and other statistics,
                            which take samples of the data array.  When
                            specifying cond_coord_ functions, make 
                            them indifferent to the length of R and C
                            so that such sampling can take place.  If
                            that is not possible, specify a seed and 
                            dimensionality in coord() to prevent coord()
                            from taking samples to calculate those stats.

            ---------------
            "weightcoord" <None, True> tells coord() to downweight coordinates
            that become large.  It is one way ("jolt_" is another) to avoid
            degenerate solutions.

            ---------------
            "jolt_" provides options for preventing or mitigating degenerate
            solutions -- coordinates that, due merely to the choice of random
            starter coordinates -- end up being unduly large.  If any coordinate
            values have distances from the origin that are more than sigma
            standard deviations from the mean, a "jolt" of randomness is
            applied to the whole coordinate array.  Syntax:

                jolt_ = None         =>  No jolts to be applied.

                jolt_ = [sigma, jolt_]

                jolt_ = [20, 1.5]    =>  If the maximum entity distance
                                         (root sum of squares of coordinates)
                                         is more than 20 standard deviations
                                         from the mean distance, add random
                                         numbers * 0.75 (1.5/2) above and below
                                         each value in the coordinates array.
                                         Do this also for the opposing
                                         facet in the next half-iteration.

            Warning:  If the jolt_ value is large and iteration ceases
            immediately after the application of random noise, the
            resulting coordinates and estimates may be inaccurate.
            Using self.verbose = True, review the printout of iterations.

            Note:  You may find that the same coordinates and large
            values get generated regularly, despite repeated jolts.
            This indicates a structural mathematical issue in the
            dataset that may have little to do with the choice of
            starter coordinates.  Extreme estimates may, in this case,
            actually be reasonable.

            In addition to jolt_, coord() offers two other methods for
            avoiding degenerate solutions:  1) conversion of one of the
            coordinate arrays to orthonormal (see condcoord arg);
            2) automatic weighting of each entity's coordinates by the
            inverse of its length (distance from origin) when calculating
            a new set of opposing coordinates, thus preventing extreme
            coordinates from self-amplifying across iterations.  This
            weighting is done automatically inside the coord() function.
            
            In practice, jolt_ is rarely used.

            ---------------
            "feather" specifies an amount of random error to add to the data
            before calculating coordinates.  Sometimes, the addition of error
            actually improves the stability of the coordinate system.  It is
            also useful to prevent rows and columns with insufficient variation.

            Benefits of Orthonormal
            -----------------------
            As mentioned, arrays converted to orthonormal are such
            that the dimensional coordinates are orthorgonal to each
            other (have dot products of 0.0) and the root sum of squares
            of each column equals 1.  Only one facet or the other can
            be specified as orthonormal.  If both facets are orthonormal
            the coord() function will lurch back and forth between
            two competing and irreconcilable coordinate spaces.  Some
            benefits of orthonormal (usually applied to the
            Facet 0 or row coordinates) are as follows:

                1)  It can be used to prevent the random starter
                    coordinates matrix from accidentally being
                    ill-conditioned (e.g., not linearly independent).

                2)  It helps prevent the coordinates from accidentally
                    exploding out of computing range during the
                    coord() iterative process.

                3)  It makes the ALS decomposition matrices
                    similar to those produced by Singular Value
                    Decomposition.

                3)  When applied to the final Fac0 (row) coordinate system,
                    the resulting column vectors of the opposite facet
                    occupy a space having a non-arbitrary origin such
                    that the coordinate vectors alone are "sufficient" to
                    calculate the objective length of each vector from
                    the origin, as well as the vector's angular relation
                    (cosine) relative to other column vectors, and other
                    statistics.  This length is the geometrical equivalent of
                    the standard deviation of the estimates for that
                    column, but freed from sampling considerations.  The
                    cosine between two column vectors is the geometrical
                    equivalent of their correlation, again freed from sampling
                    considerations.  The same properties hold for the row
                    coordinates if the orthonormal option is applied to
                    the columns (Facet 1).  The formulas are:
                    
                        length[i] = sqrt(sumsq(vector[i]))
                        
                        cosine[i] = dot(vector[i], vector[j]) /
                                       (length[i] * length[j])
                    
                    See tools.cosine() and tools.length() for more information.

                    Why this is important:

                    a)  The vector distances and cosines are "objective" --
                        independent of the sample of entities used to calculate
                        them.  That means statistical assumptions such
                        as representative samples are not required for calculating
                        these statistics.  Conventional distances and cosines
                        (a.k.a., standard deviations and correlations) are
                        extremely sample-dependent.

                    b)  They are convenient, in that they make it easy to
                        calculate means, standard deviations, and correlations
                        (actually, their more accurate and reliable objective
                        equivalents) without having to refer to the
                        observations or estimates matrices.

                    c)  This means that the coordinate vectors become,
                        in effect, "sufficient" statistics -- sufficient to
                        describe the properties of each row/column without
                        reference to the underlying data.  They are the
                        multidimensional equivalent of Fischer's definition
                        of sufficiency as used in Rasch models.


        Examples
        --------

            [under construction]


        Paste method
        ------------
            coord(ndim = [[1]],      # [<None,[[dim list],'search','homogenize']> => set dimensionality or search range, possibly homogenized]
                  runspecs = [0.0001,20],  # [<[StopWhenChange,MaxIteration]>]
                  seed = 'Auto',  #[<None,int,'Auto',{'MinR':0.90,'MaxIt':<10,[3,10]>,'Facet':<0,1>,'Stats':[<'Stab','Acc','Obj','PsMsResid','NonDegen'>],'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[...]},'Group2':{'Get':'AllExcept','Labels':'index','Entities':[...]}}>]
                  homogenize = None,    # [<None,{'ApplyAncs':<True,False>,'Facet':1,'Max':500,'Form':'Cov'} => homogenize params]
                  anchors = None,    # [<None,{'Bank':<bank,pickle file>,'Facet':<0,1>,'Entities':<['All',list entities]>,'Refresh_All':<bool>}> ]
                  quickancs = None,  # [<None,[<0,1>,ent x ndim array]> => facet, anchor array]
                  startercoord = None,    # [<None,[<0,1>,ent x ndim array]> => facet, starter array]
                  pseudomiss = None,    # [<None,True> => make cells pseudo-missing for "official" run]
                  miss_meth = 'IgnoreCells', # [<'ImputeCells' => impute iterable values for missing cells; 'IgnoreCells' => skip missing cells>]
                  solve_meth = 'LstSq', # [<'LstSq','IRLS'> => method for solving equations]
                  solve_meth_specs = None,    # [<None, spec dictionary> => specs for solve_meth (cf. solve2() docs), e.g. for IRLS -- {'runspecs':[0.001,10],'ecutmaxpos':[0.5,1.4],'pcut':0.5}]
                  condcoord = {'Fac0':'Orthonormal','Fac1':None},  # [<None,{'Fac0':<'a func',myfunc>,'Fac1':<'a func',myfunc>}> ]
                  weightcoord = True,   # [<None,True> => downweight influential coordinates]
                  jolt_ = None,  # [<None,[sigma,jolt_]> e.g., [20,1.5] => Apply 1.5 noise factor if sigma exceeds 20]
                  feather = None,     # [<None,float> => add small amount of randomness to the data]
                  )

        """
        if condcoord_ is not None:
            condcoord = condcoord_

        class coord_Error(Exception): pass

        if self.verbose is True:
            print 'coord() is working...\n'

        # Extract list of dimensionalities
        if ndim is not None:
            list_ = False
            for i in range(len(ndim)):
                if isinstance(ndim[i],list):
                    dims = ndim[i]
                    list_ = True

            if list_ is False:
                exc = 'Unable to figure out ndim.\n'
                print exc
                print 'ndim =',ndim
                raise coord_Error(exc)

            # Get best dimensionality
            if (len(dims) > 1
                or 'search' in ndim
                ):
                dmn.utils._bestdim(locals())    # bestdim and objperdim outputs are assigned to Damon as attributes
                bestdim = self.bestdim

            # Get best seed
            if (seed == 'Auto'
                  or seed == 'Auto4BestDim'
                  or seed == 'Auto4BestDim_Fast'
                  or isinstance(seed,dict)
                  ):
                if (anchors is None
                    and quickancs is None
                    and startercoord is None
                    and (homogenize is None
                         or homogenize['ApplyAncs'] is not True)
                    ):
                    dmn.utils._bestseed(locals())   # seed output dict assigned to Damon as attribute
                else:
                    pass

        # Get homogenized coordinates for final run
        if (homogenize is not None
            and homogenize['ApplyAncs'] is True
            ):

            # Get homogenize args
            h = homogenize
            h_args = {'facet':1,'form':'Cov','max_':500,'nanval':self.nanval}
            if h is not None:
                for key in h.keys():
                    if key == 'Facet':
                        h_args['facet'] = h[key]
                    if key == 'Form':
                        h_args['form'] = h[key]
                    if key == 'Max':
                        h_args['max_'] = h[key]

            # Get homogenized data
            try:
                homodata = self.homogenized

            # Extract the correct data to homogenize
            except AttributeError:

                if self.verbose is True:
                    print 'Homogenizing data...'

                try:
                    data = self.standardize_out['coredata']
                    nanval = self.standardize_out['nanval']
                except AttributeError:
                    try:
                        data = self.parse_out['coredata']
                        nanval = self.parse_out['nanval']
                    except AttributeError:
                        try:
                            data = self.score_mc_out['coredata']
                            nanval = self.score_mc_out['nanval']
                        except AttributeError:
                            try:
                                data = self.extract_valid_out['coredata']
                                nanval = self.extract_valid_out['nanval']
                            except AttributeError:
                                try:
                                    data = self.merge_info_out['coredata']
                                    data = self.merge_info_out['nanval']
                                except AttributeError:
                                    try:
                                        data = self.data_out['coredata']
                                        nanval = self.data_out['nanval']
                                    except:
                                        exc = 'Unable to find data to analyze.\n'
                                        raise coord_Error(exc)

                h_args['arr'] = data
                h_args['nanval'] = nanval
                homodata = dmn.tools.homogenize(**h_args)

                if self.verbose is True:
                    print 'Data has been homogenized.\n'

            # Calc coordinates based on homogenized data
            try:
                bestdim = [self.bestdim]
            except AttributeError:
                bestdim = dims
            hd = Damon(homodata,'array','RCD',nheaders4rows=0,nheaders4cols=0,
                            validchars=None,verbose=self.verbose)

            if self.verbose is True:
                print 'Computing coordinates of homogenized array.\n'

            hd.coord([bestdim],runspecs,seed,None,None,None,None,None,
                     miss_meth,solve_meth,solve_meth_specs,
                     condcoord,weightcoord,jolt_,feather
                     )
            hd_coord_out = hd.coord_out

            _locals = {'self':self,'ndim':None,'runspecs':runspecs,'seed':None,
                       'homogenize':None,'anchors':None,
                       'quickancs':[h_args['facet'],hd_coord_out['fac0coord']['coredata']],
                       'startercoord':None,'pseudomiss':None,
                       'miss_meth':miss_meth,'solve_meth':solve_meth,'solve_meth_specs':solve_meth_specs,
                       'condcoord':condcoord,'weightcoord':weightcoord,
                       'jolt_':jolt_,'feather':feather
                       }

            if _locals['seed'] != 'Auto4BestDim':
                coord_out = dmn.utils._coord(_locals)

                # Assign attributes
                self.coord_out = coord_out
                self.fac0coord = coord_out['fac0coord']
                self.fac1coord = coord_out['fac1coord']
                self.facs_per_ent = coord_out['facs_per_ent']

                if self.verbose is True:
                    print 'coord() is done -- see my_obj.coord_out'
                    print 'Contains:\n',self.coord_out.keys(),'\n'

            else:
                pass

        else:
            if locals()['seed'] != 'Auto4BestDim':
                coord_out = dmn.utils._coord(locals())

                # Assign attributes
                self.coord_out = coord_out
                self.fac0coord = coord_out['fac0coord']
                self.fac1coord = coord_out['fac1coord']
                self.facs_per_ent = coord_out['facs_per_ent']

                if self.verbose is True:
                    print 'coord() is done -- see my_obj.coord_out'
                    print 'Contains:\n',self.coord_out.keys(),'\n'

            else:
                pass

        return None



    ##########################################################################

    def sub_coord(self,
                  subspaces = {'row':1},    # [<{'row':int row of subspace labels}, ['key', {'sub0':['i1', i2'],...}], ['index', {'sub0':[0, 1],...}]> => identify subspaces]
                  coord_subs = {'All':{'ndim':[[2]]}},  # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for each subspace or for all subspaces]
                  coord_resids = {'All':{'ndim':[[1]]}},    # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for analyzing residuals of each subspace]
                  unique_weights = {'All':'Auto'},    # [<{'All':'Auto'} or {'sub0':<'Auto', p>, 'sub1':<'Auto', p>, where 0 < p < 1  > => how much to weight unique component for each subspace]
                  share_if = {'targ_<':30, 'pred_>': 4},   # [<{'targ_<':int, 'pred_>':int}> => when to share info between subspaces]
                  min_rel = 0.02,   # [< 0 < min_rel < 1  > => minimum reliability to use in unique weighting formula]
                  rpt_optimal = None,    # [<None, True> => calculate and return optimal unique weight]
                  ):
        """Calculate coordinates and estimates for each specified subspace.

        Returns
        -------
            The method returns None but assigns outputs to the Damon
            object, accessed by my_obj.sub_coord_out.  Outputs contain:

                {'subspace_0':{'fac0coord':_, 'fac1coord':_},
                 'subspace_1':{'fac0coord':_, 'fac1coord':_},
                 ... => coord() style outputs for each subspace
                 'estimates':_, =>  datadict of estimates for whole array
                 'optimal_weights':_ =>  optimal unique weight calculated for
                                         each subspace
                 }

             Also assigned to the Damon object:
                 self.facs_per_ent = [1, 2]

             my_obj.base_est() simply pulls and returns the 'estimates' output.

             sub_coord() runs in place of coord() and base_est().  It does
             not require use of subscale().

             Workflow:
                d = dmn.Damon(...)
                d.score_mc(...)
                d.standardize(...)
                d.sub_coord(...)
                d.base_est(...) (optional)
                d.base_resid(...)
                d.base_ear(...)
                d.base_se(...)

        Comments
        --------
            sub_coord() is the latest response to the subspace (subscale)
            problem and supersedes the use of subscale() and objectify()
            for this purpose.

            The "subspace problem" can be stated as follows:  The coord() function
            requires that all items participate in the same space, i.e., that
            the generating or natural coordinates of each item are non-zero
            for all common dimensions and zero for all other dimensions.  A
            test of word problems, each requiring some mix of math and language
            skill, is the canonical example.  This is called "within-item
            multidimensionality".

            However, in many cases tests are constructed of groups of items
            that get at some unique construct not shared by the other item
            groups.  This is a test containing a group of math items and another
            group of language items.  The two item groups may be correlated
            and have dimensions in common, but they also embody their own
            unique dimensions.  This implies that some dimensions are zeroed
            out in the generating coordinates for some groups and not others.
            This is called "between-item multidimensionality".

            Example:  Say our test has separate math and language items.  Each
            subspace will have its own unique dimension, plus a common dimension.
            When you chart it out, this actually creates a 3-dimensional space
            (and, in fact, coord() will report '3' as the optimal dimensionality).
            The chart below sketches out the dimensional schema, applied to
            all the items in each subspace.  (A value of 1 simply means "non-zero").

                {'Math':[1, 0, 1],
                 'Lang':[0, 1, 1]}

            Do you see why three dimensions?  Each subspace gets its own
            unique dimension, plus there's a third common dimension.  This
            results in a dimension being zeroed out for each subscale and three
            dimensions in all.  However, to analyze each subscale separately
            you need only two dimensions.  These dimensional schema become
            more complicated when you have more subspaces.  Here is a test
            with three subspaces:

                {'Grammar':[0,0,1,1],
                 'Vocab':  [0,1,0,1],
                 'Reading':[1,0,0,1]}

            We end up with four dimensions overall, but each subspace can be
            analyzed separately with two dimensions.

            The 'apply_zeros' parameter in create_data() is used to build
            datasets like this.

            The coord() function is good at handling within-item multidimensional
            datasets but is mathematically not suited to handle between-item
            multidimensional datasets.  sub_coord() fills the gap and permits
            analysis of between-item multidimensional datasets.

            sub_coord() requires that each item be assigned a labeled
            subspace.  In other words, you need a theoretical basis for
            grouping items and a reason to believe that they require skill
            dimensions that the other item groups do not.

            sub_coord() analyzes each group of items separately (since they
            all participate in their own common subspace), then SHARES information
            between the subspaces in order to increase the precision of each.

            The sharing process works as follows.  Say we have two subspaces,
            one called the "target" (the subspace we are trying to estimate), the
            other called the "predictor" (the subspace from which we are transferring
            information to the target).  sub_coord() analyzes the predictor items
            using coord().  The resulting row (person) coordinates are anchored
            and applied to the target items to obtain target column coordinates and
            cell estimates for each target cell.  These estimates answer the
            question:  how would each person perform on each target item given
            only information obtained from the predictor items?  They capture
            information from whatever dimensions are COMMON between the target
            and the predictor.

            But we know (or assume) that the target subspace contains UNIQUE
            dimensions that the predictors don't know about.  How do we capture
            that information?  sub_coord() calculates RESIDUALS (differences)
            between the target estimates and the observed data.  These residuals
            are the product of:  a) the unique dimensions; b) noise.  To strip
            out the noise, sub_coord() runs coord() on the residuals matrix.
            The resulting R coordinates represent the UNIQUE component of the
            target subspace.  The unique R is appended to the common R calculated
            from the predictor data and the combined R is used to calculate a new
            set of target cell estimates.  The new estimates synthesize information
            drawn from both the target subspace and the predictor subspace.

            The final solution is obtained by synthesizing target estimates
            obtained across all subscales (again, using coord()).

            One important step in the process involves how to combine the
            unique and common R's.  In the absence of noise one could give
            the two R's equal weight in computing estimates, but the presence
            of noise means that the unique R may need to be down-weighted.
            sub_coord() uses a weighting formula that relies on two statistics:
            a) the reliability of the coord() analysis of residuals to obtain
            the unique R; b) the percent of variance explained by the unique
            component.  The more reliable the analysis of the unique component
            of the data, and the more pronounced the effect of the unique
            component on the data, the higher the unique weight (maximum
            of 1.0, minimum of 0.0).  As either reliability goes to zero,
            or the role of p_unique goes to zero, the unique weight goes
            either to zero or the minimum specified by the min_rel parameter.

            While finding the optimal downweight is an intrinsically
            tricky task, the formula performs reasonably well in leading to
            cell estimates that are as close to "true" as possible.

            So, when to use coord() and when to use sub_coord()?  If you have
            grounds for believing that all items contain at least a little
            bit of each dimension, or if you think the effect of the unique
            dimensions of subspace is small relative to that of the common
            dimensions, then a regular coord() approach is preferable.  It will
            be faster and produce better estimates.  An analysis of fit (using
            the base_fit() method) will either confirm that the items participate in
            the same space or tell you which items may need to be assigned their
            own subspace.  If you have theoretical grounds for believing that
            items really should be broken into separate groups, that the effect
            of each group's unique component is likely to be large, and this is
            confirmed by doing analysis of fit based on a regular coord() analysis,
            then sub_coord() is the way to go.

            To model and practice on between-item multidimensional datasets,
            refer to the 'apply_zeros' parameter docs in dmn.create_data().

            Anchoring
            ---------
            The rasch() and coord() methods support person and item anchoring --
            applying the coordinates derived from one analysis to another
            dataset.  sub_coord() does not yet support anchoring.  While possible
            in principle, it is difficult to implement.  To compare persons across
            different analyses, there are several approaches one can take.

                1.  Merge all datasets and analyze as one big dataset, or define
                    a "reference dataset" and merge it with each new datset
                    separately and do the sub_coord() analysis.

                2.  Run sub_coord() separately on each dataset and use
                    summstat() to compute person measures for each subscale
                    with the condition that the same set of items is used
                    to define each subscale for each dataset.

                3.  Create item banks and apply anchors for each subspace
                    separately. Run sub_coord().  Get the subspace coordinates for
                    a given space -- my_obj.sub_coord_out['sub0'].  Trick the
                    bank() method into creating a bank from the 'sub0' coordinates
                    by creating a new object and assigning the subspace coordinates
                    to coord_out.

                        new_obj.coord_out = my_obj.sub_coord_out['sub0']
                        new_obj.bank()

                    The problem with this approach is that it does
                    not share information between subscales to compute
                    person measures, so they will be less precise than the
                    person measures for the original dataset.

            Standard Errors
            ---------------
            The formula for computing cell standard errors for sub_coord() is
            "under development".  In the meantime, the regular base_se()
            formula works okay for most situations.  Its limitations show up
            as a subspace becomes small, less than 10 items or so.  In that
            case, the standard error for the cells in that subspace will be
            too small and the reliabilities too high.  Above 10 items tends
            not to be a problem.

            The workflow for computing standard errors is the same as
            for coord():

                d = dmn.Damon(...)
                d.sub_coord(...)
                d.base_est()
                d.base_resid()
                d.base_ear()
                d.base_se()
                se_array = d.base_se_out['coredata']

            Computing and Reporting Subscale Measures
            -----------------------------------------
            sub_coord() computes a set of row and column coordinates for each
            subspace as well as cell estimates.  To convert these into subscale
            measures, aggregate the cell estimates (plus standard errors and
            other statistics, as desired) using summstat().  Because summstat()
            builds one stats for one construct at a time, you will need to
            run it separately for each subscale using outname. Here is a sample
            script:

                d = dmn.Damon(...)
                d.sub_coord(subspaces = {'row':1},
                            coord_subs = {'All':{'ndim':[[2]]}},
                            coord_resids = {'All':{'ndim':[[1]]}},
                            unique_weights = {'sub0':'Auto', 'sub1':'Auto'},
                            share_if = {'targ_<':70, 'pred_>': 4,},
                            min_rel = 0.02,
                            rpt_optimal = True
                            )
                d.base_est()
                d.base_resid()
                d.base_ear()
                d.base_se()
                d.est2logit()

                subs = ['sub0', 'sub1']
                for sub in subs:
                    d.summstat(data = 'est2logit_out',
                               getstats = ['Mean', 'SE', 'Count'],
                               getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]}
                               getcols = {'Get':'NoneExcept', 'Labels':1, 'Cols':[sub]},
                               outname = sub
                               )

                person_stats = d.summstat_out['0']['row_ents']
                print 'person_stats for 'sub0' =\n', person_stats
                        => This reports Measures, SE's, Counts in logits

            Performance
            -----------
            sub_coord() is computationally intensive because it requires
            multiple coord() runs for each subspace.  A dataset with
            two subspaces requires 6 coord() runs.  A dataset with
            three subspaces requires 12 coord() runs:

                n_coord()_runs = n_subs + n_subs**2

            Obviously the computational load gets out of hand quickly, even
            though a higher number of subspaces is somewhat mitigated by
            fewer items per subspace.  Therefore, if it is possible to model
            the data with vanilla coord(), that is preferable.  There
            are several ways to lighten the load using the coord_subs and
            coord_resids parameters.  Specify one dimension rather than
            a range of dimensions.  Specify an integer seed using coord()'s
            seed parameter, e.g.,

                coord_subs = {'All':{'ndim':[[2]], 'seed':1}}

            An inspection of coord() reveals other tricks to save time. The
            coord() defaults are set to maximize quality of the outputs,
            not computational speed, but in the context of sub_coord() they
            are probably overkill.

            Once you have a sense of the optimal weights for the unique
            component of each subspace, set them manually in unique_weights
            instead of using 'Auto'.

            Lower share_if's 'targ' setting.  If you can achieve an acceptable
            reliability with 15 items in the target subspace, there is no
            need to specify 'targ_<':30.  Recall that the purpose of sub_coord()
            is to transfer information between dimensionally distinct subspaces.
            If a regular analysis of a given subspace is sufficient, there's
            no need to spend computational cycles transferring information from
            other subspaces.


        Arguments
        ---------
            "subspaces" is used to assign items to each subspace, each item
            to only one subspace.  The assignment is done either by referring
            to a header subspace label row in the dataset, or by assigning
            items to each subspace explicitly.

                subspaces = {'row':1}   =>  Obtain item subspace labels from
                                            row 1 (counting from 0) of collabels.

            Explicit assignment: [reference type, subspace dict]

                subspaces = ['key', {'sub0':['i1', 'i2'], 'sub1':['i5', 'i7']}]
                                        =>  Referring to items by their key,
                                            assign items 'i1' and 'i2' to
                                            subspace 'sub0'.  Assign items
                                            'i5' and 'i7' to 'sub1'.

                subspaces = ['index', {'sub0':[1, 2], 'sub1':[5, 7]}]
                                        =>  Referring to items by their index,
                                            counting from 0 and relative to
                                            the coredata array, assign the
                                            items in position 1 and 2 to
                                            subspace 'sub0'.  Assign the items
                                            in positions 5 and 7 to 'sub1'.

            ------------
            "coord_subs" are the coord() parameters to use when analyzing
            each subspace separately (see discussion of "predictor"
            subspaces above).  They are stored in a dictionary.  Parameters
            not explicitly identified will be assigned the coord() default.
            Generally, you will only want to specify the 'ndim' parameter.
            To save space, if all subspaces can be analyzed with the same
            parameters, they can be assigned once to the 'All' key.

                coord_subs = {'All':{coord() params}},
                    or
                coord_subs = {'sub0':{sub0 params},
                              'sub1':{sub1 params},...}

                coord_subs = {'All':{'ndim':[[2]]}}
                                        =>  Analyze all subspaces with a
                                            dimensionality of 2.

                coord_subs = {'Grammar':{'ndim':[[1]]}},
                              'Vocab':{'ndim':[[2]]}},
                              'Reading':{'ndim':[[1, 2, 3]], 'seed':1},
                              'Spelling':None
                              }
                                        =>  Analyze 'Grammar' with
                                            dimensionality of 1, 'Vocab'
                                            with dimensionality of 2.  For
                                            'Reading', automatically find
                                            the best dimensionality out
                                            of 1, 2, and 3.  Use a 'seed'
                                            parameter of 1 to save time
                                            (by default, coord() searches
                                            for the best seed out of 3).

                                            'Spelling':None means don't
                                            run coord() on the 'Spelling'
                                            subspace.  Use this option
                                            when 'Spelling' has too few
                                            items to analyze, e.g., one
                                            item.  However, if you have
                                            not specifed 'share_if' to
                                            exclude 'Spelling', coord_subs()
                                            will use the raw data in
                                            'Spelling' as predictors of
                                            other subspaces, instead of
                                            its R coordinates.

                As suggested above, many subspace dimensionality schema
                will result in "2" being the optimal dimensionality when
                analyzing each subspace separately: one for the unique
                dimension, one for the common dimension.

            ------------
            "coord_resids" are the coord() parameters to use when analyzing
            the residuals array of the target subspace to obtain its
            unique component (see discussion of unique dimensions above).

                coord_resids = {'All':{'ndim':[[1]]}}
                                        =>  When analyzing the residuals
                                            of each subspace to obtain
                                            the unique dimension, use
                                            a dimensionality of 1.  In
                                            other words, we are assuming
                                            that after the common dimensions
                                            are filtered out, there is only
                                            one remaining unique dimension.

                coord_resids = {'All':{'ndim':[[0, 1, 2, 3]]}}
                                        =>  In this case, we aren't sure
                                            how many unique dimensions
                                            there are, so coord() finds
                                            the dimensionality that best
                                            models the residuals.  Note
                                            the "0".  If coord() finds a
                                            dimensionality of zero, it
                                            means there ARE NO unique
                                            dimensions aside from the
                                            common dimension.  The residuals
                                            are essentially random noise.

                                            This is one of the cases where
                                            checking for zero dimensionality
                                            actually makes a lot of sense.

                coord_resids = {'Grammar':{'ndim':[[1]]}},
                               'Vocab':{'ndim':[[2]]}},
                               'Reading':{'ndim':[[1, 2, 3]], 'seed':1},
                               'Spelling':None
                               }
                                       =>   For 'Grammar' analyze the residuals
                                            with 1 dimension.  For 'Vocab', use
                                            2.  For 'Reading' pick the best of
                                            3.  For 'Spelling', don't analyze
                                            the residuals at all; all variance
                                            will be modeled using the other
                                            subspaces.

                For many subspace dimensionality schema, "1" will be the
                optimal dimensionality for analyzing residuals -- one dimension
                to represent the unique dimensionality for that subspace. But
                it is entirely possible for a subspace to have more than one
                unique dimension.  This will depend in part on how many
                dimensions are represented in the "predictor" subspace.  The
                unique dimensionality is based on whatever is "left over"
                from the predictor subspace.  By specifying a range of
                dimensions to check, sub_coord() will automatically find
                and use the correct number of unique dimensions.

            ------------
            "unique_weights" is used to set the weight of the unique
            dimension of each subspace when combining it with the common
            dimension(s) derived from the predictor subspace (see discussion
            above).  Actually, "weight" is a misnomer.  The R coordinates
            derived from the predictor subspace always have full weight.
            However, because the R coordinates derived from analyzing the
            residuals of the target subspace may be more or less useful
            depending on how reliable the analysis and how pronounced the
            unique dimension, it may be necessary to downweight the unique
            R coordinates to yield the most accurate predictions.  The
            unique_weights parameter provides a way to specify these weights
            manually when doing simulation studies or for cases where there
            may not be enough data to estimate the weights automatically, as
            when the target subscale has only one item.

            Values range from 0.0 to 1.0, where 1.0 means the unique
            component is not downweighted at all.

                unique_weights = {'All':'Auto'}
                                        =>  For all subscales, let the
                                            function estimate the optimal
                                            downweight.

                unique_weights = {'sub0':'Auto', 'sub1':0.50}
                                        =>  For the 'sub0' subscale, let
                                            the function estimate the optimal
                                            downweight.  For the 'sub1'
                                            subscale, assign a downweight of
                                            0.50.

            ------------
            "share_if" is used to manage when to share information between
            subspaces.  For instance, if a target subspace has 95 items and
            the predictor subspace has only 5, the best results will be
            obtained by doing a regular coord() run on just the target subspace
            and ignoring the predictor subspace.  The predictor doesn't
            contain a lot of additional information, plus its estimation of
            the common component will be hampered by its small size.  The
            share_if parameter allows you to specify how many items a target
            needs to stand on its own and the number of items a predictor
            subspace needs to contain to be used.

                share_if = {'targ_<':30, 'pred_>':4}
                                        =>  Only share predictor information
                                            with the target if there are fewer
                                            than 30 items in the target AND
                                            more than 4 items in the predictor.

                                            This applies to all target/predictor
                                            combinations.

            ------------
            "min_rel" sets a minimum reliability for use in the unique component
            downweighting formula.  The formula is:

                unique_weight = sqrt(reliability * p_unique)

            where reliability is the Cronbach-alpha for the analysis of residuals
            and
                p_unique = var_explained_unique / (var_explained_unique + var_explained_common)

            When reliability goes to zero, the unique dimension is ignored entirely,
            even if p_unique is very large (close to 1.0).  Simulations suggest that
            it often makes sense to retain some influence for p_unique even when the
            reliability goes to zero.  This is controlled with min_rel.

                min_rel = 0.05          =>  Don't let reliability go below 0.05
                                            in the weighting formula.

            ------------
            "rpt_optimal" specifies whether to report the optimal weight even when
            unique_weights manually specifies the weights.  This is only useful when
            doing simulation studies to determine whether the estimated optimal weight
            actually does the best job across a range of weights of matching the
            "true" cell values.

                rpt_optimal = <None, True>

        Examples
        --------
            [under construction]


        Paste Method
        ------------
            sub_coord(subspaces = {'row':1},    # [<{'row':int row of subspace labels}, ['key', {'sub0':['i1', i2'],...}], ['index', {'sub0':[0, 1],...}]> => identify subspaces]
                      coord_subs = {'All':{'ndim':[[2]]}},  # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for each subspace or for all subspaces]
                      coord_resids = {'All':{'ndim':[[1]]}},    # [<'All' or <'sub0', 'sub1'>:<None, coord() params>> => coord() parameters for analyzing residuals of each subspace]
                      unique_weights = {'All':'Auto'},    # [<{'All':'Auto'} or {'sub0':<'Auto', p>, 'sub1':<'Auto', p>, where 0 < p < 1  > => how much to weight unique component for each subspace]
                      share_if = {'targ_<':30, 'pred_>': 4},   # [<{'targ_<':int, 'pred_>':int}> => when to share info between subspaces]
                      min_rel = 0.02,   # [< 0 < min_rel < 1  > => minimum reliability to use in unique weighting formula]
                      rpt_optimal = None,    # [<None, True> => calculate and return optimal unique weight]
                      )

        """

        if self.verbose is True:
            print 'sub_coord() is working...\n'

        # Run the damon utility
        sub_coord_out = dmn.utils._sub_coord(locals())
        self.sub_coord_out = sub_coord_out
        self.facs_per_ent = sub_coord_out['facs_per_ent']

        if self.verbose is True:
            print 'sub_coord() is done -- see my_obj.sub_coord_out'
            print 'Contains:\n',self.sub_coord_out.keys(),'\n'

        return None






    ##########################################################################

    def objectify(self,
                  targ_ents = {'Get':'AllExcept','Labels':'key','Cols':[None]},   # [{'Get':_,'Labels':_,'Cols':_} => row/col entities to objectify]
                  pred_ents = {'AllTargs':'AllExceptTarg'}, # [<{targ1:{'Get'...},...},{'AllTargs':{'Get'...}},{'AllTargs':'AllExceptTarg'},{'Subscales':{'Get'...[<sub1,'All'>]}]
                  ndim = 'Refer2Coord', # [<['Refer2Coord',[[dim list],'search','homogenize']> => n dim(s) for predicting targ_ents]
                  runspecs = [0.0001,10],   # [['StopWhenChange','MaxIteration'] => for calculating each targ_ent coordinate set]
                  seed = 'Auto',  #[<'Refer2Coord',None,int,'Auto',{'MinR':0.90,'MaxIt':<10,[3,10]>,'Facet':<0,1>,'Stats':[<'Stab','Acc','Obj','PsMsResid','NonDegen'>],'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[...]},'Group2':{'Get':'AllExcept','Labels':'index','Entities':[...]}}>]
                  starters = True,   # [<None,True> => for speed, calc starter coords and apply their dimensionality to all predictors]
                  summdim = None,      # [<None,[[dim list],'search','homogenize']> => dim to summarize all estimates in potentially higher space]
                  center = True,    # [<None,True> => center objectified estimates on observations per entity]
                  overwrite = True, # [<None,True> => overwrite coord_out and base_est_out]
                  ):
        """Objectify specified entities, return coordinates or estimates.

        Returns
        -------
            The method returns None but assigns outputs to the Damon
            object as my_obj.objectify_out.  This is a dictionary
            containing two datadicts:

            {'obj_est':         =>  datadict of mixed objectified
                                    and non-objectified estimates.
                                    Contains an 'ndim' key.

            'obj_coord':        =>  Dictionary containing:
                'fac0coord'     =>  row entity x dimension datadict
                'fac1coord',    =>  column entity x dimension datadict
                'ndim'          =>  dimensionality
            }

            'obj_coord' is set at None if the summdim argument is not
            specified.

            If summdim is specified, obj_coord overwrites (or builds if
            coord() was not run) coord_out.  Otherwise, coord_out remains
            the output of coord(), if run.

            If overwrite = True, obj_est overwrites (or builds
            if base_est() was not run) base_est_out.

            Thus, objectify() can be run in place of coord() and base_est().

            An attribute called obj_dim is added to the Damon object.

            Workflow:
                d = dmn.Damon(...)
                d.score_mc(...)
                d.subscale(...)
                d.standardize(...)
                d.objectify(...)

        Comments
        --------
            objectify() is used to do the following with specified entities:

                a)  Minimize the bias that each entity observation has
                    on the entity's coordinates,

                b)  Project each specified entity into the subspace
                    defined by a specified set of entities, or of all
                    remaining entities, stripping out all aspects
                    (extra dimensions) of the entity that are
                    not shared by the defined entities.  This is
                    good for creating subscales.

            First, the bias.  When datasets are small or the number
            of observations is small relative to the number of dimensions,
            Damon estimates suffer bias away from the 'True' value
            toward the 'observed' value.  objectify() effectively (though
            not entirely) removes this bias from specified entities.  It
            calculates the estimate that would be obtained if the corresponding
            observation had first been deleted.  It also removes the effect
            of outlying observations.

            The second property, projection of an entity into the subspace
            created by the remaining entities, has intriguing and
            useful applications.  Suppose you administered Test A
            in October.  Each student also has scores from a State test
            from last May.  You want to know how each student WOULD have
            performed on the State test last May given ONLY his performance on
            Test A.  Solution:  Include the State test scores in the Test A
            dataset as an extra column and objectify them.  This has the
            mathematical effect of projecting the State scores into the
            dimensional subspace defined by the items in Test A.  This
            means the estimates corresponding to the State scores will
            be stripped of all dimensions that are not in the Test A
            subspace, dimensions such as additional content areas,
            differential student growth from last May to October, mistakes
            the student might have made on the State test, and so on.
            It all gets filtered out.  The Damon estimates for the State
            test column now answer the question:  What is the likely
            performance of each student on the State test based on his
            or her performance on Test A?

            Subspace projection appears to offer the definitive approach
            to computing subscale measures, even when the items in the
            subscale vary on dimensions that are not captured by the
            remaining items on the test.  This phenomenon, where groups
            of items contain unique dimensions, is called "between-item
            multidimensionality".  Groups of items are sensitive to different
            dimensions though they may have some dimensions in common.
            Between-item multidimensionality is a serious problem because,
            unlike "within-item multidimensionality", it breaks Damon's common
            space requirement.  All error is a form of between-item
            multidimensionality.

            So how do we deal with it?  First, we use the subscale() method
            to specify the items in each subscale and append raw subscale
            measures to the data array.  Then we use objectify() to compute
            subscale measures specifying the 'pred_ents' Subscales option.
            Here is what objectify() does:

                1.  Say we have two subscales, Grammar and Reading, and
                    we want to calculate a Grammar measure.  objectify()
                    starts by getting row and column coordinates for just
                    the reading items.

                2.  The reading row coordinates are then anchored and
                    applied to the grammar raw subscale (calculated by
                    subscale()) to compute column coordinates for the
                    Grammar scale.

                3.  Estimates are calculated from the Grammar coordinates.
                    These are the Grammar measures.

                4.  The process is repeated to get Reading measures and
                    measures for any other subscale.  When there are more
                    than two subscales, the projection process is done
                    employing each group of predictor subscale items
                    separately.  The resulting target subscale estimates
                    are combined using coord() to build the target subscale
                    measure. So long as the items WITHIN each predictor subset
                    share a common space and are analyzed together,
                    the "between-item" dimensionality problem is side-stepped.
                    In fact, it is even possible to let Damon find the
                    optimal dimensionality for each subscale.

            Subscale measures calculated using this procedure have two
            important properties:

                1.  The reliability and precision of each subscale
                    measure is greater than could be achieved by analyzing
                    just the items in that subscale alone due to the capture
                    of (relevant) information from the rest of the test.
                    A 5 item subscale may yield measures with the precision
                    of 20 items.

                2.  The subscale is not biased by the items outside
                    the subscale due to the filtering out of all irrelevant
                    dimensions.  The Grammar measure stays strictly a
                    Grammar measure, just more precise.

            These properties of subspace projection are a known geometric
            feature of multiple regression.  Each regression estimate
            is the projection of the dependent variable into the subspace
            defined by the predictor variables.  Damon is quite similar
            to multiple regression in this respect, with this crucial
            difference:  each specified item becomes a predictor variable. More
            precisely, the ensemble of items is converted into a set of orthogonal
            coordinates of a specified number of dimensions, and THESE
            become the predictor variables used to model the "objectified"
            entity's data.  In regular regression, it would not be sensible
            to use all the items as predictor variables since regression
            requires its predictors to be orthogonal to each other or
            risk the effects of "collinearity".  Damon doesn't have this
            problem since it goes through the intermediate step of reducing
            the data to a simple factor structure first.  It also dodges a host
            of other problems such as missing data.

            The upshot is that you can use objectify() both to render
            unbiased predictions and to filter out all dimensionalities
            and attributes except those you specifically want, as defined
            by the selection of non-entity items.

            The price of these properties is a separate coord() run for
            each entity or scale to be objectified.  As long as it is just one
            entity, that is not a problem, but objectifying an entire
            100-item dataset is the computational equivalent of 100
            separate coord() runs.  That's the price.  Fortunately,
            most applications involve objectifying a small number of
            entities.

            There is an interesting side-effect of objectification.  The
            method works by calculating a new set of row coordinates for each
            column entity to be objectified.  That means there is no longer a
            single coordinate structure to model the array at the
            dimensionality specified by coord().  However, you can
            create a new coordinate structure at a HIGHER dimensionality
            to model the objectified estimates.  In fact, this dimensionality
            may be arbitrarily high given enough data, though in practice
            the original dimensionality (e.g., from a regular coord() run) usually
            suffices.

            The problem of overfit does not arise in this context because
            the estimates have already had the noise stripped out of them.
            One is simply defining a sufficiently high dimensionality to
            concisely reproduce the estimates that were already found by
            objectifying one or more column entities in the dataset.
            These higher-rank coordinates can be ported across datasets
            and used to make predictions just like the ordinary lower-rank
            coordinates produced originally by coord().

            The resulting estimates and coordinates are called "strongly
            objective", as opposed to the "regularly objective" estimates
            produced by an ordinary coord() run.

            You may wonder if objectification makes it possible to analyze
            items that do not share a common dimensionality.  So long
            as the "predictor" entities all share a common definable space,
            the "dependent" objectified entities may have extra dimensions
            (which will be filtered out) or fewer dimensions.  However, if
            the predictor entities do not participate in a common space,
            there may be issues.

            Note on "Best Dimensionality"
            -----------------------------
            When using the objectify() method, we encounter an odd phenomenon.
            Experiment suggests that it is best to set a dimensionality of 3
            or 4 (ndim = [[3]]), even when the the dimensionality of the
            predictor entities is less, such as 1.  When the dimensionality
            of the predictor entities is greater than 3, use the larger
            dimensionality; treat 3 as a minimum.

            The mathematics of why a minimum of 3 dimensions is optimal when
            predicting variables that reside outside the common space are under
            study.  When the dimensionality of the predictors is 1, using
            dimensionalities between 2 and 5 seems to work best with such
            target variables.

            Note on Standard Errors
            -----------------------
            The procedure for computing standard errors is straightforward
            and mimics the usual process:

                d = dmn.Damon(...)
                d.subscale(...)
                d.standardize(...)
                d.objectify(...)        # Instead of coord().
                d.base_est(...)         # You can skip this step, since objectify()
                                          does it automatically if overwrite = True.
                d.base_resid(...)
                d.base_ear(...)
                d.base_se(...)
                cell_err = d.base_se_out['coredata']

        Arguments
        ---------
            "targ_ents" is used to specify a set of column entities you wish
            to "objectify".  It uses Damon's extract() syntax, which provides
            a lot of options.  See

                >>>  help(core.Damon.extract)

            Examples:
                targ_ents = {'Get':'AllExcept','Labels':'key','Cols':[None]}
                                =>  Objectify all column entities.

                targ_ents = {'Get':'NoneExcept','Labels':'key','Cols':['It1','It2']}
                                =>  Objectify column entities 'It1'
                                    and 'It2'

                targ_ents = {'Get':'NoneExcept','Labels':1,'Cols':['Grammar']}
                                =>  Objectify all column entities that have
                                    the 'Grammar' label in row 1 (counting from
                                    zero) of collabels.

                targ_ents = {'Get':'NoneExcept','Labels':'key','Cols':['sub_Grammar']}
                                =>  Objectify the 'sub_Grammar' subscale
                                    appended to the data using subscale().
                                    Use the 'Subscales' option in pred_ents
                                    below.


            Note:  Entity labels must match those output by the
            most recent Damon() method, which means that if you
            ran parse(), your targ_ents must correspond to the parsed
            column labels, e.g., '1_a','1_b','2_a','2_b', etc.,
            instead of 1, 2.

            Note that if you ran subscale(), all integer column labels
            will be converted to string to accommodate the string 'sub_'
            subscale names appended to the dataset.  In this case you will
            need to put quotes around the targ_ent key names.

            ---------------
            "pred_ents" stands for "predictor" entities, the set of
            row or column entities corresponding to each specified
            targ_ent that will be used as "predictors" to construct
            the subspace into which that targ_ent will be projected.
            extract() syntax is used in nesting dictionaries to designate
            predictor entities for each targ_ent, with three additional
            syntax shortcuts.

                pred_ents = {'1':{'Get':'AllExcept','Labels':'key','Cols':['1']},
                             '2':{'Get':'NoneExcept','Labels':'key','Cols':['1','3','4']},
                              |
                              V
                             'sub_Grammar':{'Get':'AllExcept','Labels':1,'Cols':['Grammar']}
                             }
                                =>  For targ_ent '1' use all other items as predictors.
                                    For targ_ent '2' use only items '1', '3', '4' as
                                    predictors.  For the 'Grammar' subscale called 'sub_Grammar',
                                    use all except grammar items as predictors.  And so on,
                                    a separate extract() statement for each target entity.
                                    This is the most expressive syntax, capable of handling
                                    all scenarios.

                pred_ents = {'AllTargs':'AllExceptTarg'}
                                =>  This saves the work of specifying predictors
                                    for each target entity.  For each target
                                    entity, all other entities in the dataset
                                    are used as predictors.

                pred_ents = {'AllTargs':{'Get':'NoneExcept','Labels':'key','Cols':['1','2','3','4']}}
                                =>  This specifies the same set of predictor entities
                                    ('1' - '4') for each target entity.

            The 'Subscales' option is used to analyze multiple subscales
            that may or may not share a common space.  It requires reference
            to a row containing subscale labels.  The corresponding targ_ents
            keys should (generally but not necessarily) be the labels created
            automatically  by subscale(), with 'sub_' prefixes.

                targ_ents = {'Get':'NoneExcept','Labels':'key','Cols':['sub_Grammar','sub_Vocab','sub_Reading']}
                pred_ents = {'Subscales':{'Get':'NoneExcept','Labels':1,'Cols':['Grammar','Vocab','Reading']}}
                                =>  When calculating 'sub_Grammar' measures,
                                    objectify() refers separately to the group
                                    of items with the 'Vocab' descriptor and
                                    the group with the 'Reading' descriptor.
                                    It ignores the 'Grammar' items since these
                                    have already been averaged under the 'sub_Grammar'
                                    header.  The process is repeated for 'sub_Vocab'
                                    and 'sub_Reading'.  The resulting measures
                                    end up in the obj_est output as estimates
                                    in the 'sub_Grammar', 'sub_Vocab', 'sub_Reading'
                                    columns.

            ---------------
            "ndim" is used to specify, or discover, the optimal number
            of dimensions for just those entities specified in pred_ents.
            If one dimensionality is specified, it is applied to
            all groups of predictor items.  If a range of dimensionalities
            is specified, a different optimal dimensionality may be
            calculated automatically for each group of predictors.

            [ndim does not, as yet, support specifying a different dimensionality
            for each set of predictor entities when the pred_ents 'Subscales'
            option is used.  Until that is implemented, it is assumed that
            each set of predictor entities can be modeled with the same
            dimensionality, or that it's okay to let Damon estimate dimensionality
            for each set of predictor entities.]

            The ndim syntax is the mostly same as that for coord():

                ndim = 'Refer2Coord'
                                =>  objectify() will refer to the
                                    coord_out outputs to obtain starter
                                    coordinates and the "best"
                                    dimensionality.

                ndim = [[int list of dimensionalities],'search','homogenize']

                ndim = [[4]]
                                =>  Define each "predictor" subspace to
                                    be three dimensions.

                ndim = [range(4,10),'homogenize']
                                =>  find the "best" dimensionality
                                    for each defined predictor subspace,
                                    using the 'homogenize' option (see
                                    coord() docs).

                When dealing with multiple groups of predictor entities, as
                with multiple subscales, insert the ndim parameters into a
                dictionary:

                ndim = {'Grammar':[[2]], 'Reading':[[3]], 'Vocab':[range(1,5),'homogenize']}
                                =>  Analyze the Grammar items with two dimensions.
                                    Analyze the Reading items with three dimensions.
                                    Analyze the Vocabulary items with the "best"
                                    dimensionality from 1 to 5 after homogenizing.

                                    Note:  Generally subscales are built to be
                                    1-dimensional, so in real life these will
                                    almost always be:  ndim = {'Grammar':[[1]],...}

                For more ndim options, refer to:

                >>>  help(core.Damon.coord)

                Important
                ---------
                ndim is also used to calculate "background" estimates for non-target
                entities -- entities that will not be "objectified".  However, the
                dimensionality needed to calculate optimal cell estimates in this
                case may be higher than the dimensionality that is optimal for
                analzying each set of predictor entities.  The priority, here, is
                to pick the right dimensionality for the predictor entities and let
                the estimates for non-target entities fall where they will, even
                if they are not optimal.

                To get around this trade-off, you can 'search' or specify a list of
                dimensionalities and have objectify() pick the optimal number of
                dimensions in each case.  But this is computationally expensive
                and often not worth it to refine cell estimates that you probably
                aren't interested in.

            ---------------
            "runspecs" applies to the calculation of the coordinates necessary
            for modeling each individual target entity.  Set low to maximize
            speed, as starter coordinates will already have been calculated
            to get the coordinates into the right ballpark.
                runspecs = [StopWhenChange,MaxIteration]

            For instance,

                runspecs = [0.01,10]

            means keep iterating until either the percent change in the combined
            row and column coordinates between successive iterations is
            less than 0.01 or until 10 iterations have occurred, whichever
            happens first.

            ---------------
            "seed" is used to specify the coord() seed parameter when calculating
            coordinates for each group of predictor varaiables.  Set it at
            an integer for speed.  See the coord() docs for options.

                seed = 'Ref2Coord'
                                =>  objectify() will refer to the coord_out
                                    outputs to obtain the previously
                                    calculated "best" seed.

                seed = 1        =>  Use random number seed 1 for coordinates.
                                    Specifying an integer is fastest.

                seed = 'Auto'   =>  coord() will internally try out multiple
                                    seeds and pick the best.  This is slower,
                                    especially in objectify().

            ---------------
            "starters" <None,True> instructs the function to calculate starter
            coordinates for all items to speed up the process of calculating
            coordinates for each group of predictors.  It also applies the same
            dimensionality (number of dimensions, though they may not be the
            same dimensions) to all groups of predictors.  Use starters = None
            if you want to calculate a different dimensionality for each
            predictor group.

                starters = True =>  Calculate starter coordinates, same
                                    dimensionality for all predictors

                starters = None =>  Do not calculate starter coordinates.
                                    Compute fresh coordinates from scratch
                                    for each group of predictors.

            ---------------
            "summdim" specifies a new dimensionality at which to run
            coord() to "summarize" the array of mixed objectified and
            non-objectified estimates.  While it allows trying out
            different dimensionalities (like ndim), in practice this may
            not be particularly meaningful.  summdim simply needs to be large
            enough to produce estimates that are arbitrarily close to
            the "objectified + non-objectified" matrix.  The dimensionality
            specified in ndim is usually sufficient (use the 'ndim' option),
            but summdim can safely be quite a bit higher.  The goal is simply
            to produce coordinates that can be used to summarize the objectified
            estimates matrix.

            summdim's options parallel those of the coord() ndim
            argument.  Refer to coord() for more extensive documentation.

                summdim = None      =>  Do not try to summarize the
                                        mixed objectified and non-objectified
                                        estimates array with a single set
                                        of summary coordinates.
            Or:
                summdim = 'ndim'    =>  Reuse the ndim specification.

                summdim = [[int list of dimensionalities],'search','homogenize']

                summdim = [[3]]     =>  Run coord() at three dimensions, i.e.,
                                        locate each row and column entity in a
                                        3-dimensional space.

                summdim = [range(1,20)]
                                    =>  Find the best dimensionality between
                                        1 and 20.

            ---------------
            "center" specifies whether to center the objectified estimates on
            the observations corresponding to a given entity.

                center = None   =>  Objectified estimates are allowed to have
                                    a different mean for that entity.  They
                                    may actually be closer to the "True" values,
                                    but the fit with the observations won't be
                                    as good.

                center = True   =>  Objectified estimates are adjusted to have
                                    the same mean as the observations for that
                                    entity.  This adjustment is accomplished by
                                    adding a displacement dimension to the
                                    opposing coordinates.  The resulting estimates
                                    may not be as close to the "True" values,
                                    but they will fit the observations better,
                                    and sometimes that's what you want.

            ---------------
            "overwrite" <None,True> instructs the function to update (overwrite)
            the base_est_out and coord_out outputs with new estimates and
            coordinates.  All downstream methods will then refer to the
            "objectified" coordinates and estimates.

            If base_est() was not already run, overwrite=True creates the base_est_out
            output on its own, removing the need to run base_est() separately.  However,
            you might still want to run base_est() to capture the ecutmaxpos
            parameter.  If base_est() sees that a base_est() output already exists,
            it won't recalculate it but just pass it through.

        Examples
        --------

            [under construction]

        Paste method
        ------------
            objectify(targ_ents = {'Get':'AllExcept','Labels':'key','Cols':[None]},   # [{'Get':_,'Labels':_,'Cols':_} => row/col entities to objectify]
                      pred_ents = {'AllTargs':'AllExceptTarg'}, # [<{targ1:{'Get'...},...},{'AllTargs':{'Get'...}},{'AllTargs':'AllExceptTarg'},{'Subscales':{'Get'...[<sub1,'All'>]}]
                      ndim = 'Refer2Coord', # [<['Refer2Coord',[[dim list],'search','homogenize']> => n dim(s) for predicting targ_ents]
                      runspecs = [0.0001,10],   # [['StopWhenChange','MaxIteration'] => for calculating each targ_ent coordinate set]
                      seed = 'Auto',  #[<'Refer2Coord',None,int,'Auto',{'MinR':0.90,'MaxIt':<10,[3,10]>,'Facet':<0,1>,'Stats':[<'Stab','Acc','Obj','PsMsResid','NonDegen'>],'Group1':{'Get':'NoneExcept','Labels':'index','Entities':[...]},'Group2':{'Get':'AllExcept','Labels':'index','Entities':[...]}}>]
                      starters = True,   # [<None,True> => for speed, calc starter coords and apply their dimensionality to all predictors]
                      summdim = None,      # [<None,[[dim list],'search','homogenize']> => dim to summarize all estimates in potentially higher space]
                      center = True,    # [<None,True> => center objectified estimates on observations per entity]
                      overwrite = True, # [<None,True> => overwrite coord_out and base_est_out]
                      )

        """
        if self.verbose is True:
            print 'objectify() is working...\n'

        # Run the damon utility
        objectify_out = dmn.utils._objectify(locals())
        self.objectify_out = objectify_out

        if self.verbose is True:
            print 'objectify() is done -- see my_obj.objectify_out'
            print 'Contains:\n',self.objectify_out.keys(),'\n'

        return None



    #############################################################################

    def base_est(self,
                 fac_coords = 'Auto',  # [<'Auto' => use coords of current Damon, [Fac0Coords,Fac1Coords,nanval]>]
                 ecutmaxpos = None,  # [<None, [['All',[ECut,MaxPos]], ['Cols',{'It1':[ECut1,MaxPos1],'It2':[ECut2,MaxPos2],'It3':['Med','Max'],...]]> ]
                 refit = None,   # [<None,'Lstsq'> => refit estimates to coord() inputs and calc new coordinates]
                 nondegen = None,  # [<None,True> => calc a "NonDegeneracy" statistic]
                 ):
        """Calculate "base" cell estimates from row and column coordinates.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon where coredata is an array of estimates equal to the
            dot product of row and column coordinates computed by
            coord().  The base_est() outputs are accessed using:

                my_dmnobj.base_est_out

            base_est() converts row and column coordinates generated
            by the coord() and objCoord() methods into cell
            estimates in one or more output reporting metrics.

            The method can be applied automatically to the output of
            my_dmnobj.coord() or my_dmnobj.objectify, in which case
            fac_coords should be set to 'Auto', or it can be applied
            explicitly to coordinate arrays that you specify.

            If nondegen is True, base_est() will assign a new attribute
            to the Damon object:

                my_dmnobj.nondegeneracy

            NonDegeneracy ranges from 0.0 to 1.0, where a value less
            than 0.4 triggers a warning.

            Workflow:
                MyEstimates = my_dmnobj.base_est(fac0coord,fac1coord,...)

            where the expression is preceded by:
                my_dmnobj.coord(...)        or
                my_dmnobj.objCoord(...)

            The MyEstimates datadict includes an 'ecutmaxpos' key
            for use by base_resid() and est2logit().  For more information,
            see the est2logit() docs.

        Arguments
        ---------
            "fac_coords" is used if you want to get estimates for
            coordinate arrays that are not attributes of the current
            Damon obj.  Options:

                fac_coords = 'Auto'  => Let base_est() use the output of
                                        coord() in the current Damon obj.
                                        You must first have done:
                                            my_dmnobj.coord(...)   or
                                            my_dmnobj.objectify(...)

                fac_coords = [Fac0Coords,Fac1Coords,nanval]
                                    =>  Feed in coordinates that are not
                                        attributes of the current Damon.

                                        Fac0Coords is an nrows x nDims array
                                        Fac1Coords is an ncols x nDims array
                                        nanval is the Not-a-Number value used
                                            to flag invalid coordinates.

            ---------------
            "ecutmaxpos" specifies two parameters, ecut and MaxPos, used
            for computing residuals and probabilities, though not used
            by base_est to compute estimates.  It is used by downstream
            methods base_resid() and est2logit().

            ecut is a cut-point that can be assigned to each column entity
            in the estimates array to distinguish "success" from "failure".
            It is required for calculating probabilities.

            The way it works is that if an estimate is 3.6 and ecut is 3.0,
            the estimate signifies "success"; a 2.4 would signify "failure".
            ecut dichotomizes the estimates in order to make it possible
            to define probabilities.  We can now define the probability
            of success as the probability of exceeding ecut.

            MaxPos is a user-specified "maximum possible" estimate for a
            given entity.  It can be the actual maximum, or smaller or
            greater, bearing in mind that random error may cause
            the estimate to spill higher than the theoretical maximum.
            MaxPos essentially caps the estimate and is used to map
            interval metrics to sigmoid metrics.  MaxPos is optional.
            It is one of two formulas for computing probabilities.  For
            more information about these formulas see the est2logit() docs.

            ecutmaxpos has the following input options.

                ecutmaxpos = None   =>  Do not reduce estimates to dichotomous
                                        values in downstream methods.  In
                                        base_resid(), the nearest_val parameter
                                        will be used instead.  If the estimates
                                        are in the'PreLogits' standardization
                                        metric, est2logit() if called will
                                        convert them directly to probabilities or
                                        throw an error.

                ecutmaxpos = ['All',[0.65,1.5]]
                                    =>  Assign the same ecut to all
                                        columns, 0.65 in this case.  Let
                                        the maximum possible estimate be
                                        defined as 1.5.

                ecutmaxpos = ['All',[0.65,None]]
                                    =>  Assign an ecut of 0.65 to all
                                        columns.  The method for calculating
                                        probabilities will not use a
                                        MaxPos.

                ecutmaxpos = ['All',['Med',None]]
                                    =>  Assign the same ecut to all
                                        columns, in this case the median
                                        of the whole array.  Ignore
                                        MaxPos.

                ecutmaxpos = ['All',['Med','Max']]
                                    =>  Assign the same ecut to all
                                        columns, in this case the median
                                        of the whole array.  Set MaxPos
                                        equal to the maximum estimate in
                                        the estimates array.

                ecutmaxpos = ['Cols',{'It1':[0.50,1.5],'It2':[2.5,5.0],'It3':['Med','Max'],...]]
                                    =>  Assign the 'It1' estimates an
                                        ecut of 0.50 and MaxPos of 1.5.
                                        Assign 'It2' an ecut of 2.5 and
                                        a MaxPos of 5.0.  Assign 'It3' an
                                        ecut equal to the median
                                        estimate for that column and a
                                        MaxPos equal to the maximum for that
                                        column.

                ecutmaxpos = ['Cols',{'It1':[0.50,None],'It2':[2.5,None],'It3':['Med',None],...]]
                                    =>  Assign the 'It1' estimates an
                                        ecut of 0.50, 'It2' an ecut of
                                        2.5, and 'It3' an ecut equal to
                                        the median estimate for that column.
                                        Do not use the probability formula that
                                        requires MaxPos.

                                        est2logit() applies the same probability
                                        formula to all columns.  If MaxPos = None
                                        for one column, it must equal None for
                                        all columns.


                            Note:  The column IDs are drawn from
                            the original data array before parsing.  When an
                            item has been parsed into multiple columns, all
                            of those columns receive the same ecut.

                ecutmaxpos = ['Cols',['Med','Max']]
                                    =>  Assign a different ecut and MaxPos to
                                        each column, where ecut and MaxPos are
                                        the median and theoretical maximum of
                                        for that column.

            ---------------
            "refit" is used to refit the estimates to the metric of the 
            data as analyzed by coord(), generally standardized.  This really
            only comes up when coord() specifies the 'NonNeg_1D' option for
            the condcoord parameter, since the resulting coordinates
            are in an exponential metric.  When refit is applied, both the
            estimates and coordinates are converted back to the observed
            metric.  The coordinates are recalculated to maximally fit
            the new refit estimates and self.coord_out is overwritten.
            
            Format:     refit = <None, 'lstsq', or a standardize() metric>
                
                refit = None => This is for all cases where the estimates
                                metric is the same as the observed metric,
                                usually standardized, that goes into
                                coord().
                
                refit = 'LogDat'
                            =>  Use this option if the original data values
                                are on an interval scale (-inf to +inf).
                
                refit = 'PreLogit'
                            =>  Use this option if the original data values
                                are on a ratio scale (0 to +inf).
                
                refit = '0-1'
                            =>  Use this option if the original data values
                                are on a discreet ordinal or continuous 
                                sigmoid scale, specifying that the validchars
                                range is continuous sigmoid 
                                (e.g., ['0.0 -- 3.0']).  If you specify 
                                a discrete validchars range (e.g., 
                                ['0 -- 3']), the estimates will be discrete
                                as well, which you probably don't want.
                                
                 refit = other standardize() parameters
                           =>  refit accepts any standardize metric
                               parameter, but you will probably only use
                               of the three above.

                refit = 'lstsq'
                            =>  This uses ordinary least squares to relate
                                estimates to the coord() data inputs, but
                                its estimates have an exponential relationship
                                with the true values.  It will probably be
                                deprecated.
                                
            For more discussion on refit and handling 1-dimensional data,
            see the coord() docs, "The Curse of Unidimensionality."
            
            ---------------
            "nondegen" <None,True> is used to check for degenerate solutions.
            These are estimates for (genuinely) missing cells that are
            out of the range of estimates for non-missing cells.  These are
            calculated as a "NonDegeneracy" statistic which ranges from
            0.0 to 1.0, where a value less than 0.40 indicates a likely
            degenerate solution.  The NonDegeneracy statistic is assigned
            to the Damon object as an attribute:

                NonDegeneracy = Damon.NonDegeneracy

            These can only be calculated if the original dataset contains
            truly missing data.

        Examples
        --------




        Paste method
        ------------
            base_est(self,
                    fac_coords = 'Auto',  # [<'Auto' => use coords of current Damon, [Fac0Coords,Fac1Coords,nanval]>]
                    ecutmaxpos = None,  # [<None, [['All',[ECut,MaxPos]], ['Cols',{'It1':[ECut1,MaxPos1],'It2':[ECut2,MaxPos2],'It3':['Med','Max'],...]]> ]
                    refit = None,   # [<None,'Lstsq'> => refit estimates to coord() inputs and calc new coordinates]
                    nondegen = None,  # [<None,True> => calc a "NonDegeneracy" statistic]
                    )

        """
        if self.verbose is True:
            print 'base_est() is working...\n'

        # Run the damon utility
        base_est_out = dmn.utils._base_est(locals())
        self.base_est_out = base_est_out

        if self.verbose is True:
            print 'base_est() is done -- see my_obj.base_est_out'
            print 'Contains:\n',self.base_est_out.keys(),'\n'

        return None




    #############################################################################

    def base_resid(self,
                  nearest_val = None,     # [<None,'Nearest','ECut'> => how to prep estimates]
                  psmiss = None,   # [<None, True => only report for pseudo-missing cells]
                  ):
        """Calculate residuals between raw numerical cell values and cell estimates.

        Returns
        -------
            base_resid() returns None but assigns an output datadict to the
            Damon object.  The base_resid() outputs are accessed using:

            Workflow:
                d = dmn.Damon(...)
                d.standardize()
                d.coord(...)
                d.base_est(...)
                d.base_resid(...)
                resid = d.base_resid_out

        Comments
        --------
            base_resid() reports for each cell in coredata the difference
            between the raw numerical values analyzed by the coord()
            method and the estimates obtained from the coordinates using
            base_est().  The raw values may be the original data or the
            data as parsed using Damon.parse() or standardized using 
            Damon.standardize().
            
            The output of base_resid() is used to calculate expected absolute
            residuals (EAR's), standard errors (SE's), and probabilities.
            base_resid() outputs have the useful property that they are in the
            same metric for all columns, unlike fin_resid() outputs.
            
            When the original data:
            
                1) are ordinal (dichotomous or polytomous), and
                2) have been standardized to the 'PreLogit' metric
                
            an expansion factor is applied to widen the residuals at the top
            and bottom of the scale.  This is to compensate for the fact
            that dichotomous data, for example, will bias Damon cell estimates
            toward the extremes and away from the center of the scale.  For
            each cell:
            
                new_residual = residual / sqrt(p * (1 - p))
                
            where
            
                p = exp(est / (1 + est))
            
            and est is the cell estimate in logits (based on PreLogits).
            
            The same expansion factor is applied again in base_se() for
            calculating standard errors.  See the Damon.base_se() docs
            for discussion.
            
            When the original data array has a mix of ordinal and non-ordinal
            data, the non-ordinal residuals are left alone.

        Arguments
        ---------
            "nearest_val" instructs base_resid() how to prepare the
            estimates for calculating residuals.  There are three
            options:

                nearest_val = None  =>  Do not adjust estimates; use
                                        as are.

                nearest_val = 'Nearest'
                                    =>  Round or adjust all estimates to their
                                        nearest "observed" value, i.e., the
                                        nearest valid input value as it went
                                        into coord().  The function
                                        first checks whether the
                                        observed values are continuous.  If
                                        so, no adjustment is made.

                nearest_val = 'ECut'
                                    =>  Cut-points are assigned to each
                                        column (or to the whole array) to
                                        distinguish "success" from "failure".
                                        The residuals are 1 where the
                                        estimates and observations fall on opposite
                                        sides of the cut-point, 0 otherwise.
                                        There are no negative residuals in this
                                        approach.

                                        The list of ECuts values is automatically
                                        obtained from the ecutmaxpos parameter
                                        specified for the base_est() method.

            Currently, only one method can be specified for the array as
            a whole, the same for all columns.

            ---------------
            "psmiss" (<None, True>) provides the option of calculating
            residuals for only those cells whose values were made pseudomissing
            prior to estimation.  These residuals test the true predictive
            accuracy of the model.  The pseudo missing cell index is obtained
            from the output of the pseudomiss() method.

        Examples
        --------



        Paste method
        ------------
            base_resid(nearest_val = None,     # [<None,'Nearest','ECut'> => how to prep estimates]
                       psmiss = None,   # [<None, True => only report for pseudo-missing cells]
                       )

        """
        if self.verbose is True:
            print 'base_resid() is working...\n'

        # Run the damon utility
        base_resid_out = dmn.utils._base_resid(locals())
        self.base_resid_out = base_resid_out

        if self.verbose is True:
            print 'base_resid() is done -- see my_obj.base_resid_out'
            print 'Contains:\n',self.base_resid_out.keys(),'\n'

        return None




    #############################################################################

    def base_ear(self,
                 ndim = 2,   # [Number of dimensions at which to run coord() on residuals]
                 ):
        """Calculate Expected Absolute residuals.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The base_ear() outputs are accessed using:

                my_dmnobj.base_ear_out

            base_ear_out is a datadict of Expected Absolute residuals
            (EARs) between observed and estimated values as defined for the
            base_resid() method.  Included in the base_ear_out is
            the key 'ear_coord' -- the coordinates needed to estimate
            the EAR statistics in an anchored analysis.

            Workflow:
                MyEAR = my_dmnobj.base_ear(...)

            where residuals were calculated previously:
                my_dmnobj.coord(...)
                my_dmnobj.base_est()
                my_dmnobj.base_resid()

        Comments
        --------
            base_ear() calculates expected absolute residuals between
            coord() inputs and outputs for each cell.  Conceptually,
            each Expected Absolute Residual (EAR for short) is the
            Damon equivalent of a cell standard deviation.  Imagine
            a person takes an item.  He gets a score.  Now imagine that
            he takes the same item many times (his short-term memory
            erased each time).  The mean of the resulting scores would
            correspond to the Damon cell estimate.  The standard deviation
            of those scores would correspond to the EAR.  In practice,
            we only get to see each person/item interaction once, but
            Damon leverages the information in the rest of the matrix to
            obtain the equivalent of repeated observations per cell.

            EAR statistics are the basis of the standard error statistics,
            lacking only the additional element of data counts.  If
            SE = SD/sqrt(n), the EAR plays the role of the SD and the SE
            is calculated from it by dividing by the square root of
            the number of observations involved.  A more complicated form
            of this formulat is implemented by the Damon.base_se() method.
            
            base_ear() calculates the expected absolute residual per cell
            by applying Damon to the array of absolute residuals. The
            resulting cell estimates are the EARs.  The absolute residuals
            matrix, running from zero to +inf, is analyzed as a continuous
            ratio scale.  Although standard errors are 1-dimensional, Damon
            analyzes the residuals using a 2-dimensional model.  This is
            because the absolute residuals are first converted to 
            log(residuals), which means that each log(cell value) is now
            more properly modeled as a sum of its row and column coordinates 
            rather than as their product.  To simulate each cell as a sum
            rather than a product, Damon needs an additional dimension,
            which is why ndim = 2.

        Arguments
        ---------
            "ndim" is the integer number of dimensions at which to run
            coord() as applied to a matrix of absolute residuals.  The 
            ndim parameter should always be 2; the parameter is provided
            for experimentation purposes only.  
            
            In principle


        Examples
        --------


        Paste method
        ------------
            base_ear(ndim = 2,   # [Number of dimensions at which to run coord() on residuals]
                    )


        """

        if self.verbose is True:
            print 'base_ear() is working...\n'

        # Run the damon utility
        base_ear_out = dmn.utils._base_ear(locals())
        self.base_ear_out = base_ear_out

        if self.verbose is True:
            print 'base_ear() is done -- see my_obj.base_ear_out'
            print 'Contains:\n',self.base_ear_out.keys(),'\n'

        return None


    #############################################################################

    def base_se(self,
                obspercellmeth = 'CombineFacs',   # [<'PickMinFac','CombineFacs'>]
                ):
        """Calculate the standard error of each base estimate.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The base_se() outputs are accessed using:

                my_dmnobj.base_se_out

            base_se_out is a datadict of cell standard errors (SEs) based on
            cell estimates calculated by base_est(), cell residuals
            calculated by base_resid(), and cell expected absolute
            residuals calculated by base_ear().

            base_se() also returns an "obspercell_factor" array which is
            a function of the number of row and column observation per
            cell and the number of dimensions.  

            Workflow:
                my_dmnob.standardize(...)
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.base_resid(...)
                my_dmnobj.base_ear(...)
                my_dmnobj.base_se(...)
                cell_se = my_dmnobj.base_se_out

        Comments
        --------
            "base_se()" calculates the standard error of each cell estimate
            based on the outputs of base_est(), base_resid(), and base_ear().
            It may not be the final form of the standard error in the
            original metric.  (Standard errors are converted to the
            original metric using fin_est().)  But it is the preferred
            form of the standard error for conducting significance
            tests.
            
            But first, what is a standard error in Damon?  Damon produces
            two statistics for measuring cell variance:  the EAR (expected 
            absolute residual) and the SE (standard error).
            In the context of Damon, the EAR is the expected discrepancy
            between each Damon estimate and its corresponding "observed" value.
            But remember that the observed value is not necessarily the
            "true" value, since observations are themselves subject to
            error.  Thus, it is the purpose of the SE statistic to estimate
            the expected discrepancy between the Damon estimate for each
            cell and its "true" value.  Therefore:

            EAR measures how well the estimates are expected to fit
            the OBSERVATIONS.  SE measures how well the estimates are
            expected to fit the TRUTH.
            
            Before proceeding, I need to qualify the following statement.
            More precisely, SE measures how well the estimates are expected
            to fit the True, ASSUMING the true values are rescaled to be
            in the same metric as the estimates.  Since the estimates are
            usually calculated from "standardized" data, that means the
            underlying "true" values need to be rescaled accordingly.  When
            performing the operation using simulated "model" data, this 
            generally means rescaling the model/true values to have the 
            same column means and standard deviations as the standardized
            data.  Then we are in a position to make the claim that the
            standard error is the expected fit between the Damon estimates
            and the (standardized) true values.  When the data are 
            dichotomous and polytomous, the procedure for standardizing 
            the true values to be comparable to the estimates is a little
            more complicated.  But this is just for validating the method.
            In practice, you will generally be happy just to have the
            standard error in the standardized pseudo-logit estimates metric
            as a way to express how uncertain the estimate is.
            
            Why is the standard error important?  The answer is that no
            statistic or prediction is meaningful unless it is accompanied
            by some estimate of the uncertainty associated with it.  How
            "true" is this estimate?  How sure am I that Mary's test score
            exceeds the "proficient" cut-point on the scale?  The standard
            error answers that question.  The rule of thumb is that if
            a person is two standard errors above some cut-point on a 
            scale, there is a 0.95 probability that on repeated testing
            that person will get a score exceeding the cut-point.  0.05
            percent of the time, she would be expected to fall below the 
            cut-point, just by chance.  In real life, we don't get to
            test a person repeatedly.  The magic of the standard error
            formula is that it can yield the probability of exceeding the
            cut-point even though the person has only been tested ONCE,
            has only been exposed to a given question ONCE.
            
            In addition, the standard error is essential for estimating
            the reliability of a test and each item on the test.

            How, then, do we compute the SE?  The EARs are
            easy because we know what the observations are.  The SEs
            are not because we don't know what the true values
            are, hence cannot directly observe how they compare to
            the estimates.  To estimate SEs, we go to classical statistics
            and the central limit theorem, which finds that if the
            observed root variance is SD, the standard error of
            the mean is the variance of the sample mean upon repeated
            sampling, or SE = SD/sqrt(n).

            For Damon, we redo this basic formula SE = SD/sqrt(n) in
            a specific way to deal with multidimensionality and the
            fact that each cell is associated with both row and column
            observations.  The resulting formulas offered here work
            reasonably well with simulated data and have intuitive appeal.
            Here is what they look like.

            Formula 1a (regular formula):

                n = number of observations
                SE = SD/sqrt(n)

            Formula 1b (Damon version):

                Basic Idea:  SE = nFac * SD / (n - nDims)/nDims
                                = nFac * SD / (n/nDims - 1)

            where nFac is the number of unanchored facets.  This gets 
            translated into one of two formulae:

            Formula 1b.1

                SE = (nFac * EAR) / minimum( sqrt(nRow/ndim - 1), sqrt(nCol/ndim - 1) )

            Formula 1b.2

                SE = (nFac * EAR) / sqrt( sqrt(nRow/ndim - 1) * sqrt(nCol/ndim - 1) )

            The Damon SE is like the regular SE except that:

                a)  EAR is used in place of SD, being the Damon equivalent
                    of the cell standard deviation, the expected deviation between
                    the estimate and the observation.

                b)  The EAR is multiplied by the number of unanchored facets,
                    generally 2.  This is derived from the law of addition of
                    errors.

                c)  The denominator is a function of the number of observations
                    (nRow or nCol) RELATIVE TO the number of dimensions (ndim), not
                    the absolute number of observations.

                d)  The (... - 1) part of the denominator causes the denominator to go
                    to zero when the number of dimensions equals the number of
                    observations, causing the error to go to infinity, as it
                    should in this situation since Damon estimates become exactly
                    imitative of the data, and thus uninformative, when the number
                    of dimensions equals the number of row or column observations.
                    In general, Damon estimates are illegal when there are the same
                    or fewer observations than dimensions.

                e)  The denominator is calculated in one of two ways:  a) it comes
                    from from the facet that has the smallest number of observations
                    (the minimum) since this is the primary driver of error (called
                    the 'PickMinFac' formula); b) the row and column facets are
                    combined by taking a geometric mean (called the 'CombineFacs'
                    formula).
                
                f)  The number of observations "n" per row is more precisely
                    defined as the sum of the maximum number of ratings per 
                    non-missing cell in that row.  Thus, "n" is equal to
                    the number of columns only when the data are dichotomous
                    and there are no missing cells.  It attempts to count
                    the number of "bits" of information collected for that
                    person, where a bit is the maximum number of "cuts"
                    (classifications) to which a person has been exposed.
                    This can be a bit tricky to calculate.  For example,
                    a column of continuous data in principle provides an
                    infinite number of cuts, hence bits.  In practice, the
                    number is much less due to observation error.  Damon's
                    tools.obspercell() function, which the base_se() method
                    relies on, takes the most conservative position and 
                    sets a maximum of 2 possible values per cell (1 bit), even 
                    though there may be many more. 
                    
                    This is a crude way to avoid unduly small (and misleading)
                    standard errors and inflated reliabilities.  For
                    observational data subject to the limits of human
                    perception, this is probably fine.  For fine-grained
                    physical measures, or other variables known with
                    high precision, it will cause the errors to be too large
                    and reliabilities too low.

            Thus Damon replaces the factor (1 / sqrt(n)) with a new factor K
            specific to Damon:

            'PickMinFac' formula:
                K = nFac / minimum( sqrt(nRow/ndim - 1), sqrt(nCol/ndim - 1) )

            'CombineFacs' formula:
                K = nFac / sqrt( sqrt(nRow/ndim - 1) * sqrt(nCol/ndim - 1) )

            'PickMinFac' tends to be more accurate than 'CombineFacs' with
            smaller datasets where the ratio of observations to dimensions
            is less than, say, 10.  However, with large datasets it will
            tend to overestimate the standard error since it ignores
            information contributed by one of the facets.

            'CombineFacs' is more accurate with large data sets.  However, as the
            ratio of observations to dimensions drops below 10, it will tend
            to a standard error that is too small.  The reason is that
            in this situation (as the ratio of observations to dimensions
            becomes small) Damon estimates become biased toward the
            observations, resulting in an artificially low EAR and SE.
            'CombineFacs' is probably the most mathematically correct, but
            'PickMinFac' may be safer if you want to avoid false
            positive significance tests in smaller datasets.

            K is called the "obspercell_factor" and is included as an output
            for use by downstream methods.
        
            Dichotomous/Ordinal/Bounded Data
            --------------------------------
            The above formula only works when the underlying data are
            unbounded (no floor or ceiling on the values).  Dichotomous (0,1)
            data are at the extreme of bounded data, as well as being a
            common use-case in psychometrics.  Polytomous data are a less
            extreme example.  What happens with dichotomous data is that
            the input values are fixed at two values (negative and positive
            in the PreLogit metric).  Damon's coord() function, assuming
            homoscedasticity of errors which is only achieved with non-bounded
            data, leads to estimates that are biased in the direction of the
            two extremes.  That makes the residuals too small at the upper and
            lower extremes of the scale, and consequently the standard errors
            as well.
            
            To correct this, Damon's base_resid() function applies an
            expansion coefficient to widen residuals at the extremes:
                
                new_residual = residual / sqrt(p * (1 - p))
                
            where
            
                p = exp(est / (1 + est))
            
            and est is the cell estimate in logits (based on PreLogits).              
            The expanded residuals become inputs to base_ear() to get the
            expected absolute residuals (EAR).
            
            One might expect that the expansion factor as it ripples through
            the EAR statistic would be sufficient to correct the standard
            error, but it isn't.  The same factor needs to be applied again.
            The formula becomes:
            
                 SE = (nFac * EAR) / denom
                 denom = (sqrt( sqrt(nRow/ndim - 1) * sqrt(nCol/ndim - 1) ) *
                         (sqrt(p * (1 - p))))
           
            It is not yet clear mathematically why the extra factor is
            needed.  Its validity rests on empirical evidence showing that
            this formula approximates the "true" standard error in
            a wide variety of data simulations.  
            
            Note also that currently, with ordinal data, there is floor
            to how small the standard error can go.  The precise mechanism
            is still being investigated.
            
            Accuracy of Standard Errors
            ---------------------------
            Accuracy is assessed by doing simulations with dmn.create_data(). 
            We simulate a dataset with varying amounts of noise across row and
            column entities, compute Damon estimates and errors from the 
            observations, and compare the errors thus generated (averaged by
            row or column) with the "true" errors computed by comparing the
            estimates with the true or model values, which we know because
            we simulated the data.  (Remember that Damon standard
            error statistics are intended to shed light on how closely the
            estimates match the hypothetical true values.)
            
            Simulations show that the appropriate Damon error formulas
            get error estimates that are more or less in the same
            range as the true errors, but their column-by-column or
            row-by-row accuracy varies quite a bit depending on a variety
            of factors.  Rules of thumb:
                
                *   person standard errors are more accurate than item
                    standard errors (the asymmetry is due to how coord()
                    conditions coordinates and which facet finishes last)
                
                *   higher dimensionalities yield more accurate errors than
                    lower dimensionalities.  Dimensionalities 1-3 especially 
                    tend to yield errors within a small fixed range relative 
                    to the true errors.
                
                *   the variation in noise across rows or columns makes
                    a difference
                
                *   the number of rows relative to columns makes a 
                    difference.
            
            Aggregation of Standard Errors
            ------------------------------
            You will generally not be interested in the standard error
            for a particular cell but rather in the standard error
            of the mean calculating by averaging across a range of cells.
            In most statistical settings when component standard errors
            are known, the standard error of the mean is the root mean
            square error divided by the square root of n:
                
                SE = RMSE / sqrt(n)
            
            where n, if the measure is an average of several columns,
            is the number of columns.  However, this assumes that the
            standard errors for each cell are statistically independent,
            and they're not in our case.  The aggregation formula that 
            seems to work best is based on the number of dimensions 
            (degrees of freedom) associated with each cell estimate:
                
                SE = RMSE / sqrt(2 * ndim)
            
            However, because of the factors enumerated above, it may be
            necessary to tweak this number.  See the docs for 
            tools.group_se() for a more complete discussion.
        
        Arguments
        ---------
            "obspercellmeth" specifies the formula for computing the
            'obspercell_factor' used to calculate standard errors.
            The options are:

                obspercellmeth = <'PickMinFac','CombineFacs'>

            The properties of the two formulas are discussed above.


        Examples
        --------

            [under construction]

        Paste method
        ------------
            base_se(obspercellmeth = 'CombineFacs',   # [<'PickMinFac','CombineFacs'>]
                   )

        """

        if self.verbose is True:
            print 'base_se() is working...\n'

        # Run the damon utility
        base_se_out = dmn.utils._base_se(locals())
        self.base_se_out = base_se_out

        if self.verbose is True:
            print 'base_se() is done -- see my_obj.base_se_out'
            print 'Contains:\n',self.base_se_out.keys(),'\n'

        return None


    ##########################################################################

    def base_fit(self,
                 ear = None  # [<None, float, 'median', 'mean'> divide residuals by one number]
                 ):
        """Calculate "base" cell fit statistics.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The base_fit() outputs are accessed using:

                my_dmnobj.base_fit_out

            base_fit_out is a datadict whose coredata consists of
            cell-level fit statistics.  base_fit() depends on base_resid()
            and base_ear().

        Workflow:
            my_dmnobj.base_est(...)
            my_dmnobj.base_resid(...)
            my_dmnobj.base_ear(...)
            my_dmnobj.base_fit(...)

        Comments
        --------
            The formula for fit for a given cell is:

                fit[ij] = (observed - Estimate)[ij] / EAR[ij]

            where EAR is the Expected Absolute Residual.  Thus,
            fit is the ratio of the observed residual to the
            expected residual.

            More precisely, fit is interpreted as the degree to
            which the difference between an observation and
            its corresponding estimate is statistically
            significant.  If the fit is greater than 2.0 or
            less than -2.0, the observation and estimate are considered
            to be significantly different at the 95% confidence level.
            One would expect 5% of the cells in a matrix to
            exceed this threshold by chance.

            If the fit is 1.0 (the observed residual equals the
            expected residual), there is no misfit.  In a well-
            behaved dataset, the average absolute fit across all
            cells should be close to 1.0.

            fit less than 1.0 is called "overfit" and indicates
            that the observation and estimate are closer to each
            other than one would expect by chance.

            This definition of fit is conceptually similar to Rasch
            fit statistics, but there are differences.  The most
            important difference is that Damon's fit is a function of
            the observed variance, not a model-based binomial variance.
            Therefore, if a row or column has a lot of large discrepancies
            between observations and estimates, this won't necessarily
            show up in the fit statistic, because each residual viewed
            relative to the overall size of the residuals in that row
            or column may not be all that large.

            Fit so defined is not an absolute indicator of whether a
            row entity or column entity fits the model.  To get that
            kind of absolute indicator, one would need a feeling for
            what the average size of the residuals should be, and that
            is in part a subjective decision.  Use base_fit()'s ear
            parameter to redefine fit in a more absolute way.

            How to Use fit
            --------------
            The "objectivity" properties of Damon models -- the
            reproducibility of row and column parameters across
            samples -- depends on all the elements in at least one
            facet (generally items in the column facet) sharing a
            common space, varying on the same dimensions and on no
            others.  When an item does not participate in the common
            space, it misfits and the spatial structure of the dataset
            loses a little of its objectivity.  In principle, the fit
            statistics should alert us that the conditions of objectivity
            are not being met.

            There are two remedies:  a) remove the misfitting row
            or column entity from the analysis and rerun.  Remove
            more misfitting entities.  Repeat until all entities have
            an average absolute misfit (reported using summstat())
            less than some sensible cut-point, like 2.0.  b) apply the
            objectify() method to the misfitting entity.  This forces
            it to share the common space and prevents its extra dimensions
            from disturbing the rest of the analysis.

            The end result, in principle, is that the Damon estimates
            are "truer" and the coordinates more stable and reproducible
            across datasets -- the fruits of objectivity.

            I say "in principle" because in fact fit statistics are
            not an infallible indicator of fit to the model.  This is
            because it is not clear how to distinguish between residuals
            that are large because of a disturbing force outside the
            specified common space and residuals that are large simply
            because the expected variance is large due to random noise
            intrinsic to the entity.

            The reliability statistics help in this regard because they
            compare the average cell standard error to the overall spread
            of cell estimates for a given row or column entity, taking
            context into account.  Other summstat() statistics help in this
            regard, also.

            Another approach is simply to study the residuals directly,
            or the expected absolute residuals (obj.base_resid_out and
            obj.base_ear_out), and work out your own definition of
            residuals that seem unduly large, that seem to reflect some
            "outside disturbance".

            base_fit()'s ear parameter offers a way to measure fit
            by dividing all residuals by the same number, such as
            the median Expected Absolute Residual for the whole
            matrix. By forcing all row and column fits to be compared
            as if they all should have the same amount of noise, it
            becomes easier to flag entities with unusually high residuals.
            On the other hand, this approach doesn't take into account
            the fact that some rows and columns SHOULD have higher
            noise for reasons that have nothing to do with the intrusion
            of outside forces.

            For now, determining whether an entity "fits" in a specified
            common space remains as much an art as a science.

        Arguments
        ---------
            "ear" provides a way to divide all residuals by the same
            number

                ear = None  =>  (default) Divide all cell residuals by their
                                corresponding expected absolute residual.

                ear = 2.5   =>  Divide all residuals by 2.5

                ear = 'median'
                            =>  Divide all residuals by the median
                                ear of the whole ear array.

                ear = 'mean'
                            =>  Divide all residuals by the mean
                                ear of the whole ear array.

        Examples
        --------


        Paste method
        ------------
            base_fit(ear = None  # [<None, int, 'median', 'mean'> divide residuals by one number]
                    )

        """

        if self.verbose is True:
            print 'base_fit() is working...\n'

        # Run the damon utility
        base_fit_out = dmn.utils._base_fit(locals())
        self.base_fit_out = base_fit_out

        if self.verbose is True:
            print 'base_fit() is done -- see my_obj.base_fit_out'
            print 'Contains:\n',self.base_fit_out.keys(),'\n'

        return None



    ##########################################################################

    def est2logit(self,
                  estimates = 'base_est_out', # [<'base_est_out','fin_est_out','equate_out',...>]
                  ecutmaxpos = 'Auto', # [<'Auto',['All',[ECut,MaxPos]] or ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                  logitform = 'Statistical', # [<'Metric','Statistical'>]
                  obspercellmeth = 'CombineFacs',    # [<'PickMinFac',CombineFacs'> => to get obspercell_factor]
                  ):
        """Convert estimates into logits with errors, particularly useful
        when data are dichotomous.

        Returns
        -------
            The method returns None but assigns three output
            datadicts to the Damon object:

                my_dmnobj.est2logit_out        =>  logit/prob datadicts
                my_dmnobj.logit_ear_out         =>  logit EARs
                my_dmnobj.logit_se_out          =>  logit SEs

            est2logit_out is a datadict of logit estimates, also
            containing a probability datadict and the type of source
            estimates.

                my_dmnobj.est2logit_out
                my_dmnobj.est2logit_out['Prob']
                                                => Prob datadict
                my_dmnobj.est2logit_out['SourceEst']
                                                => name of source estimates
            Workflow:
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.base_resid(...)
                my_dmnobj.base_ear(...)
                my_dmnobj.fin_est(...)
                my_dmnobj.est2logit(...)

        Comments
        --------
            est2logit() converts Damon estimates, whether in
            parsed/standardized form or final estimates form,
            into logits.  logits (short for log-odds units) are
            a convenient metric for expressing probabilities in
            a linear form without a floor or ceiling.  The formulae
            for converting in and out of probabilities for each
            cell are:

                Logit = logn(Prob / (1 - Prob)) or inversely,

                Prob = exp(Logit) / (1 + exp(Logit))

            Thus, est2logit() first converts estimates into probabilities.
            But, one must ask, the probability of what? The answer
            is the probability of exceeding some estimate cut-point,
            defined using the ecutmaxpos parameter.

            While logits are a useful standardization metric, because
            they are based on probabilities they contain a possible
            ambiguity.  The probability of exceeding some estimates
            cut-point (ecut) is a function of two things:

                1.  how far above (or below) the estimate is from
                    the cut-point

                2.  the measurement error of the estimate

            As measurement error is added, all probabilities gravitate
            toward the cut-point; as it is reduced, all probabilities
            expand toward the extremes of 1 and 0.  That means the measurement
            structure of the entities -- the relative locations of the
            persons on the Item 1 variable, say -- are vulnerable to
            distortions caused by differing amounts of measurement
            error per cell.

            Thus, there is a tension between protecting measurement
            objectivity and accurately figuring the probability of
            success on a given task.  The Rasch model addresses this
            problem by, in effect, requiring each item to have the
            same measurement error (which translates to the same
            discriminating power).  Fit to the model means that Rasch
            items are constrained to have equal discriminating power so that
            logits and probabilities maintain their measurement structure
            and work as actual probabilities at the same time.

            The same approach is used Damon.  By editing the dataset to 
            maximize fit to the model, thus standardizing the error, it
            is possible to treat the resulting probabilities and logits
            as stable measurements.  In this approach, standard errors
            are calculated from the probabilities themselves using
            the binomial formula (Var = sqrt(p * (1 - p)).  The p is
            calculated by applying the cumulative normal probability distribution
            to get the probability that a cell estimate exceeds the specified
            cut-point given the cell variance as calculated by base_ear().
            
            est2logit() also offers logitform = 'Metric' which calculates
            probabilities when Damon.coord() has been applied directly to
            dichotomous data, but this is not recommended as exeriments show
            it is better to standardize into PreLogits prior to running coord().

            When to Use
            -----------
            est2logit() can be used with any type of data but is most
            useful when the observations are dichotomous.  This is because
            the standard errors generated by Damon.base_est() with dichotomous
            data, whether pre-standardized or not, will not most accurately 
            capture the (True - Estimate) error, but rather the 
            (dichotomized(True) - Estimate) error, which is generally not
            what we want.  By going into a logit/probability metric, est2logit()
            gets around this problem.
            
            However, when the observations are interval or have more than
            three or four categories, the errors output by base_se() are fine
            and est2logit() is not needed for error calculation.  It 
            nonetheless remains useful for generating linear measures that 
            can be interpreted probabilistically.  
            
            It can be used in combination with both equate() and summstat().
            If using summstat() to get summary statistics, run est2logit()
            prior to summstat().  If using equate() to get summary statistics,
            run est2logit() AFTER equate(), specifying estimates='equate_out'.
            You'll need to work from the est2logit() outputs to get further
            summary statistics or do scale transformations, which is 
            admittedly a bit awkward.

            Formulas
            --------
            est2logit() uses one of two Damon functions for calculating
            probability controlled using the logitform parameter.  
            The logitform = 'Metric' option was just mentioned. It
            does not compute probabilities per se, but numbers that
            behave like probabilities while maintaining the metric
            characteristics of the estimates, such as they are.  This
            formula is used automatically when EAR (expected absolute 
            residual) statistics are not available.  It requires both ecut
            (estimates cut-point) and maxpos (maximum possible)
            parameters.  For a derivation of the formula consult
            the tools.metricprob() documentation.

            For all other probability computations est2logit() uses
            tools.cumnormprob(), which returns the cumulative normal
            probability of an estimate relative to an estimates
            cut-point (ecut) given the variance of the estimate
            (EAR).  (This formula does not require a maxpos parameter.)
            It entails calculating the integral of the normal
            distribution curve which, though it lacks an explicit
            closed form solution, can be approximated by a number
            of methods.  One commonly used method, and the one used
            here, comes from Abramowitz and Stegun (1964) (e.g, 
            Wikipedia, Normal Distribution).

            When logitform = 'Metric' and an EAR statistic is available,
            the EAR of each cell is constrained to equal the mean EAR
            of its column.  This preserves metric integrity at the
            expense of the statistical meaning of the probability.  In
            this case, the standard error is calculated by putting
            the Estimate + EAR for each cell through the cumnormprob()
            function and getting the unsigned difference between it
            and the logit estimate.

            When logitform = 'Statistical' and an EAR statistic is
            available, the probability of each cell exceeding ecut
            is calculated based on the EAR for that cell.  This preserves
            the statistical meaning of the probability at the possible
            expense of its metric integrity, depending on fit to the model.
            In this case, the standard error of the logit estimate
            is calculated cell by cell using the binomial distribution:

                Var[p] = p * (1 - p)                                Eq. 1
                Var[logit] = 1 / p * (1 - p)                        Eq. 2
                SE[logit] = K * 1 / sqrt(p * (1 - p)),              Eq. 3

                where K = obspercell_factor and is either

                K = 2 / minimum( sqrt(nRow/ndim - 1),sqrt(nCol/ndim - 1) )
                    (if obspercellmeth = 'PickMinFac'), or          Eq. 4a

                K = 2 / sqrt( sqrt(nRow/ndim - 1) * sqrt(nCol/ndim - 1) )
                    (if obspercellmeth = 'CombineFacs')             Eq. 4b


            Eq. 1 is the variance of the probability (assuming N = 1),
            which comes from the binomial distribution.

            Eq. 2 is the variance of the logit form of p.  It is calculated
            using a property of natural logarithms in which the derivative
            d(ln(x))/dx = 1/x.  In this case, the derivative of the log
            variance equals 1/variance.  This approach to logit standard
            error is in line with Rasch-derived marginal person and item
            standard errors. However, instead of calculating errors at
            the margin as with Rasch, Damon calculates them separately
            for each cell, so that the error goes to infinity as there
            are too few row observations or too few column observations
            for a given cell.

            Eq. 3 converts variance into standard error by taking into
            account the number of independent observations associated
            with each cell, i.e., the number of observations in each
            cell's row and column. This number is part of K, the
            obspercell_factor, a statistic that is unique to Damon and
            that takes into account not only the number of observations
            but also the number of dimensions and facets.

            Equations 4a and 4b are two ways to compute K, controlled
            using the obspercellmeth argument. For more information
            about the obspercell_factor, consult the documentation in
            tools.obspercell().

        Arguments
        ---------
            "estimates" is the output of base_est(), fin_est(), equate()
            or some other datadict that you specify that is an attribute
            of the Damon object.

                estimates = 'base_est_out'
                                    =>  estimates will be drawn
                                        from base_est() outputs.
                                        Corresponding EARs, if
                                        available, will be drawn
                                        from base_ear_out.

                estimates = 'fin_est_out'
                                    =>  estimates will be drawn
                                        from fin_est() outputs.
                                        Corresponding EARs, if
                                        available, will be drawn
                                        from fin_ear_out.

                estimates = 'equate_out'
                                    =>  estimates will be drawn
                                        from equate() outputs.
                                        Corresponding EARs, if
                                        available, will be drawn
                                        from the same source.

                estimates = 'OtherDataDict_out'
                                    =>  estimates can be drawn
                                        from any other datadict
                                        that has been assigned
                                        to the Damon object.  In this
                                        case it is assumed that
                                        no EARs are available
                                        and the metricprob()
                                        function will be used.

            ------------------
            "ecutmaxpos" specifies an estimates cut-point (ecut)
            and a maximum possible estimate (MaxPos) for the array
            as a whole or for each column.  ecut is the cut-off
            above which a person is considered to have "succeeded"
            on an item, below which he is considered to have
            "failed".

            The MaxPos parameter is ignored, and can be set at None,
            whenever EAR statistics are available corresponding to
            the specified estimates.  It is only used if no EAR
            stats are available.

            ecutmaxpos is already specified as part of the base_est()
            method and is called automatically if estimates =
            'base_est_out' and ecutmaxpos = 'Auto'.

            Options:

                ecutmaxpos = 'Auto'

            means that ecutmaxpos will automatically be retrieved from the
            output of the base_est() method if estimates = 'base_est_out'.
            Otherwise it will automatically be set to ['Cols',['Med','Max']].

            When setting ecutmaxpos manually, the syntax takes one of
            the following forms:

                ecutmaxpos = ['All',[ECut,MaxPos]]

            means that the values given for ECut and MaxPos apply to the
            whole matrix, across all columns.

                ecutmaxpos = ['All',['Med','Max']]

            means that ecut should be the median ('Med') of the whole array
            and MaxPos should be the maximum ('Max') of the whole array.

                ecutmaxpos = ['Cols',['Med','Max']]

            means that the ecut and MaxPos for each column should be the
            median and maximum value of that column, calculated separately
            for each column.

                ecutmaxpos = ['Cols',{'ID1':['Med','Max'],'ID2':[10,25],...}]

            means that for the 'ID1' column ecut should be the column median
            and MaxPos should be the maximum value in the column.  For the
            'ID2' column, ecut should be 10 and MaxPos should be 25.  And so
            on for all columns.

            AGAIN:  When EAR statistics are available (either base_ear_out or
            finEAR_out exist), the MaxPos parameter is ignored and may be set
            to None.

            ------------------
            "logitform" controls the formula used for calculating
            probabilities, logits, and standard errors.

                logitform = 'Metric'
                                =>  If no EARs are available,
                                    tools.metricprob() is used to
                                    perform a metric conversion from
                                    the metric of the Damon estimates
                                    to a probability-like metric.  If
                                    the estimates are sigmoid, the
                                    probabilities will be sigmoid, too.
                                    If linear, the probabilities will
                                    be linear, too, albeit between
                                    0 and 1.

                                    If EARs are available,
                                    tools.cumnormprob() is used to
                                    compute probabilities whose error
                                    component, the EAR of each cell,
                                    is constrained to equal the mean
                                    EAR of the column, thus preserving
                                    metric objectivity.

                logitform = 'Statistical'
                                =>  If no EARs are available, the formula
                                    automatically shifts to the metricprob()
                                    function above.

                                    If EARs are available,
                                    tools.cumnormprob() is used to
                                    compute probabilities whose error
                                    component, the EAR of each cell,
                                    is allowed to differ for each cell,
                                    preserving the probabilistic meaning
                                    of each cell at the possible expense of
                                    its metric objectivity.

            ------------------
            "obspercellmeth" is the method for calculating the
            number of independent observations associated with
            each cell, i.e., the number of observations in each
            cell's row and column.  There are two options:

                obspercellmeth = 'PickMinFac', and
                obspercellmeth = 'CombineFacs'

            'PickMinFac' tends to be more accurate than 'CombineFacs' with
            smaller datasets where the ratio of observations to dimensions
            is less than, say, 10.  However, with large datasets it will
            tend to overestimate the standard error since it ignores
            information contributed by one of the facets.

            'CombineFacs' is more accurate with large data sets.  However, as the
            ratio of observations to dimensions drops below 10, it will tend
            to a standard error that is too small.  The reason is that
            in this situation (as the ratio of observations to dimensions
            becomes small) Damon estimates become biased toward the
            observations, resulting in an artificially low EAR and SE.
            'CombineFacs' is probably the most mathematically correct, but
            'PickMinFac' is safer as a default if you want to avoid false
            positive significance tests.

        Examples
        --------



        Paste method
        ------------
            est2logit(estimates = 'base_est_out', # [<'base_est_out','fin_est_out','equate_out',...>]
                      ecutmaxpos = 'Auto', # [<'Auto',['All',[ecut,MaxPos]] or ['Cols',{'ID1':[ECut1,MaxPos1],...}]> ]
                      logitform = 'Metric', # [<'Metric','Statistical'>]
                      obspercellmeth = 'PickMinFac',    # [<'PickMinFac',CombineFacs'> => to get obspercell_factor]
                      )

        """

        if self.verbose is True:
            print 'est2logit() is working...\n'

        # Run the damon utility
        est2logit_out = dmn.utils._est2logit(locals())
        self.est2logit_out = est2logit_out

        if self.verbose is True:
            print 'est2logit() is done -- see my_obj.est2logit_out, as well as logit_ear_out and logit_se_out.'
            print 'Contains:\n',self.est2logit_out.keys(),'\n'

        return None




    ##########################################################################

    def fin_est(self,
                stdmetric = 'Auto',  # [<None,'Auto','SD','LogDat','PreLogit','PLogit','Logit','0-1','Percentile','PMinMax'>]
                orig_data = 'data_out', # [<'data_out','parse_out','subscale_out','score_mc_out','std_params',...> => if latter, fill out std_params arg>]
                ents2restore = 'All',   # [<'All',[<'AllExcept','NoneExcept'>,[list of column entities to include/exclude from orig_data]]
                referto = 'Cols',    # [<None,'Whole','Cols'>]
                continuous = 'Auto',   # [<'Auto',True> => True means report continuous estimates, not rounded integers]
                std_params = None,    # [<None, standardization parameters from original data>]
                alpha = None      # [<None,True> => report predictions in original alpha instead of integers]
                ):
        """Convert Damon back to original metric.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The fin_est() outputs are accessed using:

                my_dmnobj.fin_est_out

            fin_est_out is a datadict whose 'coredata' is in the
            original metric of the Damon object for specified entities.  It
            also collapses parsed columns back to a single column.
            If base_se() has been run, fin_est() automatically
            adds "destandardized" standard errors and expected
            absolute residuals to the Damon object, formatted 
            as datadicts:

                my_dmnobj.fin_se_out
                my_dmnobj.fin_ear_out

            It also adds a "predictions dictionary" called pred_dict:

                my_dmnobj.pred_dict

            pred_dict is generated when one or more parsed columns
            (using parse()) are "deparsed" by collapsing them into
            predicted values.  The output 'coredata' array reports
            the predicted response as an integer (to avoid mixing
            non-numerical and numerical types in the same array),
            the key to which is included in pred_dict.  Thus,
            pred_dict is a dictionary containing:

                PredDict = {'pred_key':__,   =>  dictionary relating
                                                string responses to
                                                reported integers.
                                                {'a':1,'b':2,...}

                            'pred_prob':__   => (predicted response,
                                                 probability of the
                                                 predicted response)
                            }

            Also included are the probabilities of all the responses
            for each column entity.

            Workflow:
                my_dmnobj.parse(...)
                my_dmnobj.standardize(...)
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.base_resid(...)
                my_dmnobj.base_ear(...)
                my_dmnobj.base_se(...)
                my_dmnobj.fin_est(...)

            This workflow standardizes a Damon obj, calculates its
            coordinates, derives cell estimates and standard errors, then 
            destandardizes the cell estimates and standard errors to get 
            them back into the original metric.  These are the 
            "final estimates"

            How to Interpret "Deparsed" outputs
            -----------------------------------
            As long as no data columns have been parsed, the outputs are
            easy to interpret.  Each estimate is directly comparable
            to its corresponding observation, and they are equal
            if there is no error.

            However, if data columns have been parsed (see parse() docs),
            there are several types of outputs depending on how you
            set the parse() parameters:

                1.  Answer key.  If the parse() 'extractkey' option
                    was used, then outputs are the probability (or
                    logit equivalent) of the response specified in
                    the key.  If extractkey refers to an answer key,
                    then the probability is that of selecting the
                    "correct" response.  These outputs are good for
                    estimating item easiness or difficulty.

                2.  Prediction.  If parse() 'extractkey' is not used,
                    and 'ordinal' is not specified, or the data
                    are alpha, then fin_est() reports back
                    a cell prediction, either in integer or alpha
                    form, e.g., Person A on Item 1 is predicted to
                    select option 'a' (which is coded numerically
                    as an integer if alpha = None in fin_est()).

                3.  Expected Value.  If parse() 'extractkey' is
                    not used, and the responses are integers, and
                    'ordinal' is specified, then fin_est()
                    returns a continuous but bounded "expected value"
                    equal to the sum of the products of each integer
                    and its probability.  This is like a regular
                    Damon estimate, just calculated somewhat
                    differently.

                4.  Response Probabilities.  If a parsed item is
                    not included on the fin_est() 'ents2restore'
                    list, its response probabilities are returned
                    in separate columns.  This is true of both
                    integer and alpha responses.

            The parse() process can sometimes lead to confusing options,
            but it also permits quite a lot of generalizability
            and flexibility and tends to return pretty much what
            you're looking for automatically.


        Comments
        --------
            fin_est() is the inverse of standardize(). It
            converts a Damon object whose core data is an array of
            standardized values, generally estimates from base_est(),
            back to an array which approximates the "original" data
            array. "Approximates" means "in the metric of the
            original data" as well as "shaped like the original data".
            fin_est() is also the inverse of parse(), as
            it "deparses" a parsed array back to the data array's
            original dimensions.

            Mathematically, fin_est() does not try to find the
            best possible fit with the original values.  It converts
            standardized estimates into estimates and predictions that
            can be compared to the original unstandardized data.

            Destandardization involves mathematical assumptions
            that may not always be justified.  For instance, when data
            are ordinal categories, fin_est() assumes that
            the steps between successive categories space out according
            to a uniform sigmoid function whereas the steps may be
            jerkier than that in real life.  To avoid making such
            assumptions, it is always possible using the parse()
            method to "bin" all data response options for each column
            and convert them into separate column entities whose
            data are 0 or 1.  Suitably standardized to 'PreLogit'
            and run through coord() and base_est(), you can circumvent
            the need to make many standardization assumptions.

            However, while offering a general solution,
            parsing is computationally expensive and the results
            are often not as accurate as just a conventional
            standardization.  fin_est() is a quick way to obtain
            reasonably fitting estimates when the original data
            columns are in different metrics but analysis was
            done on standardized data.  However, it also is able
            to "deparse" data that was parsed into separate
            response bins, i.e., recombine them into single estimates
            or predictions.

            fin_est() is applied to a Damon object that has at some
            point been run through standardize() and perhaps parse().
            It can be applied to any comparable Damon object using
            the standardization parameters saved in its std_params outputs.
            For instance, you can restate the current estimates in
            terms of the means and standard deviations belonging to another
            data set.  You can also use std_params to convert standardized
            data to any target metric, with a different metric for
            each entity, even if there is no original dataset.  That's
            what core.create_data() does.

            Note that fin_est() can be applied to "parsed" datasets,
            one or more of whose columns have been repeated to handle
            each response option separately.  If the response options
            are ordinal, it collapses them into single "expected" values.
            An expected value is the sum of the products of the probabilities
            of a set of ordinal values by the values themselves and most
            closely matches Rasch analysis of polytomous data.  The
            relationship between the "true" and "expected" values tends
            to be slightly or moderately non-linear with an overestimate of
            the lower response categories, which tends to inflate the standard
            error.  The best standardization metric for reducing this phenomenon
            is 'PreLogit', but it can still produce biases.  It is likely that
            corrective functions can be applied to straighten out the
            relationship, but that has not been attempted.

            If data are nominal (responses are non-numerical), fin_est()
            either extracts the probabilities of a specified option (if the
            extractkey argument in parse() was used) or returns the most likely
            response coded as an integer.  The integer/response lookup key
            and response probabilities are assigned to my_dmnobj.PredDict
            by fin_est().

            fin_est() can also return the response probabilities
            without collapsing them.  The method used to deparse columns
            is not specified in fin_est() but in parse() based on
            how parse() interprets the user arguments.  These methods are
            passed from parse() to fin_est() via its 'MethDict' output.

        Arguments
        ---------
            "stdmetric" indicates the type of standardized values
            we are dealing with in the standardized array:

                None        =>  Generally, use when orig_data = 'std_params'.
                                std_params['stdmetric'] specification is
                                used to define the metric.

                                Also use None if data were not run
                                through standardize() and you are only
                                using fin_est() for its deparsing
                                capabilities.

                                Note: if you do not specify None but still
                                use std_params, std_params will over-write
                                your stdmetric parameter.

                'Auto'      =>  The 'stdmetric' parameter is chosen
                                automatically as the metric specified by
                                standardize() or over-written by est2logit().

                'SD'        =>  The inputs are in standard deviation units.

                'LogDat'    =>  The inputs are based on the log of the original
                                data, a metric used when data are on a ratio/counts
                                scale.

                'PreLogit'  =>  The inputs are in a logit metric based on
                                the standardize() 'PreLogit' metric, used to
                                standardize data sets with a mix of metrics.

                'PLogit'    =>  The inputs are based on logits from percentile
                                ranks, not to be confused with logits from
                                probabilities.

                'Logit'     =>  The inputs are logits from cell probabilities
                                calculated using est2logit().

                '0-1'       =>  The inputs range from 0 to 1 based on the
                                standardize() '0-1' metric, used to normalize
                                data sets with a mix of metrics.

                'Percentile =>  The inputs are based on the percent rank of
                                each cell value relative to others in the
                                'referto' range.

                'PMinMax'   =>  The inputs are based on converting data into
                                a percentage ranging from 0 to 1.

            For more information, see the standardize() docs.

            ----------------
            "orig_data" tells the function which array of "original
            data" fin_est() should approximate.  Suppose
            you parse the data using parse() to subdivide the
            column entities into new columns to reflect the different
            response options, and you standardize this parsed data
            and run it through coord() and get estimates using base_est().
            Now you want to convert the estimates back into the metric
            of the parsed array.  You would specify orig_data = 'parse_out'.

            If you wanted to go all the way back to the original data
            as it was initially sucked into the Damon object, you would
            specify orig_data = 'data_out'. (This works even when the data
            have been parsed; they are deparsed automatically.)

            'std_params' tells fin_est() not to refer to any of the datasets
            created for the current Damon object but to parameters
            either created in some other Damon object or specified
            manually.  In this case, you would fill out the "std_params ="
            argument below.  The std_params parameters overwrite the
            corresponding parameters of fin_est().

            Options:

                'data_out'  =>  Matches to the data you initially loaded
                                into the Damon object before applying any other
                                methods.

                'parse_out' =>  Matches to the data obtained after parsing
                                the data into extra columns to model each
                                response option separately.

                'score_mc_out'
                            =>  Matches to the data obtained after applying
                                an answer key to response data.

                'std_params' =>  Destandardize using only parameters contained
                                in the std_params argument.


            ----------------
            "ents2restore" specifies which column entities in orig_data to
            destandardize or not destandardize.  All other columns are
            left as they are, i.e., in their standardized form.  Options:

                ents2restore = 'All'
                            =>  All entities are to be destandardized.
                                If the entities were parsed, then the
                                destandardization procedure is in
                                accord with the MethDict created by
                                standardize.

                ents2restore = ['AllExcept',['It1',It3']]
                            =>  All entities are to be destandardized
                                except items 1 and 3.  These are to
                                be left in their standardized metric.

                                If items 1 and 3 were parsed and are now
                                recombined, they are reported in a
                                standardized metric (a logit, usually)
                                instead of a probability.

                ents2restore = ['NoneExcept',['It1',It3']]
                            =>  No entities are to be destandardized
                                except items 1 and 3.  All others are to
                                be left in their standardized metric.

            ----------------
            "referto" specifies whether the destandardization is
            in terms of the whole array (referto = 'Whole') or
            relies on parameters that are different for each column
            (referto = 'Cols').

                referto = None
                            =>  Use when orig_data = 'std_params'.  The
                                std_params['referto'] specification is
                                used to define the metric.

                                Note: if you do not specify None but still
                                use std_params, std_params will over-write
                                your referto parameter.

                referto = 'Cols'
                            =>  destandardize using statistics unique
                                to each column, such as column mean and
                                standard deviation.  Standardization is
                                usually in terms of columns, so this is
                                the default option.  It also supports
                                the various deparsing methods specified
                                in the MethDict output by parse().

                referto = 'Whole'
                            =>  destandardization is performed using
                                statistics, such as Mean and SD, calculated
                                from the whole array.  'Whole' doesn't
                                work unless ents2restore = 'All' and there
                                is a one-to-one correspondence betweeen
                                the original array and the standardized array.

            Usually, referto = 'Cols'.  The 'Whole' option only
            makes sense in a few situations.

            ----------------
            "continuous" specifies, when the original data are integers,
            whether to return results as rounded integers ('Auto') or
            as continuous values (True).

                continuous = 'Auto' =>  The decision whether to round estimates
                                        is decided on a column-by-column basis
                                        according to the Damon validchars
                                        specification.  For example,  [0,1,2],
                                        and ['0 -- 2'] call for integer outputs
                                        whereas ['0.0 -- 2.0'] calls for continuous
                                        outputs.

                continuous = True   =>  All columns will be reported on
                                        a continuous scale, except when data
                                        have been run through parse() and
                                        the 'Pred' method is applied (by the
                                        program) in the MethDict for a given
                                        column.

            ----------------
            "std_params" is a dictionary of standardization parameters
            that can either be specified manually or captured from
            the output of the standardize() method.  In general,
            the std_params dictionary has several uses:

                1.  As an argument to the standardize() method, it
                    standardizes the data in Damon in a way that
                    is consistent with the standardization of another
                    data set, essentially over-writing the relevant
                    arguments in the standardize() expression.  This is
                    important in Damon anchoring designs where we
                    are trying to get the estimates for specified
                    entities onto the same metric.

                2.  When RetStdParams = 1, standardize() creates a
                    std_params dictionary to be passed to other methods,
                    particularly fin_est().  fin_est() uses
                    the information in the std_params dictionary to
                    convert standardized data back to the original
                    metric of the data.

                3.  Alternatively, std_params can be used in fin_est()
                    to convert standardized data to any target metric, with
                    target means and standard deviations or target minima
                    and maxima, when there is no original data set.  It is used
                    in this way by the core.create_data() function.

            When fin_est() encounters a parsed column (as determined
            from the MethDict dictionary output by parse()) the std_params
            argument is ignored.  Parsed columns are deparsed/destandardized
            in their own way.

            When 'stdmetric' is 'Percentile' or 'PLogit', std_params['params']
            consists of the whole original dataset (needed to get
            percentiles from two datasets into the same metric).

            If orig_data = 'std_params' and std_params = None, fin_est()
            will look for the std_params dictionary in the output of
            the standardize() method.  If it can't find it, it will
            throw an error.  If orig_data = 'std_params' and std_params = a
            dictionary specified by the user, that dictionary will
            supersede any std_params dictionary that may have been created
            by standardize().

            std_params is a Python dictionary of parameters:

            {'stdmetric':   =>  <'SD','LogDat','PreLogit','PLogit','0-1','Percentile','PMinMax'>

            'validchars':   =>  <None, the validchars specification from the original
                                data set or for any target dataset>

            'referto':      =>  <'Cols','Whole'>

            'params':       =>  if 'stdmetric' == 'SD':
                                    if 'referto' == 'Whole':
                                        {'All':[ArrayMean,ArraySD]}
                                    elif 'referto' == 'Cols':
                                        {'It1':[Mean1,SD1],'It2':[Mean2,SD2],...}

                            =>  elif 'stdmetric' == 'LogDat':
                                    None (no parameters are necessary; data are log ratios/counts)

                            =>  elif 'stdmetric' == 'PreLogit':
                                    if 'referto' == 'Whole':
                                        {'All':<'VCMinMax' or [ArrayMean,ArraySD]>}
                                    elif 'referto' == 'Cols':

                    Note:  The 'PreLogit' metric can standardize datasets that are a
                    mix of interval/ratio data and ordinal/sigmoid data,
                    so it can also destandardize back to the same
                    metrics using std_params.  To destandardize back to interval/
                    ratio data, it allows you to specify a desired mean and
                    standard deviation.  To destandardize back to ordinal/sigmoid
                    data, it allows you to specify a desired minimum and maximum.
                    However, the specification of minimum and maximum is not done
                    under the 'params' key, but under the 'validchars' key, as
                    validchars contains the information necessary to infer a
                    minimum and maximum.  Therefore, when filling out the 'params'
                    values, specify a mean and standard deviation for each entity
                    that is originally in an interval/ratio metric.  Specify 'VCMinMax'
                    (validchars minimum and maximum) for each entity that is originally
                    in an ordinal/sigmoid metric.

                    It is important to note that the decision to treat a given
                    entity as interval/ratio or as ordinal/sigmoid is based
                    entirely on information in the validchars parameter.  If
                    std_params['params'] specifies a mean and standard deviation for
                    an entity that is actually ordinal/sigmoid (based on whether it
                    has a bounded range in std_params['validchars']), it will over-write
                    the mean and standard deviation with the appropriate min and max.
                    If std_params['params'] specifies 'VCMinMax' for an entity that
                    is actually ratio/interval, it will arbitrarily impose a mean of
                    0 and a standard deviation of 1.
                            {'It1':'VCMinMax','It2':[ColMean,ColSD],...}

                            =>  elif 'stdmetric' == 'PLogit':
                                    if 'referto' == 'Whole' or 'referto' = 'Cols':
                                        None (no parameters are necessary; data are probabilities)

                            =>  elif 'stdmetric' == '0-1':
                                    if 'referto' == 'Whole':
                                        {'All':<'VCMinMax' or [ArrayMean,ArraySD]>}
                                    elif 'referto' == 'Cols':
                                        {'It1':'VCMinMax','It2':[ColMean,ColSD],...}

                                Note:  The '0-1' metric, like the 'PreLogit' metric can
                                standardize datasets that are a mix of interval/ratio data
                                and ordinal/sigmoid data.  See the 'PreLogit' note for
                                details.

                            =>  elif 'stdmetric' == 'Percentile' or 'PLogit':
                                    if 'referto' == 'Whole' or 'referto' = 'Cols':
                                        whole original dataset as datadict

                            =>  elif 'stdmetric' == 'PMinMax':
                                    if 'referto' == 'Whole':
                                        {'All':[ArrayMin,ArrayMax]}
                                    elif 'referto' == 'Cols':
                                        {'It1':[ColMin,ColMax],'It2':[ColMin,ColMax],...}


            'rescale':      =>  if 'stdmetric' == 'SD','PreLogit','PLogit' or 'LogDat':
                                if 'referto' == 'Whole':
                                    {'All':[m,b]} (m = multiplier, b = intersept)
                                elif 'referto' == 'Cols':
                                    {'It1':[m1,b1],[m2,b2],...}
                                else:
                                    None

            'orig_data':     =>  orig_data':datadict
                                   (previous data set is stored under 'orig_data' as
                                    a 'datadict')
                                else:
                                    None
            }

            ----------------
            "alpha" <None,True> allows predictions of alpha characters
            ('a','b','c') to be expressed as alpha characters rather than as
            integers (1,2,3).  The reason alpha = True is not the default
            is that it forces the whole array, including numerical
            values, to be expressed as strings, which is a hindrance for
            some purposes.  For reporting, alpha = True is fine.  It
            is also readable by fin_resid() for calculating match
            statistics.

            GOTCHA
            Sometimes specifying alpha does not return alpha characters,
            even when the original data was alpha.  This is probably
            because you scored the data using score_mc() or used the
            extractkey argument in parse().  These make it impossible to
            predict the original observed alpha response.  However, the
            reported estimates do estimate whether the original response
            was the answer key's "correct" value or not.  It's just the
            "incorrect" responses that it cannot predict.  If you need
            alpha predictions, do not use score_mc() or the parse()
            extractkey argument.

        Examples
        --------

            [under construction]

        Paste method
        ------------
            fin_est(stdmetric = 'Auto',  # [<None,'Auto','SD','LogDat','PreLogit','PLogit','Logit','0-1','Percentile','PMinMax'>]
                    orig_data = 'data_out', # [<'data_out','parse_out','subscale_out','score_mc_out','std_params',...> => if latter, fill out std_params arg>]
                    ents2restore = 'All',   # [<'All',[<'AllExcept','NoneExcept'>,[list of column entities to include/exclude from orig_data]]
                    referto = 'Cols',    # [<None,'Whole','Cols'>]
                    continuous = 'Auto',   # [<'Auto',True> => True means report continuous estimates, not rounded integers]
                    std_params = None,    # [<None, standardization parameters from original data>]
                    alpha = None      # [<None,True> => report predictions in original alpha instead of integers]
                    )

        """
        if self.verbose is True:
            print 'fin_est() is working...\n'

        # Run utility
        fin_est_out = dmn.utils._fin_est(locals())
        self.fin_est_out = fin_est_out

        if self.verbose is True:
            print 'fin_est() is done -- see my_obj.fin_est_out, as well as finEAR_out and finSE_out'
            print 'Contains:\n',self.fin_est_out.keys(),'\n'

        return None


    ##########################################################################

    def fin_ear(self
                ):
        """Non-existent method.  Handled automatically by fin_est().

        """
        print ('Error:  There is no fin_ear() method.  fin_ear_out is created '
               'automatically by the fin_est() function if you have first run '
               'base_ear().\n')
        sys.exit()


    ##########################################################################

    def fin_se(self
               ):
        """Non-existent method.  Handled automatically by fin_est().

        """
        print ('Error:  There is no fin_se() method.  fin_se_out is created '
               'automatically by the fin_est() function if you have first run '
               'base_se().\n')
        sys.exit()


    ##########################################################################

    def fin_resid(self,
                  resid_type = ['All',['Diff']], # [<['All',<,['Diff'],['Nearest'],['ECut',Val],['Match']>], ['Cols',{'It1':['ECut',Val],'It2':['Diff'],...}] ]
                  psmiss = None,   # [<None, True => only report for pseudo-missing cells]
                  ):
        """Calculate residuals between original cell values and final (fin_est_out) estimates.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The fin_resid() outputs are accessed using:

                my_dmnobj.fin_resid_out

            fin_resid_out is a datadict of cell residuals -- differences
            between the original cell-level data and the final cell
            estimates as output by fin_est().

            fin_resid() requires fin_est() outputs.

            fin_resid() can output up to four types of residuals
            on a column-by-column basis depending on the fin_est()
            outputs.

            Workflow:
                MyResiduals = my_dmnobj.fin_resid(...)

            where estimates were calculated previously:
                my_dmnobj.fin_est(...)

        Comments
        --------
            fin_resid() focuses on the residuals between the
            original data values, even if they are alpha characters
            ('a','b','c'), and the final estimates as output by
            fin_est().  base_resid(), on which fin_resid() is
            modeled, focuses on the residuals between the inputs
            (perhaps standardized and parsed) that go into the
            coord() method and the estimate outputs that come out
            of the coord() and base_est() functions.  base_resid()
            residuals have the same metric across the array.
            fin_resid() residuals may have have a different metric for
            each column.

            If you have not needed to parse or standardize the data,
            base_resid() residuals are entirely sufficient and there is
            no need to run fin_resid() -- indeed, you would be unable to
            since fin_resid requires that fin_est() first have been
            run.

            Of the summstat() summary statistics, only 'Resid' is
            possibly affected by the fin_resid() outputs.  EAR and
            SE  are computed using base_resid() and transformed
            into the different observed metrics by fin_est().

            The fin_resid() outputs give a great look at "how you did".
            However, as they are not computed for missing cells, they
            can cause summary statistics to be biased or misleading
            if the missing data are not randomly missing.

        Arguments
        ---------
            "resid_type" specifies what type of residuals to calculate
            for the array as a whole or for each column separately.
            The syntax is:

                resid_type = ['All',[resid_type]] (same type for all cols)

            or...

                resid_type = ['Cols',{'It1':[RType1],'It2':[RType2],...}]

            The residual types are:
                <['ECut',Val],['Nearest'],['Match'],['Diff']>

            Options:
                ['Diff']            =>  The observed value minus the estimate.
                                        residuals may be negative.

                ['Nearest']         =>  'Nearest' means the residual will
                                        be based on the difference
                                        between the observed value and the
                                        estimate ROUNDED to the nearest valid
                                        observed value.  residuals may be
                                        negative.

                ['All',['Nearest']] or...
                ['Cols',{'It1':['Nearest'],'It2':['Nearest'],...}]
                                    =>  'All' signifies that the 'Nearest'
                                        method will be applied to all
                                        columns.  'Cols' followed by a Python
                                        dictionary signifies which specific
                                        columns (items) will use the 'Nearest'
                                        method.

                ['ECut',Val]        =>  The 'ECut' option is used to set
                                        a cut-point in the final estimates
                                        (not the base estimates) separating
                                        "success" from "failure".  When the
                                        observation and estimate fall on the
                                        same side as the cut-point, the
                                        residual is 0, otherwise 1.  There
                                        are no negative residuals in this
                                        type of residual.

                ['All',['ECut','Med']]
                                    =>  'Med' signifies that the cut-point
                                        will be the array median.  'All'
                                        signifies that the median for
                                        the whole array will be used.

                ['Cols',['ECut','Med']]
                                    =>  'Med' signifies that the cut-points
                                        will be column medians.  'Cols'
                                        signifies that the cut-point will
                                        be local to each column's individual
                                        median.

                ['All',['ECut',3.0]]
                                    =>  A cut-point of 3.0 will be applied
                                        across the whole array.

                ['Cols',{'It1':['ECut',3.0],'It2':['ECut','Med'],'It3':['Nearest'],...}]
                                    =>  Item 1 will have an ecut-style residual
                                        based on a column 3.0 cut-point.
                                        Item 2 will also be 'ECut' based on
                                        the median of the column.  Item 3
                                        will have a 'Nearest' style residual.

                ['Match']           =>  'Match' means the residual will be
                                        based on whether a predicted response
                                        exactly matches the observed response (0)
                                        or not (1).  It is useful when estimates
                                        are in the form of predictions of
                                        response values.  'Match' does not
                                        report negative residuals.

            More examples:
                ['All',['Match']]
                ['Cols',{'It1':['ECut',3.0],'It2':['Nearest'],'It3':['Match'],...}]


            ---------------
            "psmiss" (<None, True>) provides the option of calculating
            residuals for only those cells whose values were made pseudomissing
            prior to estimation.  These residuals test the true predictive
            accuracy of the model.  The pseudo missing cell index is obtained
            from the output of the pseudomiss() method.

            This option only works if the estimates array has all the same
            column keys as the observed array and the two arrays are identically
            sized.

        Examples
        --------



        Paste method
        ------------
            fin_resid(resid_type = ['All',['Diff']], # [<['All',<,['Diff'],['Nearest'],['ECut',Val],['Match']]>], ['Cols',{'It1':['ECut',Val],'It2':['Diff'],...}] ]
                      psmiss = None,   # [<None, True => only report for pseudo-missing cells]
                      )

        """
        if self.verbose is True:
            print 'fin_resid() is working...\n'

        # Run the damon utility
        fin_resid_out = dmn.utils._fin_resid(locals())
        self.fin_resid_out = fin_resid_out

        if self.verbose is True:
            print 'fin_resid() is done -- see my_obj.fin_resid_out'
            print 'Contains:\n',self.fin_resid_out.keys(),'\n'

        return None


    ##########################################################################

    def fin_fit(self,
               ):
        """Calculate "final" cell fit statistics.

        Returns
        -------
            The method returns None but assigns an output datadict to the
            Damon.  The fin_fit() outputs are accessed using:

                my_dmnobj.fin_fit_out

            fin_fit_out is a datadict whose coredata consists of
            cell-level fit statistics.  fin_fit() depends on fin_resid()
            and fin_ear(), the latter being an output of fin_est().

        Workflow:
            my_dmnobj.base_est(...)
            my_dmnobj.base_resid(...)
            my_dmnobj.base_ear(...)
            my_dmnobj.base_fit(...)

        Comments
        --------
            fin_fit() is the same as base_fit() except that it is
            applied to "final" (fin_est()) estimates and
            their corresponding EARs (expected absolute residuals).

            (For a general discussion on the use of fit statistics,
            see the comments on base_fit().)

            The only reason to use fin_fit() is to obtain fit
            statistics for the original column entities before
            they were split apart by the parse() method.  fin_fit_out
            is also called by summstat() when 'Final' is specified.

            Otherwise, base_fit() is the preferable method for
            generating fit statistics.  Its inputs have not incurred
            any errors by going through the fin_est()
            process, and because all cells are in the same metric
            it is less likely to suffer artifacts caused by
            columns being in different metrics.

        Arguments
        ---------
            None

        Examples
        --------


        Paste method
        ------------
            fin_fit()


        """

        if self.verbose is True:
            print 'fin_fit() is working...\n'

        # Run the damon utility
        fin_fit_out = dmn.utils._fin_fit(locals())
        self.fin_fit_out = fin_fit_out

        if self.verbose is True:
            print 'fin_fit() is done -- see my_obj.fin_fit_out'
            print 'Contains:\n',self.fin_fit_out.keys(),'\n'

        return None



    #############################################################################
    def fillmiss(self,
                 ):
        """Fill in the missing cells with predictions.

        Returns
        -------
            The method returns None but assigns an output datadict
            to the Damon object where coredata is an array consisting
            of the original observations where data is non-missing,
            and estimates/predictions where the data is missing.

            The estimates are those generated by fin_est()
            or, if fin_est() is not run, base_est().

            Workflow:
                my_dmnobj.standardize(...)
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.fin_est(...)
                my_dmnobj.fillmiss()

        Comments
        --------
            fillmiss() is a tool for completing incomplete data
            arrays, the array of "observed" data that is first
            loaded into Damon.  Unlike base_est() or fin_est(),
            it does not replace ALL cells with "most likely" values,
            but only those cells which are missing in the observed data
            array.

            The metric of the output estimates is controlled by
            how you choose to calculate the estimates and whether
            you use fin_est().

            fillmiss() operates only on those columns in your
            estimates that contain column keys that match those
            in the observed data.  All other columns report
            estimates rather than observations.

        Arguments
        ---------
            None

        Examples
        --------


        Paste method
        ------------
            fillmiss()

        """

        if self.verbose is True:
            print 'fillmiss() is working...\n'

        # Run the damon utility
        fillmiss_out = dmn.utils._fillmiss(locals())
        self.fillmiss_out = fillmiss_out

        if self.verbose is True:
            print 'fillmiss() is done -- see my_obj.fillmiss_out'
            print 'Contains:\n',self.fillmiss_out.keys(),'\n'

        return None


    #############################################################################

    def item_diff(self,
                  scores,  # [Ent x Item datadict of scores/estimates]
                  curve = 'Linear', # [<'Sigmoid','Linear'> => metric of scores]
                  pcut = 0.50,   # [Probability cut-point that defines minimal "success"]
                  rescale = None,   # [<None,[mean,sd]> target mean, sd of each item]
                  minmax = None, # [<None,[Min,Max]> => min and max score allowed after rescaling]
                  ):
        """Calculate item difficulty by relating scores to probability of success.

        Returns
        -------
            The method returns None but assigns an output datadict
            to the Damon object where coredata is nItems x 1 array of
            item difficulties.  Access using:

                my_dmnobj.item_diff_out

            outputs are automatically read by the summstat()
            method if 'IDiff' is specified as an output statistic.

            Workflow:
                my_dmnobj.standardize(...)
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.fin_est(...)
                my_dmnobj.est2logit(...)   This is essential
                my_dmnobj.item_diff(...)
                my_dmnobj.summstat(...)

        Comments
        --------
            item_diff() calculates a statistic called "item difficulty",
            though this statistic differs in important respects from
            item difficulty as it is defined in 1-dimensional psychometric
            models.  It also differs from the mean column scores output
            by summstat().  If you are reporting logit outputs of
            est2logit(), then each item difficulty is zero by definition
            (which may not equal the column means) and you don't need
            to run item_diff().

            In item_diff(), item difficulty is defined as that score or
            estimate that corresponds to a 0.50 probability of success on a
            given item (though other probability cut-points can be used).
            In this, it agrees with 1-dimensional psychometric models.

            Where it differs is that a 1-dimensional model constrains all
            the items to fall within a single common dimension whereas a
            multidimensional model such as NOUS does not.  This means that
            in the 1-dimensional case items and persons can be placed
            coherently on a single scale such that if Person A gets Item 1
            right, and Item 1 is harder than Item 2, then Person A can be
            predicted to get Item 2 right.  Not so in the multidimensional
            case.  Here, it is quite possible for Person A to get Item 1
            right (say, it is Math), and for Item 1 to be harder than
            Item 2 (which tests artistic ability, say), but for Person A
            to get Item 2 wrong.  Why?  Because he happens to be good at
            Math and bad at Art.

            Thus, the item difficulties calculated by this function are
            unique to the construct defined by each item.  They resemble
            Rasch difficulties only in that each difficulty corresponds to a 0.50
            probability of getting an item right.  In other words, if a
            student gets a 350 on a given item, and 350 is also the difficulty
            of the item as calculated by item_diff(), then we would
            say the student has a 0.50 probability of getting the item
            right.

            This information comes in handy when designing tests.  If we
            expect the average student to score 350 on a test, then we
            want the average item difficulty also to be 350, and we would
            pick items that are evenly spread above and below the 350 mark
            to match the expected student distribution.

            This statistic also allows items to be placed on a "Wright Map",
            a vertical difficulty scale that makes it possible to compare
            where the item difficulty distribution falls relative to the
            person ability distribution.  The only caveat is that when we
            locate students on the same Wright Map, if the items really are
            sensitive to different dimensions, we won't be able to draw
            hard and fast conclusions about how a person will perform on
            one item given his location relative to another item.  In practice,
            however, even in a multidimensional test, items tend to be
            sufficiently correlated that this is not a problem.

            Which raises the question, if the item difficulty is not sufficient
            to predict student performance on other items, what other statistics
            should Damon use to accomplish this end?  The answer is:  item
            coordinates.  Item coordinates are the Damon equivalent of item
            difficulty statistics in 1-dimensional models.  The problem is
            that these coordinates describe a multidimensional space that
            is hard to visualize, and the whole concept of difficulty
            becomes, well, difficult.  Since there is a practical need for a
            simple, tangible difficulty statistic, item_diff() creates an
            alternative to item coordinates that provides a scale score
            equivalent of a 0.50 probability of getting an item right.

            Note that as an item's difficulty approaches the top or bottom
            of the ability range of the sample, it becomes increasingly
            uncertain and large.  This also occurs in multiple choice tests
            as the probability of a correct response approaches the probability
            of getting the item correct by chance.  To prevent large, aberrant
            item difficulties, clip them using the minmax parameter.

            IMPORTANT
            ---------
            To run item_diff(), you must first have run est2logit().

        Arguments
        ---------
            "scores" is an Ent x Item datadict of scores or estimates.
            Generally this will be either the fin_est() or the base_est()
            outputs (fin_est_out, base_est_out), but it is not restricted
            to these.  Use extract() to select only those columns for
            which you want item difficulties and set scores = obj.extract_out.

            Since an important use-case is getting item difficulties into
            some outside metric, such as state scale-scores, the question
            will naturally arise, how do we get an array of estimates,
            from base_est_out, say, into a scale score metric to start
            with?  The quick answer is, use the item_diff() rescale argument.
            The specified rescale mean and standard deviation (see below)
            converts the estimates into "scale scores" in which each column
            has the same scale score mean and standard deviation.  By forcing
            each column to have the same average scale score, it seems like
            we are saying that each column has the same difficulty.  Not so.
            As we compare the array of scale scores with the corresponding
            array of logits/probabilities, we see that the scale score
            corresponding to a 0.50 probability of success differs
            from column to column, and it is this which defines the item's
            difficulty in the scale score metric, not the column average.

            It is important that the column keys in the scores matrix
            have matches in the logit estimates matrix output by est2logit().

            If the scores were calculated outside the Damon object, you will
            need to pull them inside a Damon object and do the necessary
            steps to compute probabilities of success using est2logit().
            The original scores can then be used for the scores
            parameter while item_diff() refers to the probabilities/logits
            that were computed from them.

                scores = my_obj.est2logit_out

            -----------
            "curve" describes the metric of the scores -- whether they
            are linear or sigmoid.

                curve = 'Linear'    =>  The scores have no theoretical
                                        floor or ceiling, so all units
                                        are the same size.

                curve = 'Sigmoid'   =>  The scores do have a theoretical
                                        floor and ceiling, so the units
                                        crunch at the top and bottom.
                                        This will generally occur if your
                                        scores are the outputs of fin_est()
                                        and the original data values were
                                        dichotomous or polytomous.  base_est()
                                        outputs are usually (or should be)
                                        linear, assuming you applied
                                        the standardize() method to make
                                        them so if necessary.

            The best way to determine whether to set curve as linear or
            sigmoid is to graph the scores against the corresponding
            logits.  If they form an S-shaped curve, use 'Sigmoid'.  If
            they form a straight line, use 'Linear'.

            -----------
            "pcut" is the probability cut-point used to define success.  By
            convention, this is almost always set at 0.50.  A student is said
            to be "at the item" if he has a 0.50 chance of getting it right.  But
            sometimes it makes sense to define success on an item differently,
            as having 0.75 probability of getting it right, for instance.

            -----------
            "rescale" tells the function to convert the scores and resulting
            difficulties into a new metric such that the column means have a
            specified mean and standard deviation.

            Here is how it comes up.  We have scores in logits, say, but we want
            the item difficulties to be in a scale score metric ranging from 200
            to 600.  How do we perform the conversion?

            Using rescale, we specify that we expect a range of person abilities to
            have a mean of 300, say, and a standard deviation of 60.  This comes
            from our knowledge of each student's overall performance on some outside
            test.  item_diff() takes this specification and converts the "scores"
            array into these outside scale scores such that the mean and standard
            deviation of each column matches our expectation.  We are saying, in
            effect, that regardless of the item the student's underlying ability
            as measured on the outside test is the same.  (This requires us to
            assume the items are unidimensional, and they generally are not,
            which affects the interpretation of the item difficulties as described
            in the comments above.)

            These rescaled scores are then compared with the logit/probabilities
            of success, and the score corresponding to a 0.50 probability of
            success is flagged as the item's difficulty.

                rescale = None      =>  No rescaling is necessary

                rescale = [300,60]  =>  It is expected that the items have a mean
                                        of 300 and standard deviation of 60.
                                        Item difficulties will be reported in
                                        this scale score metric.

            -----------
            "minmax" is the maximum and minimum item difficulty score allowed after
            rescaling.  It is used to limit extreme scores at the ends of the
            scale where error is high.

        Examples
        --------

            [under construction]


        Paste Method
        ------------
            item_diff(scores,  # [Ent x Item datadict of scores/estimates]
                      curve = 'Linear', # [<'Sigmoid','Linear'> => metric of scores]
                      pcut = 0.50,   # [Probability cut-point that defines minimal "success"]
                      rescale = None,   # [<None,[mean,sd]> target mean, sd of each column]
                      minmax = None, # [<None,[Min,Max]> => min and max score allowed after rescaling]
                      )

        """
        if self.verbose is True:
            print 'item_diff() is working...\n'

        # Run the damon utility
        item_diff_out = dmn.utils._item_diff(locals())
        self.item_diff_out = item_diff_out

        if self.verbose is True:
            print 'item_diff() is done -- see my_obj.item_diff_out'
            print 'Contains:\n',self.item_diff_out.keys(),'\n'

        return None




    #############################################################################

    def summstat(self,
                 data = 'base_est_out', # [<'base_est_out','fin_est_out','est2logit_out','data_out',...> => data for calculating stats]
                 getstats = ['Mean', 'SD', 'SE', 'Count', 'Outfit', 'Fit_Perc>2', 'PtBis', 'Rel'], # [Select stats from [<'Mean','SE','SD','Corr','PtBis','Resid','RMSEAR','Sep','Rel','Outfit','Fit_Perc>2','Count','Min','25Perc','Median','75Perc','Max','Coord'>] ]
                 getrows = None, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}>]
                 getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}>]
                 itemdiff = None,   # [<None,True> => express item (column) mean as a probabilistically defined item "difficulty" ]
                 outname = None,    # [<None,'cluster_name'> => name to append to summstat outputs for multiple analyses]
                 labels = None,   # [<None, {'row_ents':<None, 'person',...>, 'col_ents':<None, 'item',...>}> => to describe summarized entities]
                 group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75'],  # [<see tools.group_se docs> => factor for grouping se's]
                 correlated = None  # [<None> => deprecated]
                 ):
        """Calculate entity/attribute summary statistics.

        Returns
        -------
            The method returns None but assigns summstat_out to the
            Damon object.  summstat_out contains:

                {
                'row_ents',  =>  Damon object of stats for row entities
                'col_ents',  =>  Damon object of stats for col entities
                'descriptors', => Brief description of data source and entities
                'getrows',   =>  Repeats the 'getrows' parameter
                                 used in summstat(), as info
                'getcols',   =>  Repeats the 'getcols' parameter
                                 used in summstat(), as info
                'objperdim'  =>  objectivity at each dimensionality
                'stability'  =>  stability of coordinate structure
                'objectivity'=>  objectivity at best dimensionality
                }

            For example, to access column statistics (with outnames = None):

                my_dmnobj.summstat_out['col_ents']

            If summstat() is run multiple times for a given Damon object,
            to get stats for each subspace for instance, summstat_out will
            contain a nested dictionary of Damon objects for each subscale:

                {'sub_0':{'row_ents':_,
                          'col_ents':_, etc.
                 'sub_1':{'row_ents':_,
                          'col_ents':_, etc.
                }

            These can be combined into one report using the merge_summstat()
            method.
            
            summstat() also defines row_ents and col_ents as attributes
            of the Damon object.  They are accessed as:

                my_rowstats = my_obj.row_ents_out
                my_colstats = my_obj.col_ents_out

            Assigning them to the Damon object likes this makes it easy
            to export them using export().

            If outname is used, summstat outputs are given a suffix
            to distinguish the summstat results, making it possible
            to do multiple summstat analyses within a given Damon
            object:

            For outname1:
                my_dmn_obj.summstat_out['outname1']['col_ents']['coredata']
                my_rowstats = self.row_ents_out_outname1
                my_colstats = self.col_ents_out_outname1

            For outname2:
                my_dmn_obj.summstat_out['outname2']['col_ents']['coredata']
                my_rowstats = self.row_ents_out_outname2
                my_colstats = self.col_ents_out_outname2

            row_ents and col_ents are Damon objects of summary statistics
            for specified entities and attributes.  Statistics
            summarize in various ways the cell estimates and errors
            pertaining to the range of cells associated with each
            specified entity or entity attribute.  Besides being
            reported in a table, each entity and type of statistic
            is key-word accessible.  Here is a list of statistics
            (detailed descriptions are available below).

                'Mean',
                'SE',
                'SD',
                'Corr',
                'PtBis',
                'Resid',
                'RMSEAR',
                'Sep',
                'Rel',
                'Outfit',
                'Fit_Perc>2',
                'Count',
                'Min',
                '25Perc',
                'Median',
                '75Perc',
                'Max',
                'Coord'

            summstat() is only able to generate these statistics
            if you have already run the requisite methods.

            Workflow (methods usually run, all or in part, before
            summstat):
                my_dmnobj.score_mc(...)
                my_dmnobj.standardize(...)
                my_dmnobj.coord(...)
                    or my_dmnobj.sub_coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.base_resid(...)
                my_dmnobj.base_se(...)
                my_dmnobj.fin_est(...)
                my_dmnobj.fin_resid(...)
                my_dmnobj.fin_fit(...)
                my_dmnobj.est2logit(...)
                my_dmnobj.summstat(...)

            Rasch Summary Statistics
            ------------------------
            When the rasch() method is used, its summary statistics
            are automatically assigned to summstat_out, so you don't
            need to run summstat().  Access rasch() summary statistics
            either using my_obj.rasch_out['summstat'] or by
            my_obj.summstat_out, which contains 'row_ents' and 'col_ents'
            like regular summstat().

            Incidentally, merge_summstat() works cleanly with rasch()
            outputs.

            Printing Summary Statistics
            ---------------------------
            Summary statistics are the main output of Damon.  To convert
            them into reader-friendly tables, two approaches can be used:

                *   Export the tables using Damon.export() and format
                    them in a spreadsheet program.

                *   Pretty-print them using tools.print_summstat().

        Comments
        --------
            summstat() is used to calculate a variety of summary statistics
            to answer the following types of questions:

                What is the ability of each person?

                What is the difficulty/easiness of each item?

                What is the range of measures?

                What is the standard error of each person or item?

                How reliable is each item?

                How reliable is the test as a whole?

                How objective are the coordinates?

                How well do the observations fit the model?

                What is the average ability of girls versus boys?

                What is the average performance on math versus language?

            summstat() calculates summary statistics for rows or row clusters
            given a single range of columns, or for multiple columns or
            column clusters given a single range of rows.

            What is great about summstat() is that it is extremely flexible
            in defining how you want cells aggregated, by individual
            entity, by group, whether by row or column, and for just the
            statistics you are interested in.  However, there are forms of
            summarization that require some extra syntax.  If you
            want a mean for EACH person for EACH of a set of item clusters, for
            example, you will need to do a separate summstat() run for
            each item cluster.  This is easily done within the same Damon
            object using different "outnames" for each item cluster.  They
            can be merged into a single report using the merge_summstat()
            method.

            To get a reduced set of stats in this situation, you can also
            use equate().

            Additional summary statistics can be obtained using the score_mc()
            method.
            
            So far, the most useful workflow is to use equate to calculate
            person construct measures (with SE, test reliability, etc) and 
            reserve summstat() for just calculating item statistics.
            
            Performance
            -----------
            summstat() is unfortunately quite slow.  This is because
            it computes summary statistics separately for each range
            of cells (generally, but not necessarily, each row or column),
            without being able to take advantage of Numpy's efficiencies
            in computing row and column statistics.  This is the price
            summstat() pays for flexibility in setting the ranges
            of cells for which stats will be calculated.  If performance
            is an issue, consider computing your own summary statistics
            directly from Damon's output arrays (e.g., base_est_out).
            
            As mentioned above, a typical workflow will use equate() to
            calculate person statistics and summstat() to calculate
            item statistics, in which case performance won't be as much
            of an issue.

            Standard Error
            --------------
            The reported "SE" statistic is the Root Mean Squared Error (RMSE)
            across cells in the target range, where the individual cell
            standard errors are calculated using such methods as base_se().
            The formula for calculating cell standard errors is pretty solid
            in Damon.  The formula for combining them, as in summstat() and
            equate(), is more problematic.
            
            For one thing, the summstat() method uses a different aggregation
            formula than the equate() method, which is based not on the cell 
            standard errors calculated by base_se() but on their corresponding 
            log row and column coordinates.  This introduces a small overall 
            negative bias in the equate() stats, which is tolerated because
            of importance of this formula for equating.

            summstat()'s default SE = RMSE formula is considered "correct"
            inasmuch as it represents the canonical way of combining 
            standard errors.  But even here there is uncertainty.  Let's say
            we are aggregating errors across columns to calculate a row
            standard error.  If the columns are all positively correlated
            the RMSE formula is sufficient.  But if they include a mix
            of columns that are negatively correlated to the others, the
            effect will be to greatly reduce the range of the average
            estimate since positives and negatives cancel out.  To correct
            the standard error of such estimates, we adjust using 
            RMSE / sqrt(n).  This is specified using the "correlation"
            parameter in both summstat() and equate().  See the "correlation"
            discussion below. The upshot is that aggregating standard errors
            in Damon is still a work in progress.
            
            Item Difficulty
            ---------------
            Damon summary statistics differ from those produced by
            unidimensional psychometric programs in several important
            respects, for instance the concept of "item difficulty".

            In an IRT model such as Rasch, the reported item difficulty is
            the logit value that combines with the person ability logit
            to produce an expected value for a given cell.  In
            Damon, these values are replaced by multidimensional
            coordinate arrays that are hard to interpret by
            humans.  To address the question of item "targeting" -- the
            degree to which item difficulties mirror person abilities,
            displayed using Wright maps of person/item distributions -- a
            more human-friendly item statistic is needed.  Damon offers
            several options:

                1)  Just report the mean estimate of each item (itemdiff = None).
                    This is not a "difficulty" statistic so much as
                    an "easiness" statistic, not useful for Wright maps
                    but fine as a descriptive statistic.

                2)  Specify data = 'est2logit_out' and itemdiff = True.  (You
                    must first have run est2logit().) The mean logit person
                    scores for each item (multiplied by -1) yield
                    a useful targeting statistic to answer the question:
                    How close is the person mean to the difficulty of
                    the item?  These targeting statistics are
                    good for building Wright maps, but they are not item
                    difficulties per se, just targeting statistics.  The
                    implied item difficulties in this case are all zero, by
                    definition.  (Don't ask me to explain!)

                3)  Run item_diff() and refer to item_diff_out.  This reports
                    for each item the person score that corresponds to a 0.50
                    probability of success, which is a true item difficulty.
                    This statistic is also suitable for Wright maps.

            There are other important considerations to bear in mind
            on the subject of item difficulty:

                1)  Unlike with the Rasch model, you cannot combine the
                    person summary statistic and the item summary statistic
                    to compute an expected value for a person on an
                    item.  This can only be done using the Damon coordinates
                    calculated for that person (R[n]) and item (C[i]),
                    where:

                        Estimate[n,i] = dot(R[n],C[i]), or
                        Estimate[n,i] = tools.estimate(fac0coord,fac1coord,nanval)

                2)  All equating, predicting, anchoring, and banking is done
                    using the person and item coordinates, not the
                    item or person summary statistics.

                3)  If the dataset is multidimensional, it is still
                    possible to position persons and items on a single
                    scale (as with "Wright maps"), but one cannot make the
                    same relative inferences.  If Person A is "higher"
                    than Item 1 on the scale (prob of success > 0.50)
                    and Item 1 is higher than Item 2, that does not
                    necessarily mean that Person A is higher than Item 2.

                    Say that Item 1 is a math item and Item 2 is an
                    art item.  Because Person A is good at math, and
                    math is harder than art, it does not thereby follow
                    that Person A is good at art.  This is an inescapable
                    consequence of multidimensionality.

                    However, if Person A is higher than Person B on
                    Item 1, and Person B is higher than Person C on Item 1,
                    then Person A is necessarily higher than Person C for
                    Item1.  The logic is unidimensional because Item 1
                    embodies its own unidimensional scale.

            Other Summary Statistics
            ------------------------
            summstat() is not the only Damon method to report summary
            statistics, and it does not report all summary statistics.
            Aside from its summary statistics, summstat's most important
            feature is the ability to compute a set of summary statistics
            for a range of cells that can be specified in a variety of ways.

            Other Damon methods that report summary statistics:

                Damon.score_mc()    =>  row/column percent correct and
                                        response frequencies, point-biserial
                                        correlations

                Damon.coord()       =>  row/column coordinates, needed
                                        for equating

                Damon.sub_coord()   =>  row/column coordinates for each
                                        subspace.

                Damon.item_diff()   =>  probabilistically defined item
                                        difficulties, i.e., the score that
                                        corresponds to a 0.50 probability
                                        of success for each item.

                Damon.equate()      =>  the Mean, EAR, and SE of each
                                        person on a construct defined
                                        by a group of items, computed using
                                        only the person/item coordinates.
                                        This is a quick light-weight alternative
                                        to summstat() that provides person
                                        measures on the fly.

            How 'getrows' and 'getcols' Work
            --------------------------------
            The summstat() 'getrows' and 'getcols' arguments are used
            to accomplish two things at the same time:

                1.  row_ents.  Calculate statistics for each entity
                    specified in getrows based on the range of column
                    entities specified in getcols.

                2.  col_ents.  Calculate statistics for each entity
                    specified in getcols based on the range of row
                    entities specified in getrows.

            Thus, the getrows and getcols arguments serve double duty.
            They serve as a list of entities for EACH OF WHICH stats should
            be calculated AND they serve as a RANGE across which the
            other facet's entity statistics are calculated.

            When getrows specifies all rows and getcols specifies only
            columns that are associated with 'Language', statistics will
            be reported for each person based only on how he did
            across the 'Language' items.  Performance on 'Math' items
            will be ignored.  summstat() won't report row statistics
            for both 'Math' and 'Language' separately unless you
            run it separately for each construct, like so:

            subs = ['Math', 'Language']
            for sub in subs:
                d.summstat(data = 'base_est_out',
                           getstats = ['Mean', 'SE', 'Count'],
                           getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                           getcols = {'Get':'NoneExcept', 'Labels':1, 'Cols':[sub]},
                           outname = sub
                           )

            This will create (good for exporting):

                my_obj.row_ents_out_Math        => a datadict of row entities for Math
                my_obj.row_ents_out_Language    => a datadict of row entities for Language
                my_obj.col_ents_out_Math        => a datadict with one row of stats for Math
                my_obj.col_ents_out_Language    => a datadict with one row of stats for Language

            It will also create (in convenient Damon object form for interactive study):

                my_obj.summstat_out, which contains:

                    {'Math':{'row_ents':_,      => a Damon object of row entities for Math
                             'col_ents':_,      => a Damon object with one row of Math stats
                             ...},
                     'Language':{'row_ents':_,  => a Damon object of row entities for Language
                                 'col_ents':_,  => a Damon object with one row of stats for Language
                                 ...}
                    }

            To combine the different summstat outputs into one report, follow
            with:

                d.merge_summstat(...)
            
            What you get is a lot of flexibility in defining summary stats
            for a given set of entities across a given range.  If you don't
            want to run summstat() multiple times for multiple subscales, or
            if you need better performance, try using the equate() method.
            equate() is not as flexible but it is faster and is good with
            multiple subscales.


        Arguments
        ---------
            "data" is a string name identifying a datadict from which statistics
            are to be calculated.  It must be an attribute of the current Damon
            object.  The most commonly referred to datadicts are:

                'base_est_out','fin_est_out','est2logit_out'

            One can also refer to a datadict such as 'data_out'.  While it
            cannot be used to calculate comparison statistics like 'Corr' and
            'Resid', it can be used to get means, standard deviations, and
            other descriptive statistics.

                data = 'base_est_out'   =>  Uses the outputs of base_est(),
                                            base_resid(), base_ear(), base_se(),
                                            and base_fit().  Matching input
                                            values are automatically drawn
                                            from the standardize(), parse(),
                                            or earlier outputs, whichever
                                            has been run most recently.

                data = 'fin_est_out'    =>  Uses the outputs of fin_est()
                                            (which automatically includes
                                            finEAR_out and finSE_out), as well
                                            as fin_fit().  For matching input
                                            values, the original data is used.

                data = 'est2logit_out'  =>  Uses the outputs of est2logit(),
                                            which includes logit_ear_out and
                                            logit_se_out.  No matching input
                                            values are defined, so this
                                            data type does not report residuals,
                                            fit, and other statistics that
                                            depend on comparison with input
                                            stats.

                data = 'data_out'       =>  Pulls original data from obj.data_out.
                                            Because it does not include residuals,
                                            errors, or other statistics, only
                                            descriptive statistics such as Mean and
                                            SD can be calculated.

            ---------------
            "getstats" calls for a list of desired summary statistics.  These
            are interpreted differently depending on the data argument
            i.e. whether the statistics are based on the base_est(), fin_est(),
            or est2logit() outputs.

            IMPORTANT
            ---------
            Some of these statistics, when applied to row entities, become
            uninterpretable if data = 'fin_est_out'.  This is due to each
            column possibly having its own metric.  These differences in metric
            confound most row summary statistics, generally returning
            higher numbers than if the columns were all in the same metric.
            This problem does not occur when data = 'base_est_out' or 'est2logit_out',
            where all outputs are in a common metric.  Nor does it occur
            with data = 'fin_est_out' so long as you report only column
            statistics.  Available statistics:

            getstats = ['All']      =>  return_ all summary statistics.

            getstats = [Pick any from the following and insert in list]:

                'Mean'      =>  Mean estimated cell value for the entity.
                                If itemdiff is True, the column means are
                                multiplied by -1 to provide an estimate
                                of item difficulty.

                'SE'        =>  The standard error of the mean, equal to
                                the root mean squared standard error
                                calculated across the cells in the range.
                                Standard errors are like expected absolute
                                residuals (see RMSEAR below), except that
                                they get smaller as the number of observations
                                increases.  When they are very small, and the
                                model is correctly specified, they imply
                                that the Damon estimates are nearly "true",
                                i.e., would match what you would get if you
                                (somehow) independently administered the same
                                items to the same person a lot of times and
                                averaged the results.  Standard errors are
                                appropriate for conducting significance tests
                                and computing confidence intervals.
                                
                                Note that when aggregating standard errors
                                it is very important to specify whether all
                                the columns (and rows) are positively
                                correlated.  If they are not, the RMSR is
                                divided by sqrt(n).  See the "correlated"
                                parameter explanation.

                'SD'        =>  Standard Deviation of estimated cell values.
                                'SD' may be uninterpretable for row entities
                                if data = 'fin_est_out', as it may be confounded
                                by the differences in column metrics.

                'Corr'      =>  Correlation between observations and estimates.
                                If data = 'fin_est_out', the row entity
                                correlations are probably meaningless,
                                artificially inflated by the differences
                                in column metrics.
                
                'PtBis'     =>  Correlation between a given column entity and
                                the sum across column entities (excluding the
                                target entity).  It is only calculated for
                                column entities and uses only cell observations,
                                not estimates.  It measures the degree to which
                                a column entity is correlated with the test as
                                a whole and is traditionally used to assess
                                unidimensionality.

                'Resid'     =>  Summary of the residuals output by the
                                base_resid() or fin_resid() methods.  Depending
                                on how base_resid() or fin_resid() were
                                specified, each column may have
                                its own type of residual (e.g, 'Diff',
                                'Match', 'Nearest' -- appropriate docs),
                                aggregated in a way unique to that type.
                                They may also be "pseudo-missing" residuals,
                                based on cells that were made missing for
                                purposes of calculation.

                'RMSEAR'    =>  Root Mean Squared Expected Absolute Residual
                                (expected discrepancy between observations
                                and estimates).  It is like an average
                                residual, except that summarizes expected
                                residuals rather than raw residuals.
                                Beacause the EAR is calculated for all cells,
                                the RMSEAR has the nice property of not being
                                biased by the effects of missing values.

                'Sep'       =>  Statistical separation across estimates:
                                Spread / MeanError.  This is a ratio that
                                goes from 0 to +infinity and is interpreted
                                like a t-statistic.  A separation greater than
                                3.0 suggests that the estimates in a given
                                range are significantly different from each
                                other on the whole.  A separation of 1.0 or
                                below indicates that the entites cannot be
                                statistically distinguished from each other.
                                They are for all practical purposes more or
                                less the same.

                'Rel'       =>  reliability (Cronbach-alpha):  Separation
                                converted to a 0.0 - 1.0 metric.  In testing,
                                a reliability > 0.85 is often recommended
                                in order for a test to be usable.

                'Outfit'    =>  Mean square fit statistic across the defined
                                range, where:

                                    Cell fit = (observed - Estimate) / EAR

                                This is similar to the Rasch mean square fit
                                statistic known as "OutFit", but check the
                                distribution of fits before making a decision
                                as to what constitutes "misfit"; it may be
                                different than Rasch.

                'Fit_Perc>2'=>  Percent (actually proportion) of cells
                                whose fit is greater than 2.0 or less than
                                -2.0, where:

                                    Cell fit = (observed - Estimate) / EAR

                                By chance, we expect approximately 0.05 of
                                cells to misfit when the misfit threshold
                                is 2.  When a range of cells has a higher
                                proportion of misfits than 0.05, it signals
                                that the corresponding entit(ies) are
                                misfitting.  This means the entites
                                respond to dimensions outside the prescribed
                                dimensionality and probably don't
                                belong in the analysis, at least when items
                                are being calibrated for future use.

                                This is a different fit statistic than
                                is usually reported for Rasch fit statistics.
                                It's advantage is that the 0.05 threshold
                                has pretty much the same meaning for large and
                                small datasets and for large and small amounts
                                of random error, though it falls apart as
                                the error goes to extremes of zero or becomes
                                very large relative to the metric of the
                                observations.

                'Count'     =>  count of valid observations per entity or
                                attribute in the original dataset.

                'Min'       =>  Minimum cell estimate

                '25Perc'    =>  Cell estimate at the 25th percentile

                'Median'    =>  median_ cell estimate

                '75Perc'    =>  Cell estimate at the 75th percentile

                'Max'       =>  Maximum cell estimate

                'Coord'     =>  The Damon coordinates associated with
                                the entity.  Each cell estimate
                                is the dot product of its row and column
                                coordinates.  Coordinates are only
                                available for row and column entities,
                                not for attributes.  Also, they are
                                only reported if data = 'base_est_out'
                                and coord() is run.  sub_coord() subspace
                                coordinates are not reported.

                        The coordinate statistics describe where
                        each person and item is located in the
                        multidimensional space specified by the model.
                        They are generally uninterpretable by humans
                        because the origin of the space and the
                        orientation of the axes is arbitrary.  However,
                        they are Damon's most important statistic.
                        Aside from generating cell estimates, they are the
                        mechanism by which one dataset is anchored to
                        another.

                        When the coord() method specifies that one of the
                        facets is constrained to be orthonormal, an additional
                        property emerges.  The coordinates of the opposing
                        coordinates (generally items) can be used to compute
                        the geometric cosine between each pair of item
                        vectors, e.g.:

                            Cos[I,J] = Ix*Jx + Iy*Jy + Iz*Jz / ||I|| * ||J||

                        where x, y, and z are coordinates and ||I|| and ||J||
                        are the vector lengths (the root sum of squares of
                        the coordinates) of I and J.

                        This cosine is similar to the correlation between two
                        items, but it is better because:  a) it is impervious to
                        missing data, b) it is sample-free (objective), i.e., the
                        same for every subsample of person data, and c) it is
                        convenient (it is easily calculated directly
                        from coordinates).

                        Important Note
                        --------------
                        The 'Coord' statistic for column entities will be
                        a nanval if the coordinates were calculated using
                        parsed data but the final estimates were
                        destandardized or deparsed.  That is because in
                        this case coordinates do not exist for the original
                        column entities, only for their parsed components.

                        If you need the parsed coordinates, refer to
                        base_est_out['fac1coord']

                        In this situation, summstat() will report the
                        row entity coordinates as usual.

                summstat() automatically outputs important summary
                statistics computed by the coord() method, accessed
                as keys in the summstat_out datadict:

                    summstat_out['objperdim']
                            =>  The objectivity, stability, and accuracy
                                for each dimensionality, available only
                                if a range of dimensionalities were
                                analyzed, same as Damon.objperdim

                    summstat_out['objectivity']
                            =>  The overall objectivity of the current
                                analysis at the "official" dimensionality,
                                same as Damon.objectivity

                    summstat_out['stability']
                            =>  The stability and other statistics computed
                                for each attempted seed, same as Damon.seed

            ---------------
            "getrows" specifies which row entities or attributes should
            have summary statistics computed for them.  The syntax is
            similar to the extract() method.  getrows is a dictionary with
            three fields:  'Get', 'Labels', and 'Rows':

                getrows = {'Get':<'AllExcept','NoneExcept'>,
                           'Labels':<'key',int,'index'>,
                           'Rows':<KeyIDs,Attributes,Indices>
                           }

            All indices refer to the WHOLE array, so make sure to
            take into account rowlabels and collabels when specifying a
            row or column integer.

            Note that both:

                {'Get':'AllExcept','Labels':'keys','Rows':[None]} and
                {'Get':'NoneExcept','Labels':'keys','Rows':['All']}

            return ALL row entities.

            Note also that the Labels options 'key' and 'index' are
            special words for extract().  If they appear as keys in
            your labels, they will not be used in their special sense
            but as ordinary labels.

            getrows =
                'SummWhole'             =>  report one set of summary statistics
                                            for the whole row range across
                                            the columns specified in getcols.
                                            Do not report individual entities
                                            or attributes.

                {'Get':'AllExcept'      =>  Extract ALL rows except the ones
                                            specified in 'Rows':[...] .

                 'Get':'NoneExcept'     =>  Extract NONE of the rows except
                                            those specified in 'Rows':[...] .

                 'Labels':'key'         =>  The values listed in in 'Rows':[...]
                                            refer to rowlabel keys (unique row
                                            identifiers).

                 'Labels':<int>         =>  An integer signifying the column that
                                            contains the values listed in 'Rows':[...] .
                                            Counting starts at 0 from the left of
                                            the whole array. This makes it possible
                                            to extract on the basis of row
                                            attributes like gender or age, or on
                                            values in the core data.  More
                                            than one attribute can be specified.

                'Labels':<str>          =>  A string key label signifying the column that
                                            contains the values listed in 'Rows':[...] .
                                            This is an alternative key-based way to
                                            specify the column containing row attributes.
                                            However, it won't work if these keys are
                                            not string.  If the keys are 'key' or 'index',
                                            extract() will interpret them as labels,
                                            not in their special sense.

                 'Labels':'index'       =>  This means that the values in
                                            'Rows':[...] are not keys or labels but
                                            an integer index of rows to select (or
                                            exclude), where the counting starts
                                            with the first row (in the whole array,
                                            including labels) equal to 0.

                 'Rows':[None]          =>  If 'Get' is 'AllExcept', [None] means that
                                            all rows will be used.  If 'Get' is
                                            'NoneExcept', [None] means that no rows
                                            will be used (whatever that means).

                 'Rows':['All']         =>  Opposite of 'Rows':[None] above.

                 'Rows':['ID1','ID2',...]
                                        =>  This is the list of row entities to be
                                            extracted based on their ID.  It assumes
                                            'Labels':'key'.

                 'Rows':['Cats','Dogs',...]
                                        =>  This is the list of row attributes on
                                            the basis of which rows are to be selected
                                            (or excluded).  This says to select or
                                            exclude 'Cats' and 'Dogs'.  It assumes, e.g.,
                                            'Labels':'Species' or 'Labels':2 .

                 'Rows':[0,1,4,7]       =>  This is the index giving row numbers to
                                            be extracted.  It assumes 'Labels':'index' .
                }


            --------------
            "getcols" is the same as getrows except that it specifies columns.

            Note that both:

                {'Get':'AllExcept','Labels':'keys','Cols':[None]} and
                {'Get':'NoneExcept','Labels':'keys','Cols':['All']}

            return ALL column entities.

            getcols =
                'SummWhole'             =>  report one set of summary statistics
                                            for the whole column range across
                                            the rows specified in getcols.
                                            Do not report individual entities
                                            or attributes.

                {'Get':'AllExcept'      =>  Extract ALL columns (cols) except those
                                            specified in 'Cols':[...] .

                 'Get':'NoneExcept'     =>  Extract NONE of the cols except
                                            those specified in 'Cols':[...] .

                 'Labels':'key'         =>  The values listed in 'Cols':[...]
                                            refer to collabel keys (unique column
                                            identifiers).

                 'Labels':<int>         =>  An integer signifying the row that
                                            contains the values listed in 'Cols':[...] .
                                            Counting starts at 0 from the top of
                                            the whole array. This makes it possible
                                            to extract on the basis of column
                                            attributes like 'Spelling' or 'Vocab', or on
                                            values in the core data.  More
                                            than one attribute can be specified.

                'Labels':<str>          =>  A string key label signifying the row that
                                            contains the values listed in 'Cols':[...] .
                                            This is an alternative key-based way to
                                            specify the row containing column attributes.
                                            However, it won't work if these keys are
                                            not string.  If the keys are 'key' or 'index',
                                            extract() will interpret them as labels,
                                            not in their special sense.

                 'Labels':'index'       =>  This means that the values in
                                            'Cols':[...] are not keys or labels but
                                            an integer index of cols to select (or
                                            exclude), where the counting starts
                                            with the first col equal to 0.

                 'Cols':[None]          =>  If 'Get' is 'AllExcept', [None] means that
                                            all cols will be used.  If 'Get' is
                                            'NoneExcept', [None] means that no columns
                                            will be used (whatever that means).

                 'Cols':['All']         =>  Opposite of 'Cols':[None] above.

                 'Cols':['ID1','ID2',...]
                                        =>  This is the list of column entities to be
                                            extracted based on their ID.

                 'Cols':['Spelling','Vocab',...]
                                        =>  This is the list of col attributes on
                                            the basis of which cols are to be selected
                                            (or excluded).  This says to select or
                                            exclude 'Spelling' and 'Vocab' items.

                 'Cols':[0,1,4,7]       =>  This is the index giving col numbers to
                                            be extracted.  It assumes 'Labels':'index'.
                  }


            Important Note Regarding Column Keys
            ------------------------------------
            Damon suffers an awkward limitation when it comes to identifying
            and calling column key IDs if the parse() function has been run.
            Say the original column keys are integers (1,2,3...).  parse() is
            run on column 2 and new IDs are assigned ('1','2_a','2_b','3'...).
            Notice that integers 1 and 3 have become strings '1' and '3'.

            coord() and base_est() are run, and now we run fin_est().
            Columns '2_a' and '2_b' may or may not be rejoined.  If they
            are, the column key becomes '2'.  Otherwise, they stay '2_a'
            and '2_b'.  The awkward part is that because the column keys
            COULD be string, fin_est() forces them all to become
            string.  The integers 1,2,3 become '1','2','3'.  That means
            the column keys in the final Damon outputs may not match those
            of the inputs; they will be strings instead of ints.  Plus, there
            may be some extra keys ('2_a' and '2_b') and some missing ('2').
            These mismatches can cause problems, such as when doing lookups.

            To get around this issue, you can:  a) not use integers as
            column IDs, b) not run parse().

            What this means for summstat() is that when you specify
            column keys, make sure they are readable from the relevant
            estimates data dictionary (base_est_out or fin_est_out).

            --------------
            "itemdiff" <None,True>, if True means reverse the sign of the
            column mean estimate so that a higher mean implies higher item
            difficulty.  This option makes sense only when the estimates
            metric is centered on zero, as when estimates are in one of the
            standardized metrics.  It makes most sense when you specify
            data = 'est2logit' and run est2logit() prior to summstat().  A
            warning will print out otherwise.

            The resulting difficulties are an alternative to those computed
            using the item_diff() method, and both can be used to build
            Wright maps -- histograms that reveal how well the item distribution
            is targeted on the person distribution.  Use them under the
            following conditions:

                summstat() item difficulties
                                        =>  estimates will be reported in
                                            logits where for each item
                                            a logit value of 0.0 means a
                                            probability of success of 0.50.
                                            The item means indicate targeting
                                            of the items on the persons.

                item_diff() item difficulties
                                        =>  estimates will be reported in
                                            a non-logit metric.  item_diff()
                                            relates these scores/estimates
                                            to the probability of success
                                            and reports as the item difficulty
                                            the score corresponding to a
                                            0.50 probability of "success",
                                            as you define it.

            Note that "item diff" does not refer in this context to what
            is called in psychometrics "item differential functioning" -- the
            differential behavior of items for different subsets of persons.
            In the context of Damon, item differential functioning is the
            degree to which different types of examinees cannot be modeled
            by the same set of item coordinates -- the degree to which the
            model fails, in other words.  It can be explored by computing the
            stability statistic using different subsets of items and persons.
            See the seed parameter in coord().

            --------------
            "outname" is a suffix to append to the summstat_out attribute
            name if one is desired.  This makes it possible to run summstat
            multiple times in one Damon object (for different item clusters,
            for instance) without overwriting the summstat_out attribute.  Just
            specify a different outname for each run.

                outname = None          =>  No suffix will be appended.  Outputs
                                            will be assigned to the Damon object
                                            under the names:  summstat_out,
                                            row_ents_out, col_ents_out.

                outname = 'Math'         =>  The 'Math' suffix will be appended.
                                            Outputs will be assigned to the Damon
                                            object under the names:  summstat_out_Math,
                                            row_ents_out_Math, col_ents_out_Math.

            --------------
            "labels" specifies how to label the summarized row and column
            entities.  If None, a descriptor phrase fills the label cell, like:

                "base_est_out col ents where rows are: NoneExcept [3, 4, 5]"

            The descriptor phrase is helpful to keep track of what
            numbers are being summarized and how, but it is unwieldy for
            reporting.

                labels = None           =>  Both row and column summarized
                                            entities will be labeled with the
                                            descriptor phrase.
                
                labels = {'row_ents':None, 'col_ents':None}
                                        =>  Same as above

                labels = {'row_ents':'Person', 'col_ents':'Item'}
                                        =>  The entities are individual persons
                                            in the row_ents report and individual
                                            items in the col_ents report.

                labels = {'row_ents':'Gender', 'col_ents':'Domain'}
                                        =>  The entities are gender groups
                                            in the row_ents report and domain
                                            groups in the col_ents report.

            --------------
            "group_se" is used to generate part of the formula used to
            aggregate standard errors across cells.  The standard error
            of the mean across, say, a group of items is the root mean
            square error across the cells multiplied by some factor.
            "group_se" calculates that factor.  If the cell error for
            each item (for a given person) were statistically independent
            across items, the factor would be 1/sqrt(n_items).  However,
            this is not the case with Damon cell errors.  The appropriate
            factor is generally some version of 1/sqrt(2 * ndim), tweaked
            to account for a variety of conditions.
            
            The default is:
                 group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75']
            
            which works fairly well in reproducing the correct standard
            error across a variety of datasets, especially those associated
            with educational test data.  However, simulation experiments
            may show that you need a somewhat different formula, and you
            can specify that through this option.
            
            For a complete discussion, see the docs for tools.group_se()

            --------------
            "correlated" <None> has been deprecated and will be
            ignored internally. It is up to the user to make sure that 
            all items that go into a construct are positively correlated.
            
        Examples
        --------
        >>>  my_dmnobj.summstat(getstats = ['All'],
                                getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]},
                                outname = None,
                                labels = None
                                )
                =>  Get all summary statistics for all row entities (persons)
                    based on their performance across all column entities (items).
                    Get all summary statistics for all column entities (items)
                    based on their performance across all row entities (persons).

        >>>  my_dmnobj.summstat(getstats = ['All'],
                                getrows = 'SummWhole',
                                getcols = 'SummWhole',
                                outname = None,
                                labels = None
                                )
                =>  Get one set of all summary statistics for the data array as
                    a whole.

        >>>  my_dmnobj.summstat(getstats = ['Mean','SE','Outfit'],
                                getrows = {'Get':'AllExcept','Labels':'key','Rows':['Marc','Antony']},
                                getcols = {'Get':'NoneExcept','Labels':2,'Cols':['Math','Language']},
                                outname = None,
                                labels = None
                                )
                =>  Get the mean, standard error, and Outfit for all
                    row entities EXCEPT for 'Marc' and 'Antony' based on how
                    these other entities performed on only the 'Math' and
                    'Language' items, which are identified as such in row 2
                    of collabels.  Also get these summary statistics for
                    'Math' and 'Language' based on all persons except 'Marc'
                    and 'Antony'.

        >>>  my_dmnobj.summstat(getstats = ['Mean','SD'],
                                getrows = {'Get':'AllExcept','Labels':'index','Rows':[1,3,5]},
                                getcols = 'SummWhole',
                                outname = None,
                                labels = None
                                )
                =>  Get the mean and standard deviation for all row entities except rows 1, 3,
                    and 5 (counting from the top, including labels).  Base these
                    on performance across all items.  Get the mean and sd for the
                    column entities as a whole based on performance of all row
                    entities except those in rows 1, 3, and 5.

        >>>  for domain in ['Reading', 'Math']:
                 my_dmnobj.summstat(getstats = ['Mean','SE'],
                                    getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                    getcols = {'Get':'NoneExcept','Labels':2,'Cols':[domain]},
                                    outname = domain,
                                    labels = None
                                    )
                =>  Get the mean and standard error for each person for the 'Reading'
                    and 'Math' domains separately.  These will be stored under different
                    keynames in summstat_out.  Note that "domain" appears in both the
                    getcols and outname arguments.


        Paste method
        ------------
            summstat(data = 'base_est_out', # [<'base_est_out','fin_est_out','est2logit_out','data_out',...> => data for calculating stats]
                     getstats = ['Mean', 'SD', 'SE', 'Count', 'Outfit', 'Fit_Perc>2', 'PtBis', 'Rel'], # [Select stats from [<'Mean','SE','SD','Corr','PtBis','Resid','RMSEAR','Sep','Rel','Outfit','Fit_Perc>2','Count','Min','25Perc','Median','75Perc','Max','Coord'>] ]
                     getrows = None, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}>]
                     getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]}, # [<None,'SummWhole',{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}>]
                     itemdiff = None,   # [<None,True> => express item (column) mean as a probabilistically defined item "difficulty" ]
                     outname = None,    # [<None,'cluster_name'> => name to append to summstat outputs for multiple analyses]
                     labels = None,   # [<None, {'row_ents':<None, 'person',...>, 'col_ents':<None, 'item',...>}> => to describe summarized entities]
                     group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75'],  # [<see tools.group_se docs> => factor for grouping se's]
                     correlated = None  # [<None> => deprecated]
                     )

        """
        if self.verbose is True:
            print 'summstat() is working...\n'

        # Run the damon utility
        summstat_out = dmn.utils._summstat(locals())
        if outname is not None:
            try:
                self.summstat_out[outname] = summstat_out
            except AttributeError:
                self.__dict__['summstat_out'] = {}
                self.summstat_out[outname] = summstat_out

            # Capture outnames
            try:
                self.outnames.append(outname)
            except AttributeError:
                self.outnames = [outname]
                
        else:
            self.__dict__['summstat_out'] = summstat_out

        if self.verbose is True:
            print 'summstat() is done -- see my_obj.summstat_out.  Also, my_obj.row_ents_out and my_obj.col_ents_out.'
            print 'Contains:\n',summstat_out.keys(),'\n'

        return None



    ##########################################################################

    def merge_summstat(self,
                       merge_by = 'row',   # [<'row', 'row2col'>]
                       ):
        """Merge the results of multiple summstat() runs

        Returns
        -------
            merge_summstat() returns None but assigns to
            my_obj.merge_summstat_out a Damon object that merges
            the results of multiple summstat() runs into a single
            table.  It also may assign the Damon object to either
            my_obj.row_ents_out or my_obj.col_ents_out, depending
            on the type of entity for which stats are reported,
            unless merge_by = 'row2col'.

            Alternatively, merge_summstat() can return a merged
            row_ents and col_ents Damon object from one summstat()
            run.

            Workflow 1 (multiple subscales):

            d = dmn.Damon(...)
            d.coord(...)
            d.base_est(...)
            d.base_resid(...)
            d.base_ear(...)
            d.base_se(...)
            domains = ['Reading', 'Math', 'Science']          
            for domain in domains:
                d.summstat(..., outname=domain)
            d.merge_summstat(merge_by='row')


            Workflow 2 (merge row and col ents):

            d = dmn.Damon(...)
            d.coord(...)
            d.base_est(...)
            d.base_resid(...)
            d.base_ear(...)
            d.base_se(...)
            d.summstat(...)
            d.merge_summstat(merge_by='row2col')


            Workflow 3 (rasch):

            d = dmn.Damon(...)
            d.rasch(...)
            d.merge_summstat(merge_by='row2col')
                (No need to run summstat() first)

            merged = d.row_ents_out
              or
            merged = d.merged_summstat_out

        Comments
        --------
            The regular summstat() method summarizes data over one
            range of cells at a time, yielding one set of person
            statistics.  If you want multiple sets of person
            statistics, stats for the Reading, Math, and Science
            subscales for example, you run summstat multiple times,
            once for each subscale, yielding separate outputs for
            each subscale.  summstat() makes it easy to
            do this.

            However, if you want to compare performance on different
            subscales, or you want to graph subscale measures against
            each other using Damon.plot_two_vars(), you need to
            merge them together.  That's what merge_summstat() does.

            Thus, merge_summstat() is used when you want to get the
            output statistics for each domain into a single table.
            The resulting table would consist of, say, a list of
            persons as rows and a separate panel of statistics for
            each domain, labeled something like:

            'Persons', 'Mean_Reading', 'SE_Reading', 'Mean_Math', 'SE_Math', ...

            As mentioned, the workflow involves applying summstat()
            to a list of specific cell ranges, such as labeled items
            belonging to each of a set of domains.  Notice how "domain"
            appears in the summstat() function.  In particular,
            notice that the "outname" parameter specifies a domain.  The
            "outnames" parameter in merge_summstat() is equivalent to
            the list of domains in this case.

                domains = ['Reading', 'Math', 'Science']
                for domain in domains:
                    d.summstat(getstats = ['Mean', 'SE'],
                               getrows = {'Get':'AllExcept', 'Labels':'key', 'Rows':[None]},
                               getcols = {'Get':'NoneExcept', 'Labels':1, 'Cols':[domain]},
                               outname = domain,
                               labels = {'row_ents':'Person', 'col_ents':'Domain'})
            
            Now we run merge_summstat():

                d.merge_summstat()

            merge_summstat() looks up the results for each domain
            and merges them with the others.  In the merged table,
            domains appear in the order entered in the "domains" variable.

            Merging Row and Column Entities
            -------------------------------
            merge_summstat() can also be used to do a different type
            of merge.  Every time summstat() is run, it produces outputs
            for both row entities ("persons") and column entities ("items"),
            as separate output tables.  This is necessary because in general
            their statistics are not comparable and they may have different
            labels.  However, in some cases it is desirable to get them
            into one table.  my_obj.merge_summstat(merge_by = 'row2col) does
            that.  It strips out the different row labels and stacks the
            row entities table on top of the column entities table.  It also
            adds a column to identify the type of entity.  This table is
            necessary for the Damon.wright_map() method.

            Usually, the row and column entity stats (in particular the
            'Mean' statistic) are in different metrics and not comparable.
            So interpret the outputs accordingly.  However, in two cases
            they ARE comparable and can be graphed together:

                *   rasch() person and item stats
                *   person and item stats after running:
                        my_obj.est2logit()
                        my_obj.item_diff()

            Merging rasch() Outputs
            -----------------------
            As mentioned, merge_summstat() will automatically merge the
            row entity and column entity rasch() output arrays.  You don't
            need to run summstat() first (see workflow 3).


        Arguments
        ---------
            "merge_by" specifies the type of merge:

                merge_by = 'row'        =>  This assumes you have run summstat()
                                            multiple times for different groups
                                            of either row entities or column
                                            entities (merge_summstat automatically
                                            figures out which).  The results
                                            are merged by row with multiple panels
                                            of statistics.

                merge_by = 'row2col'    =>  This assumes you have run summstat()
                                            once (or not at all, in the case of
                                            rasch()).  The row entity stats are
                                            stacked on top of the column entity
                                            stats.

        Examples
        --------
            [build Damon object "d", run coord(), base_est(), etc.]
            
            outnames = ['Math', 'Language']
            for outname in outnames:
                d.summstat(getstats = ['Mean', 'SE'],
                           getrows = {'Get':'AllExcept', 'Labels':'key', 'Rows':[None]},
                           getcols = {'Get':'NoneExcept', 'Labels':1, 'Cols':[outname]},
                           outname = outname)

            d.merge_summstat()

            =>  Get the Mean and SE for all row entities, first for all the items
                labeled 'Math' (in the 1'th collabels row), second for all the items
                labeled 'Language'.  Then merge them into a single summary statistics
                table for both row entities and column entities.

            **********
            d.summstat()
            d.merge_summstat(merge_by='row2col')

            =>  Merge row entity and column entity stats


        Paste method
        ------------
            merge_summstat(merge_by = 'row',   # [<'row', 'row2col'>]
                           )
        """

        if self.verbose is True:
            print 'merge_summstat() is working...\n'

        # Run the damon utility
        merge_summstat_out = dmn.utils._merge_summstat(locals())
        self.merge_summstat_out = merge_summstat_out

        if self.verbose is True:
            print 'merge_summstat() is done -- see my_obj.merge_summstat_out'

        return None



    ##########################################################################

    def plot_two_vars(self,
                      xy_data, # ['my_datadict_out', e.g., 'merge_summstat_out']
                      x_name,    # [name of x variable to use for x-axis]
                      y_name,    # [name of y variable to use for y-axis]
                      ent_axis = 'col',    # [<'row', 'col'>] => how variables are situated]
                      err_data = None,  # [<None, 'my_datadict_out'> => e.g., 'merge_summstat_out']
                      x_err = None,    # [<None, size, name of x variable to use for error statistic> => create bubbles]
                      y_err = None,    # [<None, size, name of y variable to use for error statistic> => create bubbles]
                      color_by = None,  # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                      cosine_correct = 'coord|corr',    # [<None, 'coord', 'corr', 'coord|corr'> => correct for cosine between variables]
                      max_cos = 0.95,    # <unsigned corr> => trigger exception when cos(x, y) > max_cos]
                      plot = None,  # [<None, {plotting parameters}> => see docs to customize]
                      savefig = 'xy_plot.png'  # [<None, 'show', 'filename.png'> => how to output chart]
                      ):
        """Plot two variables against each other, correcting for correlation.

        Returns
        -------
            plot_two_vars() shows or saves a file as a bubble chart
            and returns the plotting information needed to make the
            chart.  It also returns a figure object.  Output:

            {'figure':_     =>  figure available for further editing
            'A_ids':_,      =>  array of point A identifiers
            'x':_,          =>  x-coordinates of each point A
            'y':_,          =>  y-coordinates of each point A
            'B_ids':_,      =>  small array of point B identifiers
            'Bx':_,         =>  x-coordinates of each point B
            'By':_,         =>  y-coordinates of each point B
            'a_err':_,      =>  array of errors for A on line a (=x)
            'b_err':_,      =>  array of errors for A on line b
            'x_name':_,     =>  label of line a (which is assigned to x)
            'y_name':_,     =>  label of line b (which provides the y orthogonal component)
            'theta':_,      =>  angle in radians between lines a, b
            'colors':_,     =>  array of colors for each point A
            'plot_params':_ =>  dictionary of plot parameters
            }

            A possible workflow:
                d = dmn.Damon(...)
                d.standardize(...)  # Often needed to get good plots
                d.coord(...)
                d.base_est(...)
                d.base_resid(...)
                d.base_ear(...)
                d.base_se(...)

                domains = ['Math', 'Language']
                for domain in domains:
                    d.summstat(...)

                d.merge_summstat(...)
                d.plot_two_vars(...)
                
        Comments
        --------
            plot_two_vars() is a way to plot variables against each
            other.  This can be done the usual way, which is to treat
            the two variables (a and b) as orthogonal axes (though they
            are not) and just plot them.  Or it can be done the Damon way,
            which is to represent a and b in their correct angular
            relationship and to represent the points (persons, say) in
            their proper locations in a defined 2-dimensional slice of
            space.  This is called "cosine correction".

            If the goal is to represent persons in space, why not just
            graph the persons using their spatial coordinates?  This
            is perfectly legitimate since the row (facet 0) coordinates
            are orthonormal by default (the column coordinates are not).
            You can even run this type of plot easily in plot_two_vars().
            The only difficulty is that the orientation of the coordinate
            axes in space is totally arbitrary.  So you'll see where the
            persons are relative to each other, great for analyzing clusters,
            but not relative to some human-understandable dimension of
            interest.

            plot_two_vars() gives you a way to get the best of two
            worlds:  you get to see how persons are really floating in
            space, but you also get to see where they fall on two
            human-defined dimensions of interest.  The space, in this case,
            is the two dimensional plane defined by the intersection
            of lines a and b.

            To pull off this trick, one of the two variables (b) has to be 
            allowed to float non-orthogonally in the plane. Plus, interpretation
            of the y-axis is a little tricky. y is interpreted as "that
            component of line b that is orthogonal to line a".  So it captures
            whatever b represents that a does not.

            plot_two_vars() is usually run after merge_summstat() and makes
            most sense when you have person measures on two subscales, along
            with their errors.  But you are not restricted to merge_summstat()
            outputs.  You can pull the two variables from anywhere.

            Standardizing
            -------------
            If you end up with charts where the points are elongated and don't
            seem to be randomly spread through space, this is generally because
            the metrics of the a and b variables are different; they imply different
            ranges.  This can be addressed by running my_obj.standardize() first.

            Visual Presentation
            -------------------
            plot_two_vars() leverages the matplotlib plotting library to
            give you extraordinary control over the appearance of your
            charts.  Fully implemented, each chart has the following features:

                *   An x-axis representing a, the primary variable of interest
                
                *   A y-axis representing the component of b orthogonal to line a
                
                *   A line b floating in the plot at an angle, with tick marks
                
                *   Each entity is a labeled point floating in space

                *   The projection of each point onto the x-axis gives its
                    measure on variable a.  Its othorgonal projection onto
                    line b gives its measure on variable b.
                
                *   Each entity is surrounded by a color-shaded, transparent
                    ellipse representing the standard error (or some other measure
                    of dispersion) of that entity's measure on both the x and y
                    axes.  The edges of each ellipse, when projected onto line b,
                    give the standard error for variable b.

                *   The color (or gray shading) of each ellipse can be customized
                    for each entity.  For instance, you can shade boys as blue
                    and girls as pink.  The colors can be random to aid visual
                    discernment.  They can also vary by intensity to reflect variation
                    on some third dimension such as age.

                *   Each entity point is represented as white against a color
                    background rather than black.

                *   Each chart is highly customizable.  You can add labels or not,
                    limit their length, add ellipses or not, and so on.

            Control of Colors
            -----------------
            The coloring of ellipses is handled using matplotlib conventions
            (http://matplotlib.org/api/colors_api.html) in the "color_by"
            parameter:
            
                * 'b': blue
                * 'g': green
                * 'r': red
                * 'c': cyan
                * 'm': magenta
                * 'y': yellow
                * 'k': black
                * 'w': white

            Gray shades are given as a string encoding a float in the
            0-1 range:

                color_by = '0.75'

            Or you can pass an R, G, B tuple or array, where each is in the
            0-1 range:

                color_by = (0.15, 0.50, 0.75)   =>  0.15 percent red, 0.50
                                                    percent green, 0.75 percent
                                                    blue.

            To get random color assignment, use:

                color_by = 'rand'


            Cosine Correction Formula
            -------------------------
            To calculate where A resides in an objective x, y
            coordinate space given that a (first variable) is
            defined as A|x, where A projects on the x-axis, and b
            (second variable) is defined as A|B where A projects on
            some vector B at angle theta from the x-axis, and
            unknown A|y is where A projects on the y-axis.  To calculate
            A|y:

            A|y = ((b - a*cos(theta)) / cos(theta)) * tan(alpha)

            Proof
            -----
            Given point A and vector B in coordinate space [x,y] with
            origin O, let observed a be OA, to where A falls on x, and
            observed b be where A falls on OB, and assume that angle
            theta between Ox and OB is known.  Find A|y where A falls on y.

            Extend the perpendicular A|x to point D on line B and
            the perpendicular A|B to point C on axis x.  Let a + c = OC
            and b + d = OD.

            cos(theta) = a / (b + d) = b / (a + c)
            tan(90 degrees - theta) = tan(alpha) = A|y / c

            Solve for c:
            
            cos(theta) * (a + c) = b
            a*cos(theta) + c*cos(theta) = b
            c*cos(theta) = b - a*cos(theta)
            c = (b - a*cos(theta)) / cos(theta)

            Solve for A|y:

            A|y = c*tan(alpha)
            A|y = ((b - a*cos(theta)) / cos(theta)) * tan(alpha)

            The solution is undefined when vector b is aligned with
            either of the axes, resulting in one of two error messages
            depending on which axis vector b is aligned to.  If b
            is aligned to y, there is no need for cosine correction
            as a and b are already orthogonal.  If b is aligned to x,
            the denominator cos(theta) goes to zero and A|y blows
            up.  In this case, a 2-dimensional mapping is not possible
            and another vector b must be chosen.

            To estimate cos(theta), one of two methods is used:

                *   Correlation Method: Compute the product-moment
                    (Pearson) correlation between variables a and b.
                    The result approximates cos(theta) to the degree
                    that the distribution of points A in space is
                    multivariate normal. This is often a reasonable
                    assumption, but it cannot be taken for granted.
                    Cherry-picking the sample, or drawing a non-random
                    sample, can result in very different correlations
                    and incorrect cos(theta)'s.

                *   Coordinates Method: Compute the cosine between two
                    column vectors (or between two means of column vectors)
                    as follows:

                    cos(theta) = ax*bx + ay*by + az*bz + ... / (dist(a) * dist(b))

                    where a and b are column vectors, x, y, z, etc. are their
                    coordinates on the x, y, z dimensions, and dist(a) and dist(b)
                    are the Euclidean distance of the a and b vectors, equal to
                    the root sum of squares of each.

            The coordinates method is only valid when the vectors of the
            opposing facet are orthonormal.  In Damon's coord() method,
            the row (facet 0) coordinates are set to orthonormal by default,
            so the coordinates method can only be applied to vectors that
            correspond to column entities (or combinations thereof), unless
            the default is changed.  The basis of this property is that
            when the row coordinates are orthonormal each dimension sums
            to one.  This has the algebraic result of producing column vectors
            that are sufficient statistics, sufficient to summarize the columns
            and their relationships without reference to the row vectors.
            Therefore, they are sufficient to calculate the cosine between two
            vectors.  One effect of this property is that, once vectors are
            calculated for a population, any subset of that population will
            result in the same vectors and vector relationships.  This means
            that cos(theta) will be the same across samples.  Which is nice.

        Arguments
        ---------
            "xy_data" is a string name referring to a datadict associated
            with the Damon object.

                xy_data = 'merge_summstat_out'
                                        =>  Get data from my_obj.merge_summstat_out,
                                            the output of the merge_summstat()
                                            method.

                xy_data = 'base_est_out'
                                        =>  Get data from my_obj.base_est_out, the
                                            output of base_est() method.

            ------------
            "x_name" is the name of the variable (a) that is used to define
            the x-axis.  It must label one of the columns (or rows) in
            the xy_data datadict.

                x_name = 'Mean_Math'    =>  Name of a column in the merge_summstat_out
                                            datadict.
                                 
            ------------
            "y_name" is the name of the variable (b) a component of which
            is used to define the y-axis.  It must label one of the columns
            (or rows) in the xy_data datadict.

                y_name = 'Mean_Reading' =>  Name of a column in the merge_summstat_out
                                            datadict.
                                            
            ------------
            "ent_axis" tells whether x_name and y_name are to be accessed
            as row entities or as column entities in xy_data.  This will
            generally be 'col', first because the summstat statistics
            are column-based, second because the coordinate method for
            calculating cosines is only valid for column entities given
            existing defaults.  (The correlation method will work for
            both types of entities, however.)

                ent_axis = 'col'   =>  x_name and y_name label columns
                                            in xy_data.

            ------------
            "err_data" is a string name referring to a datadict associated
            with the Damon object that contains standard error statistics
            or other statistics measuring dispersion around a mode.  It is
            okay (and generally is the case) for err_data to be the same as
            xy_data, but it doesn't have to be.
            
                err_data = 'merge_summstat_out'
                                        =>  Get errors from my_obj.merge_summstat_out,
                                            the output of the merge_summstat()
                                            method.
                                            
            ------------
            "x_err" is the name of the variable that contains the errors
            associated with x.  These stats give the width of the error ellipse
            for each data point (or a specified percentage of it).  x_err is
            optional and can be specified manually.

                x_err = None            =>  The x-component of each ellipse will
                                            be set to zero.

                x_err = 1.5             =>  Give all points an x error of 1.5
                                            measurement units (the horizontal radius
                                            of the ellipse).

                x_err = 'SE_Math'       =>  Name of a column containing errors in
                                            the merge_summstat_out datadict.

            ------------
            "y_err" is the name of the variable that contains the errors
            associated with y.  These will be the standard errors associated
            with the "b" variable.  They are multiplied by sine(theta) to
            give the size of the errors on the y-axis, hence the height of
            each ellipse.  When projected onto line b, the edges of the ellipse
            give the error on line be.  y_err is optional and can be specified
            manually.

                y_err = None            =>  The y-component of each ellipse will
                                            be set to zero.

                y_err = 1.5             =>  Give all points an x error of 1.5
                                            measurement units (the horizontal radius
                                            of the ellipse).

                y_err = 'SE_Math'       =>  Name of a column containing errors in
                                            the merge_summstat_out datadict.

            ------------
            "color_by" controls how colors are assigned to the ellipse
            surrounding each point.  See the discussion of colors above.
            A given color can be assigned to all points, or a unique color
            to each point, or identifying colors can be assigned to groups
            of points, or colors can vary in intensity to reflect a descriptive
            variable.  (You can also control color transparency using
            plot['transparency'].)

                color_by = None         =>  All ellipses will be colored white
                                            and be invisible.

                color_by = 'b'          =>  Color all ellipses green.

                color_by = '0.75'       =>  Color all ellipses gray with a
                                            mix of 0.75 black.  Note that
                                            this is a decimal in quotes. The
                                            number 0.75 would not work.

                color_by = (0.5, 0.25, 0.70)
                                        =>  Color all ellipses with a mix of
                                            0.50 red, 0.20 green, 0.70 blue.
                                            
                color_by = 'rand'       =>  Assign a different color to every
                                            ellipse.

                color_by = ['gender', {'male':'r', 'female':'0.50'}]
                                        =>  In xy_data, look up the column
                                            labeled 'gender'. Make each 'male'
                                            ellipse red, each 'female' ellipse
                                            50 percent grey-scale. You can also
                                            specify unique color tuples.

                color_by = ['age', 'g'] =>  In xy_data, look up the column
                                            labeled 'age'.  The higher the
                                            age, the deeper the green.
                                            
            ------------
            "cosine_correct" specifies which method to use when calculating
            cosines.

                cosine_correct = None   =>  Just plot variables x_name and y_name
                                            the usual way, each with its own axis,
                                            treating them as orthogonal.

                cosine_correct = 'coord'=>  Use only the "coordinates method"
                                            of calculating cosines. Throw an
                                            exception if it can't be used.

                cosine_correct = 'corr' =>  Use only the "correlation method"
                                            of calculating cosines.

                cosine_correct = 'coord|corr'
                                        =>  Try to use the coordinate method.
                                            Use the correlation method as a
                                            backup.
                                            
            ------------
            "max_cos" (0.0 < max_cos < 1.0) is the maximum allowable
            positive correlation between variable b and either
            axis (x (= a) or y), necessary because there is a fraction
            that blows up if b gets too close to y.  One of two exceptions
            is triggered with two possible remedies, as described above.

                max_cos = 0.95          =>  Throw an exception if the unsigned
                                            cosine exceeds 0.95.

            ------------
            "plot" is a dictionary of parameters used to control the plot.
            Any variable you specify in plot will overwrite the default
            value for that variable.  The internal plot keywords and their
            default values ('Auto' means use Damon's convention) are as
            follows:

            plot =
            {'title':'Auto', # ['Auto', 'My Title']
            'se_size':0.25,     # [0.25 => figure radius = 0.25 * SE]
            'xlabel':x_name,    # [label x variable (=a)]
            'ylabel':y_name,    # [label y variable (=b)]
            'xy_labels':False,   # [<True, False>]
            'label_offset':0,  # [<0.02,...> diagonal offset in points]
            'xy_nchars':5,  # [max number of characters in label]
            'y_line_tick':'ko', # ['ko' => black bullet ticks on y-line (=b)]
            'y_line':'k:',  # [color of y-line (=b)]
            'y_line_ncuts':4,   # [number of ticks on y-line (=b)]
            'xy_marker':'k.',   # [shape and color of xy marker, color overwritten by 'marker_color']
            'markersize':5,     # [size of marker in points]
            'marker_color':'w', # [color of marker]
            'ellipse_color':'rand',     # [color of ellipse or rectangle]
            'transparency':0.50,    # [<0.73> => proportion of transparency]
            'legend':'best',    # [<'upper right', 'lower left', 'best', ...] => location of legend]
            'xlim':'match_xy',  # [<'match_xy', 'min_max', (-3.0, 4.5)> => limit values of x-axis]
            'ylim':'match_xy',  # [See xlim]
            'savefig':'plot_xy.png',    # [<None, 'show', 'my_file.png'> => overwritten by savefig parameter]
            'x_buffer':0.05,    # [Add whitespace on x-axis so points don't hit frame. buff = x_buffer * (max_x - min_x)]
            'y_buffer':0.05,    # [See x-buffer]
            'subplot':111,  # [111 => plot proportions are equal (first 11), for subplot 1 (many possible)]
            'aspect':'equal',   # [<'auto', 'equal'> 'equal' forces square plots]
            'cosine_corrected':False,   # [<True, False>] => two-var cosine corrected plot?]
            'wright_map':False,     # [<True, False> => wright map?]
            'shape':'ellipse'   # [<'ellipse', 'circle', 'rectangle'> => shape of error shading]
            }

                plot = None                 =>  Use only the Damon defaults.

                plot['title'] = 'I like my title instead'
                                            =>  Overwrite Damon's title with
                                                'I like my title instead'

                plot['se_size'] = 0.10      =>  Make ellipses be 0.10 of one
                                                standard error to make the chart
                                                easier to read.  The SE metric
                                                is automatically given in the
                                                title.


                All variables left unspecified go to their defaults.

            This list of parameters is subject to change and a bit idiosyncratic
            to Damon, but they provide a fair degree of control over charts.
            To learn more about plotting in Python, consult the matplotlib
            documentation and look at the Damon source code.

            ------------
            "savefig" controls how to output or display the chart figure:

                savefig = None              =>  Don't output a figure, just the
                                                parameters used to create a
                                                figure.

                savefig = 'show'            =>  Display the figure on your
                                                interactive Python screen, whatever
                                                that is.  In IDLE, a window will
                                                pop up somewhere on your screen
                                                and halt the program until you
                                                close the figure.  IPython is
                                                more behaved in how it displays
                                                figures.

                savefig = 'path/to/my_graph.png'
                                            =>  Output graphic as a .png file
                                                to the indicated address.  With
                                                just 'my_graph.png', it will be
                                                dropped in the current working
                                                directory.  'my_graph.pdf' will
                                                output a .pdf file.
        Examples
        --------

            [Under construction]
            

        Paste Method
        ------------
            plot_two_vars(xy_data, # ['my_datadict_out', e.g., 'merge_summstat_out']
                          x_name,    # [name of x variable to use for x-axis]
                          y_name,    # [name of y variable to use for y-axis]
                          ent_axis = 'col',    # [<'row', 'col'>] => how variables are situated]
                          err_data = None,  # [<None, 'my_datadict_out'> => e.g., 'merge_summstat_out']
                          x_err = None,    # [<None, size, name of x variable to use for error statistic> => create bubbles]
                          y_err = None,    # [<None, size, name of y variable to use for error statistic> => create bubbles]
                          color_by = None,  # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                          cosine_correct = 'coord|corr',    # [<None, 'coord', 'corr', 'coord|corr'> => correct for cosine between variables]
                          max_cos = 0.95,    # <unsigned corr> => trigger exception when cos(x, y) > max_cos]
                          plot = None,  # [<None, {plotting parameters}> => see docs to customize]
                          savefig = 'xy_plot.png'  # [<None, 'show', 'filename.png'> => how to output chart]
                          )
            
        """

        if self.verbose is True:
            print 'plot_two_vars() is working...\n'

        # Run the damon utility
        plot_two_vars_out = dmn.utils._plot_two_vars(locals())
        self.plot_two_vars_out = plot_two_vars_out

        if self.verbose is True:
            print 'plot_two_vars() is done -- see my_obj.plot_two_vars_out'

        return None


    ##########################################################################

    def wright_map(self,
                   x_name = None,   # [<None, 'Infit', 'rand'> => name of variable to use for the x-axis]
                   y_name = 'Measure',  # [Name of variable to use for the y-axis]
                   y_err = 'SE',    # [<'SE', None, 0.40>] => name of variable containing standard errors]
                   row_ents_color_by = 'r', # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                   col_ents_color_by = 'b', # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                   plot = None, # [<None, {plotting parameters}> => see docs to customize]
                   savefig = 'wright_map.png'   # [<None, 'show', 'filename.png'> => how to output chart]
                   ):
        """Build a Wright map of persons and items.

        Returns
        -------
            wright_map() shows or saves a file as a rectangle chart
            and returns the plotting information needed to make the
            chart.  It also returns a figure object.  Output:

            {'figure':_     =>  figure available for further editing
            'A_ids':_,      =>  array of point A identifiers
            'x':_,          =>  x-coordinates of each point A
            'y':_,          =>  y-coordinates of each point A
            'B_ids':_,      =>  small array of point B identifiers
            'Bx':_,         =>  x-coordinates of each point B
            'By':_,         =>  y-coordinates of each point B
            'a_err':_,      =>  array of errors for A on line a (=x)
            'b_err':_,      =>  array of errors for A on line b
            'x_name':_,     =>  label of line a (which is assigned to x)
            'y_name':_,     =>  label of line b (which provides the y orthogonal component)
            'theta':_,      =>  angle in radians between lines a, b
            'colors':_,     =>  array of colors for each point A
            'plot_params':_ =>  dictionary of plot parameters
            }

            A possible workflow (using Rasch):
                d = dmn.Damon(...)
                d.rasch(...)
                d.merge_summstat(merg_by='row2col')
                d.wright_map(...)
                
            A possible workflow (using coord):
                d = dmn.Damon(...)
                d.coord(...)
                d.base_resid(...)
                d.base_ear(...)
                d.base_se(...)
                d.base_fit(...)
                d.summstat(...)
                d.merge_summstat(merge_by='row2col')
                d.wright_map(...)

            Wright maps can be run with coord() just fine, but the
            meaning of the item distribution is tricky and depends
            on various factors.

        Comments
        --------
            Wright maps (named after the eminent Rasch psychometrician
            Benjamin Wright who popularized them) graph person ability
            distributions against item difficulty distributions, exploiting
            a property of Rasch models that item and person parameters
            are in the same metric and can be related to each other.  They
            are used for checking how well the items are targeted on the
            items at various positions along the distribution.  One likes
            to see the two distributions mirror each other.  They are also
            used to see how persons and items cluster and to aid in understanding
            the factors that cause items to differ in difficulty.

            Damon Wright maps extend the idea by including the ability to
            graph persons and items according to other important psychometric
            statistics, such as Infit and Outfit (any non-zero statistic).
            They also include color-coded error rectangles that make it easy
            to distinguish how different person and item clusters are situated
            in the distribution.

            Wright maps make most sense with Rasch statistics, since they
            are 1-dimensional.  They can also be used with multidimensional
            NOUS statistics as well. However, the meaning of item difficulty
            is not so well defined in this case and much work remains to be
            done in understanding how best to represent item difficulty in
            this case.  Feel free to explore this question.

            Wright maps rely on the merge_summstat() and summstat() methods.

        Arguments
        ---------
            "x_name" is the name of the statistic with which to build the
            x-axis.  It should label a column in obj.merge_summstat_out.
            The statistic should consist only of positive numbers.  If no
            statistic is available, specifying None or 'rand' will create
            random numbers in their place to make the chart legible.

                x_name = 'Infit'        =>  Use Rasch infit statistics for
                                            the x-axis.

                x_name = 'Fit_Perc>2'   =>  Use one of Damon's fit statistics
                                            for the x-axis.

                x_name = None or 'rand' =>  A random number will be assigned
                                            as x.

            -------------
            "y_name" is the primary variable of interest, called "Measure" in
            the rasch() outputs and called "Mean" in the coord/summstat outputs.
            It can be positive or negative and should be accompanied by
            standard errors.  y_name must label a column in obj.merge_summstat_out.

                y_name = 'Measure'      =>  Use Rasch measures for the
                                            y-axis.

            -------------
            "y_err" is the name of the standard error of the y_name variable,
            called 'SE'.  The variable is should label a column in
            obj.merge_summstat_out.  If no SE variable is available, specifying
            None will assign a uniform rectangle size, and specifying a
            float number will size the rectangles accordingly.

                y_err = None            =>  An error rectangle equal to 0.25
                                            will be assigned to each point.

                y_err = 0.40            =>  An error rectangle equal to 0.40
                                            will be assigned to each point.

                y_err = 'SE'            =>  Errors will be pulled from the
                                            'SE' column in sumstat_merge_out.

            -------------
            "row_ents_color_by" is modeled on the "color_by" parameter in
            the plot_two_vars() method, except that it is automatically
            applied to my_obj.row_ents_out, which is output by summstat()
            This parameter controls how to assign colors to entity
            rectangles.  Read the plot_two_vars() docs for details on
            how to use this powerful parameter.
            
            -------------            
            "col_ents_color_by" is the same as row_ents_color_by, except
            that it is applied to my_obj.col_ents_out.

            -------------
            "plot" provides additional control over plotting.  See the
            plot_two_vars() docs for details on the plot parameter.

                plot = None             =>  Go with the defaults.

                plot = {'xy_labels':True}
                                        =>  Apply labels to each point,
                                            overriding the default, which
                                            is False, to leave them out.

                                            You can similary override any
                                            of the plot parameters.
            
            ------------
            "savefig" controls how to output or display the chart figure:

                savefig = None              =>  Don't output a figure, just the
                                                parameters used to create a
                                                figure.

                savefig = 'show'            =>  Display the figure on your
                                                interactive Python screen, whatever
                                                that is.  In IDLE, a window will
                                                pop up somewhere on your screen
                                                and halt the program until you
                                                close the figure.  IPython is
                                                more behaved in how it displays
                                                figures.

                savefig = 'path/to/my_graph.png'
                                            =>  Output graphic as a .png file
                                                to the indicated address.  With
                                                just 'my_graph.png', it will be
                                                dropped in the current working
                                                directory.  'my_graph.pdf' will
                                                output a .pdf file.
        Examples
        --------

            [Under construction]
            

        Paste Method
        ------------
            wright_map(x_name = None,   # [<None, 'Infit', 'rand'> => name of variable to use for the x-axis]
                       y_name = 'Measure',  # [Name of variable to use for the y-axis]
                       y_err = 'SE',    # [<'SE', None, 0.40>] => name of variable containing standard errors]
                       row_ents_color_by = 'r', # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                       col_ents_color_by = 'b', # [<None, 'g', '0.75', (0.5,0.2,0.7), 'rand', ['gender', {'male':'b', 'female':'r'}], ['age', 'g']> => color-code bubbles]
                       plot = None, # [<None, {plotting parameters}> => see docs to customize]
                       savefig = 'wright_map.png'   # [<None, 'show', 'filename.png'> => how to output chart]
                       )
            
        """


        if self.verbose is True:
            print 'wright_map() is working...\n'

        # Run the damon utility
        wright_map_out = dmn.utils._wright_map(locals())
        self.wright_map_out = wright_map_out

        if self.verbose is True:
            print 'wright_map() is done -- see my_obj.wright_map_out'

        return None


    ##########################################################################

    def equate(self,
               construct_ents = {'Get':'AllExcept','Ents':[None]},    # [<None,'Bank',{'Get':<'AllExcept','NoneExcept'>,'Ents':[<None,keys>]}>]
               label = 'Construct',  # [<None,'Bank','MyConstruct'}> => label for construct defined by ents]
               subscales = None,    # [<None,'Bank',{'Get':<'AllExcept','NoneExcept'>,'Labels':<sub int index>,'Subs':[<None,subs>]}>]
               facet = 1,    # [<None,0,1> => facet containing containing construct ents, used only if no anchoring was done]
               logits = False, # [<True, False, {'ecut':0, 'ear':1}>] => to logit metric
               rescale = None,  # [<None,'Bank',{<'All','Constr1','Constr2',...>:{'straighten':_,'mean_sd':_,'m_b':_, 'clip':_}}> => nested dict of rescale params]
               refresh = None,   # [<None, 'Bank', [constructs]> => recalc constructs for bank]
               cuts = None,  # [<None, {<'All','constr1','sub1'>:[cut0, cut1]> => performance level cutpoints]
               stats = True, # [<True, False> => compute summary stats]
               group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75'],  # [<see tools.group_se docs> => factor for grouping se's]
               correlated = None  # [<None> => deprecated]
               ):
        """Equate data to entities in a bank.

        Returns
        -------
            equate() returns None but assigns the my_obj.equate_out
            datadict to the Damon object as an attribute.  The
            coredata consists of a set of estimated scores on a
            construct defined by one or more entitites in a bank,
            along with expected absolute residuals and standard
            errors.  Scores are also computed for specified subscales,
            clusters of entities that have the same label.

            my_obj.equate_out =

                {'construct',   =>  Measures on constructs based on
                                    entities identified in construct_ents
                                    and for each specified subscale

                 'ear',         =>  Expected absolute residuals between
                                    measures and corresponding raw scores

                 'se'           =>  Standard errors of construct measures
                 
                 'pls'          =>  Student performance levels, if cuts are
                                    specified.
                 
                 'stats'        =>  A table of summary statistics for each
                                    construct measure:  
                                        Count   =>  count of persons
                                        Mean    =>  mean measure
                                        SD      =>  standard deviation
                                        SE      =>  standard error
                                        Sep     =>  person separation
                                        Rel     =>  construct reliability
                                        Min     =>  minimum measure
                                        25Perc  =>  25th percentile
                                        Median  =>  median measure
                                        75Perc  =>  75 percentile
                                        Max     =>  maximum measure
                'cuts'          =>  Measure cutpoints if the cuts option was
                                    used.
                                    
                'equate_params' =>  current equate() parameters.

            In addition, the following datadict outputs are assigned 
            directly to the Damon object for ease of access and exporting:
                
                my_obj.equate_out_construct     =>  student construct measures
                my_obj.equate_out_se            =>  their standard errors
                my_obj.equate_out_pl            =>  their performance levels
                my_obj.equate_out_stats         =>  summary statistics

            equate() assigns its input parameters to the existing
            bank under the key 'equate_params'.  Whenever equate() is run
            again, as long as either 'construct_ents' or 'subscales'
            is 'Bank', the equate() parameters will all be drawn from
            the bank.

            Workflow:  See comments.

        Comments
        --------
            equate() addresses the question:  How would a group
            of examinees perform on a test (construct) composed of
            specified items drawn from an item bank?  By using
            a calibrated item bank as an intermediary, it becomes
            possible to compare students on a common scale who
            have taken tests composed of different items.  
            
            Taking advantage of its logit and rescaling options as
            well as its facility with subscales, equate() is intended 
            to be the primary method for generating person measures that
            are equated over time and across test forms.

            summstat() can be used to answer similar questions
            but requires having all persons and items in a
            single data matrix, which can be inconvenient.  equate()
            is a light-weight alternative to summstat().

            equate() can also be used in the absence of an item bank
            to calculate measures, standard errors, and other stats
            for the main construct and subscales using coordinates instead
            of cell statistics (summstat is applied to cell stats).  The
            results are equivalent, but equate() is much faster.

            While equating is usually applied with common items
            (column entities, facet=1), there is no reason why it
            can't be done with common persons (row entities, facet=0).

            The big limitation of equate() is that it only works using the
            metric of the data as it goes into coord(), i.e., the data
            metric from which person and item coordinates are calculated,
            which are stored in a temporary or persisting "bank" along
            with other parameters and entity information.  The bank
            includes coordinates for calculating the EAR and SE statistics.
            These coordinates -- and they alone -- are used by equate() to
            calculate measures, standard errors, and other stats on the
            defined construct(s).
            
            However, equate() does contain an option for converting measures
            and standard errors into logits, similar to the 'PreLogit' metric
            but with some modest advantages.  It also allows a variety of
            other rescaling options and the assignment of performance levels.
            
            Workflow 1
            ----------
            To run equate() on the existing dataset without calling
            coordinates from pre-existing banks, equate() can create
            a temporary bank and pull parameters from it as needed:

                Step 1: Build a Damon object as usual.  Run coord()
                        and associated methods.

                Step 2: Run my_obj.equate(). equate() asks you to
                        define a common construct and subscales in terms 
                        of specific items in the bank.  The method sums up the
                        coordinates of these items to come up with
                        a single set of coordinates for the test
                        construct, called the construct vector.  Each
                        person coordinate vector is multiplied by the
                        construct vector to produce a "measure".  This
                        or some transformation of it is the reported 
                        person score.  The same procedure is used to 
                        calculate the EAR and SE statistics.  
                        
                        The summed coordinates are stored in a temporary 
                        bank called my_obj.bank_out, which you don't need
                        in this workflow but is useful for Workflow 2 below.

            This use-case makes a light, quick alternative to summstat()
            while providing measures for potentially multiple constructs
            at once.  Instead of averaging estimates over multiple ranges
            of cells, it computes them all directly from coordinates.
            While summstat() is limited to one construct at a time, equate()
            is not.  This workflow works with both coord() and sub_coord()
            coordinates. 

            Workflow 2
            ----------
            This workflow, for which equate() was originally written, 
            is to use equate() to compute measures for constructs based on 
            groups of items using pre-calibrated item coordinates and to store
            the construct coordinates in the item bank for future use, so that
            the constructs always have the same meaning and are comparable as
            the test changes:

                Calibration Phase
                -----------------
                Step 1: Build a Damon object, run coord() and (optionally)
                        base_ear() and base_se(), and the methods
                        they depend on.  The base_ear() and base_se()
                        methods calculate coordinates used
                        for calculating EARs and SEs.
                
                Step 2: Run my_obj.equate() to build an in-memory item bank
                        including coordinates for any desired constructs or
                        subscales.

                Step 3: Run my_obj.bank() to convert the in-memory item bank
                        to a pickle file ('my_bank.pkl'), modifying if desired. 

                Measurement Phase
                -----------------
                Step 4: Collect data from new examinees using items that 
                        overlap those stored in the item bank.

                Step 5: Load data as a new Damon object (my_obj2). If the 
                        calibration data was standardized, specify:
                        
                            my_obs2.standardize(std_params='my_bank.pkl')
                
                Step 6: Run coord() using the anchors parameter to refer to
                        'my_bank.pkl'.  coord() will calculate person
                        coordinates that will be anchored to the
                        coordinate system of the item bank you specify
                        in coord().  

                        equate() figures out which facet to build measures for
                        by looking at information you provided in coord()'s
                        anchors['Facet'] parameter.

                Step 7: Run base_ear() and base_se() and any intervening 
                        methods.  They will automatically refer to 'my_bank.pkl'
                        to produce person EAR and SE coordinates.

                Step 8: Run my_obj2.equate() using the construct_ents='Bank'
                        option.  For consistency, use of the 'Bank' option
                        in any argument sets all arguments to 'Bank'.  This
                        ensures that scales and subscales are comparable
                        across test runs as long as there are common items,
                        even as new students are tested and items switched
                        in and out of the test.
                        
                        Note that if the bank exists and has equate() 
                        parameters and none of the current parameters are
                        specified as 'Bank', the ones in the bank will be
                        overwritten.
                
                Step 9: Run my_obj2.bank('my_bank.pkl') to update the bank 
                        with new or refreshed item parameters that were 
                        calculated by coord() in Step 6.
                        
                        It is very important in this workflow that you
                        specify the bank file name in bank().  Otherwise,
                        bank() will create a new in-memory bank based
                        only on the current data file.
            
            Examples of Workflow 1 and Workflow 2 are provided below.
            
            Standard Errors
            ---------------
            equate() computes EAR and SE stats from base_ear() and 
            base_se() coordinates.  If logits is True, the SE stats are
            computed from the probability of success (of exceeding 0 on
            the prelogit scale) using the binomial formula for error.
            
            When the data are dichotomous or polytomous and you are graphing
            entity standard errors against their measures, expect to see a
            U- or cup-shaped distribution with error minimized at 0.0 
            and increasing toward the scales.  When logits is True, the
            graph should be a smooth line.
            
            Standard Errors: equate() vs summstat()
            ---------------------------------------
            In principle, when equate() and summstat() aggregate cell
            SE and EAR statistics to get construct measures they should
            be the same; but they're not.  This is because the aggregation
            formula is a little different for the two methods.  
            
            summstat() aggregates standard errors (and EARs) across cells.  
            equate() does all of its aggregations across coordinates, not
            cells.  This makes it possible to create abstract construct
            coordinates that can be reused across datasets even though
            the items originally used to build the construct may no longer  
            be used.  When coordinates are used to build construct 
            MEASURES, there is no mathematical difference between
            summing across a row of cell estimates and summing across the
            the corresponding column coordinates and multiplying by the
            row coordinates.  The operations are distributive.
            
            But when coordinates are used to build construct EAR and SE
            statistics, the operations are NOT distributive.  That is
            because the cell standard errors are on a ratio scale, whereas
            the corresponding coordinates are on a 2-dimensional log
            scale.  The root mean square standard error calculated across
            cells for a given row (summstat()) does not equal the exponent of 
            the mean of the column coordinates multiplied by the row 
            coordinates, which is how equate() calculates it.
            
            Fortunately, they are reasonably close.  A graph of row
            standard errors calculated using the two methods shows that
            most row SE/EAR statistics are close to the identity line
            but that the equate() standard errors are all at least slightly
            smaller than the corresponding summstat() standard errors
            with some outliers falling further below the identity line.
            This pattern holds (when calculating row standard errors) when
            the expected range of error across cells is uniform, or when
            the expected variation in errors across columns is less than
            the expected variation across rows (which is the case with
            most datasets).  However, if the range of variation in errors
            across columns is large relative to rows (which is rare), 
            other biases arise which increase the negative bias substantially.
            
            To compensate for this negative bias as well as other negative
            biases that may occur, equate() offers the 'group_se' parameter, 
            which nudges  standard errors upward by a specified percentage 
            of a standard deviation.
            
            What we lose in accuracy with a different aggregation formula in 
            equate() (assuming the summstat() error aggregation method is
            the correct one) we make up for in the convenience and 
            portability afforded by abstract SE and EAR construct coordinates.

            The 'Bank' Parameter
            --------------------         
            The 'Bank' options in the arguments described below make
            it possible to reuse the equate() parameters from previous
            analyses, as stored automatically in the bank.  The first
            time you apply equate() (during the item calibration phase), 
            do not specify 'Bank' as any of the equate() parameters.  
            Specifying 'Bank' in the measurement phase tells the function to
            use all the same parameters that you used in the calibration
            phase to ensure comparability.  Specifying 'Bank' for any argument
            automatically specifies it for all of them.  So all you have
            to type is:
            
                d.equate('Bank')

            Also, note that Workflow 2 only works in conjunction with
            coord().  It does not work with sub_coord(), rasch() or objectify().

            Adding New Items, Refreshing Old Ones
            -------------------------------------
            After the initial bank is built, whenever a new dataset is
            analyzed (using coord's anchor argument) any new items are 
            calibrated in the same coordinate system as those in the bank,
            and the new item coordinates are added to the bank.  This makes
            it easy to equate tests over time.  Every time a new test
            is given, you just go through the steps in the Measurement Phase
            described above.  Besides generating equated measures, this
            automatically updates the bank with the new items.
            
            If you have reason, such as high misfit, to believe that some of 
            the items have "drifted", have changed their psychometric behavior, 
            use the anchors parameter in coord() to specify using only
            those items as anchors that have not drifted.  For example,
            if 'It1' and It2' out of five bank items have drifted, specify:
            
                d.coord(None, anchors={'Bank':bankfile,
                                       'Facet':1,
                                       'Entities':['It3', 'It4', 'It5'],
                                       'Refresh_All':False})
                
            This specification uses only items 3 - 5 as anchors.  Note that
            Refresh_All is False.  Refresh_All = True would tell the program
            to replace all the coordinates in the bank with refreshed
            coordinates.

            See help(dmnt.flag_item_drift) for more information about
            detecting item drift.

            See helpl(dmnt.check_equating) for more information about how
            to measure the degree to which two test forms are properly
            equated.

            NOTE: When trying to replicate measures using duplicate datasets,
            you may find that the constructs/subscales do not exactly equal
            between the calibration phase and the measurement phase. This has
            to do with discrepancies in the order in which facet
            coordinates are calculated in coord(). They may differ between
            the calibration phase (iterative) and measurement phase (anchored).
            The differences should be small, however.

        Arguments
        ---------
            "construct_ents" is the list of entities you wish to
            use to define the "construct" of your test -- the measurement
            dimension of interest.  Note that it does not require
            you to identify the facet of these entities, as it
            figures this out automatically based on the facet used
            in the anchor specification when running coord() for
            the current Damon.  The terms "entities" and "attributes"
            below always refer to entities and attributes of the
            "anchored" facet.

            Examples:

                construct_ents = None
                            =>  Do not specify the construct using
                                entities.  In this case, subscales
                                cannot be None.

                construct_ents = 'Bank'
                            =>  This works if you have previously
                                defined a construct using equate(),
                                in which case the equate parameters
                                entered at that time are stored in the
                                bank and carried over to the current
                                analysis.  It you previously ran
                                equate() and do not use 'Bank', those
                                previous paramaters will be overwritten
                                in the bank by those of the current
                                analysis.

                                IMPORTANT:  Therefore, if you want
                                to preserve construct comparability
                                across analyses, use 'Bank' every
                                time you run equate() except for
                                the first time.

                construct_ents = {'Get':'AllExcept','Ents':[None]}
                            =>  Use all the entities to define the
                                construct.

                construct_ents = {'Get':'AllExcept','Ents':['It1','It4']}
                            =>  Use all entities EXCEPT Items 1 and 4
                                to define the construct.

                construct_ents = {'Get':'NoneExcept','Ents':['It1','It4']}
                            =>  Use ONLY Items 1 and 4 to define the
                                construct.

            --------------
            "label" is the label you choose to describe the
            construct created by combining the entities listed
            under construct_ents.  It only applies to the construct_ents
            argument.

                label = None
                            =>  No construct label is necessary since
                                construct_ents is None.

                label = 'Bank'
                            =>  Refer to the bank for the label
                                used in previous analyses.

                label = 'MyConstruct'
                            =>  Call the construct defined by
                                the entities in construct_ents
                                'MyConstruct'.  This is how it
                                will appear in reports.

            --------------
            "subscales" allows you to specify constructs for multiple
            subscales as labeled in one of the rows in collabels (or columns
            row rowlabels), resulting in measures for each subscale.

            Examples:
                subscales = None
                            =>  Do not specify the construct using
                                attributes.  In this case, construct_ents
                                cannot be None.

                subscales = 'Bank'
                            =>  This works if you have previously
                                defined subscales using equate(),
                                in which case the equate parameters
                                entered at that time are stored in the
                                bank and carried over to the current
                                analysis.  If you previously ran
                                equate() and do not use 'Bank', those
                                previous parameters will be overwritten
                                in the bank by those of the current
                                analysis.

                                IMPORTANT:  Therefore, if you want
                                to preserve construct comparability
                                across analyses, use 'Bank' every
                                time you run equate() except for
                                the first time.

                subscales = {'Get':'NoneExcept','Labels':1,'Subs':['Geometry']}
                            =>  To define the subscale, use ONLY the
                                entities that have the 'Geometry' label.
                                In this case the attribute label is found
                                in either column 1 of rowlabels or row 1
                                of collabels, depending on which is the
                                anchored facet.  Counting starts at 0,
                                so column 1 would be the second column from
                                the left.

                                The anchored facet is determined by the
                                'facet' arg (if coord() did not use anchoring),
                                otherwise by private inspection of the coord()
                                args.

                IMPORTANT:  Note that for the equate() method, the 'Labels'
                parameter must be an integer referring to a row in collabels or
                a column in rowlabels (counting from 0).  It can't be a string
                descriptor or it will throw an exception.

                subscales = {'Get':'NoneExcept','Labels':1,'Subs':['Geometry','Vocabulary']}
                            =>  Define two subscales, one for 'Geometry'
                                and one for 'Vocabulary'.  Since the anchored
                                facet is items (see note above), the relevant
                                entities are found in row 1 of collabels.

                subscales = {'Get':'AllExcept','Labels':1,'Subs':[None]}
                            =>  Define a unique construct for each
                                attribute listed in row 1 of collabels
                                (the anchored facet), however many that
                                may be.

            NOTE:   You can use both the construct_ents and the
            subscales arguments at the same time.  equate()
            will return one column of statistics corresponding to
            the construct_ents construct and one or more columns
            corresponding the subscales construct(s).

            --------------
            "facet" is the facet of the construct entities.
            If the coord() method invoked the "anchor" argument, equate()
            automatically gets the facet from the coord()
            arguments and facet can be set to None, as it
            will be ignored.  If anchoring was performed, the construct
            facet needs to be specified.

                facet = None
                            =>  The construct facet will be obtained
                                automatically from the anchor argument
                                in coord().

                facet = 0
                            =>  The constructs are row entities

                facet = 1
                            =>  The constructs are column entities.
            
            --------------
            "logits" = True means transform all measures to a logit
            scale.  The scale should already be in "prelogits".
            What the logit option adds is:  1) greater control over the
            probabilistic meaning of the logits, 2) logit standard errors
            that are model-based (smooth, binomial) rather than rough. The
            latter are more in line with the outputs of IRT models.  The
            rescale parameter is applied after the logits have been 
            calculated.
            
                logits = True   => tranform to logits
                
                logits = False  => do not transform
                
                logits = {'ecut':0.0, 'ear':1.0}
                                =>  calculate probability relative to the
                                    zero point of the prelogit scale -- the
                                    probability of "success".  Scale the 
                                    logits in terms of 'ear' units in
                                    tools.cumnormprob().

            --------------
            "rescale" offers a way to rescale each construct based on the 
            tools.rescale() function.  For more info:

                >>>  import damon1.tools as dmnt
                >>>  help(dmnt.rescale)

            The rescale() nested dictionary takes the following parameters
            for all constructs or each individually.  Specify only
            the parameters you need.

                {'straighten':<None,True,'Percentile'>,
                 'mean_sd':[target mean, target sd],
                 'm_b':[m,b]    # multiply by m, add b
                 'clip':<None, [min, max]>
                 'round_':<None, n decimals>
                 'reverse':<True, False> => reverse sign of construct
                 }

                'straighten' is useful when dichotomous data is used
                to estimate a linear variable.  The resulting estimates
                will sometimes have a nonlinear relationship to the true
                variable.  'straighten' converts the estimates into
                percentile logits which will often create a more linear
                relationship to the true values. 
                
                WARNING: 'straighten' computes percentile logits based
                only on the current person sample.  It doesn't know how
                to take into account persons from previous datasets. As
                it now stands, it is not appropriate for Workflow 2 above. 

                'mean_sd' and 'm_b' are applied to the straightened
                scores, if relevant.  If both are specified, 'm_b'
                is ignored.

                rescale = None
                            =>  Do not rescale.

                rescale = 'Bank'
                            =>  Use the existing rescale parameters
                                as stored in the bank.

                If you specify 'All', all possible constructs are rescaled
                with the same set of parameters:

                rescale = {'All':{'mean_sd':[300,57]}}
                            =>  Make all constructs have a mean of
                                300 and standard deviation of 57.

                rescale = {'All':{'straighten':True, 'm_b':[10,100], 
                                  'clip':[50, 200], 'round_':2}}
                            =>  "Straighten" the construct scores,
                                then multiply them by 10 and add 100.
                                Clip scores to fit between 50 and 200.
                                Round them to two decimal places (3.14).

                rescale = {'All':{'straighten':'Percentile'}}
                            =>  Convert the construct scores to percentiles.

                rescale = {'C1':{'straighten':True,'mean_sd':[300,57]},
                           'C2':{'m_b':[10,100], 'clip':[50, 150]},
                           }
                            =>  For the 'C1' construct, first straighten,
                                then scale to have mean of 300 and standard
                                deviation of 57.  For the 'C2' construct,
                                multiply by 10 and add 100.  Clip scores
                                to be between 50 and 150.

                                Let us assume there is also a 'C3'
                                construct.  Because it is not listed,
                                it is not rescaled.

            --------------
            "refresh" is used to recalculate the coordinates for one or more
            constructs and replace them in the bank, overwriting existing
            construct coordinates.  In order to preserve comparability of
            constructs and subscales as the test changes, you will generally
            want to specify refresh = None.
                        
                refresh = None
                            =>  Use the construct coordinates as they already
                                exist in the bank.  If they don't exist, they
                                will be calculated, but otherwise not.  
                
                refresh = ['construct_1', 'sub_2']
                            =>  Recalculate the coordinates for 'construct_1'
                                and 'sub_2', replacing any in the bank.  The
                                new coordinates will be used to calculate
                                construct measures.
                
                refresh = 'Bank'
                            =>  Use the refresh parameters as stored in the
                                bank.
        
            Note:  Do not confuse this usage of "refresh" with its usage in
            coord().  Here, refresh means building the construct 
            coordinates by summing a potentially different set of items.  In 
            coord(), it means actually calculating new coordinates.)
            
            --------------
            "cuts" <None, cuts> specifies cutpoints for separating the
            construct scale and subscales into performance levels.  It can
            also calculate and apply cutpoints if they aren't known, if the
            data file contains one or more columns of expert ratings that 
            classify persons.  Each cutpoint is defined as that point on 
            the measurement scale that is equidistant from the median 
            measures of students assigned to two adjoining rating scale 
            categories.
            
                cuts = None =>  Do not apply cutpoints.
                
                cuts = {'All':'my_ratings'}
                            =>  Calculate cutpoints using ratings 
                                given in the 'my_ratings' column
                                of the original data array.  The same
                                cut-points are used for all scales.

                cuts = {'construct':'construct_ratings',
                        'sub1':'sub1_ratings',
                        'sub2':'sub2_ratings'}
                            =>  Calculate cutpoints for the construct
                                scale and two subscales using ratings
                                that are specific to each scale.
                
                cuts = {'construct':'construct_ratings',
                        'sub1':'construct_ratings',
                        'sub2:'construct_ratings'}
                            =>  This is a potentially useful variation.  Here
                                we calculate different cutpoints for each
                                scale, but they are all based on a common
                                set of expert ratings.  This helps
                                make all scales interpretable according
                                to the same rating scale rubric, but taking
                                into account the metric differences in 
                                each scale.
                                
                cuts = {'All':[200, 250, 300]}
                            =>  Cut the construct scores into four categories
                                demarcated by the scores 200, 250, and 300
                                for all scales.
                
                cuts = {'construct':[200, 250, 300],
                        'sub1':[100, 150, 200'],
                        'sub2':None}
                            =>  Cut the 'construct' scores at the indicated
                                cutpoints, as well as the 'sub1' scores.  Do not
                                apply cuts to 'sub2' (the output column will
                                consist of nanvals).  If there is a 'sub3' that
                                is not specified, its output column will
                                also consist of nanvals.

            --------------
            "stats" <True, False> tells the method to calculate a set of
            summary statistics for each construct, including mean, standard
            deviation, root mean squared error, and reliability.

            --------------
            "group_se" is used to generate part of the formula used to
            aggregate standard errors across cells.  The standard error
            of the mean across, say, a group of items is the root mean
            square error across the cells multiplied by some factor.
            "group_se" calculates that factor.  If the cell error for
            each item (for a given person) were statistically independent
            across items, the factor would be 1/sqrt(n_items).  However,
            this is not the case with Damon cell errors.  The appropriate
            factor is generally some version of 1/sqrt(2 * ndim), tweaked
            to account for a variety of conditions.
            
            The default is:
                 group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75']
            
            which works fairly well in reproducing the correct standard
            error across a variety of datasets, especially those associated
            with educational test data.  However, simulation experiments
            may show that you need a somewhat different formula, and you
            can specify that through this option.
            
            For a complete discussion, see the docs for tools.group_se()

            --------------
            "correlated" <None> has been deprecated and will be
            ignored internally. It is up to the user to make sure that 
            all items that go into a construct are positively correlated.

            
        Example, Workflow 1 (build measures from the current dataset)
        -------------------
            f0 = dmn.Damon(...)
            f0.standardize()
            f0.coord([[2]])
            f0.base_est()
            f0.base_resid()
            f0.base_ear()
            f0.base_se()
            f0.equate(label='meas', facet=1)  # produces in-memory constructs
            
            # This step is optional -- converts an in-memory bank built
            # by equate() into the 'my_bank.pkl' pickle file.
            f0.bank('my_bank.pkl')   
                  
                   
        Example, Workflow 2 (build and update a bank across multiple forms)
        -------------------
        
        Phase 1, Calibration (though it also produces f0 measures)
            f0 = dmn.Damon(...)
            f0.standardize()
            f0.coord([[2]])
            f0.base_est()
            f0.base_resid()
            f0.base_ear()
            f0.base_se()
            f0.equate(label='meas', facet=1)
            f0.bank('my_bank.pkl')   
        
        Phase 2, Measurement
            f1 = dmn.Damon(...)
            f1.standardize('my_bank.pkl')  # standardizes like in f0
            f1.coord(anchors={'Bank':'my_bank.pkl'})
            f1.base_est()
            f1.base_resid()
            f1.base_ear()       # refers to ear coordinates in bank
            f1.base_se()        # refers to se coordinates in bank
            f1.equate('Bank')   # refers to equate params in bank
            f1.bank('my_bank.pkl')  # Updates bank with new items
            
            All subsequent forms are analyzed the same way.  The bank
            automatically grows as new items are absorbed into the
            coordinate system.
            
        Paste method
        ------------
            equate(self,
                   construct_ents = {'Get':'AllExcept','Ents':[None]},    # [<None,'Bank',{'Get':<'AllExcept','NoneExcept'>,'Ents':[<None,keys>]}>]
                   label = 'Construct',  # [<None,'Bank','MyConstruct'}> => label for construct defined by ents]
                   subscales = None,    # [<None,'Bank',{'Get':<'AllExcept','NoneExcept'>,'Labels':<sub int index>,'Subs':[<None,subs>]}>]
                   facet = 1,    # [<None,0,1> => facet containing containing construct ents, used only if no anchoring was done]
                   logits = False, # [<True, False, {'ecut':0, 'ear':1}>] => to logit metric
                   rescale = None,  # [<None,'Bank',{<'All','Constr1','Constr2',...>:{'straighten':_,'mean_sd':_,'m_b':_, 'clip':_}}> => nested dict of rescale params]
                   refresh = None,   # [<None, 'Bank', [constructs]> => recalc constructs for bank]
                   cuts = None,  # [<None, {<'All','constr1','sub1'>:[cut0, cut1]> => performance level cutpoints]
                   stats = True, # [<True, False> => compute summary stats]
                   group_se = ['1/sqrt(2d)n**0.1429', '1/sqrt(2d)*0.75'],  # [<see tools.group_se docs> => factor for grouping se's]
                   correlated = None  # [<None> => deprecated]
                   )
    """
        if self.verbose is True:
            print 'equate() is working...\n'

        # Run the damon utility
        equate_out = dmn.utils._equate(locals())
        self.equate_out = equate_out

        if self.verbose is True:
            print 'equate() is done -- see my_obj.equate_out'
            print 'Contains:\n',self.equate_out.keys(),'\n'

        return None




    ##########################################################################

    def bank(self,
             filename = None, # [<None,filename> => bank containing entities from both facets.  *.pkl extension is optional.]
             bankf0 = {'Remove':[None],'Add':['All']}, # [<None,{'Remove':[<None,'All', or list entities to remove>],'Add':[<None,'All', or list entities to add>]}> ]
             bankf1 = {'Remove':[None],'Add':['All']}, # [<None,{'Remove':[<None,'All', or list entities to remove>],'Add':[<None,'All', or list entities to add>]}> ]
             ):
        """Build/modify bank of entity coordinates.

        Returns
        -------
            bank() creates a new, or modifies an existing, dictionary
            of coordinates and other information for Facet 0 (row) or
            Facet 1 (column) entities.  If filename = None, bank()
            returns None but assigns bank_out to the Damon object.
            If a filename is given, the dictionary is also saved as
            a persistent pickle file.
            
            If my_obj.bank_out already exists (e.g., created by the
            equate() method), bank() modifies it if necessary and saves
            it as a pickle file.

            The bank can be accessed by the coord()and rasch() methods
            for anchoring the coordinate system of a new analysis to
            coordinates calculated in a previous analysis.  It does
            not yet support the sub_coord() and objectify() methods.

            The pickle files themselves are not human-readable
            but they are accessed and read automatically by
            the coord() and rasch() methods.  (Read the Python
            docs to learn how to open and read pickle files.)

            Here is the nested dictionary structure of a bank:

            Bank = {'facet0':{'labels':{},
                              'ent_coord':{},
                              'ear_coord':{},
                              'se_coord':{}
                              },
                    'facet1':{'labels':{},
                              'ent_coord':{},
                              'ear_coord':{},
                              'se_coord':{}
                              },
                    'anskey_param':{},
                    'std_params':{},
                    'parse_params':{},
                    'equate_params':{},
                    'refit_params':{'ear':_, 'se':_}
                    'step_coord':[[]],
                    'ndim':1,
                    'shape':(nfacet0, nfacet1)
                    }

            To access the entity coordinates for column Entity E:

                MyEntCoords = Bank['facet1']['ent_coord']['E']

            Workflow (methods usually run before bank()):
                my_dmnobj.parse(...)
                my_dmnobj.standardize(...)
                my_dmnobj.coord(...)
                my_dmnobj.base_est(...)
                my_dmnobj.base_resid(...)
                my_dmnobj.base_ear(...)
                my_dmnobj.base_se(...)
                my_dmnobj.equate(...) 
                my_dmnobj.bank(...)

        Comments
        --------
            A key feature of object-oriented statistics is the ability
            to apply the row or column coordinates generated in one
            analysis to the data of another analysis.  This forces
            both datasets to share a common space, equivalent to
            running a single large analysis with both datasets.  This
            procedure is called "anchoring".  Anchoring requires a
            set of common row entities (persons) or column entities
            (items) to link the two datasets.

            Damon supports the ability to store a lookup dictionary
            of item and person coordinates in a persistent location
            on disk called a "pickle" file.  These are sometimes referred
            to as item and person banks (Facet 1 and Facet 0, respectively).
            The coord() method can call one of these banks, open it,
            and pull the coordinates of any persons or items it shares with
            the bank.  These are used to anchor the coordinate
            space of the new entities to that of the banked entities.
            The new person and item coordinates calculated by coord()
            are thus consistent with those in the bank.  bank() can
            then be used to add the new entity coordinates to the
            bank.  Entities can also be removed from the bank as they
            become stale or produce misfits.  Existing coordinates can
            be "refreshed" in light of new data using "anchors" parameter
            in coord().

            Similar anchoring abilities are supported for rasch().

            bank() also stores coordinates for computing standard errors
            (SE) and expected absolute residuals (EAR).  These draw
            on base_ear_out and base_se_out.  base_ear() and base_se() know
            how to look up these coordinates for anchoring purposes.

            In addition to coordinates, each bank stores a score_mc answer
            key, parse, and standardization parameters (if used) which are
            drawn from the dataset used to create the bank.  This provides
            information needed to score, parse and standardize the new dataset
            so that bank coordinates can be applied to it successfully.  This
            is handled as follows:

                1.  For the score_mc step in the new Damon obj, specify:

                    my_dmnobj.score_mc(anskey='mybank.pkl')

                2.  For the parse step, specify:

                    my_dmnobj.parse(...,parse_params='mybank.pkl')

                3. For the standardize step, specify:

                    my_dmnobj.standardize(...,std_params='mybank.pkl',...)

                3.  Thanks to steps 1 -- 3, coord() can apply the banked
                    coordinates to the new dataset and: a) the column entity
                    keys will match; b) the data will be converted into
                    the metric of the original data, allowing the banked
                    coordinates and new data to be in compatible metrics.

                    If the banked coordinates are not based on scored, parsed
                    or standardized data, steps 1 -- 3 are unnecessary.

            bank() automatically retrieves the coordinates, 'anskey_param',
            'parse_params', and 'std_params'.  The 'equate_params' are
            automatically assigned to the bank at the proper time.

            Finally, bank() stores labels for persons and items.  These may
            contain, for example, the full text of each item, the correct
            response, a link to an item database -- anything that can be
            stored as a column label or row label.  To take full advantage
            of this feature, you will want to use the merge_info() function,
            which merges item information with column labels in the main
            dataset.
            
            If the equate() method is used to build a bank containing construct
            coordinates, you will want to use bank() to save this off as a
            pickle file.

        Arguments
        ---------
            "filename" is the name of the pickle file to contain the
            Bank -- a dictionary of coordinates and parameters for entities
            of both facets.  It may also be a path name.  If the bank file
            does not exist, one is created.  If it exists, it is edited as
            specified.

                filename = 'Home/Scripts/MyProject/MyBank.pkl'

            The pickle extension *.pkl is recommended but not necessary.

                filename = None     =>  Do not create a file, just
                                        hold in memory.  The bank will
                                        be deleted when the memory is
                                        released.


            -------------
            "bankf0" and "bankf1" specify how to modify or create the
            'facet0' and 'facet1' lookup dictionaries respectively.  Entity
            coordinates are loaded from coord_out and possibly base_ear_out,
            base_se_out, parse_out and standardize_out of the current
            data object.  Entities may be removed or added to the bank.
            If bank() can't find filename on your drive, it creates a new
            file and loads the latest entity dictionaries.

                bankf0 = None       =>  Do not create a dictionary of
                                        Facet 0 (row/person) entities.

                bankf1 = {'Remove':[None],'Add':['All']}
                                    =>  Do not remove any Facet 1 (column)
                                        entities from the bank.  Add all
                                        the Facet 1 entities from the current
                                        (open) Damon obj to the bank.  Where
                                        the entity from the existing Damon
                                        already exists in the bank, the bank
                                        version will be overwritten.  This is
                                        not a big deal since the two sets of
                                        coordinates are generally the same.

                bankf1 = {'Remove':['All'],'Add':['All']}
                                    =>  This cleans out the existing item
                                        bank and replaces it with all the
                                        items of the current Damon obj.

                bankf1 = {'Remove':[1,3,5],'Add':[5,6,7]}
                                    =>  Remove items (column entities) 1,3,5
                                        from the bank and add items 5,6,7 to
                                        the current Damon obj.

            Generally, banks are built for only one facet at a time,
            but this is not required.

            It is perfectly permissable to use the anchors argument
            (from coord()) and the bank() method in the same Damon
            analysis.  coord() draws anchors from the bank for analysis.
            The resulting coordinates are then stored (however desired)
            in the same bank.

            IMPORTANT
            ---------
            Equating only works when the number of dimensions is
            constant across data sets.  Within a bank, each entity
            must have the same number of coordinates, i.e., the same
            dimensionality.

        Examples
        --------


        Paste method
        ------------
            bank(filename = 'Bank.pkl', # [<None,filename> => bank containing entities from both facets.  *.pkl extension is optional.]
                 bankf0 = {'Remove':[None],'Add':['All']}, # [<None,{'Remove':[<None,'All', or list entities to remove>],'Add':[<None,'All', or list entities to add>]}> ]
                 bankf1 = {'Remove':[None],'Add':['All']}, # [<None,{'Remove':[<None,'All', or list entities to remove>],'Add':[<None,'All', or list entities to add>]}> ]
                 )
        """
        if self.verbose is True:
            print 'bank() is working...\n'

        # Run the damon utility
        bank_out = dmn.utils._bank(locals())
        self.bank_out = bank_out

        if self.verbose is True:
            print 'bank() is done -- see my_obj.bank_out'
            print 'Contains:\n',self.bank_out.keys(),'\n'

        return None

    ##################################################################################################

    def restore_invalid(self,
                        outputs,   # [['base_est_out','base_se_out',...] => string list of outputs to restore]
                        getrows = True,  # [<None,True> => restore invalid rows]
                        getcols = True,  # [<None,True> => restore invalid cols]
                        ):
        """Restore rows and columns that had insufficient data.

        Returns
        -------
            restore_invalid() returns None but replaces each
            of a list of Damon() method "outputs", stored
            as attributes of the Damon object, with a "restored"
            version of the same, i.e., one with invalid rows
            and columns included.  Thus, to access a restored
            version of the final estimates, type:

                >>>  my_dmnobj.fin_est_out

            which is also what you would type if you had not
            run restore_invalid().  The same applies to every
            other output listed in the Ouputs argument.

            Workflow:

                MyData = Damon(...)
                MyData.extract_valid(...)
                MyData.coord(...)
                MyData.base_est(...)
                MyData.fin_est(...)
                MyData.restore_invalid(['base_est_out','fin_est_out','coord_out']],...)

        Comments
        --------
            restore_invalid() reverses the operation performed
            by extract_valid().  It returns specified data arrays
            in the Damon object to their original dimensions.

            The method is straightforward when the arrays to
            be restored are the same size and have the same
            IDs as the original array.  However, when new columns
            have been added and relabeled, as with parsing, it
            may not be possible to restore invalid columns.
            Therefore, the getrows and getcols arguments allow
            you to restore just the rows or just the columns.

            restore_invalid() does not support restoring of
            missing rows and columns to all "unusually formatted"
            outputs.
            
            The method can be used to restore rows and columns
            to 'coord_out'.  In this case, however, the restored 'fac0coord'
            and 'fac1coord' are assigned directly to the Damon object 
            (self.fac0coord, self.fac1coord'), not inside self.coord_out.

        Arguments
        ---------
           "outputs" is a string list of outputs created by running
            a DamonObj method.  To get a list of available outputs,
            type:

                my_dmnobj.__dict__.keys()

            Most items on that list with a '_out' suffix -- those
            that are in datadict format and shaped like the
            original dataset except for the removal of invalid
            rows and columns -- can be restored to the dimensions
            of the original data set.

            An important requirement is that each output in the list
            must be a string name enclosed in quotes, e.g., 'base_est_out'.

            Example:
                outputs = ['base_est_out','fin_est_out','fin_se_out']

            ------------
            "getrows" <None,True> specifies whether to restore
            invalid columns.

            ------------
            "getcols" <None,True> specifies whether to restore
            invalid columns.  If you used parse() and your estimates
            array is not the same size as the original data array,
            or the columns have been renamed, you will probably
            want to set this to None.

        Examples
        --------


        Paste method
        ------------
            restore_invalid(outputs,   # [['base_est_out','base_se_out',...] => string list of outputs to restore]
                           getrows = True,  # [<None,True> => restore invalid rows]
                           getcols = True,  # [<None,True> => restore invalid cols]
                           )

        """
        if self.verbose is True:
            print 'restore_invalid() is working...\n'

        # Run the damon utility
        restore_invalid_out = dmn.utils._restore_invalid(locals())
        self.restore_invalid_out = restore_invalid_out

        if self.verbose is True:
            print 'restore_invalid() is done.'
            print 'Specified Damon outputs have been updated.\n'

        return None


    ##########################################################################

    def export(self,
               outputs,   # [['coord_out','base_est_out',...] => string list of desired datadict outputs]
               output_as = 'textfile',    # [<'textfile','hd5','pickle'> => type of output file]
               outprefix = 'aa',    # [string prefix to all file names, may be a path to a designated directory]
               outsuffix = '.csv',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
               delimiter = ',', # [<None,text delimiter, e.g. ',' or '\t'>]
               format_ = '%.60s',    # [<None, format code of cell contents>  => See numpy.savetxt() docs]
               obj_params = None,    # [<None,True> => export pickle file of Damon non-data __init__ parameters]
               ):
        """Export specified outputs as files.

        Returns
        -------
            export() returns None, but saves specified
            Damon outputs residing in memory to files in a
            designated directory.  These may be text files
            (human readable), hd5 (pytables) files (for very
            large arrays), or pickle files (for dictionaries).

            Optionally, the Damon.__init__ parameters can
            be saved off as a pickle file for future reference.

            Workflow (export() is always run last):
                Obj.coord(...)
                Obj.base_est(...)
                Obj.export(['coord_out','base_est_out'],...)

        Comments
        --------
            Having built a DamonObj and obtained results, it becomes
            necessary to save off some of these outputs as files.
            That is what export() does.  It can also save off
            useful Damon initialization parameters for future
            reference.

            If pytables was used, it moves specified datadicts
            containing PyTable arrays into a new PyTable file
            and deletes the old one, along with any temporary
            PyTable files that were created along the way.

            export() does not support all export scenarios. For
            those scenarios not supported by export(), it is always
            possible to use numpy's savetxt() function, Python's
            pickle (or better, cpickle) utility, or the tools.pytables()
            function.

            Although export() can be used to output coordinates, it
            is not the proper method for updating the coordinates
            in an item or person bank (stored as pickles).  Use
            bank() for that.

        Arguments
        ---------
            'outputs' is a string list of outputs created by running
            a DamonObj method.  To get a list of available outputs,
            type:

                my_dmnobj.__dict__.keys()

            Most items on that list with a '_out' suffix can be
            output as a file using export().  If they are datadicts,
            they can be exported in all three formats ('textfile,'hd5',
            'pickle').  Those that are not datadicts can be exported
            as "pickle" files.  datadicts containing pytables can be
            exported as 'hd5' files (though not as pickle files).

            Some of the outputs (e.g., summstat_out, coord_out) are
            not in datadict form, but in more complicated dictionaries.
            They can only be output in their complete form as pickles.
            However, you can output 'summStat_RowEnts' and 'summStat_ColEnts'
            as text files, as well as 'fac0coord' and 'fac1coord'.

            An important requirement is that each output in the list
            must be a string name enclosed in quotes, e.g., 'data_out'.

            Example:
                outputs = ['data_out','fac0coord','fin_est_out','finSE_out']

            ---------------
            "OutPutAs" specifies the output file format:

                output_as = 'textfile'   =>  Returns a text file whose
                                            rowlabels, collabels, and
                                            coredata are combined into one
                                            array and converted to string.
                                            Use the delimiter and format_
                                            arguments to describe how the
                                            file should be parsed and
                                            formatted.

                output_as = 'pickle'     =>  Returns a pickle file, a Python
                                            file format for storing Python
                                            dictionaries.  This is probably
                                            the fastest and most straightforward
                                            of the formats.

                output_as = 'hd5'        =>  Returns a file in Hierarchical data
                                            format, especially appropriate for
                                            storing very large arrays in a
                                            tree-style directory structure.  Use
                                            this if your Damon specified
                                            pytables.  It will store a datadict's
                                            rowlabels, collabels, and coredata
                                            as arrays in the group name specified
                                            by output.  All the outputs will go
                                            into the same pytables .hd5 file under
                                            separate "group" names. It will have
                                            the same name as that given by your
                                            Damon.pytables specification, minus
                                            the '_temp' suffix, plus the outprefix.

                                            You can also output regular arrays as
                                            PyTable files.

            ---------------
            "outprefix" is a string prefix that will preceed each of your
            output files, including the PyTable file.  In addition to
            organizing the outputs so that the results of the same
            analysis will all cluster together, it is the method used
            to save your files to a different folder/directory than
            your current working directory.  Example:

                outprefix = 'District1' =>  Yields filenames like:

                                            'District1_baseEst_out.csv'
                                            'District1_finEst_out.csv'

                                            (assuming outsuffix = '.csv'
                                            as described below).

                outprefix = '/Documents/MyOtherFolder/District1'
                                        =>  Saves the files to MyOtherFolder.

            ---------------
            "outsuffix" appends a string to the end of the filename,
            generally an extension.

                outsuffix = '.csv'      =>  Appends '.csv' to the filename.
                                            The choice is driven in part by
                                            the delimiter.  Comma-delimited
                                            files should have .csv extension.
                                            Tab-delimited files should have
                                            a .txt extension.  pytables should
                                            have a .hd5 extension (though this
                                            will be handled automatically).
                                            Pick what you want for pickle files.

            ---------------
            "delimiter" specifies how the columns in text file arrays
            should be delimited.  Comma (',') and tab ('\t') are the
            two most common.  Example:

                delimiter = ','

            Set delimiter = None if the format_ is not textfile.

            ---------------
            "format_" is used to specify the format of the cells in the
            output file.  The Python format syntax is a bit complex
            but is described in detail in the numpy.savetxt() documentation
            and associated links.  The % character is a format-now flag.
            Example:

                format_ = '%.10s'        =>  allocate space for 10 string
                                            characters per cell.

                format_ = '%10.5f'       =>  report as decimals with up to
                                            10 numbers to the left of the
                                            decimal point and 5 numbers to
                                            the right.

            The format_ specification is somewhat complicated by the fact
            that Damon does its own pre-formatting when it joins rowlabels,
            collabels, and coredata into one array.

            ---------------
            "obj_params" <None,True> if True, tells the function to export
            the initialization parameters of the Damon object (minus the data).
            This includes stuff like nheaders4cols, validchars, NanVal, and
            so on.  They are stored in their own pickle file under the name
            outprefix+'_ObjParams'.  You can use these later to build a new
            Damon.


        Examples
        --------


        Paste method
        ------------
            export(outputs,   # [['coord_out','base_est_out',...] => string list of desired datadict outputs]
                   output_as = 'textfile',    # [<'textfile','hd5','pickle'> => type of output file]
                   outprefix = 'aa',    # [string prefix to all file names, may be a path to a designated directory]
                   outsuffix = '.csv',  # [<'','.pkl','.csv','.txt','.hd5',...> => file extension]
                   delimiter = ',', # [<None,text delimiter, e.g. ',' or '\t'>]
                   format_ = '%.60s',    # [<None, format code of cell contents>  => See numpy.savetxt() docs]
                   obj_params = None,    # [<None,True> => export pickle file of Damon non-data __init__ parameters]
                   )


        """
        if self.verbose is True:
            print 'export() is working...\n'

        # Run the damon utility
        export_out = dmn.utils._export(locals())

        if self.verbose is True:
            print 'export() is done.\n'

        return None



    #############################################################################

    def flag(self,
             datadict = 'col_ents_out',  # [datadict or Damonobj]
             flag_rows = ('flag_items', {}),  # [<None, (func, kwargs), ('func', kwargs)> => function for flagging rows]
             flag_cols = None,  # [<None, func> => function for flagging cols]
             extract = None, # <None, {'rows':'NoneExcept', 'cols':'NoneExcept'}>]
             ):
        """Flags rows and columns that meet specified conditions.
        
        Returns
        -------
            Like extract(), flag() does not return None but returns a 
            dict while also adding self.flag_out to the Damon obj.  The
            output dict holds:

                {'rows': a list of row entities
                 'cols': a list of col entities
                 'datadict': the input datadict (for internal reference)
                 'extract': a datadict extract using 'rows' and 'cols'
                }
            
            Workflow:
                my_obj = dmn.Damon(...)
                my_obj.<methods>  # Run various Damon methods
                my_obj.flag()  # In a specified datadict, flag rows/cols
                my_obj.extract()  # Does on extract based on my_obj.flag()
        
        Comments
        --------
            flag() works in close cooperation with extract().  Whereas
            extract() creates a new datadict based on specified row and
            column entities, it does not actually identify what those
            row and column entities should be.  That's where flag() comes 
            in.  flag() allows the user to specify any criterion or 
            combination of criteria to identify rows and columns of interest 
            in a specified datadict or Damon object.  This is done by
            defining a function for flagging rows and columns or using the 
            default tools.flag_items() function.  (Flagging items is the 
            most pressing use-case.)
            
            flag() can be used both to return the desired row and column
            keys for insertion in other methods and to perform the
            actual extract itself or pass off its findings to the extract()
            method.  The default use-case -- flagging and extracting
            poorly performing items -- can be done without specifying
            any arguments for summstat(), flag(), or equate():
            
                d = dmn.Damon(...)
                d.<various methods>
                d.summstat()
                d.flag()
                d.extract()
                
                # Convert extract_out to a Damon object for easier access
                extract = dmn.Damon(d.extract, 'datadict', 'RCD_dicts_whole',
                                    verbose=None)
                print extract
            
            flag() outputs are especially useful for feeding into
            extract_valid() to remove items or persons that are 
            problematic for some reason.  It is also useful with coord()'s 
            anchors['Entities']['AllExcept'] parameter to prevent an
            item from being used as an anchor while also refreshing it.
        
        Arguments
        ---------
            "datadict" specifies a datadict or Damon object in which
            row or column entities are to be flagged.  It has to refer
            to a valid datadict.
            
                datadict = 'col_ents_out' (default)
                                =>  Use my_obj.col_ents_out, an output of
                                    the current Damon object.  When specified
                                    as a string, it is assumed that the
                                    datadict is an attribute of self, the
                                    current Damon object.
                
                datadict = my_obj.col_ents_out
                                =>  Same as above, but my_obj in this case
                                    doesn't have to be the current Damon
                                    object.
                
                datadict = your_obj.estimates_out
                                =>  Refer to the estimates output of your_obj.

            --------------
            "flag_rows" <None, (func, kwargs)> is a tuple for specifying a 
            function that returns an array of row keys that meet certain
            conditions.  Note that "row" in this context corresponds to
            items with various item statistics, as seen in the 'col_ents_out'
            file.
            
            When defining your function, the first argument is the datadict 
            specified above  after it has been converted to a Damon object 
            (which flag() does internally).  Call this Damon object "d".  It 
            provides key-based access to each row and column in d via its 
            core_row and core_col attributes.  Functions should adhere to the 
            following pattern, in which all arguments are "keyword arguments" 
            (stored in a dict).
            
            damon1.tools provides as a default a convenience function called 
            'flag_items' which you can use in lieu of rolling your own.
            
            Example functions:
                
            def mean_est(d):            # d = my_obj.estimates_out
                "List rows whose mean estimate is greater than 1.5 logits."
 
                row_means = tools.mean(d.coredata, axis=1, nanval=d.nanval)
                ix = row_means > 1.5
                return tools.getkeys(d, 'Row', 'Core')[ix]
            
            def fitting_persons(d, outfit=1.1):     # d = my_obj.row_ents_out
                "List persons with fit < outfit."
                
                row_fits = d.core_col['Outfit']
                ix = row_fits < outfit
                return tools.getkeys(d, 'Row', 'Core')[ix]
            
            def fitting_hi_performers(d, outfit=1.1, mean=2.5):
                "List persons with fit < outfit and mean score > 2.5 logits."
                
                row_fits = d.core_col['Outfit']
                row_means = d.core_col['Mean']
                ix = (row_fits < outfit) & (row_means > mean)
                return tools.getkeys(d, 'Row', 'Core')[ix]
            
            Examples of usage:
                flag_rows = None  
                                =>  Don't flag any rows.  Flag will output
                                    {...'rows':['All']}, meaning all rows
                                    are selected.  The ['All'] syntax is
                                    used in extract() to mean all entities.
                
                flag_rows = (mean_est, {})
                                =>  Get a list of row entities with means
                                    greater than 1.5 logits.  No extra args
                                    are required, but you still need to
                                    specify the empty dict {}.
                
                flag_rows = (fitting_persons, {})
                                =>  Get a list of persons with low misfit
                                    statistics.  Since the kwargs {} is
                                    empty, use the default criterion 
                                    outfit=1.1.
                
                flag_rows = (fitting_hi_performers, {'mean':3.0})
                                =>  List high performing students with outfit
                                    less than 1.1 (default, since not in
                                    the kwargs) and mean score greater than
                                    3.0 (overwrite the default 2.5).
            
                flag_rows = ('flag_items', {})  (default)
                                =>  The tools.flag_items() function is
                                    the default.  It flags items based on
                                    their reliability, point-biserial, and
                                    outfit statistics.  It requires that
                                    datadict = 'col_ents_out'.  The empty
                                    {} keyword dict instructs flag() to
                                    use the flag_items() defaults, but you
                                    can overwrite any of them.  See the
                                    tools.flag_items() docs for more info.
                                    You may also want to look at the code
                                    to help writing your own functions.
                                    
                                    Note that the function 'flag_items' is
                                    in quotes.  You could also specify
                                    tools.flag_items.  The quoted version
                                    is used as a default because specifying
                                    true function objects as defaults gets 
                                    messy during import time.  Whenever flag() 
                                    sees quoted functions, it assumes they 
                                    reside in the damon1.tools module.
                                    
            --------------
            "flag_cols" <None, (func, kwargs)> works just like flag_rows
            except that it is used to flag columns.  You probably won't
            need to use it as much.
            
            --------------
            "extract" <None, dict> is used to get flag() to extract the
            relevant rows and columns from the datadict itself rather than
            deferring it to equate().  In truth, there is probably nothing
            this option can provide that a subsequent run of the extract()
            method wouldn't do just as well.
            
                extract = None  =>  Don't extract the relevant rows/cols from
                                    the current datadict.
                
                extract = {'rows':'NoneExcept', 'cols':'NoneExcept'}
                                =>  For both rows and columns, extract just the
                                    flagged rows and columns.
                                    
                extract = {'rows':'AllExcept', 'cols':'NoneExcept'}
                                =>  For rows, extract all the rows EXCEPT those
                                    that were flagged.  Extract only the
                                    flagged columns.
                                    
                                    Similar logic applies when 'cols' is
                                    'AllExcept'.
                                    
                                    See the extract() docs for more info.
                                    
        Example of Iterative Item Flagging
        ----------------------------------
        The process of cleaning a test to fit the model (analysis of
        fit) involves analyzing the data, looking at the item stats to
        identify poorly performing items, removing them, running again,
        looking at the stats again (new items will show up as problematic),
        removing them, running again, and so on.  Damon.flag() in
        collaboration with Damon.extract_valid() can be used to do this
        programmatically, as shown below.
        
        Bear in mind that it is never safe to do analysis of fit without
        human supervision, especially with 1-dimensional models.  Every
        time an item is suspended from analysis, it subtly shifts the 
        definition of the construct or space.  It is also possible to 
        suspend too many items, weakening the reliability of the test at
        the expense of an unnecessary purity.  So if you automate this step,
        make sure that the remaining items represent the test you want
        and that you haven't weakened the overall test reliability.
        
        You can also filter out misfitting persons by a similar process,
        not shown here.  You would clean both the items and the persons,
        then anchor the remaining item parameters, then reintroduce the
        persons so that you can score them.
        
        In the code below, the key is the "rem" variable (stands for "remove").
        rem is the list of flagged items (which are rows in 'col_ents_out',
        columns in f2's data array).  This list accumulates every time you
        go through an analysis cycle.  It is then plugged back into the
        beginning of the cycle in the f2.extract_valid(rem_cols=rem) step.
        
        Expect to see a lot of items flagged in the first iteration (say, 10%), 
        a few in the second, maybe one or two on the third.  The loop stops
        iterating when no new items are flagged.  The resulting datafile
        (with the flagged items removed) can be considered reasonably
        "clean", with every item pulling its weight and more or less
        fitting into a common space.  To the degree this occurs, the resulting
        measures meet the condition of "objectivity".  Only then can we
        say that scores from linked test forms are truly "comparable".
        
        ###################
        f2 = dmn.Damon(...)
        changing = True
        rem = []
        
        while changing:
            rem_ = list(rem) # makes a copy of rem
            
            f2.extract_valid(minsd = 0.001, rem_cols=rem)  # Note "rem" here
            f2.standardize()
            f2.coord([[2]])
            f2.base_est()
            f2.base_resid()
            f2.base_ear()
            f2.base_se()
            f2.base_fit()
            f2.summstat()
            f2.flag()
            
            # rem augments, then is reused in extract_valid()
            rem.extend(f2.flag_out['rows'])  
            
            if len(rem) == len(rem_):
                changing = False

        Paste Method
        ------------
            flag(datadict = 'col_ents_out',  # [datadict or Damonobj]
                 flag_rows = ('flag_items', {}),  # [<None, (func, kwargs), ('func', kwargs)> => function for flagging rows]
                 flag_cols = None,  # [<None, func> => function for flagging cols]
                 extract = None, # <None, {'rows':'NoneExcept', 'cols':'NoneExcept'}>]
                 )        
        """

        # Run the damon utility
        flag_out = dmn.utils._flag(locals())
        self.flag_out = flag_out

        return flag_out




    #############################################################################

    def extract(self,
                datadict = 'flag_out',  # [data dictionary, e.g., self.data_out, self.coord_out]
                getrows = {'Get':'NoneExcept','Labels':'key','Rows':['All']}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                getcols = {'Get':'NoneExcept','Labels':'key','Cols':['All']}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                labels_only = None, # [<None or False,True> => extract only the relevant labels, not the core data]
                ):
        """Extracts data() array with specified rows and columns.

        Returns
        -------
            Unlike other Damon methods, extract() does not return
            None but returns a datadict as well as assigning it to the
            Damon object.  (It is best thought of as a utility function 
            that happens to double as a Damon method.)

            extract()'s datadict contains specified rows and
            columns and returns a corresponding extraction for rowlabels
            and collabels.  The elements of the new datadict are copies
            (not views) drawn from the source datadict, so they can be
            modified without affecting the source.
            
            Regular Workflow:
                my_obj = dmn.Damon(...)
                my_obj.<methods>  # Run various Damon methods

            Workflow with flag():
                my_obj.flag()  # flag rows/cols of interest
                extracted = my_obj.extract()  # refers to my_obj.flag_out,
                                              # overwrites getrows, getcols

        Comments
        --------
            extract() is used to extract specified rows and columns from
            a data dictionary or "datadict" -- a Python dictionary
            containing arrays for coredata, rowlabels, collabels,
            and other useful information.  (Every Damon method outputs
            a datadict which is assigned as an attribute to the Damon object.)
            The function offers a lot of flexibility in determining how
            rows and columns can be extracted.  They can be extracted
            by key ID, by row/column index, or by one or more attributes
            in the labels.

            extract() outputs are copies of the specified sections of
            the source data.  When one changes, the other does not follow
            suit.  The validchars specification and other specifications
            are, to the extent possible, left unchanged.  To get full
            keyword access to extract_out, you will want to convert it
            to a Damon object:
            
                my_obj.extract(...)
                d = dmn.Damon(my_obj.extract_out, 'datadict', 'RCD_dicts_whole',
                              verbose=None)
            
            extract() works in close cooperation with flag().  It extracts
            the row and column keys that flag() finds.  The default
            datadict is 'flag_out' so:
            
                my_obj.flag()
                my_obj.extract()
            
            require no extra arguments.  The extract getrows and getcols
            'Rows' and 'Cols' options are overwritten by 
            my_obj.flag_out['rows'] and my_obj.flag_out['cols'].
            
            Sorting of Keys
            ---------------
            It's important to note that (currently) the order of entities
            in the output extract does not necessarily match the order
            in which the keys were specified, as there is some sorting
            that goes on.

            Extra Header Labels
            -------------------
            If the rowlabels array contains columns in addition to the
            row keys, or the collabels array contains rows beyond the
            column keys, they will NOT automatically be included in the
            extract unless they are explicitly identified in the getrows
            and getcols parameters. However, the unique row and column
            keys will always be part of the extract.
            
            Iterative Extracts
            ------------------
            Note that every time you run extract() within the current
            Damon object, you overwrite equate_out.  To avoid confusion,
            you might not want to refer to obj.equate_out in this case but 
            assign the output to a variable you name:
            
                x = my_obj.extract(...)
            
            This comes up in several ways.  Sometimes, it is useful to
            do an iterative sequence of extractions using the same Damon
            object.  Here's an example from real life.  The test map file
            referred to is a file of items (as rows) and their attributes,
            which include domain, gradespan and form.  We want the items
            that correspond to a particular domain, gradespan, and form.
            
            # Load test maps into one item x attribute file. Items may repeat.
            tmap = load_testmaps(maps)
            
            # e.g., domain = 'Reading', gradespan = '5', form = 'Form_1'
            extracts = [('Domain', domain),
                        ('Grade', gradespan),
                        ('Form', form)]
            
            for x in extracts:
                tmap_x = tmap.extract(tmap,
                                      getrows = {'Get':'NoneExcept', 
                                                 'Labels':x[0], 
                                                 'Rows':[x[1]]},
                                                 
                                      getcols = {'Get':'AllExcept', 
                                                 'Labels':'key', 
                                                 'Cols':[None]}
                                      )
                tmap = dmn.Damon(tmap_x, 'datadict', verbose=None)
            
            Notice how tmap keeps overwriting itself with an ever narrower
            extract.  The final tmap is a Damon object of items on Form 1 for 
            just Reading and Grade 5.  Now I can look up information on just
            those items.
                

        Arguments
        ---------
            "datadict" is the data dictionary from which rows and columns
            are to be extracted.  Despite its name, it can also be a Damon
            object (extract() will convert it automatically).  All Damon
            method outputs are datadicts, but the datadict need not be from
            a Damon object.  It just needs to contain the required keys
            (go to help(dmn.Damon.__init__) for the list of required keys).
            
            datadict can also receive a string reference to a datadict in 
            the current Damon object.

            Examples:

                datadict = self.data_out
                                    =>  Refers to the starting data array of
                                        the current Damon object.
                                        
                datadict = another_obj.estimates_out
                                    =>  Doesn't have to be the current Damon
                                        object.
                datadict = 'base_se_out'
                                    =>  If in quotes, assumes that base_se_out
                                        belongs to the current Damon object.
                
                datadict = 'flag_out'
                                    =>  Use the datadict specified in
                                        my_obj.flag(datadict).  Overwrite
                                        the getrows['Rows'] and getcols['Cols']
                                        parameters with those in flag_out.
                                        
                                        This is the default.
                                        
            ---------------
            "getrows" specifies which row entities or attributes should
            be extracted from an array.  getrows is a dictionary with
            three fields:  'Get', 'Labels', and 'Rows':

                getrows = {'Get':<'AllExcept','NoneExcept'>,
                           'Labels':<'key',int,'index'>,
                           'Rows':<KeyIDs,Attributes,Indices>
                           }

            All indices refer to the WHOLE array, so make sure to
            take into account rowlabels and collabels when specifying a
            row or column integer.  Counting starts from 0.  The domain
            of rows and columns that can be extracted comprises the whole
            array, so you can extract from rowlabels and collabels, as
            well as coredata.

            Note that both:

                {'Get':'AllExcept','Labels':'keys','Rows':[None]} and
                {'Get':'NoneExcept','Labels':'keys','Rows':['All']}

            return ALL row entities.

            Note also that the Labels options 'key' and 'index' are
            special words for extract().  If they appear as keys in
            your labels, they will not be used in their special sense
            but as ordinary labels.

            getrows =
                {'Get':'AllExcept'      =>  Extract ALL rows except the ones
                                            specified in 'Rows':[...] .

                 'Get':'NoneExcept'     =>  Extract NONE of the rows except
                                            those specified in 'Rows':[...] .

                 'Labels':'key'         =>  The values listed in in 'Rows':[...]
                                            refer to rowlabel keys (unique row
                                            identifiers).

                 'Labels':<int>         =>  An integer signifying the column that
                                            contains the values listed in 'Rows':[...] .
                                            Counting starts at 0 from the left of
                                            the whole array. This makes it possible
                                            to extract on the basis of row
                                            attributes like gender or age, or on
                                            values in the core data.  More
                                            than one attribute can be specified.

                'Labels':<str>          =>  A string key label signifying the column that
                                            contains the values listed in 'Rows':[...] .
                                            This is an alternative key-based way to
                                            specify the column containing row attributes.
                                            However, it won't work if these keys are
                                            not string.  If the keys are 'key' or 'index',
                                            extract() will interpret them as labels,
                                            not in their special sense.

                 'Labels':'index'       =>  This means that the values in
                                            'Rows':[...] are not keys or labels but
                                            an integer index of rows to select (or
                                            exclude), where the counting starts
                                            with the first row (in the whole array,
                                            including labels) equal to 0.

                 'Rows':[None]          =>  If 'Get' is 'AllExcept', [None] means that
                                            all rows will be used.  If 'Get' is
                                            'NoneExcept', [None] means that no rows
                                            will be used (whatever that means).

                 'Rows':['All']         =>  Opposite of 'Rows':[None] above.

                 'Rows':['ID1','ID2',...]
                                        =>  This is the list of row entities to be
                                            extracted based on their ID.  It assumes
                                            'Labels':'key'.

                 'Rows':['Cats','Dogs',...]
                                        =>  This is the list of row attributes on
                                            the basis of which rows are to be selected
                                            (or excluded).  This says to select or
                                            exclude 'Cats' and 'Dogs'.  It assumes, e.g.,
                                            'Labels':'Species' or 'Labels':2 .

                 'Rows':[0,1,4,7]       =>  This is the index giving row numbers to
                                            be extracted.  It assumes 'Labels':'index' .
                }

            --------------
            "getcols" is the same as getrows except that it specifies columns.

            Note that both:

                {'Get':'AllExcept','Labels':'keys','Cols':[None]} and
                {'Get':'NoneExcept','Labels':'keys','Cols':['All']}

            return ALL column entities.

            getcols =
                {'Get':'AllExcept'      =>  Extract ALL columns (cols) except those
                                            specified in 'Cols':[...] .

                 'Get':'NoneExcept'     =>  Extract NONE of the cols except
                                            those specified in 'Cols':[...] .

                 'Labels':'key'         =>  The values listed in 'Cols':[...]
                                            refer to collabel keys (unique column
                                            identifiers).

                 'Labels':<int>         =>  An integer signifying the row that
                                            contains the values listed in 'Cols':[...] .
                                            Counting starts at 0 from the top of
                                            the whole array. This makes it possible
                                            to extract on the basis of column
                                            attributes like 'Spelling' or 'Vocab', or on
                                            values in the core data.  More
                                            than one attribute can be specified.

                'Labels':<str>          =>  A string key label signifying the row that
                                            contains the values listed in 'Cols':[...] .
                                            This is an alternative key-based way to
                                            specify the row containing column attributes.
                                            However, it won't work if these keys are
                                            not string.  If the keys are 'key' or 'index',
                                            extract() will interpret them as labels,
                                            not in their special sense.

                 'Labels':'index'       =>  This means that the values in
                                            'Cols':[...] are not keys or labels but
                                            an integer index of cols to select (or
                                            exclude), where the counting starts
                                            with the first col equal to 0.

                 'Cols':[None]          =>  If 'Get' is 'AllExcept', [None] means that
                                            all cols will be used.  If 'Get' is
                                            'NoneExcept', [None] means that no columns
                                            will be used (whatever that means).

                 'Cols':['All']         =>  Opposite of 'Cols':[None] above.

                 'Cols':['ID1','ID2',...]
                                        =>  This is the list of column entities to be
                                            extracted based on their ID.

                 'Cols':['Spelling','Vocab',...]
                                        =>  This is the list of col attributes on
                                            the basis of which cols are to be selected
                                            (or excluded).  This says to select or
                                            exclude 'Spelling' and 'Vocab' items.

                 'Cols':[0,1,4,7]       =>  This is the index giving col numbers to
                                            be extracted.  It assumes 'Labels':'index'.
                  }

            --------------
            "labels_only" <True, False>


        Examples
        --------
        >>>  my_dmnobj.extract(self.data_out,
                                getrows = {'Get':'AllExcept','Labels':'key','Rows':[None]},
                                getcols = {'Get':'AllExcept','Labels':'key','Cols':[None]},
                                )
                =>  Extract from the data_out datadict (the initial formatted
                    data dictionary) all row and column entities.

        >>>  my_dmnobj.extract(self.fin_est_out,
                                getrows = {'Get':'AllExcept','Labels':'key','Rows':['Marc','Antony']},
                                getcols = {'Get':'NoneExcept','Labels':2,'Cols':['Math','Language']},
                                )
                =>  Extract from the fin_est_out datadict a range comprising
                    all persons EXCEPT for 'Marc' and 'Antony' and ONLY those
                    items that are labeled 'Math' and 'Language' in row 2 of
                    collabels (counting from 0).

        >>>  my_dmnobj.extract(self.fin_est_out,
                                getrows = {'Get':'AllExcept','Labels':'index','Rows':[0,5,6,10]},
                                getcols = {'Get':'NoneExcept','Labels':'key','Cols':['Item1','Item2']},
                                )
                =>  Extract from the fin_est_out datadict a range comprising
                    rows 0, 5, 6, and 10 and Item1 and Item2.


        Paste method
        ------------
            extract(datadict,  # [data dictionary, e.g., self.data_out, self.coord_out]
                    getrows = {'Get':'NoneExcept','Labels':'key','Rows':['All']}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Rows':[<None,keys,atts,index>]}]
                    getcols = {'Get':'NoneExcept','Labels':'key','Cols':['All']}, # [{'Get':<'AllExcept','NoneExcept'>,'Labels':<'key',1,2,...,'index'>,'Cols':[<None,keys,atts,index>]}]
                    labels_only = None, # [<None or False,True> => extract only the relevant labels, not the core data]
                    )
        """
        # No messages because extract() is sometimes used iteratively.

##        if self.verbose is True:
##            print 'extract() is working...\n'

        # Run the damon utility
        extract_out = dmn.utils._extract(locals())
        self.extract_out = extract_out

##        if self.verbose is True:
##            print 'extract() is done.  There is no extract_out attribute.\n'

        return extract_out



    ##########################################################################

    def merge(self,
              source,    # [datadict FROM which rows or columns are to be extracted]
              axis = {'target':0,'source':0}, # [{'target':<0,1>,'source':<0,1>} => 0 means merge ents are rows, 1 means columns]
              targ_data = True,    # [<None,True> => include target data]
              targ_labels = True,    # [<None,True> => include target labels
              source_ids = None,  # [<None,True> => add source IDs to target IDs]
              nanval = -999,    # [Value to assign to missing fields.]
              ):
        """Extract rows or columns from source and append to current Damon.

        Returns
        -------
            merge() returns merged tables and saves an output datadict
            to my_obj.merge_out.  This is a datadict that contains
            data from a source datadict ("source") merged to a
            Damon object called the "target".  Thus, to merge two
            datadicts, one of them (the target) must first be
            initialized as a Damon object.

            Workflow:
                merged_obj = targ_obj.merge(source,...)

        Comments
        --------
            merge() is used to merge a datadict to a Damon object in
            terms of common row or column identifiers.  The first is
            the current Damon object and is called the target.
            It is the Damon object TO which data will be added.
            The second is called the source; it is the Damon object
            FROM which data is to be extracted and merged into
            the target.  The merge is done in terms of common
            row or column unique identifiers.  Thus,

                targ_obj.merge(SourceDataDict,...)

            The merge() method has some similarities with the mergetool()
            function in the tools module, but there are important
            differences.  mergetool() merges only arrays, and the
            result is an array.  This forces the row and column
            identifiers to be of the same type as the core data.
            merge(), on the other hand, is used to merge datadicts.
            This allows the row and column identifiers to be of
            a different type than the data they access.

            merge() outputs the merged datadict as an output
            attribute (my_obj.merge_out) where source data is appended
            to either the rows or columns of the target data.  The
            merge applies also to the rowlabels and collabels arrays.
            merge() includes an option to omit the target data and
            append only source data, essentially overwriting the
            target data with applicable source data.

            If multiple merges are performed within a given Damon object,
            the most recent Obj.merge_out will overwrite the previous
            ones.

            If an ID in the target is not found in the source, the
            corresponding cells are filled with nanval.

            The source and target arrays are not required to be
            aligned.  In other words, the target IDs may mark rows
            in the target datadict and columns in the source
            datadict.

            Row and column labels are merged much like the core data,
            except that the target unique row/column keys are always
            retained as the official unique keys (even when targ_labels
            is set at None), with source keys appended as necessary.

            merge() issues a warning whenever row and column keys
            are duplicated, but does not attempt to change them.
            Duplicates can automatically be renamed by converting
            the merge outputs into a Damon object using the validchars
            parameter.

            General Note
            ------------
            merge() is a complicated method -- there are many permutations,
            lots of ways to confuse it.  Make the inputs as simple as
            possible.  Check all outputs carefully.


        Arguments
        ---------
            "source" is the datadict FROM which data are to be extracted.

            --------------
            "axis" specifies the axis containing the linking entity
            IDs for both the target and source datadicts.

                axis = {'target':0,'source':0}
                                =>  The linking entities correspond
                                    to rows in both the source and
                                    the target arrays.

                axis = {'target':1,'source':0}
                                =>  The linking entities correspond
                                    to columns in the target array and
                                    rows in the source array.

            --------------
            "targ_data" <None, True> tells whether to include the core
            data from target datadict in the merged array.

                targ_data = None     =>  Do not include core target data.

                targ_data = True     =>  Do include core target data
                                        in the merged array.

            If targ_data is True, the relevant source rowlabels and
            collabels are reported alongside their target equivalents.
            Otherwise, just the target rowlabels/collabels are
            reported.

            --------------
            "targ_labels" <None, True> tells whether to append the target
            labels to the source labels.  However, regardless of how
            targ-labels is set, the unique target keys are retained
            in the labels as the primary key.

                targ_labels = True  =>  Append target labels to
                                        source labels.

                targ_labels = None  =>  Do not append the extract target
                                        label fields, but the tarket
                                        keys are retained.

            --------------
            "source_ids <None, True> tells whether to add non-overlapping
            source IDs (not already in the list of target IDs) to the list of
            target IDs to which data will be appended.  This makes
            it possible to "grow" the data set to include all unique
            IDs, both source and target, in the merged file.

                source_ids = True =>  Add non-overlapping source IDs to
                                        target.

                source_ids = None =>  Do not add non-overlapping source IDs.

            --------------
            "nanval" is the Not-a-Number-Value to be assigned to missing
            cells or to cells corresponding to target IDs that lack
            representation in the source array.  nanval must be numerical.

            If a target ID is missing among the source IDs, NaNVals are
            assigned to that target ID.

        Examples
        --------

            [under construction]


        Paste method
        ------------
            merge(source,    # [datadict FROM which rows or columns are to be extracted]
                  axis = {'target':0,'source':0}, # [{'target':<0,1>,'source':<0,1>} => 0 means merge ents are rows, 1 means columns]
                  targ_data = True,    # [<None, True> => include target data]
                  targ_labels = True,    # [<None,True> => include target labels
                  source_ids = None,  # [<None,True> => add source IDs to target IDs]
                  nanval = -999,    # [Value to assign to missing fields.]
                  )

        """
        if self.verbose is True:
            print 'merge() is working...\n'

        # Run the damon utility
        merge_out = dmn.utils._merge(locals())
        self.merge_out = merge_out

        if self.verbose is True:
            print 'merge() is done -- see my_obj.merge_out'
            print 'Contains:\n',self.merge_out.keys(),'\n'

        return None




    #############################################################################

    def transpose(self,
                  datadict = None, # [<None, datadict>, e.g., self.data_out, self.coord_out]
                  ):
        """Transpose datadict matrices.

        Returns
        -------
            Unlike other Damon methods, transpose() does not return
            None.  Nor does it assign transpose_out() to the Damon object.
            (As a data utility, it is used iteratively within the Damon object
            and is best handled like a function.)

            transpose() returns a datadict where coredata, rowlabels,
            and collabels have been transposed.  The label header fields
            are switched (e.g., nheaders4rows becomes nheaders4cols
            and vice versa).  nanval, and validchars stay the same.

            Important Note
            --------------
            The Damon.validchars attribute is only allowed to
            describe columns, hence there is no way at present to
            transpose it or apply it to a transposed array.  Any
            functions for a transposed datadict should not rely
            on the validchars specification, or your function needs
            to include code to build validchars internally.

            Workflow:
                transposed = my_dmnobj.transpose(...)

        Arguments
        ---------
            "datadict" is a DamonObj-formatted data dictionary
            such as those output by Damon methods, but
            any datadict containing the required elements will
            work.  Example:

                datadict = self.data_out, or
                datadict = MyDataDict
                datadict = None is equivalent to self.data_out.

        Examples
        --------


        Paste method
        ------------
            transpose(datadict, # [<None, datadict>, e.g., self.data_out, self.coord_out]
                      )
        """

        if self.verbose is True:
            print 'transpose() is working...\n'

        # Run the damon utility
        transpose_out = dmn.utils._transpose(locals())

        if self.verbose is True:
            print 'transpose() is done.  There is no transpose_out attribute.\n'

        return transpose_out



    ##########################################################################

    def to_dataframe(self,
                     datadict = 'data_out'  # [data dictionary, e.g., self.data_out, self.coord_out]
                     ):
        """Convert a datadict to a Pandas dataframe.
        
        Returns
        -------
            to_dataframe() returns a Pandas dataframe from a Damon
            datadict.  It does not assign an output attribute to the 
            Damon object or depend on it, so it is best thought of
            as a generic utility.
        
        Comments
        --------
            Pandas dataframes have become the go-to data structure
            for Python programmers dealing with labeled numerical
            data.  Had it been available at the time, Damon would 
            have been written on top of Pandas instead of evolving
            its own data utilities.
            
            to_dataframe() performs a simple assignment of Damon
            row keys, column keys, and coredata to Pandas index,
            columns, and data attributes.  Although most labeled
            data arrays should go back and forth between the two
            formats reasonably well, there is no guarantee that
            they will do so, and there are some cases where information
            will be dropped.  
            
            This mainly has to do with the difference between Damon 
            "rowlabels" and the Pandas "index".  Damon rowlabels (and 
            collabels) are used to store extra information, generally
            non-numerical, about each entity; they are not necessarily
            used for indexing.  Pandas indexes (and columns) are
            specifically intended for indexing.  Therefore, if a Damon
            datadict's rowlabels or collabels contain extra fields 
            (beyond row and column keys), those fields will not be
            carried over to the Pandas dataframe.  One might expect
            them to be appended to the core data array, but it was
            felt this could raise complications in trying to analyze
            the core data in Pandas, so Damon isn't going there yet.
            
            Another important difference is that Pandas promotes
            separate dtypes for each column; Damon requires that
            its coredata array all be of one type.  It deals with
            the presence of string data either by having the user
            move the string fields into rowlabels or collabels or
            by either scoring or parsing the strings to convert them 
            to numbers.  In other words, Damon was specifically
            built for numerical analysis of arrays, not as a general
            data engine.
            
            Another difference is that Pandas supports MultiIndexing,
            indexes composed of combinations of indexes.  Damon doesn't,
            really.  However, Damon can read in a multi-indexed dataframe
            by converting the multi-index into a single index.  
            to_dataframe() can then convert the single index back into
            a multi-index for Pandas.
            
            To smooth the conversion between the two formats, use
            Pandas to restrict dataframes to the rowlabel, collabel,
            coredata scheme that Damon uses, and have rowlabels and
            collabels only contain keys, not supplementary information.
            
            Note:  the method should generate, where possible, a
            df.index.name attribute for labeling index columns in the
            dataframe, for both single and multi-indexes.  However,
            for some reason it doesn't appear in the Pandas display
            in the multi-index case.  It is there nonetheless.
        
        Arguments
        ---------
            "datadict" is any Damon datadict, specified either using 
            obj.my_datadict syntax or 'my_datadict' string syntax, in
            which it is assumed to refer to the current Damon object.
            
                datadict = d.data_out 
                                    =>  Convert the initial data_out attribute
                                        of Damon object "d" into a Pandas
                                        dataframe.
                datadict = 'data_out'
                                    =>  Means the same as above.
                                    
                datadict = e.col_ents_out
                                    =>  Convert the col_ents_out datadict of
                                        Damon object "e" into a dataframe.
        
        Examples
        --------
        
            
        Paste Method
        ------------
            to_dataframe(datadict  # [data dictionary, e.g., self.data_out, self.coord_out]
                         )

        """

        if self.verbose is True:
            print 'to_dataframe() is working...\n'

        # Run the damon utility
        to_dataframe_out = dmn.utils._to_dataframe(locals())

        if self.verbose is True:
            print 'to_dataframe() is done.  There is no to_dataframe_out attribute.\n'

        return to_dataframe_out


    ##########################################################################

    def combokeys(self,
                  axis = 'Row', # [<'Row' => refer to row labels, 'Col' => refer to col labels>]
                  condarr1 = None,  # [None; ['np.where(Condition(s),then,else)'; condition(s) readable by np.where() function, to create new array 1]
                  condarr2 = None,  # [None; 'np.where(Condition(s),then,else)'; Condition(s) readable by np.where() function, to create new array 2]
                  condarr3 = None,  # [None; 'np.where(Condition(s),then,else)'; Condition(s) readable by np.where() function, to create new array 3]
                  filler = 0    # [filler to add into "cracks" created by expanding the array]
                  ):
        """Combine label elements to create arrays of keys and append to data.

        Returns
        -------
            Current Damon with augmented rowlabels or collabels.

            Workflow:
                MyAugmentedObj = my_obj.combokeys(axis,condarr1,...)

        Arguments
        ---------
            "combokeys()" is a way to create new labels for rows
            or columns based on values of existing labels.
            Sometimes row or column keys or flags need
            to be added to rowlabels or collabels to score a
            test or run SummStat.  For instance, the collabels
            output of parse() may include one row for Item IDs
            and another for the category label being evaluated
            and these need to be combined to produce a new
            identifier for each column, e.g. ['1_a','1_b','1_c',
            '2_a','2_b','2_c',...].  combokeys() provides
            a tool for doing that.

            -------------
            "axis" ('Row' or 'Col') specifies which type of
            label you will be augmenting, rowlabels or collabels.
            combokeys() can only augment one axis at a time.

            -------------
            "condarr1" specifies one or more conditions in terms
            of which the target array (rowlabels or collabels) is
            to be used to create new labels.  Say, axis = 'Col',
            so we are augmenting the collabels array.  Here is
            an example array for Damon D:

                D['whole']=
                [['0' '2' '1'   '2'   '2']
                 ['2' '0' '1'   '2'   '0']
                 ['1' '2' '1.0' '0.0' '2.0']
                 ['1' '0' '2.0' '2.0' '0.0']
                 ['1' '1' '0.0' '2.0' '0.0']]

                D['collabels']=
                [[0 2 1 2 2]
                 [2 0 1 2 0]]

                D['rowlabels']=
                [[0 2]
                 [2 0]
                 [1 2]
                 [1 0]
                 [1 1]]

            The collabels are integer values in the two rows across
            the top.  We want to create a new row in collabels that
            is '9' when the first and second columns are '2', zero
            otherwise.  How do we do that?   We write a numpy 'where'
            statement as follows, enclosing it in quotes:

            condarr1 = 'np.where(np.logical_and(collabels[0,nheaders4rows:] == 2,
                                    collabels[1,nheaders4rows:] == 2),9,0)'

            Numpy's where() function uses the syntax: np.where(Cond,Val,Else).
            We put it in quotes so that it goes into the function as
            a string where it will be evaluated with Python's eval() function.
            (See note about eval() below.)  The statement reads: "Where
            the first row of collabels (excluding its row headers) equals
            2 and the second row (excluding its row headers) is also
            2, put a 9.  Otherwise, put a zero."

            Numpy's logical_and() function allows us to specify multiple
            conditions.

            Important
            ---------
            The CondArr arguments recognizes the collabels and nheaders4rows
            variable names.  It also recognizes rowlabels, nheaders4cols,
            DatColLabels and DatRowLabels (that portion of rowlabels and
            collabels corresponding to coredata, removing the need for
            the nheaders4rows and nheaders4cols references), and coredata --
            in short, all variables that are defined inside the combokeys()
            method.  Any other arrays won't be recognized.

            The output is as follows (actually, you need an extra
            step to get the 'whole', but you get the idea):

                Out['whole']=
                [['0' '2' '1'   '2'   '2']
                 ['2' '0' '1'   '2'   '0']
                 ['0' '0' '0'   '9'   '0']
                 ['1' '2' '1.0' '0.0' '2.0']
                 ['1' '0' '2.0' '2.0' '0.0']
                 ['1' '1' '0.0' '2.0' '0.0']]

                 Out['collabels']=
                [[ 0.  2.  1.  2.  2.]
                 [ 2.  0.  1.  2.  0.]
                 [ 0.  0.  0.  9.  0.]]

            -------------
            "condarr2" and "condarr3" employ the same syntax,
            but they allow you to layer additional conditions
            on top of the first condition to create up to three
            additional rows of labels (if axis = 'Col'), or columns
            of labels (if axis = 'Row').

            Hint:  If you want to use the new labels created by
            condarr1 to set conditions for condarr2 and condarr3,
            replace 'collabels' with 'DatColLabels', which is the
            variable name inside the function.

            IMPORTANT NOTE
            --------------
            The CondArr statements only allow augmentations to
            collabels or rowlabels based on labels that pertain
            to coredata.  They don't operate on the entire collabels
            or rowlabels array.  Since the augmentations only apply
            to the labels adjoining or labeling the coredata, an
            extra "filler" value is required to fill in the cracks
            in the top-left corner of the whole data array.

            -------------
            "filler" is the value which should be used to fill
            in the cracks of the top-left corner of the whole array
            created by augmenting the coredata portion of the
            labels.  The filler value should be of a type compatible
            with the existing row labels or column labels.  Don't
            make filler a period '.', if the rowlabels and collabels
            arrays are integers.

            CAUTIONARY NOTE ON EVAL()
            -------------------------
            combokeys() uses Python's powerful eval() function to
            evaluate the CondArr statements.  Unfortunately, eval()
            can be used to execute bad or dangerous code if it
            is not properly constrained.  To address this issue,
            combokeys() won't allow CondArr to contain references
            to any of Python's __builtin__ functions and only allows
            references to attributes of Damon and Numpy.  Nonetheless,
            combokeys() should not be controllable by the end-user.

        Examples
        --------




        Paste method
        ------------
            combokeys(axis = 'Row', # [<'Row' => refer to row labels, 'Col' => refer to col labels>]
                      condarr1 = None,  # [None; 'np.where(Condition(s),then,else)'; condition(s) readable by np.where() function, to create new array 1]
                      condarr2 = None,  # [None; 'np.where(Condition(s),then,else)'; Condition(s) readable by np.where() function, to create new array 2]
                      condarr3 = None,  # [None; 'np.where(Condition(s),then,else)'; Condition(s) readable by np.where() function, to create new array 3]
                      filler = 0    # [filler to add into "cracks" created by expanding the array]
                      )

        """
        if self.verbose is True:
            print 'combokeys() is working...\n'

        # Run the damon utility
        comboKeys_out = dmn.utils._combokeys(locals())
        self.comboKeys_out = comboKeys_out

        if self.verbose is True:
            print 'combokeys() is done -- see my_obj.comboKeys_out'
            print 'Contains:\n',self.comboKeys_out.keys(),'\n'

        return None


    

##########################################################################################################
# Core Functions

def create_data(nfac0,  # [Number of facet 0 elements -- rows/persons]
                nfac1,  # [Number of facet 1 elements -- columns/items]
                ndim,   # [Number of dimensions to create]
                seed = None,  # [<None,int,{'Fac0':<None,int,array>,'Fac1':<None,int,array>}>  => generates coordinates]
                facmetric = [4,-2],  # [[m,b] => rand() * m + b, to set range of facet coordinate values]
                noise = None, # [<None, noise, {'Rows':<noise,{1:noise1,4:noise4}>,'Cols':<noise,{2:noise2,5:noise5}> => add error to rows/cols]
                validchars = None,   # [<None, ['All',[valid chars]]; or ['Cols', {1:['a','b','c','d'],2:['All'],3:['1.2 -- 3.5'],4:['0 -- '],...}]> ]
                mean_sd = None, # [<None, ['All',[Mean,SD]], or ['Cols', {1:[Mean1,SD1],2:[Mean2,SD2],3:'Refer2VC',...}]> ]
                p_nan = 0.0,  # [Proportion of cells to make missing at random]
                nanval = -999.,  # [Numeric code for designating missing values]
                condcoord = None,  # [< None, 'Orthonormal'>]nheaders4rows = 1,  # [Number of header column labels to put before each row]
                nheaders4rows = 1,  # [Number of header column labels to put before each row]
                nheaders4cols = 1,  # [Number of header row labels to put before each column]
                extra_headers = 0,  # [<0, int, {'0':0.25, '1':0.75}> => If headers > 1, range of ints for labels, randomly assigned or in blocks]
                input_array = None,   # [<None, name of data array, {'fac0coord':EntxDim row coords,'fac1coord':EntxDim col coords}>]
                apply_zeros = None, # [<None, [row, {'sub1':[0,1,1],...}> => for each item group in row, where to apply zeros to coords]
                output_as = 'Damon',  # [<'Damon','datadict','array','textfile','dataframe','Damon_textfile','datadict_textfile','array_textfile'>]
                outfile = None,    # [<None, 'my_data.csv'> => name of the output file/path when output_as includes 'textfile'>]
                delimiter = None,    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '\t'>]
                bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]> ]
                bankf1 = None,  # [<None => no bank,[<'All', list of F1 (Col) entities>]> ]
                verbose = True, # [<None, True> => print useful information and messages]
                condcoord_ = None # [<None, condcoord args> deprecated, only for backward compatibility]
                ):
    """Create simulated model and noisy data objects analyzable by Damon.

    Returns
    -------
        create_data() returns a python dictionary where each of the following
        datasets is formatted either as a DamonObj, a datadict, an array,
        or a file, or combinations thereof.  'data' and 'model' may
        also be output as '.hd5' or pytables files to conserve memory.

        {'data':,       =>  simulated 'observed' data, with error and missing added
        'model':,       =>  simulated 'True' values, no error added
        'anskey':,      =>  answer key, when data are alpha (simulating multiple choice data)
        'fac0coord':,   =>  theoretical row coordinates
        'fac1coord':    =>  theoretical column coordinates
        }

        output_as controls the output format, which can be as an array,
        a file, a DamonObj, or a datadict.

        When validchars specifies string responses, outputs are
        in a "multiple choice" alpha response format.  An "answer key" is
        included.

        Numerical and alpha files are constructed according to
        a linear model, with noise added as desired:

            data[r,c] = Row[r] * Col[c] + noise[r,c]

        Cells can also be made missing.  These create
        the output called 'data'.

        'model' is like 'data', but without noise or missing
        cells.

        "bankf0" and "bankf1" are used to create artificial
        person or item banks to test the anchoring capability of coord()
        and rasch().  They save Python "pickle" files in the current
        working directory called 'bankf0' and 'bankf1'.

    Comments
    --------

        create_data() is a function for creating artificial datasets for
        Damon.  Compliant with the assumptions of the ALS
        decomposition, the data value of each cell is the dot product
        of its row and column coordinates, plus a specified level
        of random noise.  It is important to remember that not
        all real-world datasets follow these assumptions, and
        there is an option to import datasets created using
        different rules into the function.  Nonetheless, it is
        an excellent tool for learning Damon and its capacities
        under controlled conditions without having to worry about
        collecting external datasets.

        The function can produce ratio, interval, sigmoid, ordinal,
        and dichotomous data.  It also outputs artificial item banks.

        Aside from files and arrays, create_data() can create DamonObj's
        to which Damon methods can be directly applied.  It can also
        create datadicts and hd5 (pytables-style) files.

        Nominal data
        ------------
        create_data() can create nominal data (non-numerical responses)
        in a limited sense, to mimic multiple-choice test responses.  The
        way it works is that it first creates a dichotomous (0,1) matrix.
        Then, for each designated column, it converts the "1" into the
        "correct response" for that column (say "B", out of A, B, C).
        It converts the zeros into one of the remaining response options
        at random, A or C.  That means the "correct" responses have
        an underlying mathematical meaning whereas the incorrect responses
        have no meaning at all.

        In real life, incorrect response options tend not to be random,
        so the create_data() nominal option will tend to be "noisier"
        than real life.  Nonetheless, it is a reasonable and usable
        imitation of a multiple choice dataset.  Use the score_mc()
        method to convert the alpha characters to numerical.  parse()
        also works, but not as well.

        Nominal data can also take a different form where there is
        no "correct" option, e.g., an attitude survey.  Here, each
        response has its own definite mathematical meaning.  Currently,
        create_data() is unable to create such data.  However, you
        can work around this limitation to a degree by simply creating
        a dichotomous response matrix and treating each column as
        it it were an item response, rather than an item.  This is
        not quite an accurate representation of real-world nominal
        data since it does not require a "forced choice" between
        responses, but it will work okay for playing around.
        
        WARNING:  Because equate() automatically adds new construct
        identifiers to the column or row keys, its likely to throw
        an exception if the original keys are int type.  To avoid problems, 
        create_data() forces all keys to be strings such that
        new_obj.colkeytype = 'S60'.

    Arguments
    ---------
        "nfac0" is the number of row entities to be created.  (Row
        entities are considered to belong to "facet 0".)

        ---------------
        "nfac1" is the number of column entities to be created.
        (Column entities are considered to belong to "facet 1".)

        ---------------
        "ndim" is the number of dimensions to be used in creating
        the aritifical row and column coordinate arrays.

        ---------------
        "seed" controls the selection of random numbers when
        creating row and column coordinates.  In Python, "seeds"
        are integers identifying a specific unique set of random
        numbers.  If the seed is "None", the seed integer is chosen
        at random.

        seed supports using existing arrays for row (Fac0)
        and column (Fac1) coordinates.  Both types of coordinate
        arrays are 2-D nEntities x nDims.

            seed = None     =>  Use a new, unique set of random
                                starter coordinates whenever
                                executing the function.

            seed = 1        =>  For each run, use a particular
                                set of random coordinates which is
                                the same every time.

            seed = 2, 3, ... N
                            =>  Each seed integer labels a different unique
                                set of random coordinates.

            seed = {'Fac0':MyArray0,'Fac1':MyArray1}
                            =>  Instead of generating random numbers,
                                use numpy arrays MyArray0 and
                                MyArray1.

            seed = {'Fac0':1,'Fac1':None}
                            =>  Use seed 1 for Fac0, a random selection
                                for Fac1.

            seed = {'Fac0':2,'Fac1':MyArray}
                            =>  Use seed 2 for Fac0, MyArray for Fac1.

        WARNING: Damon's coord() function also uses seeds to build
        initial random number coordinate arrays, generally starting with
        seed = 1.  If you create a Damon object with seed = 1, and coord()
        is using the same seed, you may get unreasonable agreement in
        your results.  Yes, this has bitten me more than once.  To avoid
        this, try to specify seeds in create_data (e.g., see=100) that
        coord() is not likely to use (coord() starts with seed=1 and
        iterates up from that, unless otherwise specified).

        ---------------
         "facmetric" governs the spread of the created data values.
         It multiplies the generating coordinates by a number (m) and
         adds a number (b):

             facmetric = [m,b]

             facmetric = [4,-2]

                        =>  multiply each facet value by 4 and
                            add -2.

        IMPORTANT NOTE:  facmetric allows the user to model ratio
        (count-like) data by setting the b parameter to zero.  When
        this is done, the log function is applied to the model and
        data arrays (prior to any further transformations controlled
        by validchars) and the data are interpreted as "ratio" rather
        than "interval".  If b equals any value other than zero,
        the log function is not applied:

            facmetric = [4,0]
                        =>  The log is taken of the resulting data
                            and model arrays.

            facmetric = [4,-2]
                        =>  The log is NOT taken of the resulting data
                            and model arrays.

            facmetric = [4,0.0001]
                        =>  The log is NOT taken of the resulting data
                            and model arrays, but the coordinates all
                            range from (almost) zero to four.

        ---------------
        "noise" is a way to add random error to specified rows or
        columns.  It does so by multiplying a specified number ("noise")
        by a random number between -0.5 and 0.5.  It is important to note
        that this is just one way of modeling noise and does not describe
        all possible noise scenarios.  However, it is the only noise
        scenario supported by create_data() and works well for most
        purposes.  The syntax is:

            noise = None
                        =>  No noise (0.0) is added to the
                            model array.

            noise = noise (int/float)
                        =>  The specified integer or decimal is
                            multiplied by a random number between
                            -0.5 and 0.5 and added to the model
                            array.

            noise = {'Rows':<noise,{'RowX':noiseX,...}>,
                     'Cols':<noise,{'ColY':noiseY,...}>
                     }

                        =>  noise is added to specified row
                            and column entities, starting with rows.
                            If 'Rows' is a number and specific
                            row entities are not identified,
                            the noise is added to all rows equally.
                            Same with 'Cols'.  When specific
                            entities are identified, the specified
                            noise is added to just those entities.
                            Those row/column entities that are
                            not identified are assigned a default
                            level of noise of 0.0 except (in
                            the case of 'Cols' only) where noise
                            has already been added at the row
                            level.

            Note:  the row and column identifiers are the labels
            assigned by create_data() to each row/column.

        Examples:

            noise = 4.0
                        =>  Multiply by 4.0 a random number between
                            -0.50 and 0.50 and add it to the
                            whole model array.

            noise = {'Rows':4.0,'Cols':{'Col3':2.0,'Col5':6.0}}

                        =>  Create a noise array by first adding
                            4.0 to all rows of a zero array.  Then
                            add 2.0 to all the 'Col3' cells and
                            6.0 to all the 'Col5' cells ('Col5'
                            now has 4.0 + 6.0 = 10.0).  Then
                            multiply by a random number between
                            -0.50 and 0.50.  This is the noise
                            array.  Add the noise array to the
                            model array.

            noise = {'Rows':{'Row1':4.0,'Row2':5.0},'Cols':{'Col3':2.0,'Col5':6.0}}

                        =>  Create a noise array by first adding
                            4.0 to row entities 1 and 2 of a zero
                            array.  Then add 2.0 to all the 'Col3'
                            cells and 6.0 to all the 'Col5' cells.
                            Cell['Row1','Col3'] will be 4.0 + 2.0 = 6.0.
                            All non-specified cells will be zero. Then
                            multiply by a random number between
                            -0.50 and 0.50.  This is the noise
                            array.  Add the noise array to the
                            model array.

        ---------------
        "validchars" specifies a list or range of valid characters for
        the whole coredata array or for each of the individual columns
        in the coredata array.  (It does not apply to rows.)  The
        function transforms the model values (with noise added) into
        the range implied by the validchars argument.  It does this
        for the matrix as a whole or applies a different range for each
        column, allowing the function to create very heterogeneous
        datasets.  Note the limitations on "nominal" data discussed
        in the comments.

        validchars does a lot and is important and used throughout Damon.
        In the context of this function it controls the range and metric
        of the artificially generated data values.  It can also be used
        to create nominal data in the style of a multiple-choice
        test.  Regardless of metric, the "model" output automatically
        mirrors the "data" output column by column to facilitate comparison.

        The syntax is:

            ['All',[list/range of possible responses]] or
            ['Cols',{'Col1':[list/range of responses],'Col2':[list/range of responses],...}]

            The {} brace signifies a Python dictionary.

        Example 1:

        validchars = None

        means that the existing values and ranges will be
        accepted as they are and no transformations are
        desired.

        Example 2:

        validchars = ['All',['All']]

        means that "all" values are permitted for all columns.
        If the companion specification mean_sd equals None,
        the target means and standard deviations are set at the
        existing means and standard deviations of the created data
        (i.e., there are no transformations).

        Example 2:

            ['All',['a','b','c','d']]

        means for all columns in the data array, create data in the
        form of responses from 'a' to 'd' such that one of these
        responses signifies a "correct" response.  The "correct"
        response is chosen at random by the function and recorded
        in the anskey output as the "correct" response for that
        column.  The model output in this case consists
        not of letters but of the "true" (pre-noise) probability
        of a cell containing the "correct" response.  (Bear in mind
        we assigned the response termed "correct" to those cells
        with "observed" probability (including noise) greater than
        0.50 of "succeeding" on the item.)  Therefore, where Damon
        is successful in its predictions, cells containing the correct
        response for a column should have a model probability of
        success greater than 0.50.

        The create_data() function currently only supports the
        "answer key" paradigm for creating nominal responses, i.e. where
        there is one "correct" response and this response corresponds
        to success probabilities greater than 0.50, all other responses
        being assigned at random to the probabilities less than 0.50.
        In this paradigm, non-correct responses have no intrinsic meaning
        or correct probability, aside from being less than 0.50.

        To explore other nominal scenarios, you may need to create a
        dichotomous data set, then collapse groups of columns into a
        single column assigning each cell the most likely response
        category value.

        Example 3:

            ['All',[0,1]]
            ['All',['0 -- 1']]
            ['Cols',{1:[0,1],2:[0,1],3:['0 -- 1'],...}]

        means for all columns in the data array convert the continuous
        linear data (model + noise) into the equivalent of dichotomous
        responses.  Notice that the 'All' syntax and the 'Cols' syntax yield
        the same results.  (This differs from how validchars is used in
        other Damon functions, where 'All' and 'Cols' yield mathematically
        different results.)  Also notice that [0,1] means the same
        thing as ['0 -- 1'] so long as the 0 and 1 in the second case
        are integers and not floats (with decimal points).  ['0.0 -- 1.0']
        means that results will be in a continuous range from 0 to 1
        instead of being (0/1) integers.  Note:  It is important to
        type the dash properly; it is one space followed by two hyphens
        followed by one space, enclosed in single quotes ( ' -- ' ):
        (space, hyphen, hyphen, space).  Any deviation will cause an error.

        The underlying formula for converting to dichotomous involves
        standardizing, converting to probabilities, and rounding to
        0 or 1.

        Example 4:

            ['All',[0,1,2,3]]
            ['All',['0 -- 3']]

        means for all columns transform data into integers ranging
        from 0 to 3.

        Example 5:

            ['All',['0.0 -- ']]

        means for all columns transform the data into continuous
        values ranging from 0 to infinity.  This is a ratio scale
        and behaves differently from the model's default interval scale
        which ranges from -Infinity to +Infinity.  The conversions
        are done using a log function (log(ratio) = interval).
        This is helpful to model "count" data, which starts at 0.
        But in the case of counts, you would want to specify

            ['All',['0 -- ']]

        without the decimal point, to indicate that data should
        be rounded to the nearest integer.

        Example 6:

            ['Cols',{1:['1.0 -- 10.0'],2:['1.0 -- 10.0']}]

        means for columns 1 and 2 (in a 2-column array), make the
        data values range continuously from 1.0 to 10.0.  Note that
        relative to the untransformed metric, this is a "sigmoid"
        metric which crunches at the top and bottom of the scale.

        Example 7:

            ['Cols',{1:['a','b','c'],2:[0,1],3:['1.0 -- 5.0'],4:['All']}]

        means for Column 1 make the values be 'a', 'b', or 'c'; for
        Column 2 make the data dichotomous, for Column 3 make it
        range continuously from 1.0 to 5.0, and for Column 4 let the
        data range continuously from -Infinity to +Infinity (i.e., keep
        the model + noise values as they are).

        To refrain from any transformation of the model + noise
        data, use validchars = None.

        ---------------
        "mean_sd" is used to specify a target mean and standard
        deviation for each column, or for the array as a whole, when
        the data are on an interval or ratio scale.  It is used in
        conjunction with the validchars argument.  Take care to keep
        them consistent.  Where validchars specifies 'All',
        mean_sd should provide a mean and standard deviation.
        Where mean_sd specifies 'Refer2VC', validchars should
        have a list of valid characters or a range, not 'All'.

        When the data are on a ratio scale (ranging from 0 to
        infinity), the mean and standard deviation do not apply to
        the ratio values but to the log(ratio) values.  So to obtain
        a certain mean and SD on the ratio scale, enter the log(Mean) and
        log(SD) in the mean_sd argument.  This transformation
        is necessary because means and standard deviations are
        meaningful only on an interval scale.

        Options:

            mean_sd  = None  =>  The column means and standard
                                    deviations are left unchanged.

                        = ['All',[Mean,SD]]

                                =>  Make the array as a whole have
                                    a specified mean and standard
                                    deviation.  Column means/SDs will
                                    vary.

                        = ['Cols',[Mean,SD]]

                                =>  Make each column have the specified
                                    mean and standard deviation.

                        = ['All','Refer2VC'] or ['Cols','Refer2VC']

                                =>  No Means or SDs are specified.  Use
                                    the validchars specification to decide
                                    what to do.  If validchars = 'All' for
                                    the whole array or a column, the metric
                                    is left unchanged.

                        = ['Cols',{1:[Mean1,SD1],2:[Mean2,SD2],3:'Refer2VC',...}]

                                =>  Make Column1 have Mean1 and SD1.
                                    Make Column2 have Mean2 and SD2.
                                    For Column 3, do not make it have any
                                    mean or standard deviation, presumably
                                    because it is not in an interval or
                                    ratio metric.  Instead, specify the
                                    placeholder 'Refer2VC', which means
                                    look up the minimum and maximum
                                    values in validchars and use those
                                    to specify the range of the scale.
                                    If 'Refer2VC' is specified for a column
                                    that has not been assigned a min or max
                                    in validchars, a mean of 0 and SD of 1
                                    will automatically be assigned.

        ---------------
        "p_nan" is the proportion of cells to make missing at
        random.  It actually isn't a percent:

            p_nan = 0.10
                            =>  make 0.10 or 10% of cells randomly
                                missing.

        ---------------
        "nanval" is the Not-a-Number Value used to indicate
        a missing cells.  It has to be of the same type as the
        rest of the array.  Default is -999.0 .

        ---------------
        "condcoord" provides the option of making the row
        coordinates orthonormal, or not.  Options are:

            condcoord = <None, 'Orthonormal'>

        For 'Orthonormal', the matrix procedure is numpy's QR decomposition,
        where matrix A = QR, Q is the orthonormal transformation of A, and
        R is an upper-diagonal matrix that performs the transformation.
        Q is equivalent to a "modified Gram-Schmidt" orthogonalization
        of A.

        ---------------
        "nheaders4rows" is the number of header labels to insert to the
        left of the data to label rows.  Default is 1.

        ---------------
        "nheaders4cols" is the number of header labels to insert to the
        above the data to label columns.  Default is 1.

        ---------------
        "extra_headers", short for "extra header integer range",
        is a way of inserting a specified number of integer values as
        labels in the row and column headers, but it only applies to
        those rows or columns of the header labels that are in excess
        of the leading header containing the unique ID, and does not
        apply when nheaders4rows or nheaders4rows = 1.  This argument
        is used to test Damon functions that call on row or
        column attributes.  The argument can be used to assign headers
        randomly or in specified blocks.

            extra_headers = int
                            =>  tells the function to create and
                                assign headers at random.

            extra_headers = 0, 1, -1
                            =>  The 0, 1, and -1 specifications all
                                result in only one integer value (0)
                                as an attribute, so there's really no
                                point in using them.

            extra_headers = 2
                            =>  create extra header attributes
                                consisting of integers 0 and 1

            extra_headers = 3
                            =>  create extra header attributes
                                consisting of integers 0, 1, 2.

            extra_headers = -3
                            =>  create extra header attributes
                                consisting of 0, -1, -2.

            extra_headers = {'header':proportion, ...}
                            =>  tells the function to create
                                and assign headers in blocks.

            extra_headers = {'0':0.25, '1':0.75}
                            =>  Say there is an extra header row
                                for columns and that there are 100
                                columns, i.e., nfac1 = 100.  This
                                says assign '0' to the first 25
                                columns and '1' to the remaining
                                75 columns.

                                Make sure the proportions add to
                                1.0 and that they break the columns
                                cleanly into sections.

                                The same arrangement of '0's and '1's
                                will be applied to the row headers
                                if nheaders4rows > 1.

        ---------------
        "input_array" makes it possible to import a data array or
        a set of row and column coordinates generated outside the
        function, and use those to create the model outputs.  The
        noise, validchars, and other parameters are applied to
        the resulting model values to create a data array.  This
        makes it possible to experiment with arrays built with
        nonlinear functions, to set the size of each individual
        row and column coordinate (to model a range of person
        abilities and item difficulties, for instance), and to
        experiment with setting some coordinates to zero to
        model situations where entities do not participate in
        a common D-dimensional space.  Options:

            input_array = None   =>  Do not input (import) an
                                    array or set of coordinates.

            input_array = MyArray
                                =>  Input the MyArray numpy array
                                    as the core "model" data.  Do
                                    not include row or column
                                    labels.

            input_array = {'fac0coord':MyRowCoords,'fac1coord':MyColCoords}
                                =>  MyRowCoords and MyColCoords are
                                    two nEntities x nDimensions numpy
                                    arrays.  Their dot product
                                    becomes the "model" data.  Do
                                    not include row or column labels.

        NOTE:  When input_array is used, it automatically overwrites
        the nfac0, nfac1, and ndim parameters.  input_array does not
        support pytables, so output_as = 'hd5' becomes output_as =
        'Damon'.

        ---------------
        "apply_zeros" is used to simulate data with dimensionally
        distinct subspaces.  A given group of items is said to be
        "in the same space" if they have non-zero values on the
        same dimensions and zero values for all other dimensions.
        Two subspaces differ if, for one or more dimensions, one
        of the subspaces has non-zero values while the other has
        zeros.  Damon's matrix decomposition depends strongly on
        all items sharing a common space.  If they don't, the
        individual subspaces need to be analyzed separately and
        pull information from other subspaces using a different
        procedure.  This is done using the sub_coord() method.

        The format is:
            apply_zeros = [row, {'sub1':[zeros, ones], ...}]

            apply_zeros = None  =>  Do not apply zeros to
                                    coordinates.  All items resided
                                    in a common space.

            apply_zeros = [1, {'0':[0,1,1], '1':[1,0,1]}]
                                =>  Row 1 (counting from 0) contains
                                    subscale labels '0', '1'. This
                                    is controlled using the nheaders4cols
                                    and extra_headers args.

                                    For each '0' item, zero out
                                    the first dimension and keep
                                    the remaining dimensions as they
                                    are.

                                    For each '1' item, zero out the
                                    second dimension and keep the
                                    remaining dimensions as they are.

                                    The number of zeros and ones must
                                    equal the number of dimensions
                                    specified in ndim.

        ---------------
        "output_as" specifies whether to output created data as a
        Damon object, array, or file.  Options:

            'array'             =>  Output as a numpy array (includes labels).


            'Damon'             =>  Output created data as a fully formatted
                                    data object, equivalent to Damon().

            'datadict'          =>  Output created data as a "data dictionary"
                                    but not instantiated as an object.

            'dataframe'         =>  Output data as a Pandas dataframe

            'textfile'          =>  Output as a text file.

            'array_textfile'    =>  Output as an array and a text file.

            'Damon_textfile'    =>  Output as both a DamonObj and a text file.

            'datadict_textfile'  =>  Output as both a DamonObj and a text file.
        
            [WARNING: the 'hd5' option has been deprecated.]
            'hd5'           =>  Output using pytables in Hierarchical data
                                format_, suitable for large arrays that may
                                not fit in memory.  If this option is used, the
                                file name given in outfile must have
                                a .hd5 extension.  In addition to outputting
                                a .hd5 file, a datadict is returned.

            WARNING:  When 'hd5' is specified, not all create_data()
            functionality is preserved.  The following simplifications
            are imposed:

                *   The condcoord arg is ignored.
                *   The noise arg for specified rows is ignored.

        ------------------
        "outfile" is the name of the output file or path (if saving
        to a directory that is not the current working directory),
        if 'textfile' is specified in output_as.  Options:

            None        =>  output_as does not include 'textfile'.

            'MyFileName.txt', 'MyDocs/MyFileName.txt'

                        =>  Results are output to a  file with
                            the specified name or path.

            'MyFile.hd5' =>  Results are output as a pytables
                            'hd5' file (output_as = 'hd5').

        ---------------
        "delimiter" is the character used to delimit columns when
        a file is created.  When comma (',') is used, the file
        name should have a .csv extension.  When tab ('\t') is
        used, the file name should have a .txt extension.  Use
        None when no file is specified.

        NOTE:  Tab delimiters are safer as otherwise the
        validchars column in the answer key may accidentally be
        parsed.

        ---------------
        "bankf0", when not equal to None, automatically creates
        a pickle file called 'bankf0' to store the coordinates of
        specified facet 0 (row) entities:

            bankf0 = None
                            =>  do not create a facet 0 bank.

            bankf0 = ['All']
                            =>  create a pickle file called 'bankf0'
                                and store all the row coordinates
                                in it, assigned to the entity ID.

            bankf0 = [1,3,5]
                            =>  create a pickle file called 'bankf0'
                                and store the row coordinates for
                                entities 1, 3, and 5, assigned to
                                their entity ID's.

        ---------------
        "bankf1", when not equal to None, automatically creates
        a pickle file called 'bankf1' to store the coordinates of
        specified facet 1 (column) entities:

            bankf1 = None
                            =>  do not create a facet 1 bank.

            bankf1 = ['All']
                            =>  create a pickle file called 'bankf1'
                                and store all the column coordinates
                                in it, assigned to the entity ID.

            bankf1 = [10,13,15]
                            =>  create a pickle file called 'bankf1'
                                and store the column coordinates for
                                entities 10, 13, and 15, assigned to
                                their entity ID's.

        ---------------
        "verbose" <None, True> tells create_data() to print out
        useful information and messages.  It also passes the verbose
        parameter to downstream DamonObj's.

    Examples
    --------

    Paste function
    --------------
        create_data(nfac0,  # [Number of facet 0 elements -- rows/persons]
                    nfac1,  # [Number of facet 1 elements -- columns/items]
                    ndim,   # [Number of dimensions to create]
                    seed = None,  # [<None => randomly pick starter coordinates; int => integer of "seed" random coordinates>]
                    facmetric = [4,-2],  # [[m,b] => rand() * m + b, to set range of facet coordinate values]
                    noise = None, # [<None, noise, {'Rows':<noise,{1:noise1,4:noise4}>,'Cols':<noise,{2:noise2,5:noise5}> => add error to rows/cols]
                    validchars = None,   # [<None, ['All',[valid chars]]; or ['Cols', {1:['a','b','c','d'],2:['All'],3:['1.2 -- 3.5'],4:['0 -- '],...}]> ]
                    mean_sd = None, # [<None, ['All',[Mean,SD]], or ['Cols', {1:[Mean1,SD1],2:[Mean2,SD2],3:'Refer2VC',...}]> ]
                    p_nan = 0.0,  # [Proportion of cells to make missing at random]
                    nanval = -999.,  # [Numeric code for designating missing values]
                    condcoord = None,  # [< None, 'Orthonormal'>]
                    nheaders4rows = 1,  # [Number of header column labels to put before each row]
                    nheaders4cols = 1,  # [Number of header row labels to put before each column]
                    extra_headers = 0,  # [<0, int, {'0':0.25, '1':0.75}> => If headers > 1, range of ints for labels, randomly assigned or in blocks]
                    input_array = None,   # [<None, name of data array, {'fac0coord':EntxDim row coords,'fac1coord':EntxDim col coords}>]
                    apply_zeros = None, # [<None, [row, {'sub1':[0,1,1],...}> => for each item group in row, where to apply zeros to coords]
                    output_as = 'Damon',  # [<'Damon','datadict','array','textfile','dataframe','Damon_textfile','datadict_textfile','array_textfile'>]
                    outfile = None,    # # [<None, 'my_data.csv'> => name of the output file/path when output_as includes 'textfile'>]
                    delimiter = None,    # [<None, delimiter character used to separate fields of output file, e.g., ',' or '\t'>]
                    bankf0 = None,  # [<None => no bank,[<'All', list of F0 (Row) entities>]> ]
                    bankf1 = None,  # [<None => no bank,[<'All', list of F1 (Col) entities>]> ]
                    verbose = True, # [<None, True> => print useful information and messages]
                    )

    """
    # For backward compatibility
    if condcoord_ is not None:
        condcoord = condcoord_
        
    if verbose is True:
        print 'create_data() is working...\n'

    # Run utility
    create_data_out = dmn.utils._create_data(locals())

    if verbose is True:
        print '\ncreate_data() is done.'
        print 'Contains:\n',create_data_out.keys(),'\n'

    return create_data_out




##########################################################################

def TopDamon(data,  # [file/path name or array name]
             recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
             collabels = [1,0,'S60'], # [[nheaders4cols,key4cols,colkeytype]]
             getcols = None,   # [<None,{'Get':_,'Labels':_,'Cols':_}> => select cols using extract() syntax]
             labelcols = None,   # [<None,int nheaders4rows,[list label columns by key]> => they become rowlabels]
             rename = None,  # [<None,{'Field1':'NewField1','Field2':'NewField2,...}> ]
             key4rows = ['Auto'],   # [<[<'MyID','Auto'>,rowkeytype,<'warn_dups','stop_dups',omitted>]> => name of field containing unique IDs, ID type]
             getrows = None,   # [<None,{'Get':_,'Labels':_,'Rows':_}> => select rows using extract() syntax]
             validchars = None,   # [<None,[<'All','Cols'>,<[vc's],{col vc's}>,<'Num','Guess','SkipCheck',{'nanval':n},omitted>]
             delimiter = '\t',  # [Character to delimit input file columns (e.g. ',' for .csv and '\t' for text tab-delimited files)]
             pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
             verbose = True,    # [<None,True> => print run-time messages]
             ):
    """
    Returns
    -------
        TopDamon() initializes and returns a Damon object
        consisting of specified rows and columns extracted
        from a raw data file.

        Damon() can be used to format the TopDamon() output
        further, if needed.  However, in most cases the
        Damon object initialized by TopDamon will be
        sufficient.

        Workflow
        --------
            my_obj_temp = TopDamon(...)

            # This step is optional
            my_obj = core.Damon(data = my_obj_temp.data_out,
                                format_ = 'datadict_whole',
                                etc.
                                )

            my_obj.standardize(...)
            etc.


    Comments
    --------
        TopDamon is an alternative to Damon for creating
        Damon objects, which is why it is capitalized
        like the Damon class though technically it is
        a function.  While Damon gives more control,
        TopDamon handles messier datasets.  The end
        result, in either case, is a Damon object to which
        Damon methods can be applied.

        Many datasets are not ready to be initialized
        as Damon objects because they are not organized as
        arrays with rowlabels to the left, collabels at the top,
        and coredata for the core of the array.  They may also
        lack unique IDs.  TopDamon addresses this type of
        data.  In many cases, it can serve as a replacement
        to Damon, and is often easier to use.  At the least,
        it can prepare datasets for further refinement
        in Damon.

        In the latter case, the process of creating a usable
        Damon object consists of two steps:

            1.  PrelimData = TopDamon(RawData,...)
                to get data into a preliminary RowLabel/ColLabel/
                coredata format.

            2.  core.Damon(PrelimData,format_='datadict_whole',...)
                to get data finalized for analysis.

        To deal with the possiblity that no column of unique
        row keys exists, TopDamon() adds a leading column of
        unique integer row identifiers.  You may or may not
        specify this column to be the "final" row keys using
        the key4rows = ['Auto'] option.  If ['Auto'] is not
        specified, TopDamon will remove the leading integers
        from the final Damon object.

        Example:  you have a column of student IDs
        where each student appears twice, once for a math
        test, once for science.  Damon() would ordinarily force
        you to rename the duplicate IDs. However, TopDamon()
        assigns leading unique IDs to each row to avoid this,
        and allows you the option of selecting only those rows
        corresponding to the science test.  Since this has the
        effect of removing the duplicate student IDs, you can
        now designate (using the 'key4rows' parameter) the
        StudentID column as the container of unique row identifiers.
        You can also designate the new leading column to be the
        unique row identifiers using key4rows = ['Auto'].

        Where cells are designated "not-a-number", they are
        converted to a nanval value of -999.

        For a fuller explanation of some of the parameters,
        check out:

            >>>  help(core.Damon.__init__)
            >>>  help(core.Damon.extract)

    Arguments
    ---------
        "data" is either the text file or a path name containing
        the data to be read in, or it is a numpy array.

            file = 'MyData.csv'
            file = '/Users/Me/Scripts/MyData.csv'
            file = MyArray

        Text files and arrays are the only input format supported
        by TopDamon().  The function automatically figures
        out whether data is an array or file.

        -------------
        "recode" is used to recode any number of values in one or more
        ranges that you specify.  It consists of a dictionary of ranges,
        where each range is described using Python's slice notation and
        each contains a recode dictionary.  The syntax is:

        recode = {0:[[slice(StartRow,EndRow),slice(StartCol,EndCol),{From:To,...}],
                  1:[[slice(StartRow,EndRow),slice(StartCol,EndCol),{From:To,...}],
                  ... any number of ranges ...
                  }

        recode = {0:[[slice(3,5),slice(None,None)],{'A':'a','B':'b'}],
                  1:[[slice(None,10),slice(5,6)],{'D':'d','E':'e'}]
                  }

        This means, the 0th range goes from row 3 up to (but not including)
        row 5, and from the left-most column to the right-most column ("None"
        means "go to the end").  Within that range, recode 'A' to 'a' and
        'B' to 'b'.

        Range 1 runs from all rows up to (but not including) row 10 and
        from column 5 up to (but not including) column 6 (in other words,
        just column 5).  Within range 1, recode 'D' to 'd' and 'E' to 'e'.

        Every range must be labeled by an integer, starting with 0.  They
        are recoded in that order.

        recode is applied to the dataset BEFORE anything else is done
        to it -- before a leading column of integers is added, before
        columns are shifted.  So the ranges should correspond to the
        raw dataset.

        -------------
        "collabels" describes the range of column labels of the
        source file.  The syntax is:

            collabels = [nheaders4cols,key4cols,colkeytype]

        Example
            collabels = [3,1,int]
                                =>  This means there are three rows
                                    of column labels.  The second
                                    row (row 1, counting from 0) contains
                                    the unique column identifiers.  The
                                    type of these unique column identifiers
                                    is "int" (integer).

                                    Note that the column identifiers
                                    must be unique or Damon() will
                                    force them to be unique.

        If the data does not have collumn labels, integer labels will
        be added:

            collabels = [0,None,None]
                                =>  Integer column labels will be added.
                                    If the third value (key type) is None,
                                    it will be cast to 'S60'.

        -------------
        "getcols" specifies the columns you want to extract for the
        final Damon object.  It uses the syntax of the extract() method
        to designate columns for extraction, which provides lots of
        options.  To see the full list of extraction options, see:

            >>>  help(core.Damon.extract)

        Simple examples:

            getcols = None      =>  Retain all columns as they are.

            getcols = {'Get':'NoneExcept','Labels':'key','Cols':['Item3','Item4']}
                                =>  Extract none of the columns except those
                                    labeled 'Item3' and 'Item4'.

            getcols = {'Get':'NoneExcept','Labels':1,'Cols':['Flagged']}
                                =>  Extract none of the columns except those
                                    that contain the word 'Flagged' in row
                                    1 (counting from 0).

        IMPORTANT:  Note that getcols (and getrows) is applied
        to the data array AFTER a leading column of integers
        has been inserted and BEFORE renaming (see "rename" below).
        Also, BEFORE labelcols have been shifted to the left.
        When referring to column keys, use their original names.
        Avoid referencing columns or rows using index numbers
        if you can, as it's easy to get confused.

        -------------
        "labelcols" is a list of column names (key IDs) corresponding
        to just those columns that are wanted for labeling, or it is
        an integer defining a number of left-most columns.  These will
        become a contiguous rowlabels array, even if the columns are
        not contiguous in the source data. List them in the order you
        wish them to appear.

        TopDamon automatically adds the column name given in the key4rows
        specification to labelcols in case it is omitted.


            labelcols = None    =>  None of the columns will be
                                    used as rowlabels except the
                                    automatically inserted leading
                                    column of integer row IDs.

            labelcols = 4       =>  The four columns to the left,
                                    after extracting the columns in
                                    getcols, and not including the
                                    added integer column, will be
                                    used as rowlabels.

            labelcols = ['StudentID','Grade','Gender']
                                =>  The columns labeled 'StudentID',
                                    'Grade', and 'Gender' will be
                                    moved to the left of the array,
                                    in that order, and together
                                    with a leading column of integers
                                    will constitute the rowlabels
                                    portion of the array.

        Note that labelcols should contain only the original
        column names, not their renamed versions (see "rename" below).

        -------------
        "rename" is a dictionary used to rename the column and row names
        in the collabels and rowlabels section of the array.

            rename = {'StudentID':'MyID','Grade':'Level'}
                                =>  rename 'StudentID' to 'MyID'.
                                    rename 'Grade' to 'Level'

        Note that labelcols and getcols should only contain
        original field names, not renamed fields.  If key4rows
        (see below) refers to a renamed field, it should
        use the new name.
        
        Also note that you might get bitten if the same key shows
        up in both rowlabels and collabels.  As a matter of principle,
        keys should be unique not just within a facet but across
        facets.
        
        -------------
        "key4rows" designates the (renamed) name of the column
        that contains unique row identifiers, as well as
        their type.  'Auto' makes the TopDamon()-inserted leading
        column of unique integers the key for rows.  key4rows
        also controls the handling of duplicate keys (which are
        not allowed), both for row keys and column keys. Syntax:

        key4rows =  [<'MyID','Auto'>,key type,<'warn_dups','stop_dups',omitted>]


        [NOTE:  key4rows no longer accepts int rowkeytypes.  They will
        automatically converted to string.  So ignore the references to
        "int" below.]


            key4rows = ['MyID',int]
                                =>  Make the column labeled 'MyID'
                                    (after renaming) the holder of
                                    unique row IDs.  Cast them as
                                    integers where appropriate.
                                    Do not check for row (or column)
                                    key duplicates.

            key4rows = ['MyID',int,'warn_dups']
                                =>  'MyID' holds the row IDs and
                                    Damon will look for duplicate
                                    keys.  If it finds any, it will
                                    rename them to remove duplicates
                                    and issue a warning to that effect.

            key4rows = ['MyID',int,'stop_dups']
                                =>  The same as above, but if any
                                    duplicates are found the function
                                    will return an error and a message
                                    regarding which keys have been
                                    duplicated.

            key4rows = ['Auto'] =>  Make the leading column of integers
                                    (inserted automatically by
                                    TopDamon() ) the holder of
                                    uniqe row IDs.  The type is
                                    automatically set to integer.
                                    No duplicate checking is necessary.

                                    This option allows loading datasets
                                    without an existing column of
                                    unique row identifiers, or where
                                    the row identifiers contain
                                    duplicates.

            Note that regardless of the key type (int, string, etc.)
            the keys will take on the same type as the data array
            as a whole.  The key type is applied when Damon extracts
            the keys for lookups and other purposes.

        -------------
        "getrows" specifies the rows you want to extract for the
        final Damon.  It uses the syntax of the extract() method
        to designate rows for extraction, which provides lots of
        options.  To see the full list of extraction options, see:

            >>>  help(core.Damon.extract)

        Examples:

            getrows = None      =>  Retain all rows as they are.

            getrows = {'Get':'NoneExcept','Labels':'Grade','Rows':['3','4']}
                                =>  Extract none of the rows except those
                                    corresponding to grades 3 and 4 as marked
                                    in the column labeled 'Grade'.

            getrows = {'Get':'NoneExcept','Labels':'Subject','Rows':['Math']}
                                =>  Extract none of the rows except those
                                    whose 'Subject' is 'Math'.

            getrows = {'Get':'NoneExcept','Labels':2,'Rows':['Math']}
                                =>  Extract none of the rows except those
                                    labeled 'Math' in column 1 (counting
                                    from 0) AFTER a column of integer IDs
                                    have been added.

        IMPORTANT:  Note that getrows (and getcols) is applied
        to the data array AFTER a leading column of integers
        has been inserted and BEFORE LabelsCols have been shifted
        to the left. Avoid referencing columns or rows using
        index numbers if you can, as it's easy to get confused.
        Much safer to refer to string IDs.

        -------------
        "validchars" is used to specify a list or range of valid
        characters for the array as a whole or for individual columns.
        All other characters are converted to nanval. An optional
        nanval definition can be added to validchars (default is -999)

        When specifying columns, use their final names -- AFTER renaming.
        For complete documentation:

            >>>  import damon1.core as dmn
            >>>  help(dmn.Damon.__init__)

        Syntax:
            [<'All','Cols'>,
            <[vc's],{col vc's}>,
            <'Num','Guess','SkipCheck',{'nanval':n},omitted>
            ]

        Examples:

            validchars = None   =>  All cell values are considered
                                    valid.

            validchars = ['All',[0,1,2,3],'Guess']
                                =>  After converting to nanval all values
                                    that are not 0,1,2,3, Damon will try to
                                    figure out the validchars parameter for
                                    each column by examining
                                    the first 500 rows.  If the array
                                    contains a mix of strings and integers,
                                    it is possible that the integers have
                                    been converted to str(floats), e.g.
                                    '1.0','2.0'.  Make sure to specify
                                    accordingly in the validchars list
                                    or they will be converted to nanval.

            validchars = ['All',[0,1,2,3],'Num',{'nanval':-1000}]
                                =>  Valid responses for all columns
                                    are 0,1,2,3 .  The 'Num' flag
                                    specifies that the array as
                                    a whole will be cast as float,
                                    not string.  Any alpha characters
                                    that cannot be converted to numbers,
                                    plus any invalid values, will
                                    be replaced by -1000.

            validchars = ['All',['a','b','c']]
                                =>  Valid responses for all
                                    columns are 'a','b','c'.
                                    Invalid responses will be
                                    converted by default to -999.

            validchars = ['Cols', {'ID1':['a','b'],'ID2':['All'],
                                   'ID3':['1.2 -- 3.5'], 'ID4':['0 -- ']}]

                                =>  For the 'ID1' column, valid
                                    responses are 'a' and 'b'.

                                    For the 'ID2' column, all
                                    values are allowed (e.g.
                                    negative infinity to positive
                                    infinity.

                                    For the 'ID3' column, all decimals
                                    ranging continuously from 1.2 up
                                    to and including 3.5 are valid.
                                    Note the dash: space-dash-dash-space

                                    For the 'ID4' column, all integers
                                    from 0 to infinity are valid.  Note
                                    the use of the decimal point.  When
                                    used, the range is continuous.  When
                                    omitted, the range consists only of
                                    whole number integers.


        -------------
        "delimiter" is the character used to delimit columns in the
        input text file, generally comma or tab.  If the file contains
        bracked list data, e.g., [1,2,3], comma-delimited runs into
        problems.  Tab-delimited is generally better.

            delimiter = '\t'    =>  Columns are separated by tabs
                                    (generally a .txt file).

            delimiter = ','     =>  Columns are separated by commas
                                    (generally a .csv file).

        -------------
        "pytables" is used to convert large files into "hd5" files
        using the pytables package.  This helps conserve memory.

            pytables = None     =>  Do not use pytables.

            pytables = 'MyData.hd5'
                                =>  file or pathname (string) that
                                    will hold the data in hd5 format.
                                    "hd5" stands for "hierarchical
                                    data format, version 5".

            For more information about pytables, consult the
            Damon docs.


        -------------
        "verbose" <None, True> causes messages based on the output
        Damon to be printed during run-time.

    Examples
    --------

        [under construction]


    Paste function
    --------------
        TopDamon(data,  # [file/path name or array name]
                 recode = None, # [<None,{0:[[slice(StartRow,EndRow),slice(StartCol,EndCol)],{RecodeFrom:RecodeTo,...}],...}>]
                 collabels = [1,0,'S60'], # [[nheaders4cols,key4cols,colkeytype]]
                 getcols = None,   # [<None,{'Get':_,'Labels':_,'Cols':_}> => select cols using extract() syntax]
                 labelcols = None,   # [<None,int nheaders4rows,[list label columns by key]> => they become rowlabels]
                 rename = None,  # [<None,{'Field1':'NewField1','Field2':'NewField2,...}> ]
                 key4rows = ['Auto'],   # [<[<'MyID','Auto'>,rowkeytype,<'warn_dups','stop_dups',omitted>]> => name of field containing unique IDs, ID type]
                 getrows = None,   # [<None,{'Get':_,'Labels':_,'Rows':_}> => select rows using extract() syntax]
                 validchars = None,   # [<None,[<'All','Cols'>,<[vc's],{col vc's}>,<'Num','Guess','SkipCheck',{'nanval':n},omitted>]
                 delimiter = '\t',  # [Character to delimit input file columns (e.g. ',' for .csv and '\t' for text tab-delimited files)]
                 pytables = None,    # [<None,'filename.hd5'> => Name of .hd5 file to hold Damon outputs]
                 verbose = True,    # [<None,True> => print run-time messages]
                 )

    """
    if verbose is True:
        print 'Building Damon object with TopDamon...\n'

    # Get TopDamon outputs
    TopDamon_out = dmn.utils._TopDamon(locals())

    if verbose is True:
        print 'Damon object has been built.'
        print 'Contains:\n',TopDamon_out.__dict__.keys(),'\n'

    return TopDamon_out
